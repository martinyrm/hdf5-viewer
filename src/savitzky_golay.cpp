#include "savitzky_golay.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>

namespace hdf5_plotter {
namespace {

std::size_t reflected_index(long long index, std::size_t count) {
    if (count <= 1) {
        return 0;
    }
    const long long n = static_cast<long long>(count);
    while (index < 0 || index >= n) {
        index = index < 0 ? -index - 1 : 2 * n - index - 1;
    }
    return static_cast<std::size_t>(index);
}

std::vector<double> filter_coefficients(const SavitzkyGolayParameters &parameters) {
    const int window = parameters.window_size;
    const int order = parameters.polynomial_order;
    const int derivative = parameters.derivative_order;
    const int matrix_size = order + 1;
    const int half_window = window / 2;

    std::vector<long double> augmented(static_cast<std::size_t>(matrix_size * matrix_size * 2), 0.0L);
    const int augmented_width = matrix_size * 2;
    for (int row = 0; row < matrix_size; ++row) {
        for (int col = 0; col < matrix_size; ++col) {
            long double sum = 0.0L;
            for (int offset = -half_window; offset <= half_window; ++offset) {
                const long double x = static_cast<long double>(offset) / static_cast<long double>(half_window);
                sum += std::pow(x, row + col);
            }
            augmented[static_cast<std::size_t>(row * augmented_width + col)] = sum;
        }
        augmented[static_cast<std::size_t>(row * augmented_width + matrix_size + row)] = 1.0L;
    }

    for (int pivot = 0; pivot < matrix_size; ++pivot) {
        int best_row = pivot;
        long double best_value =
            std::abs(augmented[static_cast<std::size_t>(pivot * augmented_width + pivot)]);
        for (int row = pivot + 1; row < matrix_size; ++row) {
            const long double value =
                std::abs(augmented[static_cast<std::size_t>(row * augmented_width + pivot)]);
            if (value > best_value) {
                best_value = value;
                best_row = row;
            }
        }
        if (best_value <= std::numeric_limits<long double>::epsilon()) {
            throw std::runtime_error("Savitzky-Golay coefficient matrix is singular");
        }
        if (best_row != pivot) {
            for (int col = 0; col < augmented_width; ++col) {
                std::swap(augmented[static_cast<std::size_t>(pivot * augmented_width + col)],
                          augmented[static_cast<std::size_t>(best_row * augmented_width + col)]);
            }
        }

        const long double divisor = augmented[static_cast<std::size_t>(pivot * augmented_width + pivot)];
        for (int col = 0; col < augmented_width; ++col) {
            augmented[static_cast<std::size_t>(pivot * augmented_width + col)] /= divisor;
        }
        for (int row = 0; row < matrix_size; ++row) {
            if (row == pivot) {
                continue;
            }
            const long double factor = augmented[static_cast<std::size_t>(row * augmented_width + pivot)];
            for (int col = 0; col < augmented_width; ++col) {
                augmented[static_cast<std::size_t>(row * augmented_width + col)] -=
                    factor * augmented[static_cast<std::size_t>(pivot * augmented_width + col)];
            }
        }
    }

    long double derivative_scale = 1.0L;
    for (int value = 2; value <= derivative; ++value) {
        derivative_scale *= static_cast<long double>(value);
    }
    derivative_scale /= std::pow(static_cast<long double>(half_window), derivative);

    std::vector<double> coefficients(static_cast<std::size_t>(window));
    for (int offset = -half_window; offset <= half_window; ++offset) {
        const long double x = static_cast<long double>(offset) / static_cast<long double>(half_window);
        long double coefficient = 0.0L;
        for (int power = 0; power < matrix_size; ++power) {
            const long double inverse =
                augmented[static_cast<std::size_t>(derivative * augmented_width + matrix_size + power)];
            coefficient += inverse * std::pow(x, power);
        }
        coefficients[static_cast<std::size_t>(offset + half_window)] =
            static_cast<double>(derivative_scale * coefficient);
    }
    return coefficients;
}

std::pair<float, float> finite_minmax(const std::vector<float> &values) {
    float min_value = std::numeric_limits<float>::infinity();
    float max_value = -std::numeric_limits<float>::infinity();
    for (float value : values) {
        if (std::isfinite(value)) {
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
        }
    }
    if (!std::isfinite(min_value) || !std::isfinite(max_value)) {
        return {0.0f, 1.0f};
    }
    if (min_value == max_value) {
        const float padding = std::abs(min_value) > 1.0f ? std::abs(min_value) * 0.01f : 1.0f;
        min_value -= padding;
        max_value += padding;
    }
    return {min_value, max_value};
}

} // namespace

std::size_t savitzky_golay_axis_length(std::size_t width, std::size_t height, FilterAxis axis) {
    return axis == FilterAxis::X ? width : height;
}

SavitzkyGolayOutput apply_savitzky_golay(const std::vector<float> &input, std::size_t width, std::size_t height,
                                         const SavitzkyGolayParameters &parameters, unsigned max_threads,
                                         const std::atomic<bool> *cancel_requested) {
    if (width == 0 || height == 0 || width > std::numeric_limits<std::size_t>::max() / height ||
        input.size() != width * height) {
        throw std::invalid_argument("Savitzky-Golay input dimensions do not match the data");
    }
    const std::size_t axis_length = savitzky_golay_axis_length(width, height, parameters.axis);
    if (parameters.window_size < 3 || parameters.window_size % 2 == 0 ||
        static_cast<std::size_t>(parameters.window_size) > axis_length) {
        throw std::invalid_argument("window size must be odd, at least 3, and no larger than the filtered axis");
    }
    if (parameters.polynomial_order < 0 || parameters.polynomial_order >= parameters.window_size) {
        throw std::invalid_argument("polynomial order must be smaller than the window size");
    }
    if (parameters.derivative_order < 0 || parameters.derivative_order > parameters.polynomial_order) {
        throw std::invalid_argument("derivative order must not exceed the polynomial order");
    }

    const std::vector<double> coefficients = filter_coefficients(parameters);
    const int half_window = parameters.window_size / 2;
    SavitzkyGolayOutput output;
    output.values.resize(input.size());

    const std::size_t desired_workers =
        std::max<std::size_t>(1, std::min<std::size_t>(std::max(1u, max_threads), input.size() / 262144 + 1));
    std::vector<std::thread> workers;
    workers.reserve(desired_workers);
    for (std::size_t worker = 0; worker < desired_workers; ++worker) {
        const std::size_t begin = input.size() * worker / desired_workers;
        const std::size_t end = input.size() * (worker + 1) / desired_workers;
        workers.emplace_back([&, begin, end]() {
            for (std::size_t flat_index = begin; flat_index < end; ++flat_index) {
                if ((flat_index & 1023U) == 0U && cancel_requested != nullptr &&
                    cancel_requested->load(std::memory_order_relaxed)) {
                    return;
                }
                const std::size_t row = flat_index / width;
                const std::size_t column = flat_index % width;
                double sum = 0.0;
                double finite_weight = 0.0;
                bool all_finite = true;
                for (int offset = -half_window; offset <= half_window; ++offset) {
                    std::size_t source_row = row;
                    std::size_t source_column = column;
                    if (parameters.axis == FilterAxis::X) {
                        source_column = reflected_index(static_cast<long long>(column) + offset, width);
                    } else {
                        source_row = reflected_index(static_cast<long long>(row) + offset, height);
                    }
                    const float value = input[source_row * width + source_column];
                    const double coefficient = coefficients[static_cast<std::size_t>(offset + half_window)];
                    if (std::isfinite(value)) {
                        sum += coefficient * static_cast<double>(value);
                        finite_weight += coefficient;
                    } else {
                        all_finite = false;
                    }
                }

                if (parameters.derivative_order == 0 && !all_finite && std::abs(finite_weight) > 1e-12) {
                    sum /= finite_weight;
                } else if (!all_finite) {
                    sum = std::numeric_limits<double>::quiet_NaN();
                }
                output.values[flat_index] = static_cast<float>(sum);
            }
        });
    }
    for (std::thread &worker : workers) {
        worker.join();
    }
    if (cancel_requested != nullptr && cancel_requested->load(std::memory_order_relaxed)) {
        throw std::runtime_error("filter update cancelled");
    }

    const auto [min_value, max_value] = finite_minmax(output.values);
    output.value_min = min_value;
    output.value_max = max_value;
    return output;
}

} // namespace hdf5_plotter
