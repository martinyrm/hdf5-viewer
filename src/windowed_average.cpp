#include "windowed_average.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>

namespace hdf5_plotter {
namespace {

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

std::size_t windowed_average_axis_length(std::size_t width, std::size_t height, FilterAxis axis) {
    return axis == FilterAxis::X ? width : height;
}

WindowedAverageOutput apply_windowed_average(const std::vector<float> &input, std::size_t width,
                                             std::size_t height, const WindowedAverageParameters &parameters,
                                             unsigned max_threads, const std::atomic<bool> *cancel_requested) {
    if (width == 0 || height == 0 || width > std::numeric_limits<std::size_t>::max() / height ||
        input.size() != width * height) {
        throw std::invalid_argument("windowed average input dimensions do not match the data");
    }
    const std::size_t axis_length = windowed_average_axis_length(width, height, parameters.axis);
    if (parameters.window_size < 3 || parameters.window_size % 2 == 0 ||
        static_cast<std::size_t>(parameters.window_size) > axis_length) {
        throw std::invalid_argument("window size must be odd, at least 3, and no larger than the filtered axis");
    }

    const std::size_t line_count = parameters.axis == FilterAxis::X ? height : width;
    const std::size_t half_window = static_cast<std::size_t>(parameters.window_size / 2);
    WindowedAverageOutput output;
    output.values.resize(input.size());

    const std::size_t desired_workers = std::max<std::size_t>(
        1, std::min<std::size_t>({std::max(1u, max_threads), line_count, input.size() / 262144 + 1}));
    std::vector<std::thread> workers;
    workers.reserve(desired_workers);
    for (std::size_t worker = 0; worker < desired_workers; ++worker) {
        const std::size_t begin_line = line_count * worker / desired_workers;
        const std::size_t end_line = line_count * (worker + 1) / desired_workers;
        workers.emplace_back([&, begin_line, end_line]() {
            const auto flat_index = [&](std::size_t line, std::size_t position) {
                return parameters.axis == FilterAxis::X ? line * width + position : position * width + line;
            };

            for (std::size_t line = begin_line; line < end_line; ++line) {
                if (cancel_requested != nullptr && cancel_requested->load(std::memory_order_relaxed)) {
                    return;
                }

                double sum = 0.0;
                std::size_t finite_count = 0;
                const std::size_t initial_end = std::min(axis_length - 1, half_window);
                for (std::size_t position = 0; position <= initial_end; ++position) {
                    const float value = input[flat_index(line, position)];
                    if (std::isfinite(value)) {
                        sum += static_cast<double>(value);
                        ++finite_count;
                    }
                }

                for (std::size_t position = 0; position < axis_length; ++position) {
                    if ((position & 16383U) == 0U && cancel_requested != nullptr &&
                        cancel_requested->load(std::memory_order_relaxed)) {
                        return;
                    }
                    output.values[flat_index(line, position)] =
                        finite_count > 0 ? static_cast<float>(sum / static_cast<double>(finite_count))
                                         : std::numeric_limits<float>::quiet_NaN();

                    if (position >= half_window) {
                        const float removed = input[flat_index(line, position - half_window)];
                        if (std::isfinite(removed)) {
                            sum -= static_cast<double>(removed);
                            --finite_count;
                        }
                    }
                    const std::size_t added_position = position + half_window + 1;
                    if (added_position < axis_length) {
                        const float added = input[flat_index(line, added_position)];
                        if (std::isfinite(added)) {
                            sum += static_cast<double>(added);
                            ++finite_count;
                        }
                    }
                }
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
