#include "fft_transform.h"

#include <kiss_fft.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>

namespace hdf5_plotter {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

class KissFftConfig {
  public:
    explicit KissFftConfig(int size) : config_(kiss_fft_alloc(size, 0, nullptr, nullptr)) {
        if (config_ == nullptr) {
            throw std::runtime_error("failed to allocate FFT workspace");
        }
    }

    KissFftConfig(const KissFftConfig &) = delete;
    KissFftConfig &operator=(const KissFftConfig &) = delete;
    ~KissFftConfig() { kiss_fft_free(config_); }

    kiss_fft_cfg get() const { return config_; }

  private:
    kiss_fft_cfg config_ = nullptr;
};

double window_weight(FftWindow window, std::size_t index, std::size_t count) {
    if (window == FftWindow::Rectangular || count <= 2) {
        return 1.0;
    }
    const double phase = 2.0 * kPi * static_cast<double>(index) / static_cast<double>(count - 1);
    switch (window) {
    case FftWindow::Rectangular:
        return 1.0;
    case FftWindow::Hann:
        return 0.5 - 0.5 * std::cos(phase);
    case FftWindow::Hamming:
        return 0.54 - 0.46 * std::cos(phase);
    case FftWindow::Blackman:
        return 0.42 - 0.5 * std::cos(phase) + 0.08 * std::cos(2.0 * phase);
    }
    return 1.0;
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

float spectral_value(const kiss_fft_cpx &value, FftValueMode mode, bool decibels, double normalization) {
    const double real = static_cast<double>(value.r) * normalization;
    const double imaginary = static_cast<double>(value.i) * normalization;
    double output = 0.0;
    switch (mode) {
    case FftValueMode::Magnitude:
        output = std::hypot(real, imaginary);
        if (decibels) {
            output = 20.0 * std::log10(std::max(output, 1e-12));
        }
        break;
    case FftValueMode::Power:
        output = real * real + imaginary * imaginary;
        if (decibels) {
            output = 10.0 * std::log10(std::max(output, 1e-24));
        }
        break;
    case FftValueMode::Phase:
        output = std::atan2(imaginary, real);
        break;
    case FftValueMode::Real:
        output = real;
        break;
    case FftValueMode::Imaginary:
        output = imaginary;
        break;
    }
    return static_cast<float>(output);
}

} // namespace

FftOutput apply_fft_transform(const std::vector<float> &input, std::size_t width, std::size_t height,
                              const FftParameters &parameters, double sample_spacing, unsigned max_threads,
                              const std::atomic<bool> *cancel_requested) {
    if (width == 0 || height == 0 || width > std::numeric_limits<std::size_t>::max() / height ||
        input.size() != width * height) {
        throw std::invalid_argument("FFT input dimensions do not match the data");
    }
    const std::size_t transform_size = parameters.axis == FilterAxis::X ? width : height;
    const std::size_t line_count = parameters.axis == FilterAxis::X ? height : width;
    if (transform_size < 2 || transform_size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("FFT axis length must be between 2 and INT_MAX");
    }
    if (!std::isfinite(sample_spacing) || sample_spacing <= 0.0) {
        throw std::invalid_argument("FFT sample spacing must be positive");
    }

    KissFftConfig config(static_cast<int>(transform_size));
    std::vector<double> window(transform_size);
    double coherent_gain = 0.0;
    for (std::size_t i = 0; i < transform_size; ++i) {
        window[i] = window_weight(parameters.window, i, transform_size);
        coherent_gain += window[i];
    }
    coherent_gain /= static_cast<double>(transform_size);
    if (std::abs(coherent_gain) < 1e-12) {
        coherent_gain = 1.0;
    }
    const double normalization = 1.0 / (static_cast<double>(transform_size) * coherent_gain);

    FftOutput result;
    result.values.resize(input.size());
    result.frequency_values.resize(transform_size);
    const long long center = static_cast<long long>(transform_size / 2);
    const double frequency_step = 1.0 / (static_cast<double>(transform_size) * sample_spacing);
    for (std::size_t i = 0; i < transform_size; ++i) {
        result.frequency_values[i] = static_cast<double>(static_cast<long long>(i) - center) * frequency_step;
    }

    const std::size_t worker_count =
        std::max<std::size_t>(1, std::min<std::size_t>({std::max(1u, max_threads), line_count,
                                                        input.size() / 262144 + 1}));
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
        const std::size_t begin_line = line_count * worker / worker_count;
        const std::size_t end_line = line_count * (worker + 1) / worker_count;
        workers.emplace_back([&, begin_line, end_line]() {
            std::vector<kiss_fft_cpx> time_values(transform_size);
            std::vector<kiss_fft_cpx> frequency_values(transform_size);
            for (std::size_t line = begin_line; line < end_line; ++line) {
                if (cancel_requested != nullptr && cancel_requested->load(std::memory_order_relaxed)) {
                    return;
                }

                double mean = 0.0;
                std::size_t finite_count = 0;
                if (parameters.remove_mean) {
                    for (std::size_t sample = 0; sample < transform_size; ++sample) {
                        const std::size_t source_index = parameters.axis == FilterAxis::X
                                                             ? line * width + sample
                                                             : sample * width + line;
                        const float value = input[source_index];
                        if (std::isfinite(value)) {
                            mean += static_cast<double>(value);
                            ++finite_count;
                        }
                    }
                    mean = finite_count > 0 ? mean / static_cast<double>(finite_count) : 0.0;
                }

                for (std::size_t sample = 0; sample < transform_size; ++sample) {
                    const std::size_t source_index = parameters.axis == FilterAxis::X
                                                         ? line * width + sample
                                                         : sample * width + line;
                    const float source_value = input[source_index];
                    const double value = std::isfinite(source_value) ? static_cast<double>(source_value) - mean : 0.0;
                    time_values[sample].r = static_cast<kiss_fft_scalar>(value * window[sample]);
                    time_values[sample].i = 0;
                }
                kiss_fft(config.get(), time_values.data(), frequency_values.data());

                for (std::size_t bin = 0; bin < transform_size; ++bin) {
                    const std::size_t shifted_bin = (bin + transform_size / 2) % transform_size;
                    const std::size_t destination_index = parameters.axis == FilterAxis::X
                                                              ? line * width + shifted_bin
                                                              : shifted_bin * width + line;
                    result.values[destination_index] =
                        spectral_value(frequency_values[bin], parameters.value_mode, parameters.decibels,
                                       normalization);
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

    const auto [min_value, max_value] = finite_minmax(result.values);
    result.value_min = min_value;
    result.value_max = max_value;
    return result;
}

} // namespace hdf5_plotter
