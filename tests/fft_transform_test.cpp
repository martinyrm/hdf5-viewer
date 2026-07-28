#include "fft_transform.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

bool nearly_equal(double lhs, double rhs, double tolerance = 1e-4) {
    return std::abs(lhs - rhs) <= tolerance;
}

bool check(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
    }
    return condition;
}

} // namespace

int main() {
    using namespace hdf5_plotter;

    constexpr std::size_t count = 64;
    constexpr std::size_t tone_bin = 5;
    std::vector<float> signal(count);
    for (std::size_t i = 0; i < count; ++i) {
        signal[i] = static_cast<float>(std::sin(2.0 * kPi * static_cast<double>(tone_bin * i) /
                                               static_cast<double>(count)));
    }

    FftParameters parameters;
    parameters.window = FftWindow::Rectangular;
    parameters.remove_mean = false;
    const FftOutput spectrum = apply_fft_transform(signal, count, 1, parameters, 0.5, 2);
    const std::size_t center = count / 2;
    if (!check(nearly_equal(spectrum.values[center - tone_bin], 0.5, 2e-4) &&
                   nearly_equal(spectrum.values[center + tone_bin], 0.5, 2e-4),
               "FFT magnitude did not locate the input tone")) {
        return 1;
    }
    if (!check(nearly_equal(spectrum.frequency_values[center + tone_bin],
                            static_cast<double>(tone_bin) / (static_cast<double>(count) * 0.5)),
               "FFT frequency axis used the wrong sample spacing")) {
        return 1;
    }

    std::vector<float> image(count * 3);
    for (std::size_t row = 0; row < count; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            image[row * 3 + column] = signal[row];
        }
    }
    parameters.axis = FilterAxis::Y;
    const FftOutput y_spectrum = apply_fft_transform(image, 3, count, parameters, 1.0, 3);
    for (std::size_t column = 0; column < 3; ++column) {
        if (!check(nearly_equal(y_spectrum.values[(center + tone_bin) * 3 + column], 0.5, 2e-4),
                   "Y-axis FFT did not transform image columns")) {
            return 1;
        }
    }

    parameters.axis = FilterAxis::X;
    parameters.remove_mean = true;
    const std::vector<float> constant(count, 7.0f);
    const FftOutput without_dc = apply_fft_transform(constant, count, 1, parameters, 1.0, 1);
    if (!check(std::abs(without_dc.values[center]) < 1e-5f, "FFT mean removal left a DC component")) {
        return 1;
    }

    std::atomic<bool> cancel_requested{true};
    bool cancelled = false;
    try {
        (void)apply_fft_transform(signal, count, 1, parameters, 1.0, 1, &cancel_requested);
    } catch (const std::runtime_error &) {
        cancelled = true;
    }
    if (!check(cancelled, "a cancelled FFT calculation did not stop")) {
        return 1;
    }

    return 0;
}
