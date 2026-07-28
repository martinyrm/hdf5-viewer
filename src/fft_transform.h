#pragma once

#include "filter_axis.h"

#include <atomic>
#include <cstddef>
#include <vector>

namespace hdf5_plotter {

enum class FftValueMode { Magnitude, Power, Phase, Real, Imaginary };
enum class FftWindow { Rectangular, Hann, Hamming, Blackman };

struct FftParameters {
    FilterAxis axis = FilterAxis::X;
    FftValueMode value_mode = FftValueMode::Magnitude;
    FftWindow window = FftWindow::Hann;
    bool remove_mean = true;
    bool decibels = false;
    bool automatic_spacing = true;
    double sample_spacing = 1.0;
};

struct FftOutput {
    std::vector<float> values;
    std::vector<double> frequency_values;
    float value_min = 0.0f;
    float value_max = 1.0f;
};

FftOutput apply_fft_transform(const std::vector<float> &input, std::size_t width, std::size_t height,
                              const FftParameters &parameters, double sample_spacing, unsigned max_threads,
                              const std::atomic<bool> *cancel_requested = nullptr);

} // namespace hdf5_plotter
