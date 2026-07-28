#pragma once

#include "filter_axis.h"

#include <atomic>
#include <cstddef>
#include <vector>

namespace hdf5_plotter {

struct SavitzkyGolayParameters {
    int window_size = 11;
    int polynomial_order = 3;
    int derivative_order = 0;
    FilterAxis axis = FilterAxis::X;
};

struct SavitzkyGolayOutput {
    std::vector<float> values;
    float value_min = 0.0f;
    float value_max = 1.0f;
};

std::size_t savitzky_golay_axis_length(std::size_t width, std::size_t height, FilterAxis axis);

SavitzkyGolayOutput apply_savitzky_golay(const std::vector<float> &input, std::size_t width, std::size_t height,
                                         const SavitzkyGolayParameters &parameters, unsigned max_threads,
                                         const std::atomic<bool> *cancel_requested = nullptr);

} // namespace hdf5_plotter
