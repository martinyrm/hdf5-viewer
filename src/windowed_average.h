#pragma once

#include "filter_axis.h"

#include <atomic>
#include <cstddef>
#include <vector>

namespace hdf5_plotter {

struct WindowedAverageParameters {
    int window_size = 11;
    FilterAxis axis = FilterAxis::X;
};

struct WindowedAverageOutput {
    std::vector<float> values;
    float value_min = 0.0f;
    float value_max = 1.0f;
};

std::size_t windowed_average_axis_length(std::size_t width, std::size_t height, FilterAxis axis);

WindowedAverageOutput apply_windowed_average(const std::vector<float> &input, std::size_t width,
                                             std::size_t height, const WindowedAverageParameters &parameters,
                                             unsigned max_threads,
                                             const std::atomic<bool> *cancel_requested = nullptr);

} // namespace hdf5_plotter
