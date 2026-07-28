#pragma once

#include <cstddef>
#include <vector>

namespace hdf5_plotter {

struct PreviewAxisMapping {
    double min = 0.0;
    double max = 1.0;
    bool direct_values = false;
    bool labels_from_values = false;
};

std::size_t preview_source_index(std::size_t preview_index, std::size_t stride,
                                 std::size_t source_count);

PreviewAxisMapping make_preview_axis_mapping(const std::vector<double> &sampled_values,
                                             std::size_t preview_count, std::size_t stride,
                                             std::size_t source_count);

double axis_label_value_at_coordinate(const std::vector<double> &values, double coordinate,
                                      double axis_min, double axis_max);

} // namespace hdf5_plotter
