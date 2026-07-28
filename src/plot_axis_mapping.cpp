#include "plot_axis_mapping.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace hdf5_plotter {

std::size_t preview_source_index(std::size_t preview_index, std::size_t stride,
                                 std::size_t source_count) {
    if (source_count == 0) {
        return 0;
    }
    stride = std::max<std::size_t>(1, stride);
    if (preview_index > std::numeric_limits<std::size_t>::max() / stride) {
        return source_count - 1;
    }
    return std::min(preview_index * stride, source_count - 1);
}

PreviewAxisMapping make_preview_axis_mapping(const std::vector<double> &sampled_values,
                                             std::size_t preview_count, std::size_t stride,
                                             std::size_t source_count) {
    PreviewAxisMapping mapping;
    mapping.min = 0.0;
    mapping.max = preview_count > 0
                      ? static_cast<double>(
                            preview_source_index(preview_count - 1, stride, source_count))
                      : 1.0;

    if (sampled_values.size() != preview_count || sampled_values.empty()) {
        return mapping;
    }

    bool finite_increasing = std::isfinite(sampled_values.front());
    for (std::size_t i = 1; finite_increasing && i < sampled_values.size(); ++i) {
        finite_increasing = std::isfinite(sampled_values[i]) &&
                            sampled_values[i] >= sampled_values[i - 1];
    }
    if (finite_increasing && sampled_values.front() != sampled_values.back()) {
        mapping.min = sampled_values.front();
        mapping.max = sampled_values.back();
        mapping.direct_values = true;
        return mapping;
    }

    mapping.labels_from_values = true;
    return mapping;
}

double axis_label_value_at_coordinate(const std::vector<double> &values, double coordinate,
                                      double axis_min, double axis_max) {
    if (values.empty()) {
        return coordinate;
    }
    if (values.size() == 1 || axis_min == axis_max) {
        return values.front();
    }

    const double t = (coordinate - axis_min) / (axis_max - axis_min);
    const double clamped = std::clamp(std::isfinite(t) ? t : 0.0, 0.0, 1.0);
    const std::size_t index = static_cast<std::size_t>(
        std::llround(clamped * static_cast<double>(values.size() - 1)));
    return values[std::min(index, values.size() - 1)];
}

} // namespace hdf5_plotter
