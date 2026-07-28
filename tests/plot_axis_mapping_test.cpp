#include "plot_axis_mapping.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

bool check(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
    }
    return condition;
}

bool nearly_equal(double lhs, double rhs) {
    return std::abs(lhs - rhs) <= 1e-12;
}

} // namespace

int main() {
    using namespace hdf5_plotter;

    if (!check(preview_source_index(3, 3, 11) == 9,
               "preview index was stretched to the unsampled source endpoint")) {
        return 1;
    }
    if (!check(preview_source_index(99, 3, 11) == 10,
               "preview source index was not clamped")) {
        return 1;
    }

    const PreviewAxisMapping fallback = make_preview_axis_mapping({}, 4, 3, 11);
    if (!check(nearly_equal(fallback.min, 0.0) && nearly_equal(fallback.max, 9.0) &&
                   !fallback.direct_values && !fallback.labels_from_values,
               "fallback preview axis did not use exact sampled source indices")) {
        return 1;
    }

    const std::vector<double> increasing = {10.0, 12.0, 14.0, 16.0};
    const PreviewAxisMapping physical =
        make_preview_axis_mapping(increasing, increasing.size(), 3, 11);
    if (!check(physical.direct_values && !physical.labels_from_values &&
                   nearly_equal(physical.min, 10.0) && nearly_equal(physical.max, 16.0),
               "increasing physical axis was not applied directly")) {
        return 1;
    }

    const std::vector<double> decreasing = {16.0, 14.0, 12.0, 10.0};
    const PreviewAxisMapping index_ordered =
        make_preview_axis_mapping(decreasing, decreasing.size(), 3, 11);
    if (!check(!index_ordered.direct_values && index_ordered.labels_from_values &&
                   nearly_equal(index_ordered.min, 0.0) && nearly_equal(index_ordered.max, 9.0),
               "decreasing axis did not preserve source index order")) {
        return 1;
    }
    if (!check(nearly_equal(axis_label_value_at_coordinate(
                                decreasing, 0.0, index_ordered.min, index_ordered.max),
                            16.0) &&
                   nearly_equal(axis_label_value_at_coordinate(
                                    decreasing, 9.0, index_ordered.min, index_ordered.max),
                                10.0),
               "decreasing labels were reversed relative to source indices")) {
        return 1;
    }

    return 0;
}
