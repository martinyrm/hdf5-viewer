#include "windowed_average.h"

#include <atomic>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

bool nearly_equal(float left, float right, float tolerance = 1e-5f) {
    return std::abs(left - right) <= tolerance;
}

void expect_values(const std::vector<float> &actual, const std::vector<float> &expected) {
    assert(actual.size() == expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index) {
        assert(nearly_equal(actual[index], expected[index]));
    }
}

} // namespace

int main() {
    using namespace hdf5_plotter;

    WindowedAverageParameters parameters;
    parameters.window_size = 3;
    parameters.axis = FilterAxis::X;
    const WindowedAverageOutput line =
        apply_windowed_average({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, 5, 1, parameters, 2);
    expect_values(line.values, {1.5f, 2.0f, 3.0f, 4.0f, 4.5f});

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const WindowedAverageOutput non_finite =
        apply_windowed_average({1.0f, nan, 3.0f, nan, 5.0f}, 5, 1, parameters, 1);
    expect_values(non_finite.values, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    const std::vector<float> grid = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
    };
    const WindowedAverageOutput rows = apply_windowed_average(grid, 3, 3, parameters, 3);
    expect_values(rows.values, {
                                    1.5f, 2.0f, 2.5f,
                                    4.5f, 5.0f, 5.5f,
                                    7.5f, 8.0f, 8.5f,
                                });

    parameters.axis = FilterAxis::Y;
    const WindowedAverageOutput columns = apply_windowed_average(grid, 3, 3, parameters, 3);
    expect_values(columns.values, {
                                       2.5f, 3.5f, 4.5f,
                                       4.0f, 5.0f, 6.0f,
                                       5.5f, 6.5f, 7.5f,
                                   });

    const WindowedAverageOutput constant =
        apply_windowed_average(std::vector<float>(25, 7.0f), 5, 5, parameters, 4);
    for (float value : constant.values) {
        assert(nearly_equal(value, 7.0f));
    }

    std::atomic<bool> cancelled{true};
    bool cancellation_threw = false;
    try {
        apply_windowed_average(grid, 3, 3, parameters, 1, &cancelled);
    } catch (const std::runtime_error &) {
        cancellation_threw = true;
    }
    assert(cancellation_threw);

    bool even_window_threw = false;
    try {
        parameters.window_size = 2;
        apply_windowed_average(grid, 3, 3, parameters, 1);
    } catch (const std::invalid_argument &) {
        even_window_threw = true;
    }
    assert(even_window_threw);

    bool large_window_threw = false;
    try {
        parameters.window_size = 5;
        apply_windowed_average(grid, 3, 3, parameters, 1);
    } catch (const std::invalid_argument &) {
        large_window_threw = true;
    }
    assert(large_window_threw);
}
