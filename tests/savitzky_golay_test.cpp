#include "savitzky_golay.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

bool nearly_equal(float lhs, float rhs, float tolerance = 1e-3f) {
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

    SavitzkyGolayParameters parameters;
    parameters.window_size = 5;
    parameters.polynomial_order = 2;

    const std::vector<float> constant(25, 4.5f);
    const SavitzkyGolayOutput smoothed = apply_savitzky_golay(constant, constant.size(), 1, parameters, 2);
    for (float value : smoothed.values) {
        if (!check(nearly_equal(value, 4.5f), "smoothing did not preserve a constant signal")) {
            return 1;
        }
    }

    std::vector<float> quadratic(25);
    for (std::size_t i = 0; i < quadratic.size(); ++i) {
        quadratic[i] = static_cast<float>(i * i);
    }
    parameters.derivative_order = 1;
    const SavitzkyGolayOutput first_derivative =
        apply_savitzky_golay(quadratic, quadratic.size(), 1, parameters, 2);
    for (std::size_t i = 2; i + 2 < quadratic.size(); ++i) {
        if (!check(nearly_equal(first_derivative.values[i], static_cast<float>(2 * i), 2e-3f),
                   "first derivative did not reproduce a quadratic")) {
            return 1;
        }
    }

    parameters.derivative_order = 2;
    const SavitzkyGolayOutput second_derivative =
        apply_savitzky_golay(quadratic, quadratic.size(), 1, parameters, 2);
    for (std::size_t i = 2; i + 2 < quadratic.size(); ++i) {
        if (!check(nearly_equal(second_derivative.values[i], 2.0f, 2e-3f),
                   "second derivative did not reproduce a quadratic")) {
            return 1;
        }
    }

    std::vector<float> image(7 * 9);
    for (std::size_t row = 0; row < 7; ++row) {
        for (std::size_t column = 0; column < 9; ++column) {
            image[row * 9 + column] = static_cast<float>(row * row);
        }
    }
    parameters.axis = FilterAxis::Y;
    const SavitzkyGolayOutput y_derivative = apply_savitzky_golay(image, 9, 7, parameters, 2);
    for (std::size_t row = 2; row + 2 < 7; ++row) {
        for (std::size_t column = 0; column < 9; ++column) {
            if (!check(nearly_equal(y_derivative.values[row * 9 + column], 2.0f, 2e-3f),
                       "Y-axis filtering did not operate down image columns")) {
                return 1;
            }
        }
    }

    parameters.axis = FilterAxis::X;
    std::atomic<bool> cancel_requested{true};
    bool cancelled = false;
    try {
        (void)apply_savitzky_golay(constant, constant.size(), 1, parameters, 2, &cancel_requested);
    } catch (const std::runtime_error &) {
        cancelled = true;
    }
    if (!check(cancelled, "a cancelled filter calculation did not stop")) {
        return 1;
    }

    return 0;
}
