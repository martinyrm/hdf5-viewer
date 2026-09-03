#include "plot_colormap.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace hdf5_plotter {
namespace {

struct ColorStop {
    float position;
    std::array<float, 3> color;
};

std::uint8_t to_byte(float value) {
    value = std::clamp(value, 0.0f, 1.0f);
    return static_cast<std::uint8_t>(std::lround(value * 255.0f));
}

std::array<std::uint8_t, 4> turbo_rgba(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    const float t2 = t * t;
    const float t3 = t2 * t;
    const float t4 = t3 * t;
    const float t5 = t4 * t;

    const float r = 0.13572138f + 4.61539260f * t - 42.66032258f * t2 + 132.13108234f * t3 -
                    152.94239396f * t4 + 59.28637943f * t5;
    const float g = 0.09140261f + 2.19418839f * t + 4.84296658f * t2 - 14.18503333f * t3 +
                    4.27729857f * t4 + 2.82956604f * t5;
    const float b = 0.10667330f + 12.64194608f * t - 60.58204836f * t2 + 110.36276771f * t3 -
                    89.90310912f * t4 + 27.34824973f * t5;
    return {to_byte(r), to_byte(g), to_byte(b), 255};
}

template <std::size_t Size>
std::array<std::uint8_t, 4> interpolated_rgba(const std::array<ColorStop, Size> &stops, float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    if (t <= stops.front().position) {
        return {to_byte(stops.front().color[0]), to_byte(stops.front().color[1]),
                to_byte(stops.front().color[2]), 255};
    }

    for (std::size_t index = 1; index < stops.size(); ++index) {
        if (t > stops[index].position) {
            continue;
        }
        const ColorStop &left = stops[index - 1];
        const ColorStop &right = stops[index];
        const float span = right.position - left.position;
        const float blend = span > 0.0f ? (t - left.position) / span : 0.0f;
        std::array<std::uint8_t, 4> result{};
        for (std::size_t channel = 0; channel < 3; ++channel) {
            result[channel] = to_byte(left.color[channel] + (right.color[channel] - left.color[channel]) * blend);
        }
        result[3] = 255;
        return result;
    }

    return {to_byte(stops.back().color[0]), to_byte(stops.back().color[1]),
            to_byte(stops.back().color[2]), 255};
}

double signed_log(double value, double linear_threshold) {
    return std::copysign(std::log1p(std::abs(value) / linear_threshold), value);
}

} // namespace

const char *plot_colormap_name(PlotColormap colormap) {
    switch (colormap) {
    case PlotColormap::Turbo:
        return "Turbo";
    case PlotColormap::RedBlue:
        return "Red-Blue Diverging";
    case PlotColormap::Leone:
        return "Leone";
    }
    return "Turbo";
}

std::array<std::uint8_t, 4> plot_colormap_rgba(PlotColormap colormap, float normalized_value) {
    static constexpr std::array<ColorStop, 5> red_blue = {{
        {0.00f, {0.129f, 0.400f, 0.674f}},
        {0.25f, {0.420f, 0.675f, 0.818f}},
        {0.50f, {0.969f, 0.969f, 0.969f}},
        {0.75f, {0.937f, 0.541f, 0.384f}},
        {1.00f, {0.698f, 0.094f, 0.169f}},
    }};
    static constexpr std::array<ColorStop, 8> leone = {{
        {0.00f, {0.018f, 0.012f, 0.420f}},
        {0.16f, {0.000f, 0.100f, 0.720f}},
        {0.32f, {0.000f, 0.750f, 0.950f}},
        {0.48f, {0.720f, 0.950f, 0.650f}},
        {0.62f, {1.000f, 0.910f, 0.280f}},
        {0.77f, {1.000f, 0.320f, 0.230f}},
        {0.90f, {1.000f, 0.720f, 0.670f}},
        {1.00f, {1.000f, 1.000f, 1.000f}},
    }};

    switch (colormap) {
    case PlotColormap::Turbo:
        return turbo_rgba(normalized_value);
    case PlotColormap::RedBlue:
        return interpolated_rgba(red_blue, normalized_value);
    case PlotColormap::Leone:
        return interpolated_rgba(leone, normalized_value);
    }
    return turbo_rgba(normalized_value);
}

float normalize_colormap_value(float value, float min_value, float max_value, bool logarithmic) {
    if (!std::isfinite(value) || !std::isfinite(min_value) || !std::isfinite(max_value)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (min_value > max_value) {
        std::swap(min_value, max_value);
    }
    if (min_value == max_value) {
        return 0.5f;
    }
    if (!logarithmic) {
        return std::clamp((value - min_value) / (max_value - min_value), 0.0f, 1.0f);
    }

    double transformed_min = 0.0;
    double transformed_max = 1.0;
    double transformed_value = 0.0;
    if (min_value > 0.0f) {
        transformed_min = std::log(static_cast<double>(min_value));
        transformed_max = std::log(static_cast<double>(max_value));
        transformed_value = value > 0.0f ? std::log(static_cast<double>(value)) : transformed_min;
    } else if (max_value < 0.0f) {
        transformed_min = -std::log(-static_cast<double>(min_value));
        transformed_max = -std::log(-static_cast<double>(max_value));
        transformed_value = value < 0.0f ? -std::log(-static_cast<double>(value)) : transformed_max;
    } else {
        const double max_abs = std::max(std::abs(static_cast<double>(min_value)),
                                        std::abs(static_cast<double>(max_value)));
        const double linear_threshold = std::max(max_abs * 1e-6, std::numeric_limits<double>::min());
        transformed_min = signed_log(min_value, linear_threshold);
        transformed_max = signed_log(max_value, linear_threshold);
        transformed_value = signed_log(value, linear_threshold);
    }

    const double denominator = transformed_max - transformed_min;
    if (!std::isfinite(denominator) || denominator == 0.0) {
        return 0.5f;
    }
    return static_cast<float>(std::clamp((transformed_value - transformed_min) / denominator, 0.0, 1.0));
}

} // namespace hdf5_plotter
