#pragma once

#include <array>
#include <cstdint>

namespace hdf5_plotter {

enum class PlotColormap {
    Turbo,
    RedBlue,
    Leone,
};

const char *plot_colormap_name(PlotColormap colormap);

std::array<std::uint8_t, 4> plot_colormap_rgba(PlotColormap colormap, float normalized_value);

float normalize_colormap_value(float value, float min_value, float max_value, bool logarithmic);

} // namespace hdf5_plotter
