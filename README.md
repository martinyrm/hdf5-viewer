# HDF5 ImPlot Viewer

A small C++17 desktop viewer for the HDF5 files in `stability_analyses/`.
It uses SDL2 + OpenGL, Dear ImGui, ImPlot, and HDF5 through Meson.

## Dependencies

macOS with Homebrew:

```sh
brew install meson ninja pkg-config sdl2 hdf5 freetype
```

Debian 12/13:

```sh
sudo apt install build-essential git meson ninja-build pkg-config libsdl2-dev libhdf5-dev libgl1-mesa-dev libfreetype-dev fonts-dejavu-core
```

Dear ImGui and ImPlot are pinned as Meson wrap subprojects and are downloaded
on the first `meson setup`. ImGui uses the `v1.90.9-docking` branch tag so the
app can use Dear ImGui docking with the ImPlot 0.16 API.

## Build and Run

```sh
meson setup build
meson compile -C build
./build/hdf5_plotter stability_analyses/1777392253_delay_analysis.h5
```

If no file is passed, the app tries to open
`stability_analyses/1777392253_delay_analysis.h5`.

## Plot Behavior

- Scalar datasets are displayed as text.
- Numeric 1D datasets are shown as line plots.
- Numeric 2D datasets are shown as Turbo-colored spectrogram/image plots.
- Multiple files can be opened at once. Each file gets its own tab in the
  dataset browser and its own dockable plot window.
- On startup the dataset browser is docked on the left and plot windows are
  docked on the right; focus a plot to bring its matching file tab forward.
- Use **Open File** for the built-in file picker, or edit the path in a tab and
  press **Reload**.
- Enable **Live SWMR refresh** in a file tab to poll an active SWMR writer and
  reload the selected plot when that dataset's extent changes.
- **Show plot captions** is enabled by default and draws a small caption above
  each plot from the top-most string dataset named `comment` or `comments`.
- Axis ranges are inferred from same-group 1D datasets when dimensions match,
  for example `freqs` for rows and `selected_spectrum_indices` or `timestamps`
  for columns.
- The **X axis** selector can force index/column coordinates or replace them
  with any compatible 1D dataset in the same group.
- For 2D datasets, **Flip X/Y** transposes the display so rows are shown on X
  and columns on Y. The **Y axis** selector can also replace row coordinates
  with a compatible 1D dataset. Scalar HDF5 array datatypes, such as a
  fixed-length `Wavelengths` array, are treated as 1D numeric axis datasets.
- The **Scaling** panel exposes auto/manual X and Y ranges for every plot, plus
  auto/manual color limits for 2D datasets.
- **Auto X** starts disabled after the initial data fit so mouse zooming and
  panning work immediately. Disable **Auto Y** to manually zoom Y as well; the
  edited range fields stay linked to the current plot view.
- Unix timestamp-like X axes can be formatted as local date/time or UTC.
- Line plots use a fixed high-contrast palette against the dark UI.
- Dear ImGui is built with FreeType when `freetype2` is available, and the app
  loads a platform TrueType font.

## Performance Notes

- HDF5 dataset reads run on a worker thread so the UI remains responsive.
  Access to the HDF5 C API is serialized because common HDF5 builds are not
  thread-safe.
- SWMR refresh is opt-in and polls once per second. The poll checks metadata in
  SWMR read mode and only starts a full plot reload when the selected dataset
  shape changes. The previous plot remains visible while the refresh loads, and
  the current X/Y/color ranges are preserved rather than re-autoscaled.
- Large 1D lines are cached as a viewport-aware min/max envelope, so zoomed-out
  views do not push millions of vertices through ImPlot each frame.
- The render loop caps at about 60 FPS even when the platform OpenGL driver does
  not provide vsync, avoiding runaway CPU usage while plots are idle.
- 2D datasets are sampled to a GPU texture capped by the current OpenGL maximum
  texture size and a conservative cell budget. Turbo color mapping is computed
  in parallel on CPU, then uploaded once as an RGBA texture and drawn with
  `ImPlot::PlotImage`.
- ImPlot itself is immediate-mode and CPU-driven for plot geometry. The main
  GPU acceleration here is texture-backed rendering for 2D datasets.
