# HDF5 ImPlot Viewer

HDF5 ImPlot Viewer is a fast, read-only desktop application for attosecond and
ultrafast spectroscopy data. Follow spectral features while a measurement is
still running, compare delay scans and spectrograms side-by-side, and apply
lightweight smoothing or differentiation and inspect FFTs as the data arrives.
It is built for the moment when the pulse is short, the beamtime is shorter,
and you need to know whether the signal is really there.

Files are opened locally and are never uploaded or modified.

## Download

Ready-to-run packages are available from the
[latest GitHub release](https://github.com/martinyrm/hdf5-viewer/releases/latest).

| System | Download |
| --- | --- |
| Apple Silicon Mac | `hdf5-imgui-plotter_<version>_macos-arm64.zip` |
| Intel Mac | `hdf5-imgui-plotter_<version>_macos-x86_64.zip` |
| Debian 12 | `hdf5-imgui-plotter_<version>_debian12_amd64.deb` |
| Debian 13 | `hdf5-imgui-plotter_<version>_debian13_amd64.deb` |
| 64-bit Windows | `hdf5-imgui-plotter_<version>_windows-x86_64.zip` |

### macOS

Unzip the download and open **HDF5 ImPlot Viewer.app**. The application is not
currently notarized, so macOS may block the first launch. Right-click the app
and choose **Open**, or allow it in **System Settings > Privacy & Security**.

### Debian 12/13

Install the downloaded package; `apt` will install its runtime dependencies:

```sh
sudo apt install ./hdf5-imgui-plotter_*_debian12_amd64.deb
# On Debian 13, use the corresponding *_debian13_amd64.deb file.
```

Start it by running `hdf5_plotter`.

### Windows

Extract the complete zip and run `hdf5_plotter.exe` inside the extracted
folder. Keep the accompanying DLL files beside the executable.

## Quick Start

1. Start the viewer and press `O`, or click **Open File**.
2. Select an HDF5 file. The picker previews its first `comment` or `comments`
   field before opening it.
3. Expand groups in the dataset tree, or type part of a known dataset name in
   the search box.
4. Select a numeric dataset to plot it.
5. Drag plot tabs or windows to dock them side-by-side. Each open file has its
   own browser tab and plot windows.

The dataset browser starts on the left and plots start on the right. Focusing a
plot also brings its corresponding file tab to the front.

## Supported Data

- Numeric scalar datasets are displayed as values.
- Numeric 1D datasets are shown as high-contrast line plots.
- Numeric 2D datasets are shown as Turbo-colormap image or spectrogram plots.
- String datasets named `comment` or `comments` can be shown as captions above
  plots. **Show plot captions** is enabled by default and uses the top-most
  matching comment.
- Compatible 1D numeric datasets can supply physical X or Y coordinates.
  Fixed-length numeric array datatypes, such as a `Wavelengths` array, are also
  recognized as axis candidates.

The viewer currently plots scalar, 1D, and 2D numeric datasets. Use **File
Details** to inspect datatype, dimensions, storage layout, compression filters,
chunking, and the active 2D preview plan.

## Axes, Scaling, and Navigation

- After the initial fit, **Auto X** is disabled so the mouse can zoom and pan
  immediately. Disable **Auto Y** to zoom or pan vertically.
- The **Scaling** panel provides manual X and Y limits. For 2D plots, color
  minimum and maximum have independent automatic settings and manual values.
- Use the **X axis** or **Y axis** selector to replace index coordinates with a
  compatible dataset. Values stay in their original HDF5 index order.
- Enable **Sort X ascending** when an ascending physical axis is required. The
  plotted values are reordered together with the selected X coordinates.
- Use **Flip X/Y** to transpose a 2D display.
- Unix timestamp-like X coordinates can be displayed as local date/time or UTC.
- Descending and non-monotonic 2D coordinate datasets are shown as labels over
  source-index coordinates. Applying an axis does not silently reverse the
  image.

## Colormaps and Publication Style

Open **Appearance** beside the **Filters** menu to configure plot presentation:

- **Turbo** is the default sequential colormap.
- **Red-Blue Diverging** uses a white midpoint for signed or differential data.
- **Leone** follows a deep-blue, cyan, yellow, coral, and white spectroscopy
  palette.
- **Log colour scale** applies logarithmic mapping to positive data and a
  signed-log mapping when the selected range crosses zero.
- **Publication plot style** switches plot regions to black text and axes on a
  white background using the bundled CMU Serif font. The surrounding controls
  remain in the normal dark interface.

Appearance changes are global and recolor loaded plots without re-reading the
HDF5 files or changing axis and color limits.

## Live SWMR Measurements

Enable **Live SWMR refresh** in a file tab to follow a file written using HDF5
Single Writer Multiple Reader mode.

- The file is checked approximately once per second.
- The selected plot reloads when that dataset's extent changes.
- The previous plot remains visible while new data is read.
- Axis autoscale choices and manual color limits are retained across refreshes.
- Press `Space` to pause or resume live refresh for the active file.

The acquisition software must create and flush the file in HDF5 SWMR mode.
Ordinary HDF5 files can still be viewed normally with live refresh disabled.

## 2D Slices

Enable **Slice plots** for a 2D dataset only when slices are needed. Separate,
dockable row and column plots will be created.

- Drag the row or column guide on the heatmap to update a slice.
- Enter a row or column index directly for precise selection.
- Slice plot limits follow the visible, zoomed region of the 2D plot.
- Dock slice windows beside the source heatmap for comparison.

## Display Filters

Filters are non-destructive and run in the background. The original HDF5 data
is not changed, and the previous result remains visible while a new result is
calculated.

Open the **Filters** menu to use:

- **Savitzky-Golay**: smoothing or differentiation with live window,
  polynomial-order, derivative-order, and X/Y-axis controls.
- **Windowed Average**: a centered moving average along X or Y. Edge windows
  use available samples and non-finite values are ignored.
- **FFT**: centered full-spectrum magnitude, power, phase, real, or imaginary
  output with selectable window function, mean removal, decibel scaling, and
  automatic or manual sample spacing.

The filter settings window shows stages in execution order. Use the up and down
arrow buttons to reorder them; enabled stages run from top to bottom.

## Keyboard Shortcuts

Shortcuts apply to the active file or plot when a text field is not being
edited.

| Key | Action |
| --- | --- |
| `O`, `Ctrl+O`, or `Cmd+O` | Open a file |
| `X` | Toggle X autoscaling |
| `Y` | Toggle Y autoscaling |
| `F` | Flip X/Y for a 2D plot |
| `Space` | Pause or resume live SWMR refresh |

## Large Datasets and Performance

- Large 1D plots retain the loaded data but render a cached min/max envelope
  appropriate for the current zoom level.
- Large 2D datasets are sampled directly from HDF5 to fit the GPU texture limit
  and a conservative cell budget. **File Details** reports the source shape,
  displayed shape, and row/column sampling stride.
- 2D slices and filters operate on this sampled display preview, not on every
  element of an oversized source dataset. Use a dedicated analysis pipeline
  when full-resolution numerical results are required.
- CPU work is capped to four threads by default. Set
  `HDF5_PLOTTER_MAX_THREADS=1` before starting the viewer to minimize CPU use on
  acquisition computers.
- Hidden or minimized windows skip OpenGL rendering. Unfocused windows use a
  reduced update rate while background loading and SWMR checks continue.

Enable **Performance HUD** to see frame timing and the active OpenGL vendor,
renderer, and version. On Linux, `llvmpipe` or `softpipe` means software
rendering is being used instead of the GPU.

For Linux graphics diagnostics:

```sh
sudo apt install mesa-utils
glxinfo -B
```

## Troubleshooting

**A compatible axis dataset is missing**

Check that it is numeric and contains the same number of values as the displayed
X or Y dimension. Candidates in the same HDF5 group are preferred.

**A live file does not update**

Confirm that live refresh is enabled and not paused, that the writer opened the
file in SWMR mode, and that it flushes dataset extent changes.

**Linux performance is unexpectedly poor**

Open **Performance HUD** and check the OpenGL renderer. If it reports
`llvmpipe`, install or repair the Mesa/vendor GPU driver. Debian systems
normally require `libgl1-mesa-dri`.

**A very large 2D plot has fewer displayed points than the source**

This is the expected sampled preview. Open **File Details** to see the exact
sampling stride.

## Building from Source

Prebuilt releases are recommended for laboratory users. To build locally,
install the platform dependencies first.

macOS with Homebrew:

```sh
brew install meson ninja pkg-config sdl2 hdf5 freetype
```

Debian 12/13:

```sh
sudo apt install build-essential git meson ninja-build pkg-config \
  libsdl2-dev libhdf5-dev libgl1-mesa-dev libfreetype-dev fonts-dejavu-core
```

Windows builds use the MSYS2 UCRT64 environment:

```sh
pacman -S --needed git zip \
  mingw-w64-ucrt-x86_64-gcc \
  mingw-w64-ucrt-x86_64-meson \
  mingw-w64-ucrt-x86_64-ninja \
  mingw-w64-ucrt-x86_64-pkgconf \
  mingw-w64-ucrt-x86_64-SDL2 \
  mingw-w64-ucrt-x86_64-hdf5 \
  mingw-w64-ucrt-x86_64-freetype
```

Configure and build an optimized binary:

```sh
meson setup build-release --buildtype=release
meson compile -C build-release
meson test -C build-release --print-errorlogs
```

Run it with a file path or start without one and use the file picker:

```sh
./build-release/hdf5_plotter measurement.h5
```

Dear ImGui, ImPlot, and KISS FFT are pinned Meson subprojects and are downloaded
automatically during the first setup. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
for their license notices.

Release packages are built by GitHub Actions for macOS arm64, macOS x86_64,
Debian 12, Debian 13, and Windows x86_64 whenever a `v*` tag is pushed. Maintainer
instructions are in [docs/releasing.md](docs/releasing.md).
