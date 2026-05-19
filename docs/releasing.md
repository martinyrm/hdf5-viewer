# Releasing HDF5 ImPlot Viewer

Releases are built by GitHub Actions when a tag named `v*` is pushed. The
workflow produces:

- `hdf5-imgui-plotter_<version>_macos-arm64.zip`
- `hdf5-imgui-plotter_<version>_macos-x86_64.zip`
- `hdf5-imgui-plotter_<version>_debian12_amd64.deb`
- `hdf5-imgui-plotter_<version>_debian13_amd64.deb`

Manual, non-release artifact builds can also be started from the Actions tab
with the **Release** workflow's `workflow_dispatch` button.

## Runtime Dependencies

The macOS zip contains an unsigned `.app` bundle with the Homebrew HDF5, SDL2,
FreeType, and related non-system dylibs copied into `Contents/Frameworks`.
macOS still supplies OpenGL, libc++, and the system frameworks. Because the app
is not notarized, users may need to allow it from **System Settings > Privacy &
Security** the first time they open it.

Debian 12:

```sh
sudo apt install libc6 libstdc++6 libgcc-s1 libsdl2-2.0-0 libgl1 libfreetype6 fonts-dejavu-core libhdf5-103-1
```

Debian 13:

```sh
sudo apt install libc6 libstdc++6 libgcc-s1 libsdl2-2.0-0 libgl1 libfreetype6 fonts-dejavu-core libhdf5-310
```

For hardware-accelerated OpenGL on Mesa systems, also ensure the relevant driver
stack is installed, usually:

```sh
sudo apt install libgl1-mesa-dri
```

Vendor NVIDIA/AMD/Intel packages may replace or supplement the Mesa driver
stack on lab machines with dedicated GPUs.

## Publishing A Release

1. Update `meson.build`'s project version if the release should carry a new
   application version.
2. Commit the release changes.
3. Tag the commit:

   ```sh
   git tag v0.1.0
   git push origin main --tags
   ```

4. The **Release** workflow builds on GitHub-hosted macOS 26 arm64,
   macOS 26 Intel, Debian 12, and Debian 13 runners.
5. When all build jobs pass, the workflow creates a GitHub Release and attaches
   the zip/deb artifacts plus linker diagnostic text files.

GitHub's `macos-latest` label currently points at Apple Silicon and can migrate
over time, so the workflow uses explicit `macos-26` and `macos-26-intel` labels
for reproducible architecture coverage.
