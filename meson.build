project(
  'hdf5-imgui-plotter',
  'c',
  'cpp',
  version: '0.2.0',
  default_options: [
    'cpp_std=c++17',
    'warning_level=3',
  ],
)

cpp = meson.get_compiler('cpp')

threads_dep = dependency('threads')
sdl2_dep = dependency('sdl2')
opengl_dep = dependency('opengl', required: false)
if not opengl_dep.found()
  if host_machine.system() == 'darwin'
    opengl_dep = dependency('appleframeworks', modules: ['OpenGL'])
  elif host_machine.system() == 'windows'
    opengl_dep = cpp.find_library('opengl32')
  endif
endif

hdf5_dep = dependency('hdf5', method: 'pkg-config', required: false)
if not hdf5_dep.found()
  hdf5_dep = dependency('hdf5-serial', method: 'pkg-config', required: false)
endif
if not hdf5_dep.found()
  error('HDF5 was not found. Install hdf5 with Homebrew on macOS, libhdf5-dev on Debian, or mingw-w64-ucrt-x86_64-hdf5 on Windows/MSYS2.')
endif

imgui_dep = dependency('imgui', fallback: ['imgui', 'imgui_dep'])
implot_dep = dependency('implot', fallback: ['implot', 'implot_dep'])
kissfft_dep = dependency('kissfft', fallback: ['kissfft', 'kissfft_dep'])

executable(
  'hdf5_plotter',
  files(
    'src/main.cpp',
    'src/fft_transform.cpp',
    'src/plot_axis_mapping.cpp',
    'src/savitzky_golay.cpp',
    'src/windowed_average.cpp',
  ),
  dependencies: [
    hdf5_dep,
    imgui_dep,
    implot_dep,
    kissfft_dep,
    opengl_dep,
    sdl2_dep,
    threads_dep,
  ],
  install: true,
)

savitzky_golay_test = executable(
  'savitzky_golay_test',
  files(
    'tests/savitzky_golay_test.cpp',
    'src/savitzky_golay.cpp',
  ),
  include_directories: include_directories('src'),
  dependencies: threads_dep,
)

test('savitzky-golay', savitzky_golay_test)

windowed_average_test = executable(
  'windowed_average_test',
  files(
    'tests/windowed_average_test.cpp',
    'src/windowed_average.cpp',
  ),
  include_directories: include_directories('src'),
  dependencies: threads_dep,
)

test('windowed-average', windowed_average_test)

fft_transform_test = executable(
  'fft_transform_test',
  files(
    'tests/fft_transform_test.cpp',
    'src/fft_transform.cpp',
  ),
  include_directories: include_directories('src'),
  dependencies: [
    kissfft_dep,
    threads_dep,
  ],
)

test('fft-transform', fft_transform_test)

plot_axis_mapping_test = executable(
  'plot_axis_mapping_test',
  files(
    'tests/plot_axis_mapping_test.cpp',
    'src/plot_axis_mapping.cpp',
  ),
  include_directories: include_directories('src'),
)

test('plot-axis-mapping', plot_axis_mapping_test)
