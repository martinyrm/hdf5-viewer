project(
  'hdf5-imgui-plotter',
  'c',
  'cpp',
  version: '0.1.0',
  default_options: [
    'cpp_std=c++17',
    'warning_level=3',
  ],
)

cpp = meson.get_compiler('cpp')

threads_dep = dependency('threads')
sdl2_dep = dependency('sdl2')
opengl_dep = dependency('opengl', required: false)
if not opengl_dep.found() and host_machine.system() == 'darwin'
  opengl_dep = dependency('appleframeworks', modules: ['OpenGL'])
endif

hdf5_dep = dependency('hdf5', method: 'pkg-config', required: false)
if not hdf5_dep.found()
  hdf5_dep = dependency('hdf5-serial', method: 'pkg-config', required: false)
endif
if not hdf5_dep.found()
  error('HDF5 was not found. Install hdf5 with Homebrew on macOS, or libhdf5-dev on Debian.')
endif

imgui_dep = dependency('imgui', fallback: ['imgui', 'imgui_dep'])
implot_dep = dependency('implot', fallback: ['implot', 'implot_dep'])

executable(
  'hdf5_plotter',
  files('src/main.cpp'),
  dependencies: [
    hdf5_dep,
    imgui_dep,
    implot_dep,
    opengl_dep,
    sdl2_dep,
    threads_dep,
  ],
  install: true,
)
