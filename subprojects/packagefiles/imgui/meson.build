project(
  'imgui',
  'cpp',
  default_options: [
    'cpp_std=c++17',
    'warning_level=1',
  ],
)

cpp = meson.get_compiler('cpp')

sdl2_dep = dependency('sdl2')
opengl_dep = dependency('opengl', required: false)
if not opengl_dep.found() and host_machine.system() == 'darwin'
  opengl_dep = dependency('appleframeworks', modules: ['OpenGL'])
endif
freetype_dep = dependency('freetype2', required: false)
dl_dep = cpp.find_library('dl', required: false)

imgui_inc = include_directories('.', 'backends')
imgui_sources = files(
  'imgui.cpp',
  'imgui_draw.cpp',
  'imgui_tables.cpp',
  'imgui_widgets.cpp',
  'backends/imgui_impl_opengl3.cpp',
  'backends/imgui_impl_sdl2.cpp',
)
imgui_deps = [sdl2_dep, opengl_dep, dl_dep]
imgui_args = []
if freetype_dep.found()
  imgui_sources += files('misc/freetype/imgui_freetype.cpp')
  imgui_deps += freetype_dep
  imgui_args += '-DIMGUI_ENABLE_FREETYPE'
endif

imgui_lib = static_library(
  'imgui',
  imgui_sources,
  include_directories: imgui_inc,
  dependencies: imgui_deps,
  cpp_args: imgui_args,
)

imgui_dep = declare_dependency(
  link_with: imgui_lib,
  include_directories: imgui_inc,
  dependencies: imgui_deps,
  compile_args: imgui_args,
)
