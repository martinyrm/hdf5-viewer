project(
  'implot',
  'cpp',
  default_options: [
    'cpp_std=c++17',
    'warning_level=1',
  ],
)

imgui_dep = dependency('imgui')
implot_inc = include_directories('.')
implot_sources = files(
  'implot.cpp',
  'implot_items.cpp',
)

implot_lib = static_library(
  'implot',
  implot_sources,
  include_directories: implot_inc,
  dependencies: [imgui_dep],
)

implot_dep = declare_dependency(
  link_with: implot_lib,
  include_directories: implot_inc,
  dependencies: [imgui_dep],
)
