project(
  'kissfft',
  'c',
  version: '131.2.0',
  default_options: [
    'c_std=c99',
    'warning_level=1',
  ],
)

cc = meson.get_compiler('c')
math_dep = cc.find_library('m', required: false)
kissfft_include = include_directories('.')

kissfft_library = static_library(
  'kissfft',
  files(
    'kiss_fft.c',
    'kiss_fftr.c',
  ),
  include_directories: kissfft_include,
  dependencies: math_dep,
)

kissfft_dep = declare_dependency(
  link_with: kissfft_library,
  include_directories: kissfft_include,
  dependencies: math_dep,
)
