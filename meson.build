project('SangNom', 'cpp',
        version: '40',
        default_options: ['cpp_std=c++14', 'buildtype=release'],
        meson_version: '>=0.46')


warnings = [
  '-Wall',
  '-Wextra',
  '-Wshadow',
]

cflags = [
  warnings,
]

ldflags = [
]


host_cpu_family = host_machine.cpu_family()

if host_cpu_family.startswith('x86')
  cflags += ['-mfpmath=sse', '-msse2', '-DVS_TARGET_CPU_X86=1']
endif


host_system = host_machine.system()

if host_system == 'windows' or host_system == 'cygwin'
  if host_cpu_family == 'x86'
    cflags += '-mstackrealign'
    ldflags += '-Wl,--kill-at'
  endif
endif


cxx = meson.get_compiler('cpp')


sources = [
  'src/sangnom.cpp',
]

deps = [
  dependency('vapoursynth').partial_dependency(includes: true, compile_args: true),
]

shared_module('sangnom',
              sources,
              dependencies: deps,
              link_args: ldflags,
              cpp_args: cflags,
              install: true)
