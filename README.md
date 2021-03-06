# SangNom - VapourSynth Single Field Deinterlacer #

*   SangNom is a single field deinterlacer using edge-directed interpolation but nowadays it's mainly used in anti-aliasing scripts.
*   This is a rewrite version of AviSynth SangNom2, and support more formats.

## Build ##

*   compiler with c++11 support
*   -DVS_TARGET_CPU_X86 can enable the SSE2 code path which has a large benefit on performance
***

    ./autogen.sh
    ./configure
    make
    make install

Or

    meson build && cd build
    ninja

*   build on linux
***

    g++ src/sangnom.cpp -o libsangnom.dll -std=c++1y -O3 -shared -static -I/path/to/vapoursynth/headers -DVS_TARGET_CPU_X86

*   build on windows
***

## Usage ##

    sangnom.SangNom(src, order=1, aa=48, planes=[0, 1, 2])

*   the default setting, interpolates bottom field for all planes.
***

    sangnom.SangNom(src, order=1, aa=[48, 0, 0], planes=[0, 1, 2])

*   this simulates the default setting of AviSynth SangNom2
***


## Parameter ##

    sangnom.SangNom(clip clip[, int order=1, bint dh=False, int[] aa=[48, 48, 48], int[] planes=[0, 1, 2]])

*   clip: the src clip
    *   8..16 bit integer, 32 bit float support
    *   all color family support

***
*   order: order of deinterlacing
    *   default: 1 (int)
        *   0:  double frame rate, must call std.SeparateFields().std.DoubleWeave() before processing
        *   1:  single frame rate, keep top field
        *   2:  single frame rate, keep bottom field

***
*   dh: doubles the height of the input. each line of the input is copied to every other line of the output and the missing lines are interpolated.
    *   default: False (bool)
        *   note: if dh=True, it will force all planes to be processed.

***
*   aa: the strength of anti-aliasing, this value is considered in 8 bit clip
    *   default: [48, 48, 48]  (int[])
    *   range: 0 ... 128
    *   note: the value of previous plane will be used if you don't specify it.
    *   note: in AviSynth SangNom2, the default is aa=[48, 0, 0]

***
*   planes: planes which are processed
    *   default: [0, 1, 2]
        *   note: if dh=True, it will force all planes to be processed.

***

## License ##

    SangNom - VapourSynth Single Field Deinterlacer

    Copyright (c) 2016 james1201
    Copyright (c) 2013 Victor Efimov

    This project is licensed under the MIT license. Binaries are GPL v2.
