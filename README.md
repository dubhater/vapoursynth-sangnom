# SangNom - VapourSynth SangNom #

*   SangNom is a single field deinterlacer using edge-directed interpolation but nowadays it's mainly used in anti-aliasing scripts.
*   This is a rewrite version, the algorithm is a bit different from the original AVISynth SangNom2.
*   Note if you want to use the old version, just compile the sangnom_old.cpp, which uses the old algorithm and support 8...16 bit int.

## Build ##

*   compiler with c++11 support

## Usage ##

    sangnom.SangNom(src, order=1, aa=48, planes=[0, 1, 2])

*   the default setting, interpolates bottom field for all planes.
***


## Parameter ##

    sangnom.SangNom(clip clip[, int order=1, int aa=48, int[] planes=[0, 1, 2]])

*   clip: the src clip
    *   8..16 bit integer support, 32 bit integer support
    *   all colorfamily support
    *   note: 8 bit input has sse support, others don't

***
*   order: order of deinterlacing
    *   default: 1
        *   0:  double frame rate, must call DoubleWeave() before processing
        *   1:  single frame rate, keep top field
        *   2:  single frame rate, keep bottom field

***
*   aa: the strength of anti-aliasing, this value is considered in 8 bit clip
    *   default: 48
    *   range: 0 ... 128
    *   note: don't use a too low value, it will produce crappy result, the default value is good enough

***
*   planes: planes which are processed
    *   default: all planes

***

## License ##

    SangNom

    a rewrite version of AVISynth SangNom2



    Original Author: Victor Efimov

    Copyright (c) 2013 Victor Efimov
    This project is licensed under the MIT license. Binaries are GPL v2.
