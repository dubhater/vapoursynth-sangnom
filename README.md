# SangNom - VapourSynth SangNom #

*   SangNom is a single field deinterlacer using edge-directed interpolation but nowadays it's mainly used in anti-aliasing scripts.
*   This is a rewrite version of AVISynth SangNom2, and support more formats.

## Build ##

*   compiler with c++11 support
*   CPU with SSE2 support

## Usage ##

    sangnom.SangNom(src, order=1, aa=48, algo=0, planes=[0, 1, 2])

*   the default setting, interpolates bottom field for all planes.
***


## Parameter ##

    sangnom.SangNom(clip clip[, int order=1, int aa=48, int algo=0, int[] planes=[0, 1, 2]])

*   clip: the src clip
    *   8..16 bit integer support, 32 bit float support
    *   all colorfamily support

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
    *   note: in AVISynth SangNom2, the default is aa=48, aac=0, but in this plugin default aa=48 for all planes

***
*   algo: the algorithm which to use
    *   default: 0
    *   0: the orignal one which compute the same result as AVISynth SangNom2
    *   1: the modified one which should be more accurate but much slower on performance

***
*   planes: planes which are processed
    *   default: all planes

***

## License ##

    SangNom

    a rewrite version of AVISynth SangNom2


    Copyright (c) 2016 james1201
    Copyright (c) 2013 Victor Efimov

    This project is licensed under the MIT license. Binaries are GPL v2.
