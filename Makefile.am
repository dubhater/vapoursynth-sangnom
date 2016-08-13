
warningflags = -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -Wshadow
commoncflags = -std=c++1y -O3 $(warningflags) $(MFLAGS)
AM_CXXFLAGS = $(commoncflags)
AM_CPPFLAGS = $(VapourSynth_CFLAGS)

lib_LTLIBRARIES = libsangnom.la

libsangnom_la_SOURCES = src/sangnom.cpp
libsangnom_la_CXXFLAGS = $(commoncflags)
