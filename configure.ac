
AC_INIT([sangnom], [1.0], [https://bitbucket.org/James1201/vapoursynth-sangnom])

AM_INIT_AUTOMAKE([foreign subdir-objects])
AM_SILENT_RULES([yes])

LT_INIT([disable-static win32-dll])

AC_PROG_CXX

PKG_CHECK_MODULES([VapourSynth], [vapoursynth])

AC_CANONICAL_HOST

X86="false"
PPC="false"
ARM="false"

AS_CASE(
    [$host_cpu],
    [x86_64],   [X86="true"],
    [powerpc*], [PPC="true"],
    [arm*],     [ARM="true"]
)

AS_IF(
    [test "x$X86" = "xtrue"],
    [
        AC_DEFINE([VS_TARGET_CPU_X86])

        AC_SUBST([MFLAGS], ["-mfpmath=sse -msse2"])
    ]
)

AS_IF(
    [test "x$PPC" = "xtrue"],
    [
        AC_DEFINE([VS_TARGET_CPU_POWERPC])
    ]
)

AS_IF(
    [test "x$ARM" = "xtrue"],
    [
        AC_DEFINE([VS_TARGET_CPU_ARM])
    ]
)

AC_CONFIG_FILES([Makefile])

AC_OUTPUT
