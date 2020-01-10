ACLOCAL_AMFLAGS = -I m4

lib_LTLIBRARIES = libmrutil.la

libmrutil_la_SOURCES = \
  libsharp2/sharp.cc \
  libsharp2/sharp_almhelpers.cc \
  libsharp2/sharp_core.cc \
  libsharp2/sharp_geomhelpers.cc \
  libsharp2/sharp_ylmgen.cc \
  libsharp2/sharp_internal.h \
  libsharp2/sharp_ylmgen.h \
  mr_util/math_utils.h \
  mr_util/cmplx.h \
  mr_util/error_handling.cc \
  mr_util/error_handling.h \
  mr_util/fft.h \
  mr_util/gl_integrator.h \
  mr_util/mav.h \
  mr_util/string_utils.cc \
  mr_util/string_utils.h \
  mr_util/system.cc \
  mr_util/system.h \
  mr_util/threading.cc \
  mr_util/threading.h \
  mr_util/timers.h \
  mr_util/unity_roots.h \
  mr_util/useful_macros.h

# format is "current:revision:age"
# any change: increase revision
# any interface change: increase current, revision=0
# any backward-compatible change: increase age
# any backward-incompatible change: age=0
# ==> age <= current
libmrutil_la_LDFLAGS = -version-info 0:0:0 -lpthread

AM_CXXFLAGS = @AM_CXXFLAGS@

if HAVE_MULTIARCH

libavx_la_SOURCES = libsharp2/sharp_core_inc.cc
libavx2_la_SOURCES = libsharp2/sharp_core_inc.cc
libfma_la_SOURCES = libsharp2/sharp_core_inc.cc
libfma4_la_SOURCES = libsharp2/sharp_core_inc.cc
libavx512f_la_SOURCES = libsharp2/sharp_core_inc.cc

noinst_LTLIBRARIES = libavx.la libavx2.la libfma.la libfma4.la libavx512f.la

libavx_la_CXXFLAGS = ${AM_CXXFLAGS} -mavx -DARCH=avx
libavx2_la_CXXFLAGS = ${AM_CXXFLAGS} -mavx2 -DARCH=avx2
libfma_la_CXXFLAGS = ${AM_CXXFLAGS} -mfma -DARCH=fma
libfma4_la_CXXFLAGS = ${AM_CXXFLAGS} -mfma4 -DARCH=fma4
libavx512f_la_CXXFLAGS = ${AM_CXXFLAGS} -mavx512f -DARCH=avx512f

libmrutil_la_LIBADD = libavx.la libavx2.la libfma.la libfma4.la libavx512f.la

endif

nobase_include_HEADERS = \
  libsharp2/sharp.h \
  libsharp2/sharp_geomhelpers.h \
  libsharp2/sharp_almhelpers.h

EXTRA_DIST = runtest.sh

check_PROGRAMS = sharp2_testsuite
sharp2_testsuite_SOURCES = test/sharp2_testsuite.cc
sharp2_testsuite_LDADD = libmrutil.la

TESTS = runtest.sh

pkgconfigdir = $(libdir)/pkgconfig
nodist_pkgconfig_DATA = @PACKAGE_NAME@.pc

DISTCLEANFILES=@PACKAGE_NAME@.pc @PACKAGE_NAME@.pc.in  @PACKAGE_NAME@-uninstalled.pc  @PACKAGE_NAME@-uninstalled.sh
