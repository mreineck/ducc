ACLOCAL_AMFLAGS = -I m4

lib_LTLIBRARIES = libmrutil.la

libmrutil_la_SOURCES = \
  mr_util/sharp/sharp.cc \
  mr_util/sharp/sharp_almhelpers.cc \
  mr_util/sharp/sharp_core.cc \
  mr_util/sharp/sharp_geomhelpers.cc \
  mr_util/sharp/sharp_ylmgen.cc \
  mr_util/sharp/sharp_internal.h \
  mr_util/math/math_utils.h \
  mr_util/math/cmplx.h \
  mr_util/infra/aligned_array.h \
  mr_util/infra/simd.h \
  mr_util/math/pointing.cc \
  mr_util/math/pointing.h \
  mr_util/math/vec3.h \
  mr_util/math/constants.h \
  mr_util/infra/error_handling.h \
  mr_util/math/space_filling.cc \
  mr_util/math/space_filling.h \
  mr_util/math/fft1d.h \
  mr_util/math/fft.h \
  mr_util/math/gl_integrator.h \
  mr_util/infra/mav.h \
  mr_util/infra/string_utils.cc \
  mr_util/infra/string_utils.h \
  mr_util/infra/system.cc \
  mr_util/infra/system.h \
  mr_util/infra/threading.cc \
  mr_util/infra/threading.h \
  mr_util/infra/timers.h \
  mr_util/math/unity_roots.h \
  mr_util/infra/useful_macros.h \
  mr_util/math/rangeset.h \
  mr_util/geom_utils.h \
  mr_util/math/geom_utils.cc \
  mr_util/healpix/healpix_tables.h \
  mr_util/healpix/healpix_tables.cc \
  mr_util/healpix/healpix_base.h \
  mr_util/healpix/healpix_base.cc \
  mr_util/healpix/healpix_map.h \
  mr_util/healpix/healpix_map.cc

# format is "current:revision:age"
# any change: increase revision
# any interface change: increase current, revision=0
# any backward-compatible change: increase age
# any backward-incompatible change: age=0
# ==> age <= current
libmrutil_la_LDFLAGS = -version-info 0:0:0 -lpthread

AM_CXXFLAGS = @AM_CXXFLAGS@

if HAVE_MULTIARCH

libavx_la_SOURCES = mr_util/sharp/sharp_core_inc.cc
libavx2_la_SOURCES = mr_util/sharp/sharp_core_inc.cc
libfma_la_SOURCES = mr_util/sharp/sharp_core_inc.cc
libfma4_la_SOURCES = mr_util/sharp/sharp_core_inc.cc
libavx512f_la_SOURCES = mr_util/sharp/sharp_core_inc.cc

noinst_LTLIBRARIES = libavx.la libavx2.la libfma.la libfma4.la libavx512f.la

libavx_la_CXXFLAGS = ${AM_CXXFLAGS} -mavx -DARCH=avx
libavx2_la_CXXFLAGS = ${AM_CXXFLAGS} -mavx2 -DARCH=avx2
libfma_la_CXXFLAGS = ${AM_CXXFLAGS} -mfma -DARCH=fma
libfma4_la_CXXFLAGS = ${AM_CXXFLAGS} -mfma4 -DARCH=fma4
libavx512f_la_CXXFLAGS = ${AM_CXXFLAGS} -mavx512f -DARCH=avx512f

libmrutil_la_LIBADD = libavx.la libavx2.la libfma.la libfma4.la libavx512f.la

endif

nobase_include_HEADERS = \
  mr_util/sharp/sharp.h \
  mr_util/sharp/sharp_geomhelpers.h \
  mr_util/sharp/sharp_almhelpers.h

EXTRA_DIST = test/test_libsharp.sh test/test_space_filling.sh

check_PROGRAMS = sharp2_testsuite space_filling_test hpxtest
sharp2_testsuite_SOURCES = test/sharp2_testsuite.cc
sharp2_testsuite_LDADD = libmrutil.la
space_filling_test_SOURCES = test/space_filling_test.cc
space_filling_test_LDADD = libmrutil.la
hpxtest_SOURCES = test/hpxtest.cc
hpxtest_LDADD = libmrutil.la

TESTS = test/test_libsharp.sh test/test_space_filling.sh

pkgconfigdir = $(libdir)/pkgconfig
nodist_pkgconfig_DATA = @PACKAGE_NAME@.pc

DISTCLEANFILES=@PACKAGE_NAME@.pc @PACKAGE_NAME@.pc.in  @PACKAGE_NAME@-uninstalled.pc  @PACKAGE_NAME@-uninstalled.sh
