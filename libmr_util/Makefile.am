ACLOCAL_AMFLAGS = -I m4

lib_LTLIBRARIES = libmrutil.la

libmrutil_la_SOURCES = \
  ducc0/sharp/sharp.cc \
  ducc0/sharp/sharp_almhelpers.cc \
  ducc0/sharp/sharp_core.cc \
  ducc0/sharp/sharp_geomhelpers.cc \
  ducc0/sharp/sharp_ylmgen.cc \
  ducc0/sharp/sharp_internal.h \
  ducc0/math/math_utils.h \
  ducc0/math/cmplx.h \
  ducc0/infra/aligned_array.h \
  ducc0/infra/simd.h \
  ducc0/math/pointing.cc \
  ducc0/math/pointing.h \
  ducc0/math/vec3.h \
  ducc0/math/constants.h \
  ducc0/infra/error_handling.h \
  ducc0/math/space_filling.cc \
  ducc0/math/space_filling.h \
  ducc0/math/fft1d.h \
  ducc0/math/fft.h \
  ducc0/math/gl_integrator.h \
  ducc0/infra/mav.h \
  ducc0/infra/string_utils.cc \
  ducc0/infra/string_utils.h \
  ducc0/infra/system.cc \
  ducc0/infra/system.h \
  ducc0/infra/threading.cc \
  ducc0/infra/threading.h \
  ducc0/infra/timers.h \
  ducc0/math/unity_roots.h \
  ducc0/infra/useful_macros.h \
  ducc0/math/rangeset.h \
  ducc0/geom_utils.h \
  ducc0/math/geom_utils.cc \
  ducc0/healpix/healpix_tables.h \
  ducc0/healpix/healpix_tables.cc \
  ducc0/healpix/healpix_base.h \
  ducc0/healpix/healpix_base.cc

# format is "current:revision:age"
# any change: increase revision
# any interface change: increase current, revision=0
# any backward-compatible change: increase age
# any backward-incompatible change: age=0
# ==> age <= current
libmrutil_la_LDFLAGS = -version-info 0:0:0 -lpthread

AM_CXXFLAGS = @AM_CXXFLAGS@

if HAVE_MULTIARCH

libavx_la_SOURCES = ducc0/sharp/sharp_core_inc.cc
libavx2_la_SOURCES = ducc0/sharp/sharp_core_inc.cc
libfma_la_SOURCES = ducc0/sharp/sharp_core_inc.cc
libfma4_la_SOURCES = ducc0/sharp/sharp_core_inc.cc
libavx512f_la_SOURCES = ducc0/sharp/sharp_core_inc.cc

noinst_LTLIBRARIES = libavx.la libavx2.la libfma.la libfma4.la libavx512f.la

libavx_la_CXXFLAGS = ${AM_CXXFLAGS} -mavx -DARCH=avx
libavx2_la_CXXFLAGS = ${AM_CXXFLAGS} -mavx2 -DARCH=avx2
libfma_la_CXXFLAGS = ${AM_CXXFLAGS} -mfma -DARCH=fma
libfma4_la_CXXFLAGS = ${AM_CXXFLAGS} -mfma4 -DARCH=fma4
libavx512f_la_CXXFLAGS = ${AM_CXXFLAGS} -mavx512f -DARCH=avx512f

libmrutil_la_LIBADD = libavx.la libavx2.la libfma.la libfma4.la libavx512f.la

endif

nobase_include_HEADERS = \
  ducc0/sharp/sharp.h \
  ducc0/sharp/sharp_geomhelpers.h \
  ducc0/sharp/sharp_almhelpers.h

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
