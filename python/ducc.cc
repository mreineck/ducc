#include "ducc0/bindings/pybind_utils.h"
#include "ducc0/../../python/module_adders.h"

using namespace ducc0;

#if defined (DUCC0_TARGET)
#if defined (__GNUC__)
static bool cpu_supports(const std::string &feature)
  {
  __builtin_cpu_init ();
  if (feature=="sse2")
    return __builtin_cpu_supports ("sse2");
  if (feature=="avx2")
    return __builtin_cpu_supports ("avx2");
  if (feature=="avx512f")
    return __builtin_cpu_supports ("avx512f");
  return false;
  }
#else
static bool cpu_supports(const char * /*feature*/)
  { return false; }
#endif
#endif

#ifdef DUCC0_USE_NANOBIND
NB_MODULE(PKGNAME, m)
#else
PYBIND11_MODULE(PKGNAME, m)
#endif
  {
#define DUCC0_XSTRINGIFY(s) DUCC0_STRINGIFY(s)
#define DUCC0_STRINGIFY(s) #s
#if defined (DUCC0_TARGET)
  MR_assert(cpu_supports(DUCC0_XSTRINGIFY(DUCC0_TARGET)),
    "required CPU feature not supported by this CPU");
#endif
  m.attr("__version__") = DUCC0_XSTRINGIFY(PKGVERSION);
#undef DUCC0_STRINGIFY
#undef DUCC0_XSTRINGIFY
#ifdef DUCC0_USE_NANOBIND
  m.attr("__wrapper__") = "nanobind";
#else
  m.attr("__wrapper__") = "pybind11";
#endif

  add_fft(m);
  add_sht(m);
  add_totalconvolve(m);
  add_wgridder(m);
  add_healpix(m);
  add_misc(m);
  add_pointingprovider(m);
  add_nufft(m);
  }
