#include "ducc0/infra/string_utils.cc"
#include "ducc0/infra/threading.cc"
#include "ducc0/infra/mav.cc"
#include "ducc0/math/pointing.cc"
#include "ducc0/math/geom_utils.cc"
#include "ducc0/math/space_filling.cc"
#include "ducc0/math/gl_integrator.cc"
#include "ducc0/math/gridding_kernel.cc"
#include "ducc0/sht/sht.cc"
#include "ducc0/healpix/healpix_tables.cc"
#include "ducc0/healpix/healpix_base.cc"
#include "ducc0/wgridder/wgridder.cc"

#include <pybind11/pybind11.h>
#include "python/sht_pymod.cc"
#include "python/fft_pymod.cc"
#include "python/totalconvolve_pymod.cc"
#include "python/wgridder_pymod.cc"
#include "python/healpix_pymod.cc"
#include "python/misc_pymod.cc"
#include "python/pointingprovider_pymod.cc"
#include "python/nufft_pymod.cc"

using namespace ducc0;

PYBIND11_MODULE(PKGNAME, m)
  {
#define DUCC0_XSTRINGIFY(s) DUCC0_STRINGIFY(s)
#define DUCC0_STRINGIFY(s) #s
  m.attr("__version__") = DUCC0_XSTRINGIFY(PKGVERSION);
#undef DUCC0_STRINGIFY
#undef DUCC0_XSTRINGIFY

  add_fft(m);
  add_sht(m);
  add_totalconvolve(m);
  add_wgridder(m);
  add_healpix(m);
  add_misc(m);
  add_pointingprovider(m);
  add_nufft(m);
  }
