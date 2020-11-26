#include "ducc0/infra/system.cc"
#include "ducc0/infra/string_utils.cc"
#include "ducc0/infra/threading.cc"
#include "ducc0/infra/types.cc"
#include "ducc0/infra/communication.cc"
#include "ducc0/math/pointing.cc"
#include "ducc0/math/geom_utils.cc"
#include "ducc0/math/space_filling.cc"
#include "ducc0/sharp/sharp.cc"
#include "ducc0/sharp/sharp_almhelpers.cc"
#include "ducc0/sharp/sharp_core.cc"
#include "ducc0/sharp/sharp_ylmgen.cc"
#include "ducc0/sharp/sharp_geomhelpers.cc"
#include "ducc0/healpix/healpix_tables.cc"
#include "ducc0/healpix/healpix_base.cc"

#include <pybind11/pybind11.h>
#include "python/sht.cc"
#include "python/fft.cc"
#include "python/totalconvolve.cc"
#include "python/wgridder.cc"
#include "python/healpix.cc"
#include "python/misc.cc"
#include "python/pointingprovider.cc"

using namespace ducc0;

PYBIND11_MODULE(PKGNAME, m)
  {
#if (!defined(_MSC_VER)) // no idea why this doesn't work on Windows
  m.attr("__version__") = PKGVERSION;
#endif

  add_fft(m);
  add_sht(m);
  add_totalconvolve(m);
  add_wgridder(m);
  add_healpix(m);
  add_misc(m);
  add_pointingprovider(m);
  }
