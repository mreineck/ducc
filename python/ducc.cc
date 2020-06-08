#include "mr_util/infra/system.cc"
#include "mr_util/infra/string_utils.cc"
#include "mr_util/infra/threading.cc"
#include "mr_util/math/pointing.cc"
#include "mr_util/math/geom_utils.cc"
#include "mr_util/math/space_filling.cc"
#include "mr_util/sharp/sharp.cc"
#include "mr_util/sharp/sharp_almhelpers.cc"
#include "mr_util/sharp/sharp_core.cc"
#include "mr_util/sharp/sharp_ylmgen.cc"
#include "mr_util/sharp/sharp_geomhelpers.cc"
#include "mr_util/healpix/healpix_tables.cc"
#include "mr_util/healpix/healpix_base.cc"

#include <pybind11/pybind11.h>
#include "python/sht.cc"
#include "python/fft.cc"
#include "python/totalconvolve.cc"
#include "python/wgridder.cc"
#include "python/healpix.cc"
#include "python/misc.cc"

using namespace mr;

PYBIND11_MODULE(PKGNAME, m)
  {
  m.attr("__version__") = PKGVERSION;

  add_fft(m);
  add_sht(m);
  add_totalconvolve(m);
  add_wgridder(m);
  add_healpix(m);
  add_misc(m);
  }
