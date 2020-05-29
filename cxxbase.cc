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
#include "pysharp/pysharp.cc"
#include "pypocketfft/pypocketfft.cc"
#include "pyinterpol_ng/pyinterpol_ng.cc"
#include "nifty_gridder/nifty_gridder.cc"
#include "pyHealpix/pyHealpix.cc"

using namespace mr;

PYBIND11_MODULE(PKGNAME, m)
  {
  add_pypocketfft(m);
  add_pysharp(m);
  add_pyinterpol_ng(m);
  add_nifty_gridder(m);
  add_pyHealpix(m);
  }
