#include "ducc0/bindings/pybind_utils.h"
#include "ducc0/../../python/module_adders.h"

namespace DUCC0_NAMESPACE {

void add_ducc0(py::module_ &m)
  {
  add_fft(m);
  add_sht(m);
  add_totalconvolve(m);
  add_wgridder(m);
  add_healpix(m);
  add_misc(m);
  add_pointingprovider(m);
  add_nufft(m);
  }

}
