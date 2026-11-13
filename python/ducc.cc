#include "ducc0/bindings/pybind_utils.h"

namespace DUCC0_NAMESPACE {

namespace detail_pymodule_fft { void add_fft(py::module_ &m); }
namespace detail_pymodule_sht { void add_sht(py::module_ &m); }
namespace detail_pymodule_totalconvolve { void add_totalconvolve(py::module_ &m); }
namespace detail_pymodule_wgridder { void add_wgridder(py::module_ &m); }
namespace detail_pymodule_healpix { void add_healpix(py::module_ &m); }
namespace detail_pymodule_misc { void add_misc(py::module_ &m); }
namespace detail_pymodule_pointingprovider { void add_pointingprovider(py::module_ &m); }
namespace detail_pymodule_nufft { void add_nufft(py::module_ &m); }

void add_ducc0(py::module_ &m)
  {
  detail_pymodule_fft::add_fft(m);
  detail_pymodule_sht::add_sht(m);
  detail_pymodule_totalconvolve::add_totalconvolve(m);
  detail_pymodule_wgridder::add_wgridder(m);
  detail_pymodule_healpix::add_healpix(m);
  detail_pymodule_misc::add_misc(m);
  detail_pymodule_pointingprovider::add_pointingprovider(m);
  detail_pymodule_nufft::add_nufft(m);
  }

}
