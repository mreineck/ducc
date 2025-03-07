#ifndef DUCC0_MODULE_ADDERS
#define DUCC0_MODULE_ADDERS

#include "ducc0/bindings/pybind_utils.h"

namespace ducc0 {
namespace detail_pymodule_fft { void add_fft(py::module_ &m); }
namespace detail_pymodule_sht { void add_sht(py::module_ &m); }
namespace detail_pymodule_totalconvolve { void add_totalconvolve(py::module_ &m); }
namespace detail_pymodule_wgridder { void add_wgridder(py::module_ &m); }
namespace detail_pymodule_healpix { void add_healpix(py::module_ &m); }
namespace detail_pymodule_misc { void add_misc(py::module_ &m); }
namespace detail_pymodule_pointingprovider { void add_pointingprovider(py::module_ &m); }
namespace detail_pymodule_nufft { void add_nufft(py::module_ &m); }

using detail_pymodule_fft::add_fft;
using detail_pymodule_sht::add_sht;
using detail_pymodule_totalconvolve::add_totalconvolve;
using detail_pymodule_wgridder::add_wgridder;
using detail_pymodule_healpix::add_healpix;
using detail_pymodule_misc::add_misc;
using detail_pymodule_pointingprovider::add_pointingprovider;
using detail_pymodule_nufft::add_nufft;
}

#endif
