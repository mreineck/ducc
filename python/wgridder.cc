/*
 *  This file is part of nifty_gridder.
 *
 *  nifty_gridder is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  nifty_gridder is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with nifty_gridder; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/* Copyright (C) 2019-2020 Max-Planck-Society
   Author: Martin Reinecke */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "ducc0/bindings/pybind_utils.h"
#include "python/gridder_cxx.h"

namespace ducc0 {

namespace detail_pymodule_wgridder {

using namespace std;

namespace py = pybind11;

auto None = py::none();

template<typename T> py::array ms2dirty2(const py::array &uvw_,
  const py::array &freq_, const py::array &ms_, const py::object &wgt_,
  size_t npix_x, size_t npix_y, double pixsize_x, double pixsize_y, size_t nu,
  size_t nv, double epsilon, bool do_wstacking, size_t nthreads,
  size_t verbosity)
  {
  auto uvw = to_mav<double,2>(uvw_, false);
  auto freq = to_mav<double,1>(freq_, false);
  auto ms = to_mav<complex<T>,2>(ms_, false);
  auto wgt = get_optional_const_Pyarr<T>(wgt_, {ms.shape(0),ms.shape(1)});
  auto wgt2 = to_mav<T,2>(wgt, false);
  auto dirty = make_Pyarr<T>({npix_x,npix_y});
  auto dirty2 = to_mav<T,2>(dirty, true);
  {
  py::gil_scoped_release release;
  ms2dirty(uvw,freq,ms,wgt2,pixsize_x,pixsize_y,nu,nv,epsilon,
    do_wstacking,nthreads,dirty2,verbosity);
  }
  return move(dirty);
  }
py::array Pyms2dirty(const py::array &uvw,
  const py::array &freq, const py::array &ms, const py::object &wgt,
  size_t npix_x, size_t npix_y, double pixsize_x, double pixsize_y, size_t nu,
  size_t nv, double epsilon, bool do_wstacking, size_t nthreads,
  size_t verbosity)
  {
  if (isPyarr<complex<float>>(ms))
    return ms2dirty2<float>(uvw, freq, ms, wgt, npix_x, npix_y,
      pixsize_x, pixsize_y, nu, nv, epsilon, do_wstacking, nthreads, verbosity);
  if (isPyarr<complex<double>>(ms))
    return ms2dirty2<double>(uvw, freq, ms, wgt, npix_x, npix_y,
      pixsize_x, pixsize_y, nu, nv, epsilon, do_wstacking, nthreads, verbosity);
  MR_fail("type matching failed: 'ms' has neither type 'c8' nor 'c16'");
  }
constexpr auto ms2dirty_DS = R"""(
Converts an MS object to dirty image.

Parameters
==========
uvw: np.array((nrows, 3), dtype=np.float64)
    UVW coordinates from the measurement set
freq: np.array((nchan,), dtype=np.float64)
    channel frequencies
ms: np.array((nrows, nchan,), dtype=np.complex64 or np.complex128)
    the input measurement set data.
    Its data type determines the precision in which the calculation is carried
    out.
wgt: np.array((nrows, nchan), float with same precision as `ms`), optional
    If present, its values are multiplied to the output
npix_x, npix_y: int
    dimensions of the dirty image
pixsize_x, pixsize_y: float
    angular pixel size (in radians) of the dirty image
nu, nv: int
    dimensions of the (oversampled) intermediate uv grid
    These values must be >= 1.2*the dimensions of the dirty image; tupical
    oversampling values lie between 1.5 and 2.
    Increasing the oversampling factor decreases the kernel support width
    required for the desired accuracy, so it typically reduces run-time; on the
    other hand, this will increase memory consumption.
    If at least one of these two values is 0, the library will automatically
    pick values that result in a fast computation.
epsilon: float
    accuracy at which the computation should be done. Must be larger than 2e-13.
    If `ms` has type np.complex64, it must be larger than 1e-5.
do_wstacking: bool
    if True, the full improved w-stacking algorithm is carried out, otherwise
    the w values are assumed to be zero.
nthreads: int
    number of threads to use for the calculation
verbosity: int
    0: no output
    1: some output
    2: detailed output

Returns
=======
np.array((nxdirty, nydirty), dtype=float of same precision as `ms`)
    the dirty image
)""";

template<typename T> py::array dirty2ms2(const py::array &uvw_,
  const py::array &freq_, const py::array &dirty_, const py::object &wgt_,
  double pixsize_x, double pixsize_y, size_t nu, size_t nv, double epsilon,
  bool do_wstacking, size_t nthreads, size_t verbosity)
  {
  auto uvw = to_mav<double,2>(uvw_, false);
  auto freq = to_mav<double,1>(freq_, false);
  auto dirty = to_mav<T,2>(dirty_, false);
  auto wgt = get_optional_const_Pyarr<T>(wgt_, {uvw.shape(0),freq.shape(0)});
  auto wgt2 = to_mav<T,2>(wgt, false);
  auto ms = make_Pyarr<complex<T>>({uvw.shape(0),freq.shape(0)});
  auto ms2 = to_mav<complex<T>,2>(ms, true);
  {
  py::gil_scoped_release release;
  dirty2ms(uvw,freq,dirty,wgt2,pixsize_x,pixsize_y,nu,nv,epsilon,
    do_wstacking,nthreads,ms2,verbosity);
  }
  return move(ms);
  }
py::array Pydirty2ms(const py::array &uvw,
  const py::array &freq, const py::array &dirty, const py::object &wgt,
  double pixsize_x, double pixsize_y, size_t nu, size_t nv, double epsilon,
  bool do_wstacking, size_t nthreads, size_t verbosity)
  {
  if (isPyarr<float>(dirty))
    return dirty2ms2<float>(uvw, freq, dirty, wgt,
      pixsize_x, pixsize_y, nu, nv, epsilon, do_wstacking, nthreads, verbosity);
  if (isPyarr<double>(dirty))
    return dirty2ms2<double>(uvw, freq, dirty, wgt,
      pixsize_x, pixsize_y, nu, nv, epsilon, do_wstacking, nthreads, verbosity);
  MR_fail("type matching failed: 'dirty' has neither type 'f4' nor 'f8'");
  }
constexpr auto dirty2ms_DS = R"""(
Converts a dirty image to an MS object.

Parameters
==========
uvw: np.array((nrows, 3), dtype=np.float64)
    UVW coordinates from the measurement set
freq: np.array((nchan,), dtype=np.float64)
    channel frequencies
dirty: np.array((nxdirty, nydirty), dtype=np.float32 or np.float64)
    dirty image
    Its data type determines the precision in which the calculation is carried
    out.
wgt: np.array((nrows, nchan), same dtype as `dirty`), optional
    If present, its values are multiplied to the output
pixsize_x, pixsize_y: float
    angular pixel size (in radians) of the dirty image
nu, nv: int
    dimensions of the (oversampled) intermediate uv grid
    These values must be >= 1.2*the dimensions of the dirty image; tupical
    oversampling values lie between 1.5 and 2.
    Increasing the oversampling factor decreases the kernel support width
    required for the desired accuracy, so it typically reduces run-time; on the
    other hand, this will increase memory consumption.
    If at least one of these two values is 0, the library will automatically
    pick values that result in a fast computation.
epsilon: float
    accuracy at which the computation should be done. Must be larger than 2e-13.
    If `dirty` has type np.float32, it must be larger than 1e-5.
do_wstacking: bool
    if True, the full improved w-stacking algorithm is carried out, otherwise
    the w values are assumed to be zero.
nthreads: int
    number of threads to use for the calculation
verbosity: int
    0: no output
    1: some output
    2: detailed output

Returns
=======
np.array((nrows, nchan,), dtype=complex of same precision as `dirty`)
    the measurement set data.
)""";

void add_wgridder(py::module &msup)
  {
  using namespace pybind11::literals;
  auto m = msup.def_submodule("wgridder");

  m.def("ms2dirty", &Pyms2dirty, ms2dirty_DS, "uvw"_a, "freq"_a, "ms"_a,
    "wgt"_a=None, "npix_x"_a, "npix_y"_a, "pixsize_x"_a, "pixsize_y"_a, "nu"_a, "nv"_a,
    "epsilon"_a, "do_wstacking"_a=false, "nthreads"_a=1, "verbosity"_a=0);
  m.def("dirty2ms", &Pydirty2ms, dirty2ms_DS, "uvw"_a, "freq"_a, "dirty"_a,
    "wgt"_a=None, "pixsize_x"_a, "pixsize_y"_a, "nu"_a, "nv"_a, "epsilon"_a,
    "do_wstacking"_a=false, "nthreads"_a=1, "verbosity"_a=0);
  }

}

using detail_pymodule_wgridder::add_wgridder;

}
