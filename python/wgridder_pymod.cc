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

/* Copyright (C) 2019-2024 Max-Planck-Society
   Author: Martin Reinecke */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "ducc0/bindings/pybind_utils.h"
#include "ducc0/wgridder/wgridder.h"
#include "ducc0/wgridder/wgridder_sycl.h"

namespace ducc0 {

namespace detail_pymodule_wgridder {

using namespace std;

namespace py = pybind11;

auto None = py::none();

template<typename T> py::array Py2_vis2dirty_tuning(const py::array &uvw_,
  const py::array &freq_, const py::array &vis_, const py::object &wgt_, const py::object &mask_,
  size_t npix_x, size_t npix_y, double pixsize_x, double pixsize_y,
  double epsilon, bool do_wgridding, size_t nthreads, size_t verbosity,
  bool flip_u, bool flip_v, bool flip_w, bool divide_by_n, py::object &dirty_, double sigma_min,
  double sigma_max, double center_x, double center_y,
  bool double_precision_accumulation)
  {
  auto uvw = to_cmav<double,2>(uvw_);
  auto freq = to_cmav<double,1>(freq_);
  auto vis = to_cmav<complex<T>,2>(vis_);
  auto wgt = get_optional_const_Pyarr<T>(wgt_, {vis.shape(0),vis.shape(1)});
  auto wgt2 = to_cmav<T,2>(wgt);
  auto mask = get_optional_const_Pyarr<uint8_t>(mask_, {uvw.shape(0),freq.shape(0)});
  auto mask2 = to_cmav<uint8_t,2>(mask);
  // sizes must be either both zero or both nonzero
  MR_assert((npix_x==0)==(npix_y==0), "inconsistent dirty image dimensions");
  auto dirty = (npix_x==0) ? get_Pyarr<T>(dirty_, 2)
                           : get_optional_Pyarr<T>(dirty_, {npix_x, npix_y});
  auto dirty2 = to_vmav<T,2>(dirty);
  {
  py::gil_scoped_release release;
  double_precision_accumulation ?
    ms2dirty_tuning<T,double>(uvw,freq,vis,wgt2,mask2,pixsize_x,pixsize_y,epsilon,
      do_wgridding,nthreads,dirty2,verbosity,flip_u,flip_v,flip_w,divide_by_n, sigma_min,
      sigma_max, center_x, center_y) :
    ms2dirty_tuning<T,T>(uvw,freq,vis,wgt2,mask2,pixsize_x,pixsize_y,epsilon,
      do_wgridding,nthreads,dirty2,verbosity,flip_u,flip_v,flip_w,divide_by_n, sigma_min,
      sigma_max, center_x, center_y);
  }
  return dirty;
  }
py::array Py_vis2dirty_tuning(const py::array &uvw,
  const py::array &freq, const py::array &vis, const py::object &wgt,
  size_t npix_x, size_t npix_y, double pixsize_x, double pixsize_y,
  double epsilon, bool do_wgridding, size_t nthreads,
  size_t verbosity, const py::object &mask, bool flip_u, bool flip_v, bool flip_w, bool divide_by_n,
  py::object &dirty, double sigma_min, double sigma_max,
  double center_x, double center_y,
  bool double_precision_accumulation)
  {
  if (isPyarr<complex<float>>(vis))
    return Py2_vis2dirty_tuning<float>(uvw, freq, vis, wgt, mask, npix_x, npix_y,
      pixsize_x, pixsize_y, epsilon, do_wgridding, nthreads, verbosity,
      flip_u, flip_v, flip_w, divide_by_n, dirty, sigma_min, sigma_max, center_x, center_y,
      double_precision_accumulation);
  if (isPyarr<complex<double>>(vis))
    return Py2_vis2dirty_tuning<double>(uvw, freq, vis, wgt, mask, npix_x, npix_y,
      pixsize_x, pixsize_y, epsilon, do_wgridding, nthreads, verbosity,
      flip_u, flip_v, flip_w, divide_by_n, dirty, sigma_min, sigma_max, center_x, center_y,
      double_precision_accumulation);
  MR_fail("type matching failed: 'vis' has neither type 'c8' nor 'c16'");
  }
constexpr auto vis2dirty_tuning_DS = R"""(
Converts visibilities to a dirty image.

This implementation tries to subdivide the operation into smaller tasks to
optimize performance.

Parameters
----------
uvw: numpy.ndarray((nrows, 3), dtype=numpy.float64)
    UVW coordinates from the measurement set
freq: numpy.ndarray((nchan,), dtype=numpy.float64)
    channel frequencies
vis: numpy.ndarray((nrows, nchan), dtype=numpy.complex64 or numpy.complex128)
    the input visibilities.
    Its data type determines the precision in which the calculation is carried
    out.
wgt: numpy.ndarray((nrows, nchan), float with same precision as `vis`), optional
    If present, its values are multiplied to the input before gridding
mask: numpy.ndarray((nrows, nchan), dtype=numpy.uint8), optional
    If present, only visibilities are processed for which mask!=0
npix_x, npix_y: int
    dimensions of the dirty image (must both be at least 32)
    If the `dirty` argument is provided, image dimensions will be inferred from
    the passed array; in this case npix_x and npix_y must be either consistent
    with these dimensions, or be zero.
pixsize_x, pixsize_y: float
    angular pixel size (in projected radians) of the dirty image
center_x, center_y: float
    center of the dirty image relative to the phase center
    (in projected radians)
epsilon: float
    accuracy at which the computation should be done. Must be larger than 2e-13.
    If `vis` has type numpy.complex64, it must be larger than 1e-5.
do_wgridding: bool
    if True, the full w-gridding algorithm is carried out, otherwise
    the w values are assumed to be zero.
flip_[uvw]: bool
    if True, all [uvw] coordinates in uvw are multiplied by -1
divide_by_n: bool
    if True, the dirty image pixels are divided by n
sigma_min, sigma_max: float
    minimum and maximum allowed oversampling factors
nthreads: int
    number of threads to use for the calculation
verbosity: int
    0: no output
    1: some diagnostic output and timings
dirty: numpy.ndarray((npix_x, npix_y), dtype=float of same precision as `vis`),
    optional
    If provided, the dirty image will be written to this array and a handle
    to it will be returned.
double_precision_accumulation: bool
    If True, always use double precision for accumulating operations onto the
    uv grid. This is necessary to reduce numerical errors in special cases.

Returns
-------
numpy.ndarray((npix_x, npix_y), dtype=float of same precision as `vis`)
    the dirty image

Notes
-----
The input arrays should be contiguous and in C memory order.
Other strides will work, but can degrade performance significantly.
)""";

template<typename T> py::array Py2_vis2dirty(const py::array &uvw_,
  const py::array &freq_, const py::array &vis_, const py::object &wgt_, const py::object &mask_,
  size_t npix_x, size_t npix_y, double pixsize_x, double pixsize_y,
  double epsilon, bool do_wgridding, size_t nthreads, size_t verbosity,
  bool flip_u, bool flip_v, bool flip_w, bool divide_by_n, py::object &dirty_, double sigma_min,
  double sigma_max, double center_x, double center_y, bool allow_nshift,
  bool gpu, bool double_precision_accumulation)
  {
  auto uvw = to_cmav<double,2>(uvw_);
  auto freq = to_cmav<double,1>(freq_);
  auto vis = to_cmav<complex<T>,2>(vis_);
  auto wgt = get_optional_const_Pyarr<T>(wgt_, {vis.shape(0),vis.shape(1)});
  auto wgt2 = to_cmav<T,2>(wgt);
  auto mask = get_optional_const_Pyarr<uint8_t>(mask_, {uvw.shape(0),freq.shape(0)});
  auto mask2 = to_cmav<uint8_t,2>(mask);
  // sizes must be either both zero or both nonzero
  MR_assert((npix_x==0)==(npix_y==0), "inconsistent dirty image dimensions");
  auto dirty = (npix_x==0) ? get_Pyarr<T>(dirty_, 2)
                           : get_optional_Pyarr<T>(dirty_, {npix_x, npix_y});
  auto dirty2 = to_vmav<T,2>(dirty);
  {
  py::gil_scoped_release release;
  if (gpu)
    double_precision_accumulation ?
      ms2dirty_sycl<T,double>(uvw,freq,vis,wgt2,mask2,pixsize_x,pixsize_y,epsilon,
        do_wgridding,nthreads,dirty2,verbosity,flip_u,flip_v,flip_w,divide_by_n, sigma_min,
        sigma_max, center_x, center_y, allow_nshift) :
      ms2dirty_sycl<T,T>(uvw,freq,vis,wgt2,mask2,pixsize_x,pixsize_y,epsilon,
        do_wgridding,nthreads,dirty2,verbosity,flip_u,flip_v,flip_w,divide_by_n, sigma_min,
        sigma_max, center_x, center_y, allow_nshift);
  else
    double_precision_accumulation ?
      ms2dirty<T,double>(uvw,freq,vis,wgt2,mask2,pixsize_x,pixsize_y,epsilon,
        do_wgridding,nthreads,dirty2,verbosity,flip_u,flip_v,flip_w,divide_by_n, sigma_min,
        sigma_max, center_x, center_y, allow_nshift) :
      ms2dirty<T,T>(uvw,freq,vis,wgt2,mask2,pixsize_x,pixsize_y,epsilon,
        do_wgridding,nthreads,dirty2,verbosity,flip_u,flip_v,flip_w,divide_by_n, sigma_min,
        sigma_max, center_x, center_y, allow_nshift);
  }
  return dirty;
  }
py::array Py_vis2dirty(const py::array &uvw,
  const py::array &freq, const py::array &vis, const py::object &wgt,
  size_t npix_x, size_t npix_y, double pixsize_x, double pixsize_y,
  double epsilon, bool do_wgridding, size_t nthreads,
  size_t verbosity, const py::object &mask, bool flip_u, bool flip_v, bool flip_w, bool divide_by_n,
  py::object &dirty, double sigma_min, double sigma_max,
  double center_x, double center_y, bool allow_nshift,
  bool gpu, bool double_precision_accumulation)
  {
  if (isPyarr<complex<float>>(vis))
    return Py2_vis2dirty<float>(uvw, freq, vis, wgt, mask, npix_x, npix_y,
      pixsize_x, pixsize_y, epsilon, do_wgridding, nthreads, verbosity,
      flip_u, flip_v, flip_w, divide_by_n, dirty, sigma_min, sigma_max, center_x, center_y,
      allow_nshift, gpu, double_precision_accumulation);
  if (isPyarr<complex<double>>(vis))
    return Py2_vis2dirty<double>(uvw, freq, vis, wgt, mask, npix_x, npix_y,
      pixsize_x, pixsize_y, epsilon, do_wgridding, nthreads, verbosity,
      flip_u, flip_v, flip_w, divide_by_n, dirty, sigma_min, sigma_max, center_x, center_y,
      allow_nshift, gpu, double_precision_accumulation);
  MR_fail("type matching failed: 'vis' has neither type 'c8' nor 'c16'");
  }
constexpr auto vis2dirty_DS = R"""(
Converts visibilities to a dirty image.

Parameters
----------
uvw: numpy.ndarray((nrows, 3), dtype=numpy.float64)
    UVW coordinates from the measurement set
freq: numpy.ndarray((nchan,), dtype=numpy.float64)
    channel frequencies
vis: numpy.ndarray((nrows, nchan), dtype=numpy.complex64 or numpy.complex128)
    the input visibilities.
    Its data type determines the precision in which the calculation is carried
    out.
wgt: numpy.ndarray((nrows, nchan), float with same precision as `vis`), optional
    If present, its values are multiplied to the input before gridding
mask: numpy.ndarray((nrows, nchan), dtype=numpy.uint8), optional
    If present, only visibilities are processed for which mask!=0
npix_x, npix_y: int
    dimensions of the dirty image (must both be even and at least 32)
    If the `dirty` argument is provided, image dimensions will be inferred from
    the passed array; in this case npix_x and npix_y must be either consistent
    with these dimensions, or be zero.
pixsize_x, pixsize_y: float
    angular pixel size (in projected radians) of the dirty image
center_x, center_y: float
    center of the dirty image relative to the phase center
    (in projected radians)
epsilon: float
    accuracy at which the computation should be done. Must be larger than 2e-13.
    If `vis` has type numpy.complex64, it must be larger than 1e-5.
do_wgridding: bool
    if True, the full w-gridding algorithm is carried out, otherwise
    the w values are assumed to be zero.
flip_[uvw]: bool
    if True, all [uvw] coordinates in uvw are multiplied by -1
divide_by_n: bool
    if True, the dirty image pixels are divided by n
sigma_min, sigma_max: float
    minimum and maximum allowed oversampling factors
nthreads: int
    number of threads to use for the calculation
verbosity: int
    0: no output
    1: some diagnostic output and timings
dirty: numpy.ndarray((npix_x, npix_y), dtype=float of same precision as `vis`),
    optional
    If provided, the dirty image will be written to this array and a handle
    to it will be returned.
double_precision_accumulation: bool
    If True, always use double precision for accumulating operations onto the
    uv grid. This is necessary to reduce numerical errors in special cases.

Returns
-------
numpy.ndarray((npix_x, npix_y), dtype=float of same precision as `vis`)
    the dirty image

Notes
-----
The input arrays should be contiguous and in C memory order.
Other strides will work, but can degrade performance significantly.
)""";

template<typename T> py::array Py2_dirty2vis_tuning(const py::array &uvw_,
  const py::array &freq_, const py::array &dirty_, const py::object &wgt_, const py::object &mask_,
  double pixsize_x, double pixsize_y, double epsilon, bool do_wgridding,
  size_t nthreads, size_t verbosity, bool flip_u, bool flip_v, bool flip_w, bool divide_by_n,
  py::object &vis_, double sigma_min, double sigma_max, double center_x, double center_y)
  {
  auto uvw = to_cmav<double,2>(uvw_);
  auto freq = to_cmav<double,1>(freq_);
  auto dirty = to_cmav<T,2>(dirty_);
  auto wgt = get_optional_const_Pyarr<T>(wgt_, {uvw.shape(0),freq.shape(0)});
  auto wgt2 = to_cmav<T,2>(wgt);
  auto mask = get_optional_const_Pyarr<uint8_t>(mask_, {uvw.shape(0),freq.shape(0)});
  auto mask2 = to_cmav<uint8_t,2>(mask);
  auto vis = get_optional_Pyarr<complex<T>>(vis_, {uvw.shape(0),freq.shape(0)});
  auto vis2 = to_vmav<complex<T>,2>(vis);
  {
  py::gil_scoped_release release;
  dirty2ms_tuning<T,T>(uvw,freq,dirty,wgt2,mask2,pixsize_x,pixsize_y,epsilon,
    do_wgridding,nthreads,vis2,verbosity,flip_u,flip_v,flip_w,divide_by_n, sigma_min,
    sigma_max, center_x, center_y);
  }
  return vis;
  }
py::array Py_dirty2vis_tuning(const py::array &uvw,
  const py::array &freq, const py::array &dirty, const py::object &wgt,
  double pixsize_x, double pixsize_y, double epsilon, bool do_wgridding,
  size_t nthreads, size_t verbosity, const py::object &mask,
  bool flip_u, bool flip_v, bool flip_w, bool divide_by_n, py::object &vis, double sigma_min,
  double sigma_max, double center_x, double center_y)
  {
  if (isPyarr<float>(dirty))
    return Py2_dirty2vis_tuning<float>(uvw, freq, dirty, wgt, mask,
      pixsize_x, pixsize_y, epsilon, do_wgridding, nthreads, verbosity,
      flip_u, flip_v, flip_w, divide_by_n, vis, sigma_min, sigma_max, center_x, center_y);
  if (isPyarr<double>(dirty))
    return Py2_dirty2vis_tuning<double>(uvw, freq, dirty, wgt, mask,
      pixsize_x, pixsize_y, epsilon, do_wgridding, nthreads, verbosity,
      flip_u, flip_v, flip_w, divide_by_n, vis, sigma_min, sigma_max, center_x, center_y);
  MR_fail("type matching failed: 'dirty' has neither type 'f4' nor 'f8'");
  }
constexpr auto dirty2vis_tuning_DS = R"""(
Converts a dirty image to visibilities.

This implementation tries to subdivide the operation into smaller tasks to
optimize performance.

Parameters
----------
uvw: numpy.ndarray((nrows, 3), dtype=numpy.float64)
    UVW coordinates from the measurement set
freq: numpy.ndarray((nchan,), dtype=numpy.float64)
    channel frequencies
dirty: numpy.ndarray((npix_x, npix_y), dtype=numpy.float32 or numpy.float64)
    dirty image
    Its data type determines the precision in which the calculation is carried
    out.
    Both dimensions must be least 32.
wgt: numpy.ndarray((nrows, nchan), same dtype as `dirty`), optional
    If present, its values are multiplied to the output
mask: numpy.ndarray((nrows, nchan), dtype=numpy.uint8), optional
    If present, only visibilities are processed for which mask!=0
pixsize_x, pixsize_y: float
    angular pixel size (in projected radians) of the dirty image
center_x, center_y: float
    center of the dirty image relative to the phase center
    (in projected radians)
epsilon: float
    accuracy at which the computation should be done. Must be larger than 2e-13.
    If `dirty` has type numpy.float32, it must be larger than 1e-5.
do_wgridding: bool
    if True, the full w-gridding algorithm is carried out, otherwise
    the w values are assumed to be zero.
flip_[uvw]: bool
    if True, all [uvw] coordinates in uvw are multiplied by -1
divide_by_n: bool
    if True, the dirty image pixels are divided by n
sigma_min, sigma_max: float
    minimum and maximum allowed oversampling factors
nthreads: int
    number of threads to use for the calculation
verbosity: int
    0: no output
    1: some diagnostic output and timings
vis: numpy.ndarray((nrows, nchan), dtype=complex of same precision as `dirty`),
    optional
    If provided, the computed visibilities will be stored in this array, and
    a handle to it will be returned.

Returns
-------
numpy.ndarray((nrows, nchan), dtype=complex of same precision as `dirty`)
    the computed visibilities.

Notes
-----
The input arrays should be contiguous and in C memory order.
Other strides will work, but can degrade performance significantly.
)""";
template<typename T> py::array Py2_dirty2vis(const py::array &uvw_,
  const py::array &freq_, const py::array &dirty_, const py::object &wgt_, const py::object &mask_,
  double pixsize_x, double pixsize_y, double epsilon, bool do_wgridding,
  size_t nthreads, size_t verbosity, bool flip_u, bool flip_v, bool flip_w, bool divide_by_n,
  py::object &vis_, double sigma_min, double sigma_max, double center_x, double center_y, bool allow_nshift,
  bool gpu)
  {
  auto uvw = to_cmav<double,2>(uvw_);
  auto freq = to_cmav<double,1>(freq_);
  auto dirty = to_cmav<T,2>(dirty_);
  auto wgt = get_optional_const_Pyarr<T>(wgt_, {uvw.shape(0),freq.shape(0)});
  auto wgt2 = to_cmav<T,2>(wgt);
  auto mask = get_optional_const_Pyarr<uint8_t>(mask_, {uvw.shape(0),freq.shape(0)});
  auto mask2 = to_cmav<uint8_t,2>(mask);
  auto vis = get_optional_Pyarr<complex<T>>(vis_, {uvw.shape(0),freq.shape(0)});
  auto vis2 = to_vmav<complex<T>,2>(vis);
  {
  py::gil_scoped_release release;
  if (gpu)
    dirty2ms_sycl<T,T>(uvw,freq,dirty,wgt2,mask2,pixsize_x,pixsize_y,epsilon,
      do_wgridding,nthreads,vis2,verbosity,flip_u,flip_v,flip_w,divide_by_n, sigma_min,
      sigma_max, center_x, center_y, allow_nshift);
  else
    dirty2ms<T,T>(uvw,freq,dirty,wgt2,mask2,pixsize_x,pixsize_y,epsilon,
      do_wgridding,nthreads,vis2,verbosity,flip_u,flip_v,flip_w,divide_by_n, sigma_min,
      sigma_max, center_x, center_y, allow_nshift);
  }
  return vis;
  }
py::array Py_dirty2vis(const py::array &uvw,
  const py::array &freq, const py::array &dirty, const py::object &wgt,
  double pixsize_x, double pixsize_y, double epsilon, bool do_wgridding,
  size_t nthreads, size_t verbosity, const py::object &mask,
  bool flip_u, bool flip_v, bool flip_w, bool divide_by_n, py::object &vis, double sigma_min,
  double sigma_max, double center_x, double center_y, bool allow_nshift,
  bool gpu)
  {
  if (isPyarr<float>(dirty))
    return Py2_dirty2vis<float>(uvw, freq, dirty, wgt, mask,
      pixsize_x, pixsize_y, epsilon, do_wgridding, nthreads, verbosity,
      flip_u, flip_v, flip_w, divide_by_n, vis, sigma_min, sigma_max, center_x, center_y, allow_nshift, gpu);
  if (isPyarr<double>(dirty))
    return Py2_dirty2vis<double>(uvw, freq, dirty, wgt, mask,
      pixsize_x, pixsize_y, epsilon, do_wgridding, nthreads, verbosity,
      flip_u, flip_v, flip_w, divide_by_n, vis, sigma_min, sigma_max, center_x, center_y, allow_nshift, gpu);
  MR_fail("type matching failed: 'dirty' has neither type 'f4' nor 'f8'");
  }
constexpr auto dirty2vis_DS = R"""(
Converts a dirty image to visibilities.

Parameters
----------
uvw: numpy.ndarray((nrows, 3), dtype=numpy.float64)
    UVW coordinates from the measurement set
freq: numpy.ndarray((nchan,), dtype=numpy.float64)
    channel frequencies
dirty: numpy.ndarray((npix_x, npix_y), dtype=numpy.float32 or numpy.float64)
    dirty image
    Its data type determines the precision in which the calculation is carried
    out.
    Both dimensions must be even and at least 32.
wgt: numpy.ndarray((nrows, nchan), same dtype as `dirty`), optional
    If present, its values are multiplied to the output
mask: numpy.ndarray((nrows, nchan), dtype=numpy.uint8), optional
    If present, only visibilities are processed for which mask!=0
pixsize_x, pixsize_y: float
    angular pixel size (in projected radians) of the dirty image
center_x, center_y: float
    center of the dirty image relative to the phase center
    (in projected radians)
epsilon: float
    accuracy at which the computation should be done. Must be larger than 2e-13.
    If `dirty` has type numpy.float32, it must be larger than 1e-5.
do_wgridding: bool
    if True, the full w-gridding algorithm is carried out, otherwise
    the w values are assumed to be zero.
flip_[uvw]: bool
    if True, all [uvw] coordinates in uvw are multiplied by -1
divide_by_n: bool
    if True, the dirty image pixels are divided by n
sigma_min, sigma_max: float
    minimum and maximum allowed oversampling factors
nthreads: int
    number of threads to use for the calculation
verbosity: int
    0: no output
    1: some diagnostic output and timings
vis: numpy.ndarray((nrows, nchan), dtype=complex of same precision as `dirty`),
    optional
    If provided, the computed visibilities will be stored in this array, and
    a handle to it will be returned.

Returns
-------
numpy.ndarray((nrows, nchan), dtype=complex of same precision as `dirty`)
    the computed visibilities.

Notes
-----
The input arrays should be contiguous and in C memory order.
Other strides will work, but can degrade performance significantly.
)""";

py::array Py_ms2dirty(const py::array &uvw,
  const py::array &freq, const py::array &ms, const py::object &wgt,
  size_t npix_x, size_t npix_y, double pixsize_x, double pixsize_y, size_t /*nu*/,
  size_t /*nv*/, double epsilon, bool do_wgridding, size_t nthreads,
  size_t verbosity, const py::object &mask,
  bool double_precision_accumulation, bool gpu)
  {
  return Py_vis2dirty(uvw, freq, ms, wgt, npix_x, npix_y, pixsize_x, pixsize_y,
    epsilon, do_wgridding, nthreads, verbosity, mask, false, false, false, true, None, 1.1,
    2.6, 0., 0., true, gpu, double_precision_accumulation);
  }

constexpr auto ms2dirty_DS = R"""(
Converts an MS object to dirty image.

Parameters
----------
uvw: numpy.ndarray((nrows, 3), dtype=numpy.float64)
    UVW coordinates from the measurement set
freq: numpy.ndarray((nchan,), dtype=numpy.float64)
    channel frequencies
ms: numpy.ndarray((nrows, nchan), dtype=numpy.complex64 or numpy.complex128)
    the input measurement set data.
    Its data type determines the precision in which the calculation is carried
    out.
wgt: numpy.ndarray((nrows, nchan), float with same precision as `ms`), optional
    If present, its values are multiplied to the input before gridding
npix_x, npix_y: int
    dimensions of the dirty image (must both be even and at least 32)
pixsize_x, pixsize_y: float
    angular pixel size (in projected radians) of the dirty image
nu, nv: int
    obsolete, ignored
epsilon: float
    accuracy at which the computation should be done. Must be larger than 2e-13.
    If `ms` has type numpy.complex64, it must be larger than 1e-5.
do_wstacking: bool
    if True, the full w-gridding algorithm is carried out, otherwise
    the w values are assumed to be zero.
nthreads: int
    number of threads to use for the calculation
verbosity: int
    0: no output
    1: some output
mask: numpy.ndarray((nrows, nchan), dtype=numpy.uint8), optional
    If present, only visibilities are processed for which mask!=0
double_precision_accumulation: bool
    If True, always use double precision for accumulating operations onto the
    uv grid. This is necessary to reduce numerical errors in special cases.

Returns
-------
numpy.ndarray((npix_x, npix_y), dtype=float of same precision as `ms`)
    the dirty image

Notes
-----
The input arrays should be contiguous and in C memory order.
Other strides will work, but can degrade performance significantly.
)""";

py::array Py_dirty2ms(const py::array &uvw,
  const py::array &freq, const py::array &dirty, const py::object &wgt,
  double pixsize_x, double pixsize_y, size_t /*nu*/, size_t /*nv*/, double epsilon,
  bool do_wgridding, size_t nthreads, size_t verbosity, const py::object &mask, bool gpu)
  {
  return Py_dirty2vis(uvw, freq, dirty, wgt, pixsize_x, pixsize_y, epsilon, do_wgridding, nthreads, verbosity, mask, false, false, false, true, None, 1.1, 2.6, 0, 0, true, gpu);
  }

constexpr auto dirty2ms_DS = R"""(
Converts a dirty image to an MS object.

Parameters
----------
uvw: numpy.ndarray((nrows, 3), dtype=numpy.float64)
    UVW coordinates from the measurement set
freq: numpy.ndarray((nchan,), dtype=numpy.float64)
    channel frequencies
dirty: numpy.ndarray((npix_x, npix_y), dtype=numpy.float32 or numpy.float64)
    dirty image
    Its data type determines the precision in which the calculation is carried
    out.
    Both dimensions must be even and at least 32.
wgt: numpy.ndarray((nrows, nchan), same dtype as `dirty`), optional
    If present, its values are multiplied to the output
pixsize_x, pixsize_y: float
    angular pixel size (in projected radians) of the dirty image
nu, nv: int
    obsolete, ignored
epsilon: float
    accuracy at which the computation should be done. Must be larger than 2e-13.
    If `dirty` has type numpy.float32, it must be larger than 1e-5.
do_wstacking: bool
    if True, the full w-gridding algorithm is carried out, otherwise
    the w values are assumed to be zero.
nthreads: int
    number of threads to use for the calculation
verbosity: int
    0: no output
    1: some output
mask: numpy.ndarray((nrows, nchan), dtype=numpy.uint8), optional
    If present, only visibilities are processed for which mask!=0

Returns
-------
numpy.ndarray((nrows, nchan), dtype=complex of same precision as `dirty`)
    the measurement set data.

Notes
-----
The input arrays should be contiguous and in C memory order.
Other strides will work, but can degrade performance significantly.
)""";

constexpr const char *wgridder_experimental_DS = R"""(
Experimental, more powerful interface to the gridding code

Notes
-----

The functionality in this module is not considered to have a stable interface
and also may be moved to other modules in the future. If you use it, be prepared
to adjust your code at some point ion the future!
)""";

void add_pythonfuncs(py::module_ &m)
  {
  using namespace pybind11::literals;

  m.def("vis2dirty", &Py_vis2dirty, vis2dirty_DS, py::kw_only(), "uvw"_a, "freq"_a, "vis"_a,
    "wgt"_a=None, "npix_x"_a=0, "npix_y"_a=0, "pixsize_x"_a, "pixsize_y"_a,
    "epsilon"_a, "do_wgridding"_a=false, "nthreads"_a=1, "verbosity"_a=0,
    "mask"_a=None, "flip_u"_a=false, "flip_v"_a=false, "flip_w"_a=false, "divide_by_n"_a=true, "dirty"_a=None,
    "sigma_min"_a=1.1, "sigma_max"_a=2.6, "center_x"_a=0., "center_y"_a=0.,
    "allow_nshift"_a=true, "gpu"_a=false, "double_precision_accumulation"_a=false);
  m.def("dirty2vis", &Py_dirty2vis, dirty2vis_DS, py::kw_only(), "uvw"_a, "freq"_a, "dirty"_a,
    "wgt"_a=None, "pixsize_x"_a, "pixsize_y"_a, "epsilon"_a,
    "do_wgridding"_a=false, "nthreads"_a=1, "verbosity"_a=0, "mask"_a=None,
    "flip_u"_a=false, "flip_v"_a=false, "flip_w"_a=false, "divide_by_n"_a=true, "vis"_a=None, "sigma_min"_a=1.1,
    "sigma_max"_a=2.6, "center_x"_a=0., "center_y"_a=0., "allow_nshift"_a=true, "gpu"_a=false);
  }

void add_wgridder(py::module_ &msup)
  {
  using namespace pybind11::literals;
  auto m = msup.def_submodule("wgridder");
  auto m2 = m.def_submodule("experimental", wgridder_experimental_DS);

  add_pythonfuncs(m);
  add_pythonfuncs(m2);

  m2.def("sycl_active", &ducc0::sycl_active);

  m2.def("vis2dirty_tuning", &Py_vis2dirty_tuning, vis2dirty_tuning_DS, py::kw_only(), "uvw"_a, "freq"_a, "vis"_a,
    "wgt"_a=None, "npix_x"_a=0, "npix_y"_a=0, "pixsize_x"_a, "pixsize_y"_a,
    "epsilon"_a, "do_wgridding"_a=false, "nthreads"_a=1, "verbosity"_a=0,
    "mask"_a=None, "flip_u"_a=false, "flip_v"_a=false, "flip_w"_a=false,
    "divide_by_n"_a=true, "dirty"_a=None,
    "sigma_min"_a=1.1, "sigma_max"_a=2.6, "center_x"_a=0., "center_y"_a=0.,
    "double_precision_accumulation"_a=false);
  m2.def("dirty2vis_tuning", &Py_dirty2vis_tuning, dirty2vis_tuning_DS, py::kw_only(), "uvw"_a, "freq"_a, "dirty"_a,
    "wgt"_a=None, "pixsize_x"_a, "pixsize_y"_a, "epsilon"_a,
    "do_wgridding"_a=false, "nthreads"_a=1, "verbosity"_a=0, "mask"_a=None,
    "flip_u"_a=false, "flip_v"_a=false, "flip_w"_a=false, "divide_by_n"_a=true, "vis"_a=None, "sigma_min"_a=1.1,
    "sigma_max"_a=2.6, "center_x"_a=0., "center_y"_a=0.);

  m.def("ms2dirty", &Py_ms2dirty, ms2dirty_DS, "uvw"_a, "freq"_a, "ms"_a,
    "wgt"_a=None, "npix_x"_a, "npix_y"_a, "pixsize_x"_a, "pixsize_y"_a, "nu"_a=0, "nv"_a=0,
    "epsilon"_a, "do_wstacking"_a=false, "nthreads"_a=1, "verbosity"_a=0, "mask"_a=None,
    "double_precision_accumulation"_a=false, "gpu"_a=false);
  m.def("dirty2ms", &Py_dirty2ms, dirty2ms_DS, "uvw"_a, "freq"_a, "dirty"_a,
    "wgt"_a=None, "pixsize_x"_a, "pixsize_y"_a, "nu"_a=0, "nv"_a=0, "epsilon"_a,
    "do_wstacking"_a=false, "nthreads"_a=1, "verbosity"_a=0, "mask"_a=None, "gpu"_a=false);
  }

}

using detail_pymodule_wgridder::add_wgridder;

}
