/*
 *  This code is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This code is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this code; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/* Copyright (C) 2019-2022 Max-Planck-Society
   Author: Martin Reinecke */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "ducc0/bindings/pybind_utils.h"
#include "ducc0/nufft/nufft.h"

namespace ducc0 {

namespace detail_pymodule_nufft {

using namespace std;

namespace py = pybind11;

auto None = py::none();

template<typename Tgrid, typename Tcoord> py::array Py2_u2nu(const py::array &grid_,
  const py::array &coord_, bool forward, double epsilon, size_t nthreads,
  py::object &out__, size_t verbosity, double sigma_min, double sigma_max,
  double periodicity, bool fft_order)
  {
  using Tpoints = Tgrid;
  auto coord = to_cmav<Tcoord,2>(coord_);
  auto grid = to_cfmav<complex<Tgrid>>(grid_);
  auto out_ = get_optional_Pyarr<complex<Tpoints>>(out__, {coord.shape(0)});
  auto out = to_vmav<complex<Tpoints>,1>(out_);
  {
  py::gil_scoped_release release;
  u2nu<Tgrid,Tgrid>(coord,grid,forward,epsilon,nthreads,out,verbosity,
                    sigma_min,sigma_max, periodicity, fft_order);
  }
  return out_;
  }
py::array Py_u2nu(const py::array &grid,
  const py::array &coord, bool forward, double epsilon, size_t nthreads,
  py::object &out, size_t verbosity, double sigma_min, double sigma_max,
  double periodicity, bool fft_order)
  {
  if (isPyarr<double>(coord))
    {
    if (isPyarr<complex<double>>(grid))
      return Py2_u2nu<double, double>(grid, coord, forward, epsilon, nthreads,
        out, verbosity, sigma_min, sigma_max, periodicity, fft_order);
    else if (isPyarr<complex<float>>(grid))
      return Py2_u2nu<float, double>(grid, coord, forward, epsilon, nthreads,
        out, verbosity, sigma_min, sigma_max, periodicity, fft_order);
    }
  else if (isPyarr<float>(coord))
    {
    if (isPyarr<complex<double>>(grid))
      return Py2_u2nu<double, float>(grid, coord, forward, epsilon, nthreads,
        out, verbosity, sigma_min, sigma_max, periodicity, fft_order);
    else if (isPyarr<complex<float>>(grid))
      return Py2_u2nu<float, float>(grid, coord, forward, epsilon, nthreads,
        out, verbosity, sigma_min, sigma_max, periodicity, fft_order);
    }
  MR_fail("not yet supported");
  }

template<typename Tpoints, typename Tcoord> py::array Py2_nu2u(const py::array &points_,
  const py::array &coord_, bool forward, double epsilon, size_t nthreads,
  py::array &out_, size_t verbosity, double sigma_min, double sigma_max,
  double periodicity, bool fft_order)
  {
  using Tgrid = Tpoints;
  auto coord = to_cmav<Tcoord,2>(coord_);
  auto points = to_cmav<complex<Tpoints>,1>(points_);
  auto out = to_vfmav<complex<Tgrid>>(out_);
  {
  py::gil_scoped_release release;
  nu2u<Tgrid,Tgrid>(coord,points,forward,epsilon,nthreads,out,verbosity,
                    sigma_min,sigma_max, periodicity, fft_order);
  }
  return out_;
  }
py::array Py_nu2u(const py::array &points,
  const py::array &coord, bool forward, double epsilon, size_t nthreads,
  py::array &out, size_t verbosity, double sigma_min, double sigma_max,
  double periodicity, bool fft_order)
  {
  if (isPyarr<double>(coord))
    {
    if (isPyarr<complex<double>>(points))
      return Py2_nu2u<double, double>(points, coord, forward, epsilon, nthreads,
        out, verbosity, sigma_min, sigma_max, periodicity, fft_order);
    else if (isPyarr<complex<float>>(points))
      return Py2_nu2u<float, double>(points, coord, forward, epsilon, nthreads,
        out, verbosity, sigma_min, sigma_max, periodicity, fft_order);
    }
  else if (isPyarr<float>(coord))
    {
    if (isPyarr<complex<double>>(points))
      return Py2_nu2u<double, float>(points, coord, forward, epsilon, nthreads,
        out, verbosity, sigma_min, sigma_max, periodicity, fft_order);
    else if (isPyarr<complex<float>>(points))
      return Py2_nu2u<float, float>(points, coord, forward, epsilon, nthreads,
        out, verbosity, sigma_min, sigma_max, periodicity, fft_order);
    }
  MR_fail("not yet supported");
  }

class Py_Nufftplan
  {
  private:
    vector<size_t> uniform_shape;
    size_t npoints;

    unique_ptr<Nufft< float,  float,  float, 1>> pf1;
    unique_ptr<Nufft<double, double, double, 1>> pd1;
    unique_ptr<Nufft< float,  float,  float, 2>> pf2;
    unique_ptr<Nufft<double, double, double, 2>> pd2;
    unique_ptr<Nufft< float,  float,  float, 3>> pf3;
    unique_ptr<Nufft<double, double, double, 3>> pd3;

    template<typename T, size_t ndim> void construct(
      unique_ptr<Nufft<T,T,T,ndim>> &ptr,
      bool gridding, const py::array &coord_,
      const py::object &uniform_shape_,
      double epsilon_, 
      size_t nthreads_, 
      double sigma_min, double sigma_max,
      double periodicity, bool fft_order_)
      {
      auto coord = to_cmav<T,2>(coord_);
      auto shp = to_array<size_t,ndim>(uniform_shape_);
      {
      py::gil_scoped_release release;
      ptr = make_unique<Nufft<T,T,T,ndim>> (gridding, coord, shp,
        epsilon_, nthreads_, sigma_min, sigma_max, periodicity, fft_order_);
      }
      }
    template<typename T, size_t ndim> py::array do_nu2u(
      const unique_ptr<Nufft<T,T,T,ndim>> &ptr,
      bool forward, size_t verbosity, const py::array &points_,
      py::object &uniform__) const
      {
      auto points = to_cmav<complex<T>,1>(points_);
      auto uniform_ = get_optional_Pyarr<complex<T>>(uniform__, uniform_shape);
      auto uniform = to_vmav<complex<T>,ndim>(uniform_);
      {
      py::gil_scoped_release release;
      ptr->nu2u(forward, verbosity, points, uniform);
      }
      return uniform_;
      }
    template<typename T, size_t ndim> py::array do_u2nu(
      const unique_ptr<Nufft<T,T,T,ndim>> &ptr,
      bool forward, size_t verbosity, const py::array &uniform_,
      py::object &points__) const
      {
      auto uniform = to_cmav<complex<T>,ndim>(uniform_);
      auto points_ = get_optional_Pyarr<complex<T>>(points__, {npoints});
      auto points = to_vmav<complex<T>,1>(points_);
      {
      py::gil_scoped_release release;
      ptr->u2nu(forward, verbosity, uniform, points);
      }
      return points_;
      }

  public:
    Py_Nufftplan(bool gridding, const py::array &coord_,
                 const py::object &uniform_shape_,
                 double epsilon_, 
                 size_t nthreads_, 
                 double sigma_min, double sigma_max,
                 double periodicity, bool fft_order_)
      : uniform_shape(py::cast<vector<size_t>>(uniform_shape_)),
        npoints(coord_.shape(0))
      {
      auto ndim = uniform_shape.size();
      MR_assert((ndim>=1)&&(ndim<=3), "unsupported dimensionality");
      if (isPyarr<double>(coord_))
        {
        if (ndim==1)
          construct(pd1, gridding, coord_, uniform_shape_, epsilon_, nthreads_,
                    sigma_min, sigma_max, periodicity, fft_order_);
        else if (ndim==2)
          construct(pd2, gridding, coord_, uniform_shape_, epsilon_, nthreads_,
            sigma_min, sigma_max, periodicity, fft_order_);
        else if (ndim==3)
          construct(pd3, gridding, coord_, uniform_shape_, epsilon_, nthreads_,
            sigma_min, sigma_max, periodicity, fft_order_);
        }
      else if (isPyarr<float>(coord_))
        {
        if (ndim==1)
          construct(pf1, gridding, coord_, uniform_shape_, epsilon_, nthreads_,
            sigma_min, sigma_max, periodicity, fft_order_);
        else if (ndim==2)
          construct(pf2, gridding, coord_, uniform_shape_, epsilon_, nthreads_,
            sigma_min, sigma_max, periodicity, fft_order_);
        else if (ndim==3)
          construct(pf3, gridding, coord_, uniform_shape_, epsilon_, nthreads_,
            sigma_min, sigma_max, periodicity, fft_order_);
        }
      else
        MR_fail("unsupported");
      }

    py::array nu2u(bool forward, size_t verbosity,
      const py::array &points_, py::object &uniform_)
      {
      if (pd1) return do_nu2u(pd1, forward, verbosity, points_, uniform_);
      if (pf1) return do_nu2u(pf1, forward, verbosity, points_, uniform_);
      if (pd2) return do_nu2u(pd2, forward, verbosity, points_, uniform_);
      if (pf2) return do_nu2u(pf2, forward, verbosity, points_, uniform_);
      if (pd3) return do_nu2u(pd3, forward, verbosity, points_, uniform_);
      if (pf3) return do_nu2u(pf3, forward, verbosity, points_, uniform_);
      MR_fail("unsupported");
      }
    py::array u2nu(bool forward, size_t verbosity,
      const py::array &uniform_, py::object &points_)
      {
      if (pd1) return do_u2nu(pd1, forward, verbosity, uniform_, points_);
      if (pf1) return do_u2nu(pf1, forward, verbosity, uniform_, points_);
      if (pd2) return do_u2nu(pd2, forward, verbosity, uniform_, points_);
      if (pf2) return do_u2nu(pf2, forward, verbosity, uniform_, points_);
      if (pd3) return do_u2nu(pd3, forward, verbosity, uniform_, points_);
      if (pf3) return do_u2nu(pf3, forward, verbosity, uniform_, points_);
      MR_fail("unsupported");
      }
  };


constexpr const char *u2nu_DS = R"""(
Type 2 non-uniform FFT (uniform to non-uniform)

Parameters
----------
grid : numpy.ndarray(1D/2D/3D, dtype=complex)
    the grid of input data
coord : numpy.ndarray((npoints, ndim), dtype=numpy.float32 or numpy.float64)
    the coordinates of the npoints non-uniform points.
    ndim must be the same as grid.ndim
    Periodicity is assumed; the coordinates don't have to lie inside a
    particular interval, but smaller absolute coordinate values help accuracy
forward : bool
    if True, perform the FFT with exponent -1, else +1.
epsilon : float
    desired accuracy
    for single precision inputs, this must be >1e-6, for double precision it
    must be >2e-13
nthreads : int >= 0
    the number of threads to use for the computation
    if 0, use as many threads as there are hardware threads available on the system
out : numpy.ndarray((npoints,), same data type as grid), optional
    if provided, this will be used to store the result
verbosity: int
    0: no console output
    1: some diagnostic console output
sigma_min, sigma_max: float
    minimum and maximum allowed oversampling factors
periodicity: float
    periodicity of the coordinates
fft_order: bool
    if False, grids start with the most negative Fourier node
    if True, grids start with the zero Fourier mode

Returns
-------
numpy.ndarray((npoints,), same data type as grid)
    the computed values at the specified non-uniform grid points.
    Identical to `out` if it was provided
)""";

constexpr const char *nu2u_DS = R"""(
Type 1 non-uniform FFT (non-uniform to uniform)

Parameters
----------
points : numpy.ndarray((npoints,), dtype=numpy.complex)
    The input values at the specified non-uniform grid points
coord : numpy.ndarray((npoints, ndim), dtype=numpy.float32 or numpy.float64)
    the coordinates of the npoints non-uniform points.
    ndim must be the same as out.ndim
    Periodicity is assumed; the coordinates don't have to lie inside a
    particular interval, but smaller absolute coordinate values help accuracy
forward : bool
    if True, perform the FFT with exponent -1, else +1.
epsilon : float
    desired accuracy
    for single precision inputs, this must be >1e-6, for double precision it
    must be >2e-13
nthreads : int >= 0
    the number of threads to use for the computation
    if 0, use as many threads as there are hardware threads available on the system
out : numpy.ndarray(1D/2D/3D, same dtype as points)
    the grid of output data
    Note: this is a mandatory parameter, since its shape defines the grid dimensions!
verbosity: int
    0: no console output
    1: some diagnostic console output
sigma_min, sigma_max: float
    minimum and maximum allowed oversampling factors
    1.2 <= sigma_min < sigma_max <= 2.5
periodicity: float
    periodicity of the coordinates
fft_order: bool
    if False, grids start with the most negative Fourier node
    if True, grids start with the zero Fourier mode

Returns
-------
numpy.ndarray(1D/2D/3D, same dtype as points)
    the computed grid values.
    Identical to `out`.
)""";

constexpr const char *plan_init_DS = R"""(
Nufft plan constructor

Parameters
----------
nu2u : bool
    True: plan will be used for nu2u transforms
    False: plan will be used for u2nu transforms
    The resulting plan can actually be used for both transform types, but
    optimization will be better for the requested type.
coord : numpy.ndarray((npoints, ndim), dtype=numpy.float32 or numpy.float64)
    the coordinates of the npoints non-uniform points.
    Periodicity is assumed; the coordinates don't have to lie inside a
    particular interval, but smaller absolute coordinate values help accuracy
grid_shape : tuple(int) of length ndim
    the shape of the uniform grid
epsilon : float
    desired accuracy
    for single precision inputs, this must be >1e-6, for double precision it
    must be >2e-13
nthreads : int >= 0
    the number of threads to use for the computation
    if 0, use as many threads as there are hardware threads available on the system
sigma_min, sigma_max: float
    minimum and maximum allowed oversampling factors
    1.2 <= sigma_min < sigma_max <= 2.5
periodicity: float
    periodicity of the coordinates
fft_order: bool
    if False, grids start with the most negative Fourier node
    if True, grids start with the zero Fourier mode
)""";

constexpr const char *plan_nu2u_DS = R"""(
Perform a pre-planned nu2u transform.

Parameters
----------
forward : bool
    if True, perform the FFT with exponent -1, else +1.
verbosity: int
    0: no console output
    1: some diagnostic console output
points : numpy.ndarray((npoints,), dtype=numpy.complex)
    The input values at the specified non-uniform grid points
out : numpy.ndarray(1D/2D/3D, same dtype as points)
    if provided, this will be used to store he result.

Returns
-------
numpy.ndarray(1D/2D/3D, same dtype as points)
    the computed grid values.
    Identical to `out` if it was provided.
)""";

constexpr const char *plan_u2nu_DS = R"""(
Perform a pre-planned u2nu transform.

Parameters
----------
forward : bool
    if True, perform the FFT with exponent -1, else +1.
verbosity: int
    0: no console output
    1: some diagnostic console output
grid : numpy.ndarray(1D/2D/3D, dtype=complex)
    the grid of input data
out : numpy.ndarray((npoints,), same data type as grid), optional
    if provided, this will be used to store the result

Returns
-------
numpy.ndarray((npoints,), same data type as grid)
    the computed values at the specified non-uniform grid points.
    Identical to `out` if it was provided.
)""";

constexpr const char *bestEpsilon_DS = R"""(
Computes the smallest possible error for the given NUFFT parameters.

Parameters
----------
ndim : int (1-3)
    the dimensionality of the transform
singleprec : bool
    True if np.float32/np.complex64 are used, otherwise False
sigma_min, sigma_max: float
    minimum and maximum allowed oversampling factors
    1.2 <= sigma_min < sigma_max <= 2.5

Returns
-------
float
    the smallest possible error that can be achieved for the given parameters.
)""";


void add_nufft(py::module_ &msup)
  {
  using namespace pybind11::literals;
  auto m = msup.def_submodule("nufft");

  m.def("u2nu", &Py_u2nu, u2nu_DS,  py::kw_only(), "grid"_a, "coord"_a,
        "forward"_a, "epsilon"_a, "nthreads"_a=1, "out"_a=None, "verbosity"_a=0,
        "sigma_min"_a=1.2, "sigma_max"_a=2.51, "periodicity"_a=2*pi,
        "fft_order"_a=false);
  m.def("nu2u", &Py_nu2u, nu2u_DS, py::kw_only(), "points"_a, "coord"_a,
        "forward"_a, "epsilon"_a, "nthreads"_a=1, "out"_a=None, "verbosity"_a=0,
        "sigma_min"_a=1.2, "sigma_max"_a=2.51, "periodicity"_a=2*pi,
        "fft_order"_a=false);
  m.def("bestEpsilon", &bestEpsilon, bestEpsilon_DS, py::kw_only(),
        "ndim"_a, "singleprec"_a, "sigma_min"_a=1.1, "sigma_max"_a=2.6);

  py::class_<Py_Nufftplan> (m, "plan", py::module_local())
    .def(py::init<bool, const py::array &, const py::object &,
                  double, size_t, double, double, double, bool>(),
      plan_init_DS, py::kw_only(), "nu2u"_a, "coord"_a, "grid_shape"_a,
        "epsilon"_a, "nthreads"_a=0, "sigma_min"_a=1.1, "sigma_max"_a=2.6,
        "periodicity"_a=2*pi, "fft_order"_a=false)
    .def("nu2u", &Py_Nufftplan::nu2u, plan_nu2u_DS, py::kw_only(), "forward"_a,
      "verbosity"_a=0, "points"_a, "out"_a=None)
    .def("u2nu", &Py_Nufftplan::u2nu, py::kw_only(), "forward"_a,
      "verbosity"_a=0, "grid"_a, "out"_a=None);
  }

}

using detail_pymodule_nufft::add_nufft;

}
