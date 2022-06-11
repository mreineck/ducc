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
  py::object &out__)
  {
  using Tpoints = decltype(conj(Tgrid(0)));

  auto coord = to_cmav<Tcoord,2>(coord_);
  auto ndim = coord.shape(1);
  if (ndim==2)
    {
    auto grid = to_cmav<complex<Tgrid>,2>(grid_);
    auto out_ = get_optional_Pyarr<Tpoints>(out__, {coord.shape(0)});
    auto out = to_vmav<Tpoints,1>(out_);
    {
    py::gil_scoped_release release;
    dirty2ms_nufft<Tgrid,Tgrid>(coord,grid,forward,epsilon,nthreads,out,1,1.2,2.0);
    }
    return move(out_);
    }
  MR_fail("unsupported");
  }
py::array Py_u2nu(const py::array &grid,
  const py::array &coord, bool forward, double epsilon, size_t nthreads,
  py::object &out)
  {
  if (isPyarr<double>(coord))  // double precision coordinates
    {
    if (isPyarr<complex<double>>(grid))
      return Py2_u2nu<double, double>(grid, coord, forward, epsilon, nthreads, out);
    else if (isPyarr<complex<float>>(grid))  // double precision R2C
      return Py2_u2nu<float, double>(grid, coord, forward, epsilon, nthreads, out);
    }
  else if (isPyarr<double>(coord))  // single precision coordinates
    {
    if (isPyarr<complex<double>>(grid))
      return Py2_u2nu<double, float>(grid, coord, forward, epsilon, nthreads, out);
    else if (isPyarr<complex<float>>(grid))
      return Py2_u2nu<float, float>(grid, coord, forward, epsilon, nthreads, out);
    }
  MR_fail("not yet supported");
  }

template<typename Tpoints, typename Tcoord> py::array Py2_nu2u(const py::array &points_,
  const py::array &coord_, bool forward, double epsilon, size_t nthreads,
  py::object &out__)
  {
  using Tgrid = Tpoints;
  auto coord = to_cmav<Tcoord,2>(coord_);
  auto ndim = coord.shape(1);
  if (ndim==2)
    {
    auto points = to_cmav<complex<Tpoints>,1>(points_);
  //  auto out_ = make_Pyarr<Tgrid>(out__, {coord.shape(0)});
    auto out = to_vmav<complex<Tgrid>,2>(out__);
    {
    py::gil_scoped_release release;
    ms2dirty_nufft<Tgrid,Tgrid>(coord,points,forward,epsilon,nthreads,out,1,1.2,2.0);
    }
    return move(out__);
    }
  MR_fail("unsupported");
  }
py::array Py_nu2u(const py::array &points,
  const py::array &coord, bool forward, double epsilon, size_t nthreads,
  py::object &out)
  {
  if (isPyarr<double>(coord))  // double precision coordinates
    {
    if (isPyarr<complex<double>>(points))
      return Py2_nu2u<double, double>(points, coord, forward, epsilon, nthreads, out);
    else if (isPyarr<complex<float>>(points))  // double precision R2C
      return Py2_nu2u<float, double>(points, coord, forward, epsilon, nthreads, out);
    }
  else if (isPyarr<double>(coord))  // single precision coordinates
    {
    if (isPyarr<complex<double>>(points))
      return Py2_nu2u<double, float>(points, coord, forward, epsilon, nthreads, out);
    else if (isPyarr<complex<float>>(points))
      return Py2_nu2u<float, float>(points, coord, forward, epsilon, nthreads, out);
    }
  MR_fail("not yet supported");
  }

void add_nufft(py::module_ &msup)
  {
  using namespace pybind11::literals;
  auto m = msup.def_submodule("nufft");

  m.def("u2nu", &Py_u2nu, py::kw_only(), "grid"_a, "coord"_a, "forward"_a,
        "epsilon"_a, "nthreads"_a=1, "out"_a=None);
  m.def("nu2u", &Py_nu2u, py::kw_only(), "points"_a, "coord"_a, "forward"_a,
        "epsilon"_a, "nthreads"_a=1, "out"_a=None);
  }

}

using detail_pymodule_nufft::add_nufft;

}
