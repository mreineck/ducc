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
MR_assert(!forward, "only backward transforms supported at the moment");
  using Tpoints = decltype(conj(Tgrid(0)));

  auto coord = to_cmav<Tcoord,2>(coord_);
  auto ndim = coord.shape(1);
  if (ndim==2)
    {
    auto grid = to_cmav<Tgrid,2>(grid_);
    auto out_ = get_optional_Pyarr<Tpoints>(out__, {coord.shape(0)});
    auto out = to_vmav<Tpoints,1>(out_);
    vmav<Tpoints,2> out2(out.data(), mav_info<2>::shape_t{out.shape(0), 1}, mav_info<2>::stride_t{out.stride(0), 0});
    {
    py::gil_scoped_release release;
    auto wgt = cmav<Tgrid, 2>::build_uniform({out.shape(0),1}, Tgrid(1));
    auto mask = cmav<uint8_t, 2>::build_uniform({out.shape(0),1}, uint8_t(1));
    vmav<double,2> uvw({coord.shape(0),3});
Tgrid SPEEDOFLIGHT = Tgrid(299792458.);
auto pixsize_x = 2*pi/grid.shape(0);
auto pixsize_y = 2*pi/grid.shape(1);
    for (size_t i=0; i<coord.shape(0); ++i)
      {
      uvw(i,0) = coord(i,0)/(2*pi)*SPEEDOFLIGHT/pixsize_x;
      uvw(i,1) = coord(i,1)/(2*pi)*SPEEDOFLIGHT/pixsize_y;
      uvw(i,2) = 0;
      }
    auto freq = cmav<double, 1>::build_uniform({1}, 1.);
    dirty2ms_nufft<Tgrid,Tgrid>(cmav<double,2>(uvw),freq,grid,wgt,mask,pixsize_x,pixsize_y,epsilon,
      nthreads,out2,1,false,false,1.2,2.0,0.,0.);
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
    if (isPyarr<double>(grid))  // double precision R2C
      return Py2_u2nu<double, double>(grid, coord, forward, epsilon, nthreads, out);
    }
  MR_fail("not yet supported");
  }

void add_nufft(py::module_ &msup)
  {
  using namespace pybind11::literals;
  auto m = msup.def_submodule("nufft");

  m.def("u2nu", &Py_u2nu, py::kw_only(), "grid"_a, "coord"_a, "forward"_a,
        "epsilon"_a, "nthreads"_a=1, "out"_a=None);
  }

}

using detail_pymodule_nufft::add_nufft;

}
