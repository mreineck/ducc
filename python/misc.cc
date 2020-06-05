/*
 *  This file is part of DUCC.
 *
 *  DUCC is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  DUCC is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with DUCC; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 *  For more information about HEALPix, see http://healpix.sourceforge.net
 */

/*
 *  DUCC is being developed at the Max-Planck-Institut fuer Astrophysik
 */

/*
 *  Copyright (C) 2020 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

#include "mr_util/math/constants.h"
#include "mr_util/math/gl_integrator.h"

namespace mr {

namespace detail_pymodule_misc {

using namespace std;
namespace py = pybind11;

using a_d_c = py::array_t<double, py::array::c_style | py::array::forcecast>;

a_d_c GL_weights(int64_t nlat, int64_t nlon)
  {
  a_d_c res(nlat);
  auto rr=res.mutable_unchecked<1>();
  GL_Integrator integ(nlat);
  auto wgt = integ.weights();
  for (size_t i=0; i<size_t(rr.shape(0)); ++i)
    rr[i] = wgt[i]*twopi/nlon;
  return res;
  }

a_d_c GL_thetas(int64_t nlat)
  {
  a_d_c res(nlat);
  auto rr=res.mutable_unchecked<1>();
  GL_Integrator integ(nlat);
  auto x = integ.coords();
  for (size_t i=0; i<size_t(rr.shape(0)); ++i)
    rr[i] = acos(-x[i]);
  return res;
  }

const char *misc_DS = R"""(
Various unsorted utilities
)""";

void add_misc(py::module &msup)
  {
  using namespace pybind11::literals;
  auto m = msup.def_submodule("misc");
  m.doc() = misc_DS;

  m.def("GL_weights",&GL_weights, "nlat"_a, "nlon"_a);
  m.def("GL_thetas",&GL_thetas, "nlat"_a);
  }

}

using detail_pymodule_misc::add_misc;

}

