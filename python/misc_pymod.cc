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
 */

/*
 *  DUCC is being developed at the Max-Planck-Institut fuer Astrophysik
 */

/*
 *  Copyright (C) 2020-2021 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

#include "ducc0/infra/mav.h"
#include "ducc0/infra/transpose.h"
#include "ducc0/math/fft.h"
#include "ducc0/math/constants.h"
#include "ducc0/math/gl_integrator.h"
#include "ducc0/bindings/pybind_utils.h"

namespace ducc0 {

namespace detail_pymodule_misc {

using namespace std;
namespace py = pybind11;

py::array Py_GL_weights(size_t nlat, size_t nlon)
  {
  auto res = make_Pyarr<double>({nlat});
  auto res2 = to_mav<double,1>(res, true);
  GL_Integrator integ(nlat);
  auto wgt = integ.weights();
  for (size_t i=0; i<res2.shape(0); ++i)
    res2.v(i) = wgt[i]*twopi/nlon;
  return move(res);
  }

py::array Py_GL_thetas(size_t nlat)
  {
  auto res = make_Pyarr<double>({nlat});
  auto res2 = to_mav<double,1>(res, true);
  GL_Integrator integ(nlat);
  auto x = integ.coords();
  for (size_t i=0; i<res2.shape(0); ++i)
    res2.v(i) = acos(-x[i]);
  return move(res);
  }

template<typename T> py::array Py2_transpose(const py::array &in, py::array &out)
  {
  auto in2 = to_fmav<T>(in, false);
  auto out2 = to_fmav<T>(out, true);
  transpose(in2, out2, [](const T &in, T &out){out=in;});
  return out;
  }

py::array Py_transpose(const py::array &in, py::array &out)
  {
  if (isPyarr<float>(in))
    return Py2_transpose<float>(in, out);
  if (isPyarr<double>(in))
    return Py2_transpose<double>(in, out);
  if (isPyarr<complex<float>>(in))
    return Py2_transpose<complex<float>>(in, out);
  if (isPyarr<complex<double>>(in))
    return Py2_transpose<complex<double>>(in, out);
  if (isPyarr<int>(in))
    return Py2_transpose<int>(in, out);
  if (isPyarr<long>(in))
    return Py2_transpose<long>(in, out);
  MR_fail("unsupported datatype");
  }


const char *misc_DS = R"""(
Various unsorted utilities

Notes
-----

The functionality in this module is not considered to have a stable interface
and also may be moved to other modules in the future. If you use it, be prepared
to adjust your code at some point ion the future!
)""";

void add_misc(py::module_ &msup)
  {
  using namespace pybind11::literals;
  auto m = msup.def_submodule("misc");
  m.doc() = misc_DS;

  m.def("GL_weights",&Py_GL_weights, "nlat"_a, "nlon"_a);
  m.def("GL_thetas",&Py_GL_thetas, "nlat"_a);

  m.def("transpose",&Py_transpose, "in"_a, "out"_a);
  }

}

using detail_pymodule_misc::add_misc;

}

