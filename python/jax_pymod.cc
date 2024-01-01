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
 *  Copyright (C) 2023 Max-Planck-Society
 *  Authors: Martin Reinecke, Jakob Roth, Gordian Edenhofer 
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>

#include "ducc0/infra/error_handling.h"

namespace ducc0 {

namespace detail_pymodule_jax {

namespace py = pybind11;
using namespace std;
using namespace ducc0;

template<typename T> vector<T> tuple2vector (const py::tuple &tp)
  {
  vector<size_t> res;
  for (auto v:tp)
    res.push_back(v.cast<size_t>());
  return res;
  }

// https://en.cppreference.com/w/cpp/numeric/bit_cast
template <class To, class From>
typename std::enable_if<sizeof(To) == sizeof(From) && std::is_trivially_copyable<From>::value &&
                            std::is_trivially_copyable<To>::value,
                        To>::type
bit_cast(const From& src) noexcept
  {
  static_assert(
      std::is_trivially_constructible<To>::value,
      "This implementation additionally requires destination type to be trivially constructible");

  To dst;
  memcpy(&dst, &src, sizeof(To));
  return dst;
  }

template <typename T>
pybind11::capsule EncapsulateFunction(T* fn)
  {
  return pybind11::capsule(bit_cast<void*>(fn), "xla._CUSTOM_CALL_TARGET");
  }

void linop(void *out, void **in, bool adjoint)
  {
  py::gil_scoped_acquire get_GIL;
  py::handle hnd(*reinterpret_cast<PyObject **>(in[0]));
  auto obj = py::reinterpret_borrow<py::object>(hnd);
  const py::dict state(obj);
  auto shape_in = tuple2vector<size_t>(state[adjoint ? "_shape_out" : "_shape_in"]);
  auto shape_out = tuple2vector<size_t>(state[adjoint ? "_shape_in" : "_shape_out"]);
  auto dtin = state[adjoint ? "_dtype_out" : "_dtype_in"];
  auto dtout = state[adjoint ? "_dtype_in" : "_dtype_out"];

  py::str dummy;
  py::array pyin (dtin, shape_in, in[1], dummy);
MR_assert(!pyin.owndata(), "oops1");
  pyin.attr("flags").attr("writeable")=false;
MR_assert(!pyin.writeable(), "oops40");
  py::array pyout (dtout, shape_out, out, dummy);
MR_assert(!pyout.owndata(), "oops3");
MR_assert(pyout.writeable(), "oops40");
  state["_func"](pyin, pyout, adjoint, state);
  }

void linop_forward(void *out, void **in) { linop(out, in, false); }
void linop_adjoint(void *out, void **in) { linop(out, in, true); }

pybind11::dict Registrations()
  {
  pybind11::dict dict;
  dict["cpu_linop_forward"] = EncapsulateFunction(linop_forward);
  dict["cpu_linop_adjoint"] = EncapsulateFunction(linop_adjoint);
  return dict;
  }

void add_jax(py::module_ &msup)
  {
  auto m = msup.def_submodule("jax");
  m.def("registrations", &Registrations);
  }

}

using detail_pymodule_jax::add_jax;

}
