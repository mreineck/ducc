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
#include "ducc0/bindings/typecode.h"
#include "ducc0/bindings/pybind_utils.h"

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

py::object typecode2dtype(uint8_t typecode)
  {
  if (isTypecode<float>(typecode)) return Dtype<float>();
  if (isTypecode<double>(typecode)) return Dtype<double>();
  if (isTypecode<complex<float>>(typecode)) return Dtype<complex<float>>();
  if (isTypecode<complex<double>>(typecode)) return Dtype<complex<double>>();
  MR_fail("unsupported data type");
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
  py::handle hnd(*reinterpret_cast<PyObject **>(in[2]));
  auto obj = py::reinterpret_borrow<py::object>(hnd);
  const py::dict state(obj);
  size_t idx = 3;
  auto dtin = typecode2dtype(uint8_t(*reinterpret_cast<int64_t *>(in[idx++])));
  size_t ndim_in = *reinterpret_cast<uint64_t *>(in[idx++]);
  vector<size_t> shape_in;
  for (size_t i=0; i<ndim_in; ++i)
    shape_in.push_back(*reinterpret_cast<uint64_t *>(in[idx++]));
  auto dtout = typecode2dtype(uint8_t(*reinterpret_cast<int64_t *>(in[idx++])));
  size_t ndim_out = *reinterpret_cast<uint64_t *>(in[idx++]);
  vector<size_t> shape_out;
  for (size_t i=0; i<ndim_out; ++i)
    shape_out.push_back(*reinterpret_cast<uint64_t *>(in[idx++]));

  py::str dummy;
  py::array pyin (dtin, shape_in, in[0], dummy);
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
