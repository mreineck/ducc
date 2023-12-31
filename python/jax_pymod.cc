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

#include <vector>
#include <cmath>
#include <cstdint>

#include "ducc0/bindings/pybind_utils.h"
#include "ducc0/bindings/array_descriptor.h"

namespace ducc0 {

namespace detail_pymodule_jax {

namespace py = pybind11;
using namespace std;
using namespace ducc0;

uint8_t dtype2typecode(const py::object &type)
  {
  auto type2 = normalizeDtype(type);
  if (isDtype<float>(type2)) return Typecode<float>::value;
  if (isDtype<double>(type2)) return Typecode<double>::value;
  if (isDtype<complex<float>>(type2)) return Typecode<complex<float>>::value;
  if (isDtype<complex<double>>(type2)) return Typecode<complex<double>>::value;
  MR_fail("unsupported data type");
  }

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
  auto shape_in = tuple2vector<size_t>(state["shape_in"]);
  auto shape_out = tuple2vector<size_t>(state["shape_out"]);
  auto dtin = dtype2typecode(state["dtype_in"]);
  auto dtout = dtype2typecode(state["dtype_out"]);

  if (adjoint) { swap(shape_in, shape_out); swap(dtin, dtout); }

  ArrayDescriptor ain(in[1], shape_in, dtin),
                  aout(out, shape_out, dtout);

  auto tin = ain.typecode;
  auto tout = aout.typecode;

  py::array pyin, pyout;
  if (isTypecode<float>(tin))
    pyin = make_Pyarr_from_cfmav(ain.to_cfmav<false,float>());
  else if (isTypecode<double>(tin))
    pyin = make_Pyarr_from_cfmav(ain.to_cfmav<false,double>());
  else if (isTypecode<complex<float>>(tin))
    pyin = make_Pyarr_from_cfmav(ain.to_cfmav<false,complex<float>>());
  else if (isTypecode<complex<double>>(tin))
    pyin = make_Pyarr_from_cfmav(ain.to_cfmav<false,complex<double>>());
  else
    MR_fail("unsupported input array type");

  if (isTypecode<float>(tout))
    pyout = make_Pyarr_from_vfmav(aout.to_vfmav<false,float>());
  else if (isTypecode<double>(tout))
    pyout = make_Pyarr_from_vfmav(aout.to_vfmav<false,double>());
  else if (isTypecode<complex<float>>(tout))
    pyout = make_Pyarr_from_vfmav(aout.to_vfmav<false,complex<float>>());
  else if (isTypecode<complex<double>>(tout))
    pyout = make_Pyarr_from_vfmav(aout.to_vfmav<false,complex<double>>());
  else
    MR_fail("unsupported output array type");

  state["func"](pyin, pyout, adjoint, state);
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
