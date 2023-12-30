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

#include "ducc0/math/constants.h"
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
bit_cast(const From& src) noexcept {
  static_assert(
      std::is_trivially_constructible<To>::value,
      "This implementation additionally requires destination type to be trivially constructible");

  To dst;
  memcpy(&dst, &src, sizeof(To));
  return dst;
}

template <typename T>
pybind11::capsule EncapsulateFunction(T* fn) {
  return pybind11::capsule(bit_cast<void*>(fn), "xla._CUSTOM_CALL_TARGET");
}

template<typename Tin,typename Tout> void do_pyfunc2(const ArrayDescriptor &ain,
  ArrayDescriptor &aout, const py::dict &state,  bool adjoint)
  {
  auto in(ain.to_cfmav<false,Tin>());
  auto out(aout.to_vfmav<false,Tout>());

  auto Pyin = make_Pyarr_from_cfmav(in);
  auto Pyout = make_Pyarr_from_vfmav(out);
  state["func"](Pyin, Pyout, adjoint, state);
  }
void do_pyfunc(const ArrayDescriptor &ain, ArrayDescriptor &aout,
  const py::dict &state, bool adjoint)
  {
  auto tin = ain.typecode;
  auto tout = aout.typecode;
  if (isTypecode<float>(tin))
    {
    if (isTypecode<float>(tout))
      do_pyfunc2<float,float>(ain, aout, state, adjoint);
    else if (isTypecode<complex<float>>(tout))
      do_pyfunc2<float,complex<float>>(ain, aout, state, adjoint);
    else
      MR_fail("unsupported data types for pyfunc");
    }
  else if (isTypecode<double>(tin))
    {
    if (isTypecode<double>(tout))
      do_pyfunc2<double,double>(ain, aout, state, adjoint);
    else if (isTypecode<complex<double>>(tout))
      do_pyfunc2<double,complex<double>>(ain, aout, state, adjoint);
    else
      MR_fail("unsupported data types for pyfunc");
    }
  else if (isTypecode<complex<float>>(tin))
    {
    if (isTypecode<float>(tout))
      do_pyfunc2<complex<float>,float>(ain, aout, state, adjoint);
    else if (isTypecode<complex<float>>(tout))
      do_pyfunc2<complex<float>,complex<float>>(ain, aout, state, adjoint);
    else
      MR_fail("unsupported data types for pyfunc");
    }
  else if (isTypecode<complex<double>>(tin))
    {
    if (isTypecode<double>(tout))
      do_pyfunc2<complex<double>,double>(ain, aout, state, adjoint);
    else if (isTypecode<complex<double>>(tout))
      do_pyfunc2<complex<double>,complex<double>>(ain, aout, state, adjoint);
    else
      MR_fail("unsupported data types for pyfunc");
    }
  else    
    MR_fail("Bad types for Healpix SHT");
  }

void linop(void *out, void **in, bool adjoint) {
  // Parse the inputs
  py::gil_scoped_acquire get_GIL;
  py::handle hnd(*reinterpret_cast<PyObject **>(in[0]));
  auto obj = py::reinterpret_borrow<py::object>(hnd);
  const py::dict state(obj);
  auto shape_in = tuple2vector<size_t>(state["shape_in"]);
  auto shape_out = tuple2vector<size_t>(state["shape_out"]);
  auto dtin = dtype2typecode(state["dtype_in"]);
  auto dtout = dtype2typecode(state["dtype_out"]);

  if (adjoint) swap(shape_in, shape_out);
  if (adjoint) swap(dtin, dtout);

  ArrayDescriptor ain(in[1], shape_in, dtin),
                  aout(out, shape_out, dtout);

  do_pyfunc(ain, aout, state, adjoint); 
}

void linop_forward(void *out, void **in) { linop(out, in, false); }
void linop_adjoint(void *out, void **in) { linop(out, in, true); }

pybind11::dict Registrations() {
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
