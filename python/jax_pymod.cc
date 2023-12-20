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
#include "ducc0/fft/fft.h"
#include "ducc0/fft/fftnd_impl.h"
#include "ducc0/sht/sht.h"
#include "ducc0/healpix/healpix_base.h"

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

template<typename T> void do_fht2(const ArrayDescriptor &ain,
  ArrayDescriptor &aout, const py::dict &state, const vector<size_t> &axes,
  size_t nthreads)
  {
  auto ain2(ain.to_cfmav<false,T>());
  auto aout2(aout.to_vfmav<false,T>());
  auto fct(state["fct"].cast<T>());
  {
  py::gil_scoped_release release;
  r2r_genuine_fht(ain2, aout2, axes, fct, nthreads);
  }
  }

void do_fht(const ArrayDescriptor &ain, ArrayDescriptor &aout,
  const py::dict &state, bool /*adjoint*/)
  {
  auto axes = tuple2vector<size_t>(state["axes"]);
  auto nthreads = state["nthreads"].cast<size_t>();
  if (isTypecode<float>(ain.typecode))
    do_fht2<float>(ain, aout, state, axes, nthreads);
  else if (isTypecode<double>(ain.typecode))
    do_fht2<double>(ain, aout, state, axes, nthreads);
  else    
    MR_fail("Bad types for FHT");
  }

template<typename T> void do_sht2d2(const ArrayDescriptor &ain,
  ArrayDescriptor &aout, const py::dict &/*state*/, size_t spin, size_t lmax,
  size_t mmax, const string &geometry, size_t nthreads, bool adjoint)
  {
  vmav<size_t,1> mstart({mmax+1}, UNINITIALIZED);
  for (size_t m=0, idx=0; m<=mmax; ++m, idx+=lmax+1-m)
    mstart(m) = idx;

  if (adjoint)
    {
    auto ain2(ain.to_cmav<false,T,3>());
    auto aout2(aout.to_vmav<false,complex<T>,2>());
    {
    py::gil_scoped_release release;
    adjoint_synthesis_2d(aout2, ain2, spin, lmax, mstart, 1, geometry, 0., nthreads, STANDARD);
    }
    }
  else
    {
    auto ain2(ain.to_cmav<false,complex<T>,2>());
    auto aout2(aout.to_vmav<false,T,3>());
    {
    py::gil_scoped_release release;
    synthesis_2d(ain2, aout2, spin, lmax, mstart, 1, geometry, 0., nthreads, STANDARD);
    }
    }
  }
void do_sht2d(const ArrayDescriptor &ain, ArrayDescriptor &aout,
  const py::dict &state, bool adjoint)
  {
  auto spin = state["spin"].cast<size_t>();
  auto lmax = state["lmax"].cast<size_t>();
  auto mmax = state["mmax"].cast<size_t>();
  auto geometry = state["geometry"].cast<string>();
  auto nthreads = state["nthreads"].cast<size_t>();

  auto floattype = adjoint ? ain.typecode : aout.typecode;
  if (isTypecode<float>(floattype))
    do_sht2d2<float>(ain, aout, state, spin, lmax, mmax, geometry, nthreads, adjoint);
  else if (isTypecode<double>(floattype))
    do_sht2d2<double>(ain, aout, state, spin, lmax, mmax, geometry, nthreads, adjoint);
  else    
    MR_fail("Bad types for SHT2D transform");
  }

template<typename T> void do_sht_healpix2(const ArrayDescriptor &ain,
  ArrayDescriptor &aout, const py::dict &/*state*/, size_t spin, size_t lmax,
  size_t mmax, size_t nside, size_t nthreads, bool adjoint)
  {
  vmav<size_t,1> mstart({mmax+1}, UNINITIALIZED);
  for (size_t m=0, idx=0; m<=mmax; ++m, idx+=lmax+1-m)
    mstart(m) = idx;

  Healpix_Base2 base(nside, RING, SET_NSIDE);
  auto nrings = size_t(4*nside-1);
  vmav<double,1> theta({nrings}, UNINITIALIZED), phi0({nrings}, UNINITIALIZED);
  vmav<size_t,1> nphi({nrings}, UNINITIALIZED), ringstart({nrings}, UNINITIALIZED);
  for (size_t r=0, rs=nrings-1; r<=rs; ++r, --rs)
    {
    int64_t startpix, ringpix;
    double ringtheta;
    bool shifted;
    base.get_ring_info2 (r+1, startpix, ringpix, ringtheta, shifted);
    theta(r) = ringtheta;
    theta(rs) = pi-ringtheta;
    nphi(r) = nphi(rs) = size_t(ringpix);
    phi0(r) = phi0(rs) = shifted ? (pi/ringpix) : 0.;
    ringstart(r) = size_t(startpix);
    ringstart(rs) = size_t(base.Npix() - startpix - ringpix);
    }

  if (adjoint)
    {
    auto ain2(ain.to_cmav<false,T,2>());
    auto aout2(aout.to_vmav<false,complex<T>,2>());
    {
    py::gil_scoped_release release;
    adjoint_synthesis(aout2, ain2, spin, lmax, mstart, 1, theta, nphi, phi0, ringstart, 1, nthreads, STANDARD);
    }
    }
  else
    {
    auto ain2(ain.to_cmav<false,complex<T>,2>());
    auto aout2(aout.to_vmav<false,T,2>());
    {
    py::gil_scoped_release release;
    synthesis(ain2, aout2, spin, lmax, mstart, 1, theta, nphi, phi0, ringstart, 1, nthreads, STANDARD);
    }
    }
  }
void do_sht_healpix(const ArrayDescriptor &ain, ArrayDescriptor &aout,
  const py::dict &state, bool adjoint)
  {
  auto spin = state["spin"].cast<size_t>();
  auto lmax = state["lmax"].cast<size_t>();
  auto mmax = state["mmax"].cast<size_t>();
  auto nside = state["nside"].cast<size_t>();
  auto nthreads = state["nthreads"].cast<size_t>();

  auto floattype = adjoint ? ain.typecode : aout.typecode;
  if (isTypecode<float>(floattype))
    do_sht_healpix2<float>(ain, aout, state, spin, lmax, mmax, nside, nthreads, adjoint);
  else if (isTypecode<double>(floattype))
    do_sht_healpix2<double>(ain, aout, state, spin, lmax, mmax, nside, nthreads, adjoint);
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

  auto job = state["job"].cast<string>();
  if (job=="FHT")
    do_fht(ain, aout, state, adjoint); 
  else if (job=="SHT2D")
    do_sht2d(ain, aout, state, adjoint); 
  else if (job=="SHT_Healpix")
    do_sht_healpix(ain, aout, state, adjoint); 
  else
    MR_fail("unsupported operator job type");
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
