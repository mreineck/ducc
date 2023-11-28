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
 *  Author: Martin Reinecke
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <complex>

#include "ducc0/infra/string_utils.h"
#include "ducc0/infra/error_handling.h"
#include "ducc0/math/constants.h"
#include "ducc0/bindings/pybind_utils.h"
#include "ducc0/bindings/array_descriptor.h"

namespace ducc0 {

namespace detail_pymodule_jax {

using namespace std;

namespace py = pybind11;

auto None = py::none();

// abstract base class providing a unified interface for various linear operators
class LinOp
  {
  public:
    virtual void apply(const ArrayDescriptor &in, const ArrayDescriptor &out) const = 0;
    virtual void applyAdjoint(const ArrayDescriptor &in, const ArrayDescriptor &out) const = 0;
    virtual vector<size_t> shape_apply(const vector<size_t> &shp_in) const = 0;
    virtual uint8_t type_apply(const uint8_t &type_in) const = 0;
    virtual vector<size_t> shape_applyAdjoint(const vector<size_t> &shp_in) const = 0;
    virtual uint8_t type_applyAdjoint(const uint8_t &type_in) const = 0;

    virtual ~LinOp() {}
  };

class C2C: public LinOp
  {
  private:
    bool fwd;
    vector<size_t> axes;
    size_t nthreads;

  public:
    C2C(bool fwd_, const vector<size_t> &axes_, size_t nthreads_)
      : fwd(fwd_), axes(axes_), nthreads(nthreads_) {}


    // virtual
    void apply(const ArrayDescriptor &in, const ArrayDescriptor &out) const
      {
      if (isDtype<complex<float>>(in.dtype))
        c2c(to_cfmav<false, complex<float>>(in),
            to_vfmav<false, complex<float>>(out),
            axes, fwd, 1.f, nthreads);
      else if (isDtype<complex<double>>(in.dtype))
        c2c(to_cfmav<false, complex<double>>(in),
            to_vfmav<false, complex<double>>(out),
            axes, fwd, 1., nthreads);
      else
        MR_fail("bad invocation");
      }
    // virtual
    void applyAdjoint(const ArrayDescriptor &in, const ArrayDescriptor &out) const
      {
      if (isDtype<complex<float>>(in.dtype))
        c2c(to_cfmav<false, complex<float>>(in),
            to_vfmav<false, complex<float>>(out),
            axes, !fwd, 1.f, nthreads);
      else if (isDtype<complex<double>>(in.dtype))
        c2c(to_cfmav<false, complex<double>>(in),
            to_vfmav<false, complex<double>>(out),
            axes, !fwd, 1., nthreads);
      else
        MR_fail("bad invocation");
      }
    // virtual
    vector<size_t> shape_apply(const vector<size_t> &shp_in) const { return shp_in; }
    // virtual
    uint8_t type_apply(const uint8_t &type_in) const { return type_in; }
    // virtual
    vector<size_t> shape_applyAdjoint(const vector<size_t> &shp_in) const { return shp_in; }
    // virtual
    uint8_t type_applyAdjoint(const uint8_t &type_in) const { return type_in; }
  };

uint8_t dtype2typecode(const py::array &arr)
  {
  if (isPyarr<float>(arr)) return Typecode<float>::value;
  if (isPyarr<double>(arr)) return Typecode<double>::value;
  if (isPyarr<complex<float>>(arr)) return Typecode<complex<float>>::value;
  if (isPyarr<complex<double>>(arr)) return Typecode<complex<double>>::value;
  MR_fail("unsupported data type");
  }

ArrayDescriptor arrdesc(const py::array &arr)
  {
  ArrayDescriptor res;
  res.ndim = arr.ndim();
  MR_assert(res.ndim<=ArrayDescriptor::maxdim, "dimensionality too high");
  res.dtype = dtype2typecode(arr);
  res.data = const_cast<void *>(arr.data());
  for (size_t i=0; i<res.ndim; ++i)
    res.shape[i] = size_t(arr.shape(int(i)));
  auto st = ptrdiff_t(typeSize(res.dtype));
  for (size_t i=0; i<res.ndim; ++i)
    {
    auto tmp = arr.strides(int(i));
    MR_assert((tmp/st)*st==tmp, "bad stride");
    res.stride[i] = tmp/st;
    }
  res.readonly = 0;
  return res;
  }

py::array makeFlexiblePyarr(const vector<size_t> &dims, uint8_t typecode)
  {
  if (typecode==Typecode<float>::value) return make_Pyarr<float>(dims);
  if (typecode==Typecode<double>::value) return make_Pyarr<double>(dims);
  if (typecode==Typecode<complex<float>>::value) return make_Pyarr<complex<float>>(dims);
  if (typecode==Typecode<complex<double>>::value) return make_Pyarr<complex<double>>(dims);
  MR_fail("unsupported data type");
  }

class Py_Linop
  {
  private:
    unique_ptr<LinOp> op;

  public:
    Py_Linop(const string &job, const py::dict &params)
      {
      if (job=="c2c")
        {
        op = make_unique<C2C>(
          params["forward"].cast<bool>(),
          params["axes"].cast<vector<size_t>>(),
          params["nthreads"].cast<size_t>());
        }
      else
        MR_fail("unrecognized kind of operator");
      }

    py::array apply(const py::array &in) const
      {
      auto desc_in = arrdesc(in);
      vector<size_t> shp;
      for (size_t i=0; i<desc_in.ndim; ++i)
        shp.push_back(size_t(desc_in.shape[i]));
      auto out = makeFlexiblePyarr(op->shape_apply(shp), op->type_apply(desc_in.dtype));
      auto desc_out = arrdesc(out);
      op->apply(desc_in, desc_out);
      return out;
      }
    py::array applyAdjoint(const py::array &in) const
      {
      auto desc_in = arrdesc(in);
      vector<size_t> shp;
      for (size_t i=0; i<desc_in.ndim; ++i)
        shp.push_back(size_t(desc_in.shape[i]));
      auto out = makeFlexiblePyarr(op->shape_applyAdjoint(shp), op->type_applyAdjoint(desc_in.dtype));
      auto desc_out = arrdesc(out);
      op->applyAdjoint(desc_in, desc_out);
      return out;
      }
  };

void add_jax(py::module_ &msup)
  {
  using namespace pybind11::literals;
  auto m = msup.def_submodule("jax");

  py::class_<Py_Linop> (m, "Linop", py::module_local())
    .def(py::init<const string &, const py::dict &>(), "job"_a, "params"_a)
    .def("apply", &Py_Linop::apply, "in"_a)
    .def("applyAdjoint", &Py_Linop::applyAdjoint, "in"_a);
  }

}

using detail_pymodule_jax::add_jax;

}

