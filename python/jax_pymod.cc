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
    virtual void apply(const ArrayDescriptor &in, const ArrayDescriptor &out, bool adjoint) const = 0;
    virtual vector<size_t> shape_out(const vector<size_t> &shp_in, bool adjoint) const = 0;
    virtual uint8_t type_out(const uint8_t &type_in, bool adjoint) const = 0;

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
    void apply(const ArrayDescriptor &in, const ArrayDescriptor &out, bool adjoint) const
      {
      bool direction = adjoint ? (!fwd) : fwd;
      if (isTypecode<complex<float>>(in.typecode))
        c2c(in.to_cfmav<false, complex<float>>(),
            out.to_vfmav<false, complex<float>>(),
            axes, direction, 1.f, nthreads);
      else if (isTypecode<complex<double>>(in.typecode))
        c2c(in.to_cfmav<false, complex<double>>(),
            out.to_vfmav<false, complex<double>>(),
            axes, direction, 1., nthreads);
      else
        MR_fail("bad invocation");
      }
    // virtual
    vector<size_t> shape_out(const vector<size_t> &shp_in, bool /*adjoint*/) const { return shp_in; }
    // virtual
    uint8_t type_out(const uint8_t &type_in,bool /*adjoint*/) const { return type_in; }
  };

uint8_t nparr2typecode(const py::array &arr)
  {
  if (isPyarr<float>(arr)) return Typecode<float>::value;
  if (isPyarr<double>(arr)) return Typecode<double>::value;
  if (isPyarr<complex<float>>(arr)) return Typecode<complex<float>>::value;
  if (isPyarr<complex<double>>(arr)) return Typecode<complex<double>>::value;
  MR_fail("unsupported data type");
  }
uint8_t dtype2typecode(const py::object &type)
  {
  auto type2 = normalizeDtype(type);
  if (isDtype<float>(type2)) return Typecode<float>::value;
  if (isDtype<double>(type2)) return Typecode<double>::value;
  if (isDtype<complex<float>>(type2)) return Typecode<complex<float>>::value;
  if (isDtype<complex<double>>(type2)) return Typecode<complex<double>>::value;
  MR_fail("unsupported data type");
  }
py::object typecode2dtype(uint8_t typecode)
  {
  if (isTypecode<float>(typecode)) return Dtype<float>();
  if (isTypecode<double>(typecode)) return Dtype<double>();
  if (isTypecode<complex<float>>(typecode)) return Dtype<complex<float>>();
  if (isTypecode<complex<double>>(typecode)) return Dtype<complex<double>>();
  MR_fail("unsupported data type");
  }

ArrayDescriptor arrdesc(const py::array &arr)
  {
  ArrayDescriptor res;
  res.ndim = arr.ndim();
  MR_assert(res.ndim<=ArrayDescriptor::maxdim, "dimensionality too high");
  res.typecode = nparr2typecode(arr);
  res.data = const_cast<void *>(arr.data());
  for (size_t i=0; i<res.ndim; ++i)
    res.shape[i] = size_t(arr.shape(int(i)));
  auto st = ptrdiff_t(typeSize(res.typecode));
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

    py::array apply(const py::array &in, bool adjoint) const
      {
      auto desc_in = arrdesc(in);
      vector<size_t> shp;
      for (size_t i=0; i<desc_in.ndim; ++i)
        shp.push_back(size_t(desc_in.shape[i]));
      auto out = makeFlexiblePyarr(op->shape_out(shp,adjoint), op->type_out(desc_in.typecode,adjoint));
      auto desc_out = arrdesc(out);
      op->apply(desc_in, desc_out, adjoint);
      return out;
      }
    py::tuple shape_out(const py::tuple &shape_in, bool adjoint) const
      {
      vector<size_t> shp;
      for (size_t i=0; i<py::len(shape_in); ++i)
        shp.push_back(shape_in[i].cast<size_t>());
      auto shp2 = op->shape_out(shp, adjoint);
      py::list shp3;
      for (auto num : shp2)
        shp3.append(py::cast(num));
      return py::tuple(shp3);
      }
    py::object type_out(const py::object &type_in, bool adjoint) const
      {
      return typecode2dtype(op->type_out(dtype2typecode(type_in), adjoint));
      }
  };

void add_jax(py::module_ &msup)
  {
  using namespace pybind11::literals;
  auto m = msup.def_submodule("jax");

  py::class_<Py_Linop> (m, "Linop", py::module_local())
    .def(py::init<const string &, const py::dict &>(), "job"_a, "params"_a)
    .def("apply", &Py_Linop::apply, "in"_a, "adjoint"_a)
    .def("shape_out", &Py_Linop::shape_out, "shape_in"_a, "adjoint"_a)
    .def("type_out", &Py_Linop::type_out, "type_in"_a, "adjoint"_a);
  }

}

using detail_pymodule_jax::add_jax;

}

