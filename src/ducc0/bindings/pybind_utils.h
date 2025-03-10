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

/* Copyright (C) 2020-2025 Max-Planck-Society
   Author: Martin Reinecke */

#ifndef DUCC0_PYBIND_UTILS_H
#define DUCC0_PYBIND_UTILS_H

#include <cstddef>
#include <string>
#include <array>
#include <vector>
#include <optional>
#include <variant>
#ifdef DUCC0_USE_NANOBIND
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/function.h>
#else
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#endif

#include "ducc0/infra/error_handling.h"
#include "ducc0/infra/mav.h"
#include "ducc0/infra/misc_utils.h"

namespace ducc0 {

#ifdef DUCC0_USE_NANOBIND
namespace py = nanobind;
#else
namespace py = pybind11;
#endif

namespace detail_pybind {

using namespace std;

using shape_t=fmav_info::shape_t;
using stride_t=fmav_info::stride_t;

static const auto None = py::none();

#ifdef DUCC0_USE_NANOBIND
using NpArr = py::ndarray<py::numpy>;
using CNpArr = py::ndarray<py::numpy, py::ro>;
#else
using NpArr = py::array;
using CNpArr = py::array;
#endif

using OptNpArr = optional<NpArr>;
using OptCNpArr = optional<CNpArr>;

static inline string makeSpec(const string &name)
  { return (name=="") ? "" : name+": "; }

template<typename T> bool isPyarr(const CNpArr &obj)
#ifdef DUCC0_USE_NANOBIND
  { return obj.dtype()==py::dtype<T>(); }
#else
  { return py::isinstance<py::array_t<T>>(obj); }
#endif

static inline shape_t copy_shape(const CNpArr &arr, const string &/*spec*/="")
  {
  shape_t res(size_t(arr.ndim()));
  for (size_t i=0; i<res.size(); ++i)
    res[i] = size_t(arr.shape(int(i)));
  return res;
  }

template<typename T, bool rw> stride_t copy_strides(const CNpArr &arr,
  const string &spec="")
  {
  stride_t res(size_t(arr.ndim()));
  for (size_t i=0; i<res.size(); ++i)
    {
#ifdef DUCC0_USE_NANOBIND
    auto tmp = arr.stride(int(i));
    res[i] = tmp;
#else
    auto tmp = arr.strides(int(i));
    constexpr auto st = ptrdiff_t(sizeof(T));
    MR_assert((tmp/st)*st==tmp, spec, "bad stride");
    res[i] = tmp/st;
#endif
    if constexpr(rw)
      MR_assert((arr.shape(int(i))==1) || (tmp!=0),
        spec, "detected zero stride in writable array");
    }
  return res;
  }

template<typename T> cfmav<T> to_cfmav(const CNpArr &obj, const string &name="")
  {
  const auto spec = makeSpec(name);
  MR_assert(isPyarr<const T>(obj), "data type mismatch");
  return cfmav<T>(reinterpret_cast<const T *>(obj.data()),
    copy_shape(obj, spec), copy_strides<T,false>(obj, spec));
  }
template<typename T, size_t ndim> cmav<T,ndim> to_cmav(const CNpArr &obj,
  const string &name="")
  { return cmav<T,ndim>(to_cfmav<T>(obj, name)); }

static inline auto extend_axes(fmav_info &info, size_t ndim, const string &name="")
  {
  const auto spec = makeSpec(name);
  MR_assert(info.ndim()<=ndim, spec, "array has too many dimensions");
  shape_t newshape(ndim, 1);
  stride_t newstride(ndim, 0);
  size_t add=ndim-info.ndim();
  for (size_t i=0; i<info.ndim(); ++i)
    { newshape[i+add]=info.shape(i); newstride[i+add]=info.stride(i); }
  return make_tuple(newshape, newstride);
  }

template<typename T> cfmav<T> to_cfmav_with_optional_leading_dimensions(const CNpArr &obj, size_t ndim,
  const string &name="")
  {
  auto tmp = to_cfmav<T>(obj, name); 
  auto [newshape, newstride] = extend_axes(tmp, ndim, name);
  return cfmav<T>(tmp.data(), newshape, newstride);
  }
template<typename T, size_t ndim> cmav<T,ndim> to_cmav_with_optional_leading_dimensions(const CNpArr &obj,
  const string &name="")
  { return cmav<T,ndim>(to_cfmav_with_optional_leading_dimensions<T>(obj, ndim, name)); }

#ifdef DUCC0_USE_NANOBIND  // with nanobind, we need extra functions working on nonconst Python arrays
template<typename T> bool isPyarr(const NpArr &obj)
  { return isPyarr<T>(CNpArr(obj)); }
template<typename T> cfmav<T> to_cfmav(const NpArr &obj, const string &name="")
  { return to_cfmav<T>(CNpArr(obj), name); }
template<typename T, size_t ndim> cmav<T,ndim> to_cmav(const NpArr &obj, const string &name="")
  { return to_cmav<T,ndim>(CNpArr(obj), name); }
template<typename T> cfmav<T> to_cfmav_with_optional_leading_dimensions(const NpArr &obj, size_t ndim,
  const string &name="")
  { return to_cfmav_with_optional_leading_dimensions<T>(CNpArr(obj), ndim, name); }
template<typename T, size_t ndim> cmav<T,ndim> to_cmav_with_optional_leading_dimensions(const NpArr &obj,
  const string &name="")
  { return to_cmav_with_optional_leading_dimensions<T, ndim>(CNpArr(obj), name); }
#endif

template<typename T> vfmav<T> to_vfmav(const NpArr &obj, const string &name="")
  {
  const auto spec = makeSpec(name);
  MR_assert(isPyarr<T>(obj), "data type mismatch");
#ifdef DUCC0_USE_NANOBIND
  return vfmav<T>(reinterpret_cast<T *>(obj.data()),
    copy_shape(CNpArr(obj), spec), copy_strides<T,true>(CNpArr(obj), spec));
#else
  auto arr = py::array_t<T>(obj);
  return vfmav<T>(reinterpret_cast<T *>(arr.mutable_data()),
    copy_shape(CNpArr(obj), spec), copy_strides<T,true>(CNpArr(obj), spec));
#endif
  }

template<typename T, size_t ndim> vmav<T,ndim> to_vmav(const NpArr &obj,
  const string &name="")
  { return vmav<T,ndim>(to_vfmav<T>(obj, name)); }

template<typename T> vfmav<T> to_vfmav_with_optional_leading_dimensions(const NpArr &obj, size_t ndim,
  const string &name="")
  {
  auto tmp = to_vfmav<T>(obj, name);
  auto [newshape, newstride] = extend_axes(tmp, ndim, name);
  return vfmav<T>(tmp.data(), newshape, newstride);
  }
template<typename T, size_t ndim> vmav<T,ndim> to_vmav_with_optional_leading_dimensions(const NpArr &obj,
  const string &name="")
  { return vmav<T,ndim>(to_vfmav_with_optional_leading_dimensions<T>(obj, ndim, name)); }

template<typename T> void zero_Pyarr(const NpArr &arr, size_t nthreads=1)
  { mav_apply([](T &v){ v=T(0); }, nthreads, to_vfmav<T>(arr)); }

template<typename T> NpArr make_Pyarr(const shape_t &dims, bool zero=false)
  {
#ifdef DUCC0_USE_NANOBIND
  auto *res = new vfmav<T>(dims);
  py::capsule owner(res, [](void *p) noexcept {
      delete reinterpret_cast<vfmav<T> *>(p);
    });
  NpArr res_(py::ndarray<py::numpy,T>(res->data(), dims.size(), dims.data(), owner));
#else
  auto res_=NpArr(py::array_t<T>(dims));
#endif
  if (zero) zero_Pyarr<T>(res_);
  return res_;
  }
template<typename T, size_t ndim> NpArr make_Pyarr
  (const array<size_t,ndim> &dims, bool zero=false)
  { return make_Pyarr<T>(shape_t(dims.begin(), dims.end()), zero); }

template<typename T> NpArr make_noncritical_Pyarr(const shape_t &shape, bool zero=false)
  {
  auto ndim = shape.size();
  if (ndim==1) return make_Pyarr<T>(shape);
  auto shape2 = noncritical_shape(shape, sizeof(T));
#ifdef DUCC0_USE_NANOBIND
  auto *tmp = new vfmav<T>(shape2, UNINITIALIZED);
  py::capsule owner(tmp, [](void *p) noexcept {
      delete reinterpret_cast<vfmav<T> *>(p);
    });
  py::ndarray<py::numpy,T> res(tmp->data(), shape.size(), shape.data(), owner, tmp->stride().data());
#else
  py::array_t<T> tmp(shape2);
  py::list slices;
  for (size_t i=0; i<ndim; ++i)
    slices.append(py::slice(0, shape[i], 1));
  py::array_t<T> res(tmp[py::tuple(slices)]);
#endif
  if (zero) zero_Pyarr<T>(NpArr(res));
  return NpArr(res);
  }

template<typename T> NpArr get_optional_Pyarr(const OptNpArr &arr_,
  const shape_t &dims, const string &name="")
  {
  if (!arr_) return make_Pyarr<T>(dims, false);
  const auto spec = makeSpec(name);
  auto val = arr_.value();
  MR_assert(isPyarr<T>(val), spec, "incorrect data type");
  MR_assert(dims.size()==size_t(val.ndim()), spec, "dimension mismatch");
  for (size_t i=0; i<dims.size(); ++i)
    MR_assert(dims[i]==size_t(val.shape(int(i))), spec, "dimension mismatch");
  return val;
  }

template<typename T> NpArr get_optional_Pyarr_minshape
  (OptNpArr &arr_, const shape_t &dims, const string &name="")
  {
  if (!arr_) return make_Pyarr<T>(dims, false);
  const auto spec = makeSpec(name);
  auto val = arr_.value();
  MR_assert(isPyarr<T>(val), spec, "incorrect data type");
  MR_assert(dims.size()==size_t(val.ndim()), spec, "dimension mismatch");
  for (size_t i=0; i<dims.size(); ++i)
    MR_assert(dims[i]<=size_t(val.shape(int(i))), spec, "array shape too small");
  return val;
  }

template<typename T> CNpArr get_optional_const_Pyarr(
  const OptCNpArr &arr_, const shape_t &dims, const string &name="")
  {
  if (!arr_) return CNpArr(make_Pyarr<T>(shape_t(dims.size(), 0)));
  const auto spec = makeSpec(name);
  auto val = arr_.value();
  MR_assert(isPyarr<T>(val), spec, "incorrect data type");
  MR_assert(dims.size()==size_t(val.ndim()), spec, "dimension mismatch");
  for (size_t i=0; i<dims.size(); ++i)
    MR_assert(dims[i]==size_t(val.shape(int(i))), spec, "dimension mismatch");
  return val;
  }

#ifdef DUCC0_USE_NANOBIND
inline py::object normalizeDtype(const py::object &dtype)
  {
  static py::object converter = py::module_::import_("numpy").attr("dtype");
  return converter(dtype);
  }
template<typename T> inline py::object Dtype();
template<> inline py::object Dtype<float>()
  { static auto res = normalizeDtype(py::cast("f4")); return res; }
template<> inline py::object Dtype<double>()
  { static auto res = normalizeDtype(py::cast("f8")); return res; }
template<> inline py::object Dtype<complex<float>>()
  { static auto res = normalizeDtype(py::cast("c8")); return res; }
template<> inline py::object Dtype<complex<double>>()
  { static auto res = normalizeDtype(py::cast("c16")); return res; }
template<typename T> bool isDtype(const py::object &dtype)
  { return Dtype<T>().equal(dtype); }
#else
inline py::dtype normalizeDtype(const py::object &dtype)
  {
  static py::object converter = py::module_::import("numpy").attr("dtype");
  return converter(dtype);
  }
template<typename T> py::dtype Dtype()
  { return py::dtype::of<T>(); }
template<typename T> bool isDtype(const py::dtype &dtype)
  { return Dtype<T>().equal(dtype); }
#endif
}

using detail_pybind::NpArr;
using detail_pybind::OptNpArr;
using detail_pybind::CNpArr;
using detail_pybind::OptCNpArr;
using detail_pybind::None;
using detail_pybind::isPyarr;
using detail_pybind::make_Pyarr;
using detail_pybind::make_noncritical_Pyarr;
using detail_pybind::get_optional_Pyarr;
using detail_pybind::get_optional_Pyarr_minshape;
using detail_pybind::get_optional_const_Pyarr;
using detail_pybind::to_cfmav;
using detail_pybind::to_vfmav;
using detail_pybind::to_cmav;
using detail_pybind::to_cmav_with_optional_leading_dimensions;
using detail_pybind::to_cfmav_with_optional_leading_dimensions;
using detail_pybind::to_vmav;
using detail_pybind::to_vmav_with_optional_leading_dimensions;
using detail_pybind::to_vfmav_with_optional_leading_dimensions;
using detail_pybind::normalizeDtype;
using detail_pybind::isDtype;

}

#endif
