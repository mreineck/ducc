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

using shape_t=fmav_info::shape_t;
using stride_t=fmav_info::stride_t;

#ifdef DUCC0_USE_NANOBIND
using NpArr = py::ndarray<py::numpy>;
using CNpArr = py::ndarray<py::numpy, py::ro>;
template<typename T> using NpArrT = py::ndarray<py::numpy,T>;
#else
using NpArr = py::array;
using CNpArr = py::array;
template<typename T> using NpArrT = py::array_t<T>;
#endif

using OptNpArr = std::optional<NpArr>;
using OptCNpArr = std::optional<CNpArr>;

static inline string makeSpec(const string &name)
  { return (name=="") ? "" : name+": "; }

#ifdef DUCC0_USE_NANOBIND
template<typename T> bool isPyarr(const CNpArr &obj)
  { return obj.dtype()==py::dtype<T>(); }
#endif
template<typename T> bool isPyarr(const NpArr &obj)
#ifdef DUCC0_USE_NANOBIND
  { return obj.dtype()==py::dtype<T>(); }
#else
  { return py::isinstance<py::array_t<T>>(obj); }
#endif

shape_t copy_shape(const CNpArr &arr, const string &/*spec*/="")
  {
  shape_t res(size_t(arr.ndim()));
  for (size_t i=0; i<res.size(); ++i)
    res[i] = size_t(arr.shape(int(i)));
  return res;
  }
#ifdef DUCC0_USE_NANOBIND
shape_t copy_shape(const NpArr &arr, const string &spec="")
  { return copy_shape(CNpArr(arr), spec); }
#endif

template<typename T> stride_t copy_strides_rw(const NpArr &arr,
  const string &spec="")
  {
  stride_t res(size_t(arr.ndim()));
  for (size_t i=0; i<res.size(); ++i)
    {
#ifdef DUCC0_USE_NANOBIND
    auto tmp = arr.stride(int(i));
    MR_assert((arr.shape(int(i))==1) || (tmp!=0),
      spec, "detected zero stride in writable array");
    res[i] = tmp;
#else
    constexpr auto st = ptrdiff_t(sizeof(T));
    auto tmp = arr.strides(int(i));
    MR_assert((arr.shape(int(i))==1) || (tmp!=0),
      spec, "detected zero stride in writable array");
    MR_assert((tmp/st)*st==tmp, spec, "bad stride");
    res[i] = tmp/st;
#endif
    }
  return res;
  }

template<typename T> stride_t copy_strides_ro(const CNpArr &arr,
#ifdef DUCC0_USE_NANOBIND
  const string &/*spec*/="")
#else
  const string &spec="")
#endif
  {
  stride_t res(size_t(arr.ndim()));
  for (size_t i=0; i<res.size(); ++i)
    {
#ifdef DUCC0_USE_NANOBIND
    auto tmp = arr.stride(int(i));
    res[i] = tmp;
#else
    constexpr auto st = ptrdiff_t(sizeof(T));
    auto tmp = arr.strides(int(i));
    MR_assert((tmp/st)*st==tmp, spec, "bad stride");
    res[i] = tmp/st;
#endif
    }
  return res;
  }
#ifdef DUCC0_USE_NANOBIND
template<typename T> stride_t copy_strides_ro(const NpArr &arr,
  const string &spec="")
  { return copy_strides_ro<T>(CNpArr(arr), spec); }
#endif

template<typename T> cfmav<T> to_cfmav(const CNpArr &obj, const string &name="")
  {
  const auto spec = makeSpec(name);
  MR_assert(isPyarr<const T>(obj), "data type mismatch");
  return cfmav<T>(reinterpret_cast<const T *>(obj.data()),
    copy_shape(obj, spec), copy_strides_ro<T>(obj, spec));
  }
#ifdef DUCC0_USE_NANOBIND
template<typename T> cfmav<T> to_cfmav(const NpArr &obj, const string &name="")
  { return to_cfmav<T>(CNpArr(obj), name); }
#endif

template<typename T> vfmav<T> to_vfmav(const NpArr &obj,
  const string &name="")
  {
  const auto spec = makeSpec(name);
  MR_assert(isPyarr<T>(obj), "data type mismatch");
#ifdef DUCC0_USE_NANOBIND
  return vfmav<T>(reinterpret_cast<T *>(obj.data()),
    copy_shape(obj, spec), copy_strides_rw<T>(obj, spec));
#else
  auto arr = NpArrT<T>(obj);
  return vfmav<T>(reinterpret_cast<T *>(arr.mutable_data()),
    copy_shape(obj, spec), copy_strides_rw<T>(obj, spec));
#endif
  }

template<typename T, size_t ndim> cmav<T,ndim> to_cmav(const CNpArr &obj,
  const string &name="")
  { return cmav<T,ndim>(to_cfmav<T>(obj, name)); }
#ifdef DUCC0_USE_NANOBIND
template<typename T, size_t ndim> cmav<T,ndim> to_cmav(const NpArr &obj,
  const string &name="")
  { return to_cmav<T,ndim>(CNpArr(obj), name); }
#endif

template<typename T> cfmav<T> to_cfmav_with_optional_leading_dimensions(const CNpArr &obj, size_t ndim,
  const string &name="")
  {
  const auto spec = makeSpec(name);
  auto tmp = to_cfmav<T>(obj, name); 
  MR_assert(tmp.ndim()<=ndim, spec, "array has too many dimensions");
  typename cfmav<T>::shape_t newshape(ndim);
  typename cfmav<T>::stride_t newstride(ndim);
  size_t add=ndim-tmp.ndim();
  for (size_t i=0; i<add; ++i)
    { newshape[i]=1; newstride[i]=0; }
  for (size_t i=0; i<tmp.ndim(); ++i)
    { newshape[i+add]=tmp.shape(i); newstride[i+add]=tmp.stride(i); }
  return cfmav<T>(tmp.data(), newshape, newstride);
  }
#ifdef DUCC0_USE_NANOBIND
template<typename T> cfmav<T> to_cfmav_with_optional_leading_dimensions(const NpArr &obj, size_t ndim,
  const string &name="")
  { return to_cfmav_with_optional_leading_dimensions<T>(CNpArr(obj), name); }
#endif
template<typename T, size_t ndim> cmav<T,ndim> to_cmav_with_optional_leading_dimensions(const CNpArr &obj,
  const string &name="")
  { return cmav<T,ndim>(to_cfmav_with_optional_leading_dimensions<T>(obj, ndim, name)); }
#ifdef DUCC0_USE_NANOBIND
template<typename T, size_t ndim> cmav<T,ndim> to_cmav_with_optional_leading_dimensions(const NpArr &obj,
  const string &name="")
  { return to_cmav_with_optional_leading_dimensions<T, ndim>(CNpArr(obj), name); }
#endif
template<typename T, size_t ndim> vmav<T,ndim> to_vmav(const NpArr &obj,
  const string &name="")
  { return vmav<T,ndim>(to_vfmav<T>(obj, name)); }

template<typename T> vfmav<T> to_vfmav_with_optional_leading_dimensions(const NpArr &obj, size_t ndim,
  const string &name="")
  {
  const auto spec = makeSpec(name);
  auto tmp = to_vfmav<T>(obj, name); 
  MR_assert(tmp.ndim()<=ndim, spec, "array has too many dimensions");
  typename vfmav<T>::shape_t newshape(ndim);
  typename vfmav<T>::stride_t newstride(ndim);
  size_t add=ndim-tmp.ndim();
  for (size_t i=0; i<add; ++i)
    { newshape[i]=1; newstride[i]=0; }
  for (size_t i=0; i<tmp.ndim(); ++i)
    { newshape[i+add]=tmp.shape(i); newstride[i+add]=tmp.stride(i); }
  return vfmav<T>(tmp.data(), newshape, newstride);
  }
template<typename T, size_t ndim> vmav<T,ndim> to_vmav_with_optional_leading_dimensions(const NpArr &obj,
  const string &name="")
  { return vmav<T,ndim>(to_vfmav_with_optional_leading_dimensions<T>(obj, ndim, name)); }

template<typename T> void zero_Pyarr(NpArr &arr, size_t nthreads=1)
  {
  auto arr2 = to_vfmav<T>(arr);
  mav_apply([](T &v){ v=T(0); }, nthreads, arr2);
  }

template<typename T> NpArr make_Pyarr(const shape_t &dims, bool zero=false)
  {
#ifdef DUCC0_USE_NANOBIND
  auto *res = new vfmav<T>(dims);
  py::capsule owner(res, [](void *p) noexcept {
       delete reinterpret_cast<vfmav<T> *>(p);
    });
  NpArr res_(NpArrT<T>(res->data(), dims.size(), dims.data(), owner));
#else
  auto res_=NpArr(NpArrT<T>(dims));
#endif
  if (zero) zero_Pyarr<T>(res_);
  return res_;
  }

template<typename T, size_t ndim> NpArr make_Pyarr
  (const std::array<size_t,ndim> &dims, bool zero=false)
  {
  auto res=NpArr(NpArrT<T>(shape_t(dims.begin(), dims.end())));
  if (zero) zero_Pyarr<T>(res);
  return res;
  }

template<typename T> NpArr make_noncritical_Pyarr(const shape_t &shape)
  {
  auto ndim = shape.size();
  if (ndim==1) return make_Pyarr<T>(shape);
  auto shape2 = noncritical_shape(shape, sizeof(T));
#ifdef DUCC0_USE_NANOBIND
  auto *res = new vfmav<T>(shape2);
  py::capsule owner(res, [](void *p) noexcept {
       delete reinterpret_cast<vfmav<T> *>(p);
    });
  NpArrT<T> res_(res->data(), shape.size(), shape.data(), owner, res->stride().data());
  return NpArr(res_);
#else
  NpArrT<T> tarr(shape2);
  py::list slices;
  for (size_t i=0; i<ndim; ++i)
    slices.append(py::slice(0, shape[i], 1));
  NpArrT<T> sub(tarr[py::tuple(slices)]);
  return NpArr(sub);
#endif
  }

template<typename T> NpArr get_Pyarr(const NpArr &arr_, size_t ndims,
  const string &name="")
  {
  const auto spec = makeSpec(name);
  MR_assert(isPyarr<T>(arr_), spec, "incorrect data type");
  MR_assert(ndims==size_t(arr_.ndim()), spec, "dimension mismatch");
  return arr_;
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
  if (!arr_) return make_Pyarr<T>(dims);
  const auto spec = makeSpec(name);
  auto val = arr_.value();
  MR_assert(isPyarr<T>(val), spec, "incorrect data type");
  MR_assert(dims.size()==size_t(val.ndim()), spec, "dimension mismatch");
  for (size_t i=0; i<dims.size(); ++i)
    MR_assert(dims[i]<=size_t(val.shape(int(i))), spec, "array shape too small");
  return val;
  }

#ifdef DUCC0_USE_NANOBIND
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
#endif
template<typename T> NpArr get_optional_const_Pyarr(
  const OptNpArr &arr_, const shape_t &dims, const string &name="")
  {
  if (!arr_) return make_Pyarr<T>(shape_t(dims.size(), 0));
  const auto spec = makeSpec(name);
  auto val = arr_.value();
  MR_assert(isPyarr<T>(val), spec, "incorrect data type");
  MR_assert(dims.size()==size_t(val.ndim()), spec, "dimension mismatch");
  for (size_t i=0; i<dims.size(); ++i)
    MR_assert(dims[i]==size_t(val.shape(int(i))), spec, "dimension mismatch");
  return val;
  }

#ifndef DUCC0_USE_NANOBIND
py::object normalizeDtype(const py::object &dtype)
  {
  static py::object converter = py::module_::import("numpy").attr("dtype");
  return converter(dtype);
  }
template<typename T> py::object Dtype()
  { return py::dtype::of<T>(); }
template<typename T> bool isDtype(const py::object &dtype)
  { return Dtype<T>().equal(dtype); }
#endif
}

using detail_pybind::NpArr;
using detail_pybind::OptNpArr;
using detail_pybind::CNpArr;
using detail_pybind::OptCNpArr;
using detail_pybind::isPyarr;
using detail_pybind::make_Pyarr;
using detail_pybind::make_noncritical_Pyarr;
using detail_pybind::get_Pyarr;
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
#ifndef DUCC0_USE_NANOBIND
using detail_pybind::normalizeDtype;
using detail_pybind::isDtype;
using detail_pybind::Dtype;
#endif

}

#endif
