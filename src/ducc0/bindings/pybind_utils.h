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
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "ducc0/infra/error_handling.h"
#include "ducc0/infra/mav.h"
#include "ducc0/infra/misc_utils.h"

namespace ducc0 {

namespace py = pybind11;

namespace detail_pybind {

using shape_t=fmav_info::shape_t;
using stride_t=fmav_info::stride_t;

using NpArr = py::array;
template<typename T> using NpArrT = py::array_t<T>;

static inline string makeSpec(const string &name)
  { return (name=="") ? "" : name+": "; }

py::object normalizeDtype(const py::object &dtype)
  {
  static py::object converter = py::module_::import("numpy").attr("dtype");
  return converter(dtype);
  }

bool isPyarr(const py::object &obj)
  { return py::isinstance<NpArr>(obj); }

template<typename T> bool isPyarr(const py::object &obj)
  { return py::isinstance<NpArrT<T>>(obj); }

template<typename T> NpArrT<T> toPyarr(const py::object &obj,
  const string &spec="")
  {
  auto tmp = obj.cast<NpArrT<T>>();
  MR_assert(tmp.is(obj), spec, "error during array conversion");
  return tmp;
  }

shape_t copy_shape(const NpArr &arr, const string &/*spec*/="")
  {
  shape_t res(size_t(arr.ndim()));
  for (size_t i=0; i<res.size(); ++i)
    res[i] = size_t(arr.shape(int(i)));
  return res;
  }

template<typename T> stride_t copy_strides(const NpArr &arr, bool rw,
  const string &spec="")
  {
  stride_t res(size_t(arr.ndim()));
  constexpr auto st = ptrdiff_t(sizeof(T));
  for (size_t i=0; i<res.size(); ++i)
    {
    auto tmp = arr.strides(int(i));
    MR_assert((!rw) || (arr.shape(int(i))==1) || (tmp!=0),
      spec, "detected zero stride in writable array");
    MR_assert((tmp/st)*st==tmp, spec, "bad stride");
    res[i] = tmp/st;
    }
  return res;
  }

template<size_t ndim>
  std::array<size_t, ndim> copy_fixshape(const NpArr &arr,
  const string &spec="")
  {
  MR_assert(size_t(arr.ndim())==ndim, spec, "incorrect number of dimensions");
  std::array<size_t, ndim> res;
  for (size_t i=0; i<ndim; ++i)
    res[i] = size_t(arr.shape(int(i)));
  return res;
  }

template<typename T, size_t ndim>
  std::array<ptrdiff_t, ndim> copy_fixstrides(const NpArr &arr, bool rw,
  const string &spec="")
  {
  MR_assert(size_t(arr.ndim())==ndim, spec, "incorrect number of dimensions");
  std::array<ptrdiff_t, ndim> res;
  constexpr auto st = ptrdiff_t(sizeof(T));
  for (size_t i=0; i<ndim; ++i)
    {
    auto tmp = arr.strides(int(i));
    MR_assert((!rw) || (arr.shape(int(i))==1) || (tmp!=0),
      spec, "detected zero stride in writable array");
    MR_assert((tmp/st)*st==tmp, spec, "bad stride");
    res[i] = tmp/st;
    }
  return res;
  }

template<typename T> cfmav<T> to_cfmav(const py::object &obj,
  const string &name="")
  {
  const auto spec = makeSpec(name);
  auto arr = toPyarr<T>(obj, spec);
  return cfmav<T>(reinterpret_cast<const T *>(arr.data()),
    copy_shape(arr, spec), copy_strides<T>(arr, false, spec));
  }
template<typename T> vfmav<T> to_vfmav(const py::object &obj,
  const string &name="")
  {
  const auto spec = makeSpec(name);
  auto arr = toPyarr<T>(obj, spec);
  return vfmav<T>(reinterpret_cast<T *>(arr.mutable_data()),
    copy_shape(arr, spec), copy_strides<T>(arr, true, spec));
  }

template<typename T, size_t ndim> cmav<T,ndim> to_cmav(const NpArr &obj,
  const string &name="")
  {
  const auto spec = makeSpec(name);
  auto arr = toPyarr<T>(obj, spec);
  return cmav<T,ndim>(reinterpret_cast<const T *>(arr.data()),
    copy_fixshape<ndim>(arr, spec), copy_fixstrides<T,ndim>(arr, false, spec));
  }
template<typename T, size_t ndim> cmav<T,ndim> to_cmav_with_optional_leading_dimensions(const NpArr &obj,
  const string &name="")
  {
  const auto spec = makeSpec(name);
  auto tmp = to_cfmav<T>(obj, name); 
  MR_assert(tmp.ndim()<=ndim, spec, "array has too many dimensions");
  typename cmav<T,ndim>::shape_t newshape;
  typename cmav<T,ndim>::stride_t newstride;
  size_t add=ndim-tmp.ndim();
  for (size_t i=0; i<add; ++i)
    { newshape[i]=1; newstride[i]=0; }
  for (size_t i=0; i<tmp.ndim(); ++i)
    { newshape[i+add]=tmp.shape(i); newstride[i+add]=tmp.stride(i); }
  return cmav<T,ndim>(tmp.data(), newshape, newstride);
  }
template<typename T> cfmav<T> to_cfmav_with_optional_leading_dimensions(const NpArr &obj, size_t ndim,
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
template<typename T, size_t ndim> vmav<T,ndim> to_vmav(const NpArr &obj,
  const string &name="")
  {
  const auto spec = makeSpec(name);
  auto arr = toPyarr<T>(obj, spec);
  return vmav<T,ndim>(reinterpret_cast<T *>(arr.mutable_data()),
    copy_fixshape<ndim>(arr, spec), copy_fixstrides<T,ndim>(arr, true, spec));
  }
template<typename T, size_t ndim> vmav<T,ndim> to_vmav_with_optional_leading_dimensions(const NpArr &obj,
  const string &name="")
  {
  const auto spec = makeSpec(name);
  auto tmp = to_vfmav<T>(obj, name); 
  MR_assert(tmp.ndim()<=ndim, spec, "array has too many dimensions");
  typename vmav<T,ndim>::shape_t newshape;
  typename vmav<T,ndim>::stride_t newstride;
  size_t add=ndim-tmp.ndim();
  for (size_t i=0; i<add; ++i)
    { newshape[i]=1; newstride[i]=0; }
  for (size_t i=0; i<tmp.ndim(); ++i)
    { newshape[i+add]=tmp.shape(i); newstride[i+add]=tmp.stride(i); }
  return vmav<T,ndim>(tmp.data(), newshape, newstride);
  }
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

template<typename T, size_t len> std::array<T,len> to_array(const py::object &obj,
  const string &name="")
  {
  const auto spec = makeSpec(name);
  auto vec = py::cast<std::vector<T>>(obj);
  MR_assert(vec.size()==len, spec, "unexpected number of elements");
  std::array<T,len> res;
  for (size_t i=0;i<len; ++i) res[i] = vec[i];
  return res;
  }

template<typename T> void zero_Pyarr(NpArrT<T> &arr, size_t nthreads=1)
  {
  auto arr2 = to_vfmav<T>(arr);
  mav_apply([](T &v){ v=T(0); }, nthreads, arr2);
  }

template<typename T> NpArrT<T> make_Pyarr(const shape_t &dims, bool zero=false)
  {
  auto res=NpArrT<T>(dims);
  if (zero) zero_Pyarr(res);
  return res;
  }
template<typename T, size_t ndim> NpArrT<T> make_Pyarr
  (const std::array<size_t,ndim> &dims, bool zero=false)
  {
  auto res=NpArrT<T>(shape_t(dims.begin(), dims.end()));
  if (zero) zero_Pyarr(res);
  return res;
  }

template<typename T> NpArrT<T> make_noncritical_Pyarr(const shape_t &shape)
  {
  auto ndim = shape.size();
  if (ndim==1) return make_Pyarr<T>(shape);
  auto shape2 = noncritical_shape(shape, sizeof(T));
  NpArrT<T> tarr(shape2);
  py::list slices;
  for (size_t i=0; i<ndim; ++i)
    slices.append(py::slice(0, shape[i], 1));
  NpArrT<T> sub(tarr[py::tuple(slices)]);
  return sub;
  }

template<typename T> NpArrT<T> get_Pyarr(py::object &arr_, size_t ndims,
  const string &name="")
  {
  const auto spec = makeSpec(name);
  MR_assert(isPyarr<T>(arr_), spec, "incorrect data type");
  auto tmp = toPyarr<T>(arr_, spec);
  MR_assert(ndims==size_t(tmp.ndim()), spec, "dimension mismatch");
  return tmp;
  }

template<typename T> NpArrT<T> get_optional_Pyarr(py::object &arr_,
  const shape_t &dims, const string &name="")
  {
  if (arr_.is_none()) return make_Pyarr<T>(dims, false);
  const auto spec = makeSpec(name);
  MR_assert(isPyarr<T>(arr_), spec, "incorrect data type");
  auto tmp = toPyarr<T>(arr_, spec);
  MR_assert(dims.size()==size_t(tmp.ndim()), spec, "dimension mismatch");
  for (size_t i=0; i<dims.size(); ++i)
    MR_assert(dims[i]==size_t(tmp.shape(int(i))), spec, "dimension mismatch");
  return tmp;
  }

template<typename T> NpArrT<T> get_optional_Pyarr_minshape
  (py::object &arr_, const shape_t &dims, const string &name="")
  {
  if (arr_.is_none()) return make_Pyarr<T>(dims);
  const auto spec = makeSpec(name);
  MR_assert(isPyarr<T>(arr_), spec, "incorrect data type");
  auto tmp = toPyarr<T>(arr_, spec);
  MR_assert(dims.size()==size_t(tmp.ndim()), spec, "dimension mismatch");
  for (size_t i=0; i<dims.size(); ++i)
    MR_assert(dims[i]<=size_t(tmp.shape(int(i))), spec, "array shape too small");
  return tmp;
  }

template<typename T> NpArrT<T> get_optional_const_Pyarr(
  const py::object &arr_, const shape_t &dims, const string &name="")
  {
  if (arr_.is_none()) return NpArrT<T>(shape_t(dims.size(), 0));
  const auto spec = makeSpec(name);
  MR_assert(isPyarr<T>(arr_), spec, "incorrect data type");
  auto tmp = toPyarr<T>(arr_, spec);
  MR_assert(dims.size()==size_t(tmp.ndim()), spec, "dimension mismatch");
  for (size_t i=0; i<dims.size(); ++i)
    MR_assert(dims[i]==size_t(tmp.shape(int(i))), spec, "dimension mismatch");
  return tmp;
  }

template<typename T> py::object Dtype()
  { return py::dtype::of<T>(); }
template<typename T> bool isDtype(const py::object &dtype)
  { return Dtype<T>().equal(dtype); }

}

using detail_pybind::NpArr;
using detail_pybind::NpArrT;
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
using detail_pybind::to_array;
using detail_pybind::normalizeDtype;
using detail_pybind::isDtype;
using detail_pybind::Dtype;

}

#endif
