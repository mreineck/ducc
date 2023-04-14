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

/* Copyright (C) 2020-2023 Max-Planck-Society
   Author: Martin Reinecke */


#ifndef DUCC0_PYBIND_UTILS_H
#define DUCC0_PYBIND_UTILS_H

#include <cstddef>
#include <array>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "ducc0/infra/error_handling.h"
#include "ducc0/infra/mav.h"
#include "ducc0/infra/misc_utils.h"

namespace ducc0 {

namespace detail_pybind {

using shape_t=fmav_info::shape_t;
using stride_t=fmav_info::stride_t;

namespace py = pybind11;

py::object normalizeDtype(const py::object &dtype)
  {
  static py::object converter = py::module_::import("numpy").attr("dtype");
  return converter(dtype);
  }

bool isPyarr(const py::object &obj)
  { return py::isinstance<py::array>(obj); }

template<typename T> bool isPyarr(const py::object &obj)
  { return py::isinstance<py::array_t<T>>(obj); }

template<typename T> py::array_t<T> toPyarr(const py::object &obj)
  {
  auto tmp = obj.cast<py::array_t<T>>();
  MR_assert(tmp.is(obj), "error during array conversion");
  return tmp;
  }

shape_t copy_shape(const py::array &arr)
  {
  shape_t res(size_t(arr.ndim()));
  for (size_t i=0; i<res.size(); ++i)
    res[i] = size_t(arr.shape(int(i)));
  return res;
  }

template<typename T> stride_t copy_strides(const py::array &arr, bool rw)
  {
  stride_t res(size_t(arr.ndim()));
  constexpr auto st = ptrdiff_t(sizeof(T));
  for (size_t i=0; i<res.size(); ++i)
    {
    auto tmp = arr.strides(int(i));
    MR_assert((!rw) || (tmp!=0), "detected zero stride in writable array");
    MR_assert((tmp/st)*st==tmp, "bad stride");
    res[i] = tmp/st;
    }
  return res;
  }

template<size_t ndim>
  std::array<size_t, ndim> copy_fixshape(const py::array &arr)
  {
  MR_assert(size_t(arr.ndim())==ndim, "incorrect number of dimensions");
  std::array<size_t, ndim> res;
  for (size_t i=0; i<ndim; ++i)
    res[i] = size_t(arr.shape(int(i)));
  return res;
  }

template<typename T, size_t ndim>
  std::array<ptrdiff_t, ndim> copy_fixstrides(const py::array &arr, bool rw)
  {
  MR_assert(size_t(arr.ndim())==ndim, "incorrect number of dimensions");
  std::array<ptrdiff_t, ndim> res;
  constexpr auto st = ptrdiff_t(sizeof(T));
  for (size_t i=0; i<ndim; ++i)
    {
    auto tmp = arr.strides(int(i));
    MR_assert((!rw) || (tmp!=0), "detected zero stride in writable array");
    MR_assert((tmp/st)*st==tmp, "bad stride");
    res[i] = tmp/st;
    }
  return res;
  }

template<typename T> cfmav<T> to_cfmav(const py::object &obj)
  {
  auto arr = toPyarr<T>(obj);
  return cfmav<T>(reinterpret_cast<const T *>(arr.data()),
    copy_shape(arr), copy_strides<T>(arr, false));
  }
template<typename T> vfmav<T> to_vfmav(const py::object &obj)
  {
  auto arr = toPyarr<T>(obj);
  return vfmav<T>(reinterpret_cast<T *>(arr.mutable_data()),
    copy_shape(arr), copy_strides<T>(arr, true));
  }

template<typename T, size_t ndim> cmav<T,ndim> to_cmav(const py::array &obj)
  {
  auto arr = toPyarr<T>(obj);
  return cmav<T,ndim>(reinterpret_cast<const T *>(arr.data()),
    copy_fixshape<ndim>(arr), copy_fixstrides<T,ndim>(arr, false));
  }
template<typename T, size_t ndim> cmav<T,ndim> to_cmav_with_optional_leading_dimensions(const py::array &obj)
  {
  auto tmp = to_cfmav<T>(obj); 
  MR_assert(tmp.ndim()<=ndim, "array has too many dimensions");
  typename cmav<T,ndim>::shape_t newshape;
  typename cmav<T,ndim>::stride_t newstride;
  size_t add=ndim-tmp.ndim();
  for (size_t i=0; i<add; ++i)
    { newshape[i]=1; newstride[i]=0; }
  for (size_t i=0; i<tmp.ndim(); ++i)
    { newshape[i+add]=tmp.shape(i); newstride[i+add]=tmp.stride(i); }
  return cmav<T,ndim>(tmp.data(), newshape, newstride);
  }
template<typename T, size_t ndim> vmav<T,ndim> to_vmav(const py::array &obj)
  {
  auto arr = toPyarr<T>(obj);
  return vmav<T,ndim>(reinterpret_cast<T *>(arr.mutable_data()),
    copy_fixshape<ndim>(arr), copy_fixstrides<T,ndim>(arr, true));
  }
template<typename T, size_t ndim> vmav<T,ndim> to_vmav_with_optional_leading_dimensions(const py::array &obj)
  {
  auto tmp = to_vfmav<T>(obj); 
  MR_assert(tmp.ndim()<=ndim, "array has too many dimensions");
  typename vmav<T,ndim>::shape_t newshape;
  typename vmav<T,ndim>::stride_t newstride;
  size_t add=ndim-tmp.ndim();
  for (size_t i=0; i<add; ++i)
    { newshape[i]=1; newstride[i]=0; }
  for (size_t i=0; i<tmp.ndim(); ++i)
    { newshape[i+add]=tmp.shape(i); newstride[i+add]=tmp.stride(i); }
  return vmav<T,ndim>(tmp.data(), newshape, newstride);
  }

template<typename T, size_t len> array<T,len> to_array(const py::object &obj)
  {
  auto vec = py::cast<vector<T>>(obj);
  MR_assert(vec.size()==len, "unexpected number of elements");
  array<T,len> res;
  for (size_t i=0;i<len; ++i) res[i] = vec[i];
  return res;
  }

template<typename T> void zero_Pyarr(py::array_t<T> &arr, size_t nthreads=1)
  {
  auto arr2 = to_vfmav<T>(arr);
  mav_apply([](T &v){ v=T(0); }, nthreads, arr2);
  }

template<typename T> py::array_t<T> make_Pyarr(const shape_t &dims, bool zero=false)
  {
  auto res=py::array_t<T>(dims);
  if (zero) zero_Pyarr(res);
  return res;
  }
template<typename T, size_t ndim> py::array_t<T> make_Pyarr
  (const std::array<size_t,ndim> &dims, bool zero=false)
  {
  auto res=py::array_t<T>(shape_t(dims.begin(), dims.end()));
  if (zero) zero_Pyarr(res);
  return res;
  }

template<typename T> py::array_t<T> make_noncritical_Pyarr(const shape_t &shape)
  {
  auto ndim = shape.size();
  if (ndim==1) return make_Pyarr<T>(shape);
  auto shape2 = noncritical_shape(shape, sizeof(T));
  py::array_t<T> tarr(shape2);
  py::list slices;
  for (size_t i=0; i<ndim; ++i)
    slices.append(py::slice(0, shape[i], 1));
  py::array_t<T> sub(tarr[py::tuple(slices)]);
  return sub;
  }

template<typename T> py::array_t<T> get_Pyarr(py::object &arr_, size_t ndims)
  {
  MR_assert(isPyarr<T>(arr_), "incorrect data type");
  auto tmp = toPyarr<T>(arr_);
  MR_assert(ndims==size_t(tmp.ndim()), "dimension mismatch");
  return tmp;
  }

template<typename T> py::array_t<T> get_optional_Pyarr(py::object &arr_,
  const shape_t &dims, bool zero_if_new=false)
  {
  if (arr_.is_none()) return make_Pyarr<T>(dims, zero_if_new);
  MR_assert(isPyarr<T>(arr_), "incorrect data type");
  auto tmp = toPyarr<T>(arr_);
  MR_assert(dims.size()==size_t(tmp.ndim()), "dimension mismatch");
  for (size_t i=0; i<dims.size(); ++i)
    MR_assert(dims[i]==size_t(tmp.shape(int(i))), "dimension mismatch");
  return tmp;
  }

template<typename T> py::array_t<T> get_optional_Pyarr_minshape
  (py::object &arr_, const shape_t &dims)
  {
  if (arr_.is_none()) return make_Pyarr<T>(dims);
  MR_assert(isPyarr<T>(arr_), "incorrect data type");
  auto tmp = toPyarr<T>(arr_);
  MR_assert(dims.size()==size_t(tmp.ndim()), "dimension mismatch");
  for (size_t i=0; i<dims.size(); ++i)
    MR_assert(dims[i]<=size_t(tmp.shape(int(i))), "array shape too small");
  return tmp;
  }

template<typename T> py::array_t<T> get_optional_const_Pyarr(
  const py::object &arr_, const shape_t &dims)
  {
  if (arr_.is_none()) return py::array_t<T>(shape_t(dims.size(), 0));
  MR_assert(isPyarr<T>(arr_), "incorrect data type");
  auto tmp = toPyarr<T>(arr_);
  MR_assert(dims.size()==size_t(tmp.ndim()), "dimension mismatch");
  for (size_t i=0; i<dims.size(); ++i)
    MR_assert(dims[i]==size_t(tmp.shape(int(i))), "dimension mismatch");
  return tmp;
  }

template<typename T> bool isDtype(const py::object &dtype)
  {
  static const auto tmp = make_Pyarr<T>({}).dtype();
  return tmp.is(dtype);
  }

}

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
using detail_pybind::to_vmav;
using detail_pybind::to_vmav_with_optional_leading_dimensions;
using detail_pybind::to_array;
using detail_pybind::normalizeDtype;
using detail_pybind::isDtype;

}

#endif
