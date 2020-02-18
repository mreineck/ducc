#ifndef MRUTIL_PYBIND_UTILS_H
#define MRUTIL_PYBIND_UTILS_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "mr_util/mav.h"

namespace mr {

namespace detail_pybind {

namespace py = pybind11;

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

template<typename T> stride_t copy_strides(const py::array &arr)
  {
  stride_t res(size_t(arr.ndim()));
  constexpr auto st = ptrdiff_t(sizeof(T));
  for (size_t i=0; i<res.size(); ++i)
    {
    auto tmp = arr.strides(int(i));
    MR_assert((tmp/st)*st==tmp, "bad stride");
    res[i] = tmp/st;
    }
  return res;
  }

template<size_t ndim> std::array<size_t, ndim> copy_fixshape(const py::array &arr)
  {
  MR_assert(size_t(arr.ndim())==ndim, "incorrect number of dimensions");
  std::array<size_t, ndim> res;
  for (size_t i=0; i<ndim; ++i)
    res[i] = size_t(arr.shape(int(i)));
  return res;
  }

template<typename T, size_t ndim> std::array<ptrdiff_t, ndim> copy_fixstrides(const py::array &arr)
  {
  MR_assert(size_t(arr.ndim())==ndim, "incorrect number of dimensions");
  std::array<ptrdiff_t, ndim> res;
  constexpr auto st = ptrdiff_t(sizeof(T));
  for (size_t i=0; i<ndim; ++i)
    {
    auto tmp = arr.strides(int(i));
    MR_assert((tmp/st)*st==tmp, "bad stride");
    res[i] = tmp/st;
    }
  return res;
  }

template<typename T> py::array_t<T> make_Pyarr(const shape_t &dims)
  { return py::array_t<T>(dims); }

template<typename T> py::array_t<T> get_optional_Pyarr(py::object &arr_,
  const shape_t &dims)
  {
  if (arr_.is_none()) return py::array_t<T>(dims);
  MR_assert(isPyarr<T>(arr_), "incorrect data type");
  auto tmp = toPyarr<T>(arr_);
  MR_assert(dims.size()==size_t(tmp.ndim()), "dimension mismatch");
  for (size_t i=0; i<dims.size(); ++i)
    MR_assert(dims[i]==size_t(tmp.shape(int(i))), "dimension mismatch");
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

template<typename T> fmav<T> to_fmav(py::object &obj)
  {
  auto arr = toPyarr<T>(obj);
  return fmav<T>(reinterpret_cast<T *>(arr.mutable_data()),
    copy_shape(arr), copy_strides<T>(arr));
  }

template<typename T> cfmav<T> to_cfmav(const py::object &obj)
  {
  auto arr = toPyarr<T>(obj);
  return cfmav<T>(reinterpret_cast<const T *>(arr.data()),
    copy_shape(arr), copy_strides<T>(arr));
  }

template<typename T, size_t ndim> mav<T,ndim> to_mav(py::array &obj)
  {
  auto arr = toPyarr<T>(obj);
  return mav<T,ndim>(reinterpret_cast<T *>(arr.mutable_data()),
    copy_fixshape<ndim>(arr), copy_fixstrides<T,ndim>(arr));
  }

template<typename T, size_t ndim> cmav<T,ndim> to_cmav(const py::array &obj)
  {
  auto arr = toPyarr<T>(obj);
  return cmav<T,ndim>(reinterpret_cast<const T *>(arr.data()),
    copy_fixshape<ndim>(arr), copy_fixstrides<T,ndim>(arr));
  }

}

using detail_pybind::isPyarr;
using detail_pybind::make_Pyarr;
using detail_pybind::get_optional_Pyarr;
using detail_pybind::get_optional_const_Pyarr;
using detail_pybind::to_fmav;
using detail_pybind::to_cfmav;
using detail_pybind::to_mav;
using detail_pybind::to_cmav;

}

#endif
