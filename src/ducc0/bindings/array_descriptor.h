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

/* Copyright (C) 2022-2023 Max-Planck-Society
   Author: Martin Reinecke */

#ifndef DUCC0_ARRAY_DESCRIPTOR_H
#define DUCC0_ARRAY_DESCRIPTOR_H

#include <array>
#include <iostream>
#include "ducc0/infra/error_handling.h"
#include "ducc0/infra/mav.h"
#include "ducc0/bindings/typecode.h"

#include "src/main.rs.h"

namespace ducc0 {

namespace detail_array_descriptor {

using namespace std;

struct ArrayDescriptor
  {
  static constexpr size_t maxdim=10;

  array<uint64_t, maxdim> shape;
  array<int64_t, maxdim> stride;

  void *data;
  uint8_t ndim;
  uint8_t dtype;
  };

template<bool swapdims, typename T1, typename T2> void copy_data
  (const ArrayDescriptor &desc, T1 &shp, T2 &str)
  {
  auto ndim = desc.ndim;
  if constexpr (swapdims)
    for (size_t i=0; i<ndim; ++i)
      {
      shp[i] = desc.shape[ndim-1-i];
      str[i] = desc.stride[ndim-1-i];
      }
  else
    for (size_t i=0; i<ndim; ++i)
      {
      shp[i] = desc.shape[i];
      str[i] = desc.stride[i];
      }
  }

template<bool swapdims, typename T, size_t ndim> auto prep1
  (const ArrayDescriptor &desc)
  {
  static_assert(ndim<=ArrayDescriptor::maxdim, "dimensionality too high");
  MR_assert(ndim==desc.ndim, "dimensionality mismatch");
  MR_assert(Typecode<T>::value==desc.dtype, "data type mismatch");
  typename mav_info<ndim>::shape_t shp;
  typename mav_info<ndim>::stride_t str;
  copy_data<swapdims>(desc, shp, str);
  return make_tuple(shp, str);
  }
template<bool swapdims, typename T, size_t ndim> cmav<T,ndim> to_cmav(const ArrayDescriptor &desc)
  {
  auto [shp, str] = prep1<swapdims, T, ndim>(desc);
  return cmav<T, ndim>(reinterpret_cast<const T *>(desc.data), shp, str);
  }
template<bool swapdims, typename T, typename T2, size_t ndim> cmav<T2,ndim> to_cmav_with_typecast(const ArrayDescriptor &desc)
  {
  static_assert(sizeof(T)==sizeof(T2), "type size mismatch");
  auto [shp, str] = prep1<swapdims, T, ndim>(desc);
  return cmav<T2, ndim>(reinterpret_cast<const T2 *>(desc.data), shp, str);
  }
template<bool swapdims, typename T, size_t ndim> vmav<T,ndim> to_vmav(ArrayDescriptor &desc)
  {
  auto [shp, str] = prep1<swapdims, T, ndim>(desc);
  return vmav<T, ndim>(reinterpret_cast<T *>(desc.data), shp, str);
  }
template<bool swapdims, typename T> auto prep2(const ArrayDescriptor &desc)
  {
  MR_assert(Typecode<T>::value==desc.dtype, "data type mismatch");
  typename fmav_info::shape_t shp(desc.ndim);
  typename fmav_info::stride_t str(desc.ndim);
  copy_data<swapdims>(desc, shp, str);
  return make_tuple(shp, str);
  }
template<bool swapdims, typename T> cfmav<T> to_cfmav(const ArrayDescriptor &desc)
  {
  auto [shp, str] = prep2<swapdims, T>(desc);
  return cfmav<T>(reinterpret_cast<const T *>(desc.data), shp, str);
  }
template<bool swapdims, typename T> vfmav<T> to_vfmav(ArrayDescriptor &desc)
  {
  auto [shp, str] = prep2<swapdims, T>(desc);
  return vfmav<T>(reinterpret_cast<T *>(desc.data), shp, str);
  }

template<bool swap_content, typename Tin, typename Tout> vector<Tout> to_vector
  (const ArrayDescriptor &desc)
  {
  MR_assert(Typecode<Tin>::value==desc.dtype, "data type mismatch");
  MR_assert(desc.ndim==1, "need 1D array for conversion to vector");
  vector<Tout> res;
  res.reserve(desc.shape[0]);
  auto data = reinterpret_cast<const Tin *>(desc.data);
  for (size_t i=0; i<desc.shape[0]; ++i)
    res.push_back(swap_content ? data[(desc.shape[0]-1-i)*desc.stride[0]]
                               : data[i*desc.stride[0]]);
  return res;
  }
template<bool swap_content, typename Tin, typename Tout> vector<Tout> to_vector_subtract_1
  (const ArrayDescriptor &desc)
  {
  static_assert(is_integral<Tin>::value, "need an integral type for this");
  MR_assert(Typecode<Tin>::value==desc.dtype, "data type mismatch");
  MR_assert(desc.ndim==1, "need 1D array for conversion to vector");
  vector<Tout> res;
  res.reserve(desc.shape[0]);
  auto data = reinterpret_cast<const Tin *>(desc.data);
  for (size_t i=0; i<desc.shape[0]; ++i)
    {
    Tin val = (swap_content ? data[(desc.shape[0]-1-i)*desc.stride[0]]
                            : data[i*desc.stride[0]]) - Tin(1);
    res.push_back(Tout(val));
    }
  return res;
  }

template<typename T, size_t ndim> cmav<T,ndim> subtract_1(const cmav<T,ndim> &inp)
  {
  vmav<T,ndim> res(inp.shape());
  mav_apply([](T &v1, const T &v2){v1=v2-T(1);}, 1, res, inp);
  return res;
  }
}

using detail_array_descriptor::ArrayDescriptor;
using detail_array_descriptor::to_cmav;
using detail_array_descriptor::to_cmav_with_typecast;
using detail_array_descriptor::to_vmav;
using detail_array_descriptor::to_cfmav;
using detail_array_descriptor::to_vfmav;
using detail_array_descriptor::to_vector;
using detail_array_descriptor::to_vector_subtract_1;
using detail_array_descriptor::subtract_1;

uint8_t get_ndim(const RustArrayDescriptor &arg) {
  return arg.ndim;
}

void set_ndim(RustArrayDescriptor &arg, uint8_t ndim) {
  arg.ndim = ndim;
}

uint64_t get_shape(const RustArrayDescriptor &arg, const uint8_t idim) {
  return arg.shape[idim];
}

// void set_shape(const RustArrayDescriptor &arg, const uint8_t idim, const uint64_t val) {
//   arg.shape[idim] = val;
// };

}

#endif
