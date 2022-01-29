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

/* Copyright (C) 2022 Max-Planck-Society
   Author: Martin Reinecke */

#ifndef DUCC0_SYCL_UTILS_H
#define DUCC0_SYCL_UTILS_H

#if (__has_include("CL/sycl.hpp"))
#include "CL/sycl.hpp"
#define DUCC0_HAVE_SYCL
using namespace cl;
#endif

#if (defined(DUCC0_HAVE_SYCL))
#if (__has_include(<cufft.h>))
#include <cufft.h>
#define DUCC0_HAVE_CUFFT
#endif
#endif

#include <vector>
#include "ducc0/infra/mav.h"
#include "ducc0/infra/error_handling.h"

namespace ducc0 {

namespace detail_sycl_utils {

using namespace std;

template<typename T, size_t ndim> inline sycl::buffer<T, ndim> make_sycl_buffer
  (const cmav<T,ndim> &arr)
  {
  MR_assert(arr.contiguous(), "mav must be contiguous");
  if constexpr (ndim==1)
    return sycl::buffer<T, ndim> {arr.data(),
           sycl::range<ndim>(arr.shape(0)),
           {sycl::property::buffer::use_host_ptr()}};
  if constexpr (ndim==2)
    return sycl::buffer<T, ndim> {arr.data(),
           sycl::range<ndim>(arr.shape(0), arr.shape(1)),
           {sycl::property::buffer::use_host_ptr()}};
  if constexpr (ndim==3)
    return sycl::buffer<T, ndim> {arr.data(),
           sycl::range<ndim>(arr.shape(0), arr.shape(1), arr.shape(2)),
           {sycl::property::buffer::use_host_ptr()}};
  if constexpr (ndim==4)
    return sycl::buffer<T, ndim> {arr.data(),
           sycl::range<ndim>(arr.shape(0), arr.shape(1), arr.shape(2), arr.shape(3)),
           {sycl::property::buffer::use_host_ptr()}};
  MR_fail("dimensionality too high");
  }

template<typename T, size_t ndim> inline sycl::buffer<T, ndim> make_sycl_buffer
  (vmav<T,ndim> &arr)
  {
  MR_assert(arr.contiguous(), "mav must be contiguous");
  if constexpr (ndim==1)
    return sycl::buffer<T, ndim> {arr.data(),
           sycl::range<ndim>(arr.shape(0)),
           {sycl::property::buffer::use_host_ptr()}};
  if constexpr (ndim==2)
    return sycl::buffer<T, ndim> {arr.data(),
           sycl::range<ndim>(arr.shape(0), arr.shape(1)),
           {sycl::property::buffer::use_host_ptr()}};
  if constexpr (ndim==3)
    return sycl::buffer<T, ndim> {arr.data(),
           sycl::range<ndim>(arr.shape(0), arr.shape(1), arr.shape(2)),
           {sycl::property::buffer::use_host_ptr()}};
  if constexpr (ndim==4)
    return sycl::buffer<T, ndim> {arr.data(),
           sycl::range<ndim>(arr.shape(0), arr.shape(1), arr.shape(2), arr.shape(3)),
           {sycl::property::buffer::use_host_ptr()}};
  MR_fail("dimensionality too high");
  }

template<typename T> inline sycl::buffer<T,1> make_sycl_buffer
  (const vector<T> &arr)
  {
  return sycl::buffer<T, 1> {arr.data(),
         sycl::range<1>(arr.size()),
         {sycl::property::buffer::use_host_ptr()}};
  }
template<typename T> inline sycl::buffer<T,1> make_sycl_buffer
  (vector<T> &arr)
  {
  return sycl::buffer<T, 1> {arr.data(),
         sycl::range<1>(arr.size()),
         {sycl::property::buffer::use_host_ptr()}};
  }

}

using detail_sycl_utils::make_sycl_buffer;

}

#endif
