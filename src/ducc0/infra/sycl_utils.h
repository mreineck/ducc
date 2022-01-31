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
#include "ducc0/fft/fft.h"

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

#if (defined (DUCC0_HAVE_CUFFT))
template<typename T, int ndim> void sycl_c2c(sycl::queue &q, sycl::buffer<complex<T>,ndim> &buf, bool forward)
  {
  // This should not be needed, but without it tests fail when optimization is off
  q.wait();
  q.submit([&](sycl::handler &cgh)
    {
    auto acc{buf.template get_access<sycl::access::mode::read_write>(cgh)};
    cgh.hipSYCL_enqueue_custom_operation([=](sycl::interop_handle &h) {
      void *native_mem = h.get_native_mem<sycl::backend::cuda>(acc);
      cufftHandle plan;
#define DUCC0_CUDACHECK(cmd, err) { auto res=cmd; MR_assert(res==CUFFT_SUCCESS, err); }
      DUCC0_CUDACHECK(cufftCreate(&plan), "plan creation failed")
//      DUCC0_CUDACHECK(cufftSetStream(plan, h.get_native_queue<sycl::backend::cuda>()), "could not set stream");
      auto direction = forward ? CUFFT_FORWARD : CUFFT_INVERSE;
      if constexpr (is_same<T,double>::value)
        {
        if constexpr(ndim==2)
          {
          DUCC0_CUDACHECK(cufftPlan2d(&plan, buf.get_range().get(0), buf.get_range().get(1), CUFFT_Z2Z),
            "double precision planning failed")
          }
        else
          MR_fail("unsupported dimensionality");
        auto* cu_d = reinterpret_cast<cufftDoubleComplex *>(native_mem);
        DUCC0_CUDACHECK(cufftExecZ2Z(plan, cu_d, cu_d, direction),
          "double precision FFT failed")
        }
      else if constexpr (is_same<T,float>::value)
        {
        if constexpr(ndim==2)
          {
          DUCC0_CUDACHECK(cufftPlan2d(&plan, buf.get_range().get(0), buf.get_range().get(1), CUFFT_C2C),
            "double precision planning failed")
          }
        else
          MR_fail("unsupported dimensionality");
        auto* cu_d = reinterpret_cast<cufftComplex *>(native_mem);
        DUCC0_CUDACHECK(cufftExecC2C(plan, cu_d, cu_d, direction),
          "single precision FFT failed")
        }
      else
        MR_fail("unsupported data type");
      DUCC0_CUDACHECK(cufftDestroy(plan), "plan destruction failed")
#undef DUCC0_CUDACHECK
      });
    });
//q.wait();
  }
#else
template<typename T, int ndim> void sycl_c2c(sycl::queue &q, sycl::buffer<complex<T>,ndim> &buf, bool forward)
  {
  sycl::host_accessor<complex<T>,ndim,sycl::access::mode::read_write> acc{buf};
  complex<T> *ptr = acc.get_pointer();
  if constexpr(ndim==2)
    {
    vfmav<complex<T>> arr(ptr, {buf.get_range().get(0), buf.get_range().get(1)});
    c2c (arr, arr, {0,1}, forward, T(1));
    }
  else
    MR_fail("unsupported dimensionality");
  }
#endif
}

using detail_sycl_utils::make_sycl_buffer;
using detail_sycl_utils::sycl_c2c;

}

#endif
