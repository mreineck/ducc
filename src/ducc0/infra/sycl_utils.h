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

#if defined(DUCC0_USE_SYCL)
//#if (defined(SYCL_LANGUAGE_VERSION) && (SYCL_LANGUAGE_VERSION>=202001))
//#if (__has_include("CL/sycl.hpp"))
#include "CL/sycl.hpp"
//#define DUCC0_USE_SYCL
//#endif
//#endif

#if (defined(__HIPSYCL_ENABLE_CUDA_TARGET__))
#if (__has_include(<cufft.h>))
#include <cufft.h>
#define DUCC0_HAVE_CUFFT
#endif
#endif

#include <vector>
#include "ducc0/infra/mav.h"
#include "ducc0/infra/error_handling.h"
#include "ducc0/fft/fft.h"

#endif

namespace ducc0 {

#if defined(DUCC0_USE_SYCL)

namespace detail_sycl_utils {

using namespace std;
using namespace cl;

template<typename T, size_t ndim> inline sycl::buffer<T, ndim> make_sycl_buffer
  (const cmav<T,ndim> &arr)
  {
  MR_assert(arr.contiguous(), "mav must be contiguous");
#if 0
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
#else  // hack to avoid unnecessary copies with hipSYCL
  if constexpr (ndim==1)
    {
    sycl::buffer<T, ndim> res {const_cast<T *>(arr.data()),
      sycl::range<ndim>(arr.shape(0)),
      {sycl::property::buffer::use_host_ptr()}};
    res.set_write_back(false);
    return res;
    }
  if constexpr (ndim==2)
    {
    sycl::buffer<T, ndim> res {const_cast<T *>(arr.data()),
      sycl::range<ndim>(arr.shape(0), arr.shape(1)),
      {sycl::property::buffer::use_host_ptr()}};
    res.set_write_back(false);
    return res;
    }
  if constexpr (ndim==3)
    {
    sycl::buffer<T, ndim> res {const_cast<T *>(arr.data()),
      sycl::range<ndim>(arr.shape(0), arr.shape(1),arr.shape(2)),
      {sycl::property::buffer::use_host_ptr()}};
    res.set_write_back(false);
    return res;
    }
  if constexpr (ndim==4)
    {
    sycl::buffer<T, ndim> res {const_cast<T *>(arr.data()),
      sycl::range<ndim>(arr.shape(0), arr.shape(1),arr.shape(2),arr.shape(3)),
      {sycl::property::buffer::use_host_ptr()}};
    res.set_write_back(false);
    return res;
    }
#endif
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
#if 0
  return sycl::buffer<T, 1> {arr.data(),
         sycl::range<1>(arr.size()),
         {sycl::property::buffer::use_host_ptr()}};
#else  // hack to avoid unnecessary copies with hipSYCL
  sycl::buffer<T, 1> res{const_cast<T *>(arr.data()),
    sycl::range<1>(arr.size()),
    {sycl::property::buffer::use_host_ptr()}};
  res.set_write_back(false);
  return res;
#endif
  }
template<typename T> inline sycl::buffer<T,1> make_sycl_buffer
  (vector<T> &arr)
  {
  return sycl::buffer<T, 1> {arr.data(),
         sycl::range<1>(arr.size()),
         {sycl::property::buffer::use_host_ptr()}};
  }

#if (defined (DUCC0_HAVE_CUFFT))
#define DUCC0_CUDACHECK(cmd, err) { auto res=cmd; MR_assert(res==CUFFT_SUCCESS, err, "\nError code: ", res); }

template<typename T> class sycl_fft_plan
  {
  private:
    static_assert(is_same<T,double>::value || is_same<T,float>::value, "unsupported data type");
    cufftHandle plan;

  public:
    sycl_fft_plan(sycl::buffer<complex<T>,2> &buf)
      {
      auto transtype = is_same<T,double>::value ? CUFFT_Z2Z : CUFFT_C2C;
      DUCC0_CUDACHECK(cufftPlan2d(&plan, buf.get_range().get(0), buf.get_range().get(1), transtype),
            "planning failed")
      DUCC0_CUDACHECK(cudaDeviceSynchronize(), "synchronization problem")
      }
    void exec(sycl::queue &q, sycl::buffer<complex<T>,2> &buf, bool forward)
      {
      q.wait();
      q.submit([&](sycl::handler &cgh)
        {
        sycl::accessor acc{buf, cgh, sycl::read_write};
        auto direction = forward ? CUFFT_FORWARD : CUFFT_INVERSE;
        cgh.hipSYCL_enqueue_custom_operation([acc,direction,plan=plan](sycl::interop_handle &h) {
          DUCC0_CUDACHECK(cufftSetStream(plan, h.get_native_queue<sycl::backend::cuda>()), "could not set stream")
          void *native_mem = h.get_native_mem<sycl::backend::cuda>(acc);
          if constexpr(is_same<T,double>::value)
            {
            auto* cu_d = reinterpret_cast<cufftDoubleComplex *>(native_mem);
            DUCC0_CUDACHECK(cufftExecZ2Z(plan, cu_d, cu_d, direction),
              "double precision FFT failed")
            }
          else
            {
            auto* cu_d = reinterpret_cast<cufftComplex *>(native_mem);
            DUCC0_CUDACHECK(cufftExecC2C(plan, cu_d, cu_d, direction),
              "single precision FFT failed")
            }
          });
        });
      q.wait();
      }
    ~sycl_fft_plan()
      {
      DUCC0_CUDACHECK(cufftDestroy(plan), "plan destruction failed")
      DUCC0_CUDACHECK(cudaDeviceSynchronize(), "synchronization problem")
      }
  };

#undef DUCC0_CUDACHECK
#endif

template<typename T, int ndim> void sycl_zero_buffer(sycl::queue &q, sycl::buffer<T,ndim> &buf)
  {
  q.submit([&](sycl::handler &cgh)
    {
    sycl::accessor acc{buf, cgh, sycl::write_only, sycl::no_init};
    if constexpr(ndim==1)
      cgh.parallel_for(sycl::range<1>(acc.get_range().get(0)), [acc](sycl::item<1> item)
        { acc[item.get_id(0)] = T(0); });
    if constexpr(ndim==2)
       cgh.parallel_for(sycl::range<2>(acc.get_range().get(0), acc.get_range().get(1)), [acc](sycl::item<2> item)
        { acc[item.get_id(0)][item.get_id(1)] = T(0); });
    if constexpr(ndim==3)
       cgh.parallel_for(sycl::range<3>(acc.get_range().get(0), acc.get_range().get(1), acc.get_range().get(2)), [acc](sycl::item<3> item)
        { acc[item.get_id(0)][item.get_id(1)][item.get_id(2)] = T(0); });
    });
  }

void print_device_info(const sycl::device &device)
  {
  cout << "max_compute_units: " << device.template get_info<sycl::info::device::max_compute_units>() << endl;
  cout << "max_work_group_size: " << device.template get_info<sycl::info::device::max_work_group_size>() << endl;
  cout << "max_work_item_dimensions: " << device.template get_info<sycl::info::device::max_work_item_dimensions>() << endl;
//using blah = sycl::info::device::max_work_item_sizes<1>;
//cout << "max_work_item_sizes<1>: " << device.template get_info<blah>() << endl;
  auto has_local_mem = device.is_host()
    || (device.get_info<sycl::info::device::local_mem_type>()
    != sycl::info::local_mem_type::none);
  auto local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
  cout << "local memory size: " << local_mem_size << endl;
  auto subgroupsizes = device.get_info<sycl::info::device::sub_group_sizes>();
  cout << "sub group sizes: ";
  for (auto i:subgroupsizes) cout << i << " ";
  cout << endl;
  }

#ifndef __INTEL_LLVM_COMPILER
template<typename T> using my_atomic_ref = sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,sycl::access::address_space::global_space>;
template<typename T> using my_atomic_ref_l = sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::work_group,sycl::access::address_space::local_space>;
#else
template<typename T> using my_atomic_ref = sycl::ext::oneapi::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,sycl::access::address_space::global_space>;
template<typename T> using my_atomic_ref_l = sycl::ext::oneapi::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,sycl::access::address_space::local_space>;
#endif

#ifndef __INTEL_LLVM_COMPILER
template<typename T, size_t ndim> using my_local_accessor = sycl::local_accessor<T,ndim>;
#else
template<typename T, size_t ndim> using my_local_accessor = sycl::accessor<T,ndim,sycl::access::mode::read_write, sycl::access::target::local>;
#endif

template<typename T, int ndim> void ensure_device_copy(sycl::queue &q, sycl::buffer<T,ndim> &buf)
  {
  q.submit([&](sycl::handler &cgh)
    {
    sycl::accessor acc{buf, cgh, sycl::read_only};
    cgh.single_task([acc](){});
    });
  }
}

using detail_sycl_utils::make_sycl_buffer;
using detail_sycl_utils::sycl_zero_buffer;
using detail_sycl_utils::sycl_fft_plan;
using detail_sycl_utils::print_device_info;
using detail_sycl_utils::my_atomic_ref;
using detail_sycl_utils::my_atomic_ref_l;
using detail_sycl_utils::my_local_accessor;
using detail_sycl_utils::ensure_device_copy;

#endif

inline bool sycl_active()
  {
#if defined(DUCC0_USE_SYCL)
  return true;
#else
  return false;
#endif
  }

}

#endif
