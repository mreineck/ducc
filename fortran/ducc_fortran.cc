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

/* Copyright (C) 2023 Max-Planck-Society
   Author: Martin Reinecke */

/*
Compilation: (the -I path must point to the src/ directory in the ducc0 checkout)

g++ -O3 -march=native -ffast-math -I ../src/ ducc_fortran.cc -Wfatal-errors -pthread -std=c++17 -fPIC -c -fvisibility=hidden

Creating the shared library:

g++ -O3 -march=native -o ducc_fortran.so ducc_fortran.o -Wfatal-errors -pthread -std=c++17 -shared -fPIC

CONVENTIONS USED FOR THIS WRAPPER:
 - passed ArrayDescriptors are in Fortran order, i.e. all axis-swapping etc. will
   take place on the C++ side, if necessary
 - if axis indices or array indices are passed, they are assumed to be one-based
*/

#include <ISO_Fortran_binding.h>
#include "ducc0/infra/threading.cc"
#include "ducc0/infra/mav.cc"
//#include "ducc0/math/gl_integrator.cc"
//#include "ducc0/math/gridding_kernel.cc"
#include "ducc0/fft/fft.h"
#include "ducc0/fft/fft1d_impl.h"
#include "ducc0/fft/fftnd_impl.h"
//#include "ducc0/nufft/nufft.h"
#include "ducc0/bindings/typecode.h"
#include "ducc0/bindings/array_descriptor.h"
//#include "ducc0/sht/sht.cc"

#include <iostream>

using namespace ducc0;
using namespace std;

//using namespace Fortran::ISO;  // for flang in the future?

#if defined _WIN32 || defined __CYGWIN__
#define DUCC0_INTERFACE_FUNCTION extern "C" [[gnu::dllexport]]
#else
#define DUCC0_INTERFACE_FUNCTION extern "C" [[gnu::visibility("default")]]
#endif

void arraytodesc_c(const CFI_cdesc_t *arr, ducc0::ArrayDescriptor *desc)
  {
  MR_assert((arr!=nullptr) && (desc!=nullptr), "Null pointer found");
  switch (arr->type)
    {
    case CFI_type_int32_t: desc->dtype=19; break;
    case CFI_type_int64_t: desc->dtype=23; break;
    case CFI_type_float: desc->dtype=3; break;
    case CFI_type_double: desc->dtype=7; break;
    case CFI_type_float_Complex: desc->dtype=67; break;
    case CFI_type_double_Complex: desc->dtype=71; break;
    default: MR_fail("unsupported data type");
    }
  desc->ndim = arr->rank;
  MR_assert(desc->ndim<=10, "array rank too large");
  desc->data = arr->base_addr; // FIXME: perhaps needs correction!
  size_t nbytes = typeSize(desc->dtype);
  for (size_t i=0; i<size_t(desc->ndim); ++i)
    {
    desc->shape[i] = arr->dim[i].extent;
    desc->stride[i] = arr->dim[i].sm/nbytes; //??
    MR_assert(desc->stride[i]*ptrdiff_t(nbytes)==arr->dim[i].sm, "bad stride");
    }
  }

// FFT

DUCC0_INTERFACE_FUNCTION
void fft_c2c_c(const CFI_cdesc_t *in__, CFI_cdesc_t *out__,
  const CFI_cdesc_t *axes__, int forward, double fct, size_t nthreads)
  {
  ArrayDescriptor in_, out_, axes_;
  arraytodesc_c(in__, &in_);
  arraytodesc_c(out__, &out_);
  arraytodesc_c(axes__, &axes_);
  auto &in(in_);
  auto &out(out_);
  const auto &axes(axes_);
  auto myaxes(axes.to_vector_subtract_1<false, size_t>());
  for (auto &a: myaxes) a = in.ndim-1-a;
  if (isTypecode<complex<double>>(in.dtype))
    {
    auto myin(in.to_cfmav<true,complex<double>>());
    auto myout(out.to_vfmav<true,complex<double>>());
    c2c(myin, myout, myaxes, bool(forward), fct, nthreads);
    }
  else if (isTypecode<complex<float>>(in.dtype))
    {
    auto myin(in.to_cfmav<true,complex<float>>());
    auto myout(out.to_vfmav<true,complex<float>>());
    c2c(myin, myout, myaxes, bool(forward), float(fct), nthreads);
    }
  else
    MR_fail("bad datatype");
  }

