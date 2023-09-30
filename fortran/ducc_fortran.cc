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
   Authors: Martin Reinecke */

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

#if defined _WIN32 || defined __CYGWIN__
#define DUCC0_INTERFACE_FUNCTION extern "C" [[gnu::dllexport]]
#else
#define DUCC0_INTERFACE_FUNCTION extern "C" [[gnu::visibility("default")]]
#endif

// FFT

DUCC0_INTERFACE_FUNCTION
void fft_c2c(const ArrayDescriptor *in_, ArrayDescriptor *out_,
  const ArrayDescriptor *axes_, int forward, double fct, size_t nthreads)
  {
  const auto &in(*in_);
  auto &out(*out_);
  const auto &axes(*axes_);
  auto myaxes(to_vector_subtract_1<false, uint64_t, size_t>(axes));
  for (auto &a: myaxes) a = in.ndim-1-a;
  if (in.dtype==Typecode<complex<double>>::value)
    {
    auto myin(to_cfmav<true,complex<double>>(in));
    auto myout(to_vfmav<true,complex<double>>(out));
    c2c(myin, myout, myaxes, forward, fct, nthreads);
    }
  else if (in.dtype==Typecode<complex<float>>::value)
    {
    auto myin(to_cfmav<true,complex<float>>(in));
    auto myout(to_vfmav<true,complex<float>>(out));
    c2c(myin, myout, myaxes, forward, float(fct), nthreads);
    }
  else
    MR_fail("bad datatype");
  }

DUCC0_INTERFACE_FUNCTION
void fft_c2c_inplace(ArrayDescriptor *inout_,
  const ArrayDescriptor *axes_, int forward, double fct, size_t nthreads)
  {
  auto &inout(*inout_);
  const auto &axes(*axes_);
  auto myaxes(to_vector_subtract_1<false, uint64_t, size_t>(axes));
  for (auto &a: myaxes) a = inout.ndim-1-a;
  if (isDtype<complex<double>>(inout.dtype))
    {
    auto myinout(to_vfmav<true,complex<double>>(inout));
    c2c(cfmav(myinout), myinout, myaxes, forward, fct, nthreads);
    }
  else if (isDtype<complex<float>>(inout.dtype))
    {
    auto myinout(to_vfmav<true,complex<float>>(inout));
    c2c(cfmav(myinout), myinout, myaxes, forward, float(fct), nthreads);
    }
  else
    MR_fail("bad datatype");
  }

DUCC0_INTERFACE_FUNCTION
void print_array(const ducc0::ArrayDescriptor *desc)
  {
  cout << "ndim: " << int(desc->ndim) << endl;
  cout << "type: " << int(desc->dtype) << endl;
  for (size_t i=0;i<desc->ndim; ++i)
    {
    cout << desc->shape[i] << " " << int(desc->stride[i]) << endl;
    }
  }

DUCC0_INTERFACE_FUNCTION
ptrdiff_t get_stride (const char *p1, const char *p2, uint8_t dtype)
  {
  size_t nbytes = ((dtype&15)+1)*((dtype>>6)+1);
  ptrdiff_t res = ptrdiff_t(p2-p1)/ptrdiff_t(nbytes);
  MR_assert(res*nbytes==p2-p1, "bad stride");
  return res;
  }
