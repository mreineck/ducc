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

/* Copyright (C) 2022-2023 Max-Planck-Society, Philipp Arras
   Authors: Philipp Arras */

#include <cstdint>
#include <complex>
#include <vector>
using namespace std;
using shape_t = vector<size_t>;

#include "ducc0/infra/threading.cc"
#include "ducc0/infra/mav.cc"
#include "ducc0/bindings/typecode.h"
#include "ducc0/bindings/array_descriptor.h"
#include "ducc0/fft/fft.h"

static constexpr size_t MAXDIM=10;

template<typename T> void square_impl(ducc0::ArrayDescriptor &arg) {
   auto bar = ducc0::to_vfmav<false, T>(arg);
   ducc0::mav_apply([](T &v1){v1*=v1;}, 1, bar);
}

template<typename T>
void c2c(const ducc0::ArrayDescriptor &in, ducc0::ArrayDescriptor &out, const ducc0::ArrayDescriptor &axes, const bool forward, const T fct, const size_t nthreads) {
  auto in_mav = ducc0::to_cfmav<false, complex<T>>(in);
  auto out_mav = ducc0::to_vfmav<false, complex<T>>(out);

  auto axes_mav = ducc0::to_cfmav<false, size_t>(axes);
  // TODO Check if 1d etc.

  shape_t axes1;
  for (size_t i=0; i<axes_mav.shape(0); i++)
    axes1.push_back(axes_mav(i));

  ducc0::c2c(in_mav, out_mav, axes1, forward, fct, nthreads);
}

extern "C" {

void c_square(ducc0::ArrayDescriptor &arg) {
  auto typec = arg.dtype;
  if (typec == ducc0::Typecode<double>::value)
    square_impl<double>(arg);
  else if(typec == ducc0::Typecode<float>::value)
    square_impl<float>(arg);
  else if(typec == ducc0::Typecode<complex<double>>::value)
    square_impl<complex<double>>(arg);
  else if(typec == ducc0::Typecode<complex<float>>::value)
    square_impl<complex<float>>(arg);
  else
    MR_fail("asdf");
 }

void c2c_double(const ducc0::ArrayDescriptor &in, ducc0::ArrayDescriptor &out, const ducc0::ArrayDescriptor &axes, const bool forward, const double fct, const size_t nthreads) {
  c2c(in, out, axes, forward, fct, nthreads);
}
// void c2c_float(const ducc0::ArrayDescriptor &in, ducc0::ArrayDescriptor &out, const bool axes[MAXDIM], const bool forward, const float fct, const size_t nthreads) {
//   c2c(in, out, axes, forward, fct, nthreads);
// }
// void c2c_inplace_double(ducc0::ArrayDescriptor &inout, const bool axes[MAXDIM], const bool forward, const double fct, const size_t nthreads) {
//   c2c(inout, inout, axes, forward, fct, nthreads);
// }
// void c2c_inplace_float(ducc0::ArrayDescriptor &inout, const bool axes[MAXDIM], const bool forward, const float fct, const size_t nthreads) {
//   c2c(inout, inout, axes, forward, fct, nthreads);
// }

}
