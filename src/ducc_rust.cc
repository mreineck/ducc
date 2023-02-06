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

#include "ducc0/infra/threading.cc"
#include "ducc0/infra/mav.cc"
#include "ducc0/bindings/typecode.h"
#include "ducc0/bindings/array_descriptor.h"

template<typename T> void square_impl(ducc0::ArrayDescriptor &arg) {
   auto bar = ducc0::to_vfmav<false, T>(arg);
   ducc0::mav_apply([](T &v1){v1*=v1;}, 1, bar);
}

extern "C" {

void square(ducc0::ArrayDescriptor &arg) {
  auto typec = arg.dtype;
  if (typec == ducc0::Typecode<double>::value)
    square_impl<double>(arg);
  else if(typec == ducc0::Typecode<float>::value)
    square_impl<float>(arg);
  else if(typec == ducc0::Typecode<std::complex<double>>::value)
    square_impl<std::complex<double>>(arg);
  else if(typec == ducc0::Typecode<std::complex<float>>::value)
    square_impl<std::complex<float>>(arg);
  else
    MR_fail("asdf");
 }

}

