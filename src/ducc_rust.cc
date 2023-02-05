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

#include "ducc_rust.h"
#include "src/main.rs.h"

#include "ducc0/infra/threading.cc"
#include "ducc0/infra/mav.cc"
#include "ducc0/bindings/typecode.h"
#include "ducc0/bindings/array_descriptor.h"

namespace ducc0 {
  namespace rustInterface {

uint8_t get_ndim(const RustArrayDescriptor &arg) {
  return arg.ndim;
}

uint8_t get_dtype(const RustArrayDescriptor &arg) {
  return arg.dtype;
}

void set_ndim(RustArrayDescriptor &arg, uint8_t ndim) {
  arg.ndim = ndim;
}

uint64_t get_shape(const RustArrayDescriptor &arg, const uint8_t idim) {
  return arg.shape[idim];
}

int64_t get_stride(const RustArrayDescriptor &arg, const uint8_t idim) {
  return arg.stride[idim];
}

void square(RustArrayDescriptor &arg) {
  auto &ad(reinterpret_cast<ArrayDescriptor &>(arg));
  ad.dtype = Typecode<double>::value;  // TODO

  // struct ArrayDescriptor ad;
  // ad.data = const_cast<double*>(arg.data); // THIS IS NOT NICE ;)
  // ad.ndim = arg.ndim;
  // // ad.dtype = arg.dtype;
  // ad.dtype = Typecode<double>::value;  // TODO
  // for (auto i=0; i<10; i++) {
  //   ad.shape[i] = get_shape(arg, i);
  //   ad.stride[i] = get_stride(arg, i);
  // }

  auto bar = to_vfmav<false, double>(ad);  // TODO
  mav_apply([](double &v1){v1*=v1;}, 1, bar);
}
}
}
