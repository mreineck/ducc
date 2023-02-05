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

namespace ducc0 {
namespace rustInterface {

  struct RustArrayDescriptor;

uint8_t get_ndim(const RustArrayDescriptor &arg);
uint8_t get_dtype(const RustArrayDescriptor &arg);
void set_ndim(RustArrayDescriptor &arg, uint8_t ndim);
uint64_t get_shape(const RustArrayDescriptor &arg, const uint8_t idim);
int64_t get_stride(const RustArrayDescriptor &arg, const uint8_t idim);
void square(RustArrayDescriptor &arg);

}
}
