/*
 *  This file is part of libc_utils.
 *
 *  libc_utils is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  libc_utils is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with libc_utils; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/*
 *  libc_utils is being developed at the Max-Planck-Institut fuer Astrophysik
 *  and financially supported by the Deutsches Zentrum fuer Luft- und Raumfahrt
 *  (DLR).
 */

/*
 *  Utilities for conversion between coordinates, Morton, and Peano indices
 *
 *  Copyright (C) 2015-2020 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#ifndef MRUTIL_MORTON_UTILS_H
#define MRUTIL_MORTON_UTILS_H

#include <cstdint>

#ifdef __BMI2__
#include <x86intrin.h>
#endif

namespace mr {

#ifndef __BMI2__
uint32_t block2morton2D_32 (uint32_t v);
uint32_t coord2morton2D_32 (uint32_t x, uint32_t y);
uint32_t morton2block2D_32 (uint32_t v);
void morton2coord2D_32 (uint32_t v, uint32_t *x, uint32_t *y);
uint64_t block2morton2D_64 (uint64_t v);
uint64_t coord2morton2D_64 (uint64_t x, uint64_t y);
uint64_t morton2block2D_64 (uint64_t v);
void morton2coord2D_64 (uint64_t v, uint64_t *x, uint64_t *y);

uint32_t block2morton3D_32 (uint32_t v);
uint32_t coord2morton3D_32 (uint32_t x, uint32_t y, uint32_t z);
uint32_t morton2block3D_32 (uint32_t v);
void morton2coord3D_32 (uint32_t v, uint32_t *x, uint32_t *y, uint32_t *z);
uint64_t block2morton3D_64 (uint64_t v);
uint64_t coord2morton3D_64 (uint64_t x, uint64_t y, uint64_t z);
uint64_t morton2block3D_64 (uint64_t v);
void morton2coord3D_64 (uint64_t v, uint64_t *x, uint64_t *y, uint64_t *z);
#else
inline uint32_t block2morton2D_32 (uint32_t v)
  { return _pdep_u32(v,0x55555555u)|_pdep_u32(v>>16,0xaaaaaaaau); }
inline uint32_t coord2morton2D_32 (uint32_t x, uint32_t y)
  { return _pdep_u32(x,0x55555555u)|_pdep_u32(y,0xaaaaaaaau); }
inline uint32_t morton2block2D_32 (uint32_t v)
  { return _pext_u32(v,0x55555555u)|(_pext_u32(v,0xaaaaaaaau)<<16); }
inline void morton2coord2D_32 (uint32_t v, uint32_t *x, uint32_t *y)
  { *x=_pext_u32(v,0x55555555u); *y=_pext_u32(v,0xaaaaaaaau); }
inline uint64_t block2morton2D_64 (uint64_t v)
  {
  return _pdep_u64(v,0x5555555555555555u)
        |_pdep_u64(v>>32,0xaaaaaaaaaaaaaaaau);
  }
inline uint64_t coord2morton2D_64 (uint64_t x, uint64_t y)
  { return _pdep_u64(x,0x5555555555555555u)|_pdep_u64(y,0xaaaaaaaaaaaaaaaau); }
inline uint64_t morton2block2D_64 (uint64_t v)
  {
  return _pext_u64(v,0x5555555555555555u)
       |(_pext_u64(v,0xaaaaaaaaaaaaaaaau)<<32);
  }
inline void morton2coord2D_64 (uint64_t v, uint64_t *x, uint64_t *y)
  {
  *x=_pext_u64(v,0x5555555555555555u);
  *y=_pext_u64(v,0xaaaaaaaaaaaaaaaau);
  }

inline uint32_t block2morton3D_32 (uint32_t v)
  {
  return _pdep_u32(v    ,0x09249249u)
        |_pdep_u32(v>>10,0x12492492u)
        |_pdep_u32(v>>20,0x24924924u);
  }
inline uint32_t coord2morton3D_32 (uint32_t x, uint32_t y, uint32_t z)
  {
  return _pdep_u32(x,0x09249249u)
        |_pdep_u32(y,0x12492492u)
        |_pdep_u32(z,0x24924924u);
  }
inline uint32_t morton2block3D_32 (uint32_t v)
  {
  return _pext_u32(v,0x9249249u)
       |(_pext_u32(v,0x12492492u)<<10)
       |(_pext_u32(v,0x24924924u)<<20);
  }
inline void morton2coord3D_32 (uint32_t v,
  uint32_t *x, uint32_t *y, uint32_t *z)
  {
  *x = _pext_u32(v,0x09249249u);
  *y = _pext_u32(v,0x12492492u);
  *z = _pext_u32(v,0x24924924u);
  }
inline uint64_t block2morton3D_64 (uint64_t v)
  {
  return _pdep_u64(v    ,0x1249249249249249u)
        |_pdep_u64(v>>21,0x2492492492492492u)
        |_pdep_u64(v>>42,0x4924924924924924u);
  }
inline uint64_t coord2morton3D_64 (uint64_t x, uint64_t y, uint64_t z)
  {
  return _pdep_u64(x,0x1249249249249249u)
        |_pdep_u64(y,0x2492492492492492u)
        |_pdep_u64(z,0x4924924924924924u);
  }
inline uint64_t morton2block3D_64 (uint64_t v)
  {
  return _pext_u64(v,0x1249249249249249u)
       |(_pext_u64(v,0x2492492492492492u)<<21)
       |(_pext_u64(v,0x4924924924924924u)<<42);
  }
inline void morton2coord3D_64 (uint64_t v,
  uint64_t *x, uint64_t *y, uint64_t *z)
  {
  *x = _pext_u64(v,0x1249249249249249u);
  *y = _pext_u64(v,0x2492492492492492u);
  *z = _pext_u64(v,0x4924924924924924u);
  }
#endif

uint32_t morton2peano2D_32(uint32_t v, int bits);
uint32_t peano2morton2D_32(uint32_t v, int bits);

uint64_t morton2peano2D_64(uint64_t v, int bits);
uint64_t peano2morton2D_64(uint64_t v, int bits);

uint32_t morton2peano3D_32(uint32_t v, int bits);
uint32_t peano2morton3D_32(uint32_t v, int bits);

uint64_t morton2peano3D_64(uint64_t v, int bits);
uint64_t peano2morton3D_64(uint64_t v, int bits);

inline uint32_t coord2block2D_32(uint32_t x, uint32_t y)
  { return (x&0xffff) | (y<<16); }
inline void block2coord2D_32(uint32_t v, uint32_t *x, uint32_t *y)
  { *x=v&0xffff; *y=v>>16; }
inline uint32_t coord2block3D_32(uint32_t x, uint32_t y, uint32_t z)
  { return (x&0x3ff) | ((y&0x3ff)<<10) | ((z&0x3ff)<<20); }
inline void block2coord3D_32(uint32_t v,
  uint32_t *x, uint32_t *y, uint32_t *z)
  { *x=v&0x3ff; *y=(v>>10)&0x3ff; *z=(v>>20)&0x3ff; }

inline uint64_t coord2block2D_64(uint64_t x, uint64_t y)
  { return (x&0xffffffff) | (y<<32); }
inline void block2coord2D_64(uint64_t v, uint64_t *x, uint64_t *y)
  { *x=v&0xffffffff; *y=v>>32; }
inline uint64_t coord2block3D_64(uint64_t x, uint64_t y, uint64_t z)
  { return (x&0x1fffff) | ((y&0x1fffff)<<21) | ((z&0x1fffff)<<42); }
inline void block2coord3D_64(uint64_t v,
  uint64_t *x, uint64_t *y, uint64_t *z)
  { *x=v&0x1fffff; *y=(v>>21)&0x1fffff; *z=(v>>42)&0x1fffff; }

}

#endif
