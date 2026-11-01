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

/*
 *  Utilities for conversion between integer coordinates, Morton indices,
 *  and Peano-Hilbert indices
 *
 *  Copyright (C) 2015-2023 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#include "ducc0/math/space_filling.h"
#include "ducc0/infra/error_handling.h"

namespace DUCC0_NAMESPACE {

using namespace std;

#ifndef DUCC0_USE_PDEP_PEXT

#if 0
// this variant is sometimes faster than the functions in the else-branch, but
// its tables need valuable L1 space.

namespace {

static constexpr const array<uint16_t, 256> utab = []()
  {
  array<uint16_t, 256> res;
  for (uint16_t i=0; i<256; ++i)
    res[i] = (i&1) + ((i&2)<<1) + ((i&4)<<2) + ((i&8)<<3)
           + ((i&16)<<4) + ((i&32)<<5) + ((i&64)<<6) + ((i&128)<<7);
  return res;
  }();

static constexpr const array<uint16_t, 256> ctab = []()
  {
  array<uint16_t, 256> res;
  for (uint16_t i=0; i<256; ++i)
    res[i] = ((i&1)<<0) + ((i&2)<<7) + ((i&4)>>1) + ((i&8)<<6)
           + ((i&16)>>2) + ((i&32)<<5) + ((i&64)>>3) + ((i&128)<<4);
  return res;
  }();

} // unnamed namespace

uint32_t spread_bits_2D_32 (uint32_t v)
  {
  using I=uint32_t;
  return  I(utab[ v     &0xff])     | (I(utab[(v>> 8)&0xff])<<16);
  }
uint64_t spread_bits_2D_64 (uint64_t v)
  {
  using I=uint64_t;
  return  I(utab[ v     &0xff])      | (I(utab[(v>> 8)&0xff])<<16)
       | (I(utab[(v>>16)&0xff])<<32) | (I(utab[(v>>24)&0xff])<<48);
  }
uint32_t block2morton2D_32 (uint32_t v)
  {
  using I=uint32_t;
  return  I(utab[ v     &0xff])     | (I(utab[(v>> 8)&0xff])<<16)
       | (I(utab[(v>>16)&0xff])<<1) | (I(utab[(v>>24)&0xff])<<17);
  }
uint32_t coord2morton2D_32 (array<uint32_t,2> xy)
  {
  using I=uint32_t;
  return  I(utab[xy[0]&0xff])     | (I(utab[(xy[0]>>8)&0xff])<<16)
       | (I(utab[xy[1]&0xff])<<1) | (I(utab[(xy[1]>>8)&0xff])<<17);
  }
uint32_t morton2block2D_32 (uint32_t v)
  {
  using I=uint32_t;
  I raw1 = v&0x55555555, raw2 = (v>>1)&0x55555555;
  raw1|=raw1>>15;
  raw2|=raw2>>15;
  return I(ctab[raw1&0xff]    ) | I(ctab[(raw1>>8)&0xff]<< 4)
       | I(ctab[raw2&0xff]<<16) | I(ctab[(raw2>>8)&0xff]<<20);
  }
array<uint32_t,2> morton2coord2D_32 (uint32_t v)
  {
  using I=uint32_t;
  I raw1 = v&0x55555555U, raw2 = (v>>1)&0x55555555U;
  raw1|=raw1>>15;
  raw2|=raw2>>15;
  return {I(ctab[raw1&0xff]) | I(ctab[(raw1>>8)&0xff]<<4),
          I(ctab[raw2&0xff]) | I(ctab[(raw2>>8)&0xff]<<4)};
  }
uint64_t block2morton2D_64 (uint64_t v)
  {
  using I=uint64_t;
  return  I(utab[ v     &0xff])      | (I(utab[(v>> 8)&0xff])<<16)
       | (I(utab[(v>>16)&0xff])<<32) | (I(utab[(v>>24)&0xff])<<48)
       | (I(utab[(v>>32)&0xff])<< 1) | (I(utab[(v>>40)&0xff])<<17)
       | (I(utab[(v>>48)&0xff])<<33) | (I(utab[(v>>56)&0xff])<<49);
  }
uint64_t coord2morton2D_64 (array<uint64_t,2> xy)
  {
  using I=uint64_t;
  return  I(utab[ xy[0]     &0xff])      | (I(utab[(xy[0]>> 8)&0xff])<<16)
       | (I(utab[(xy[0]>>16)&0xff])<<32) | (I(utab[(xy[0]>>24)&0xff])<<48)
       | (I(utab[ xy[1]     &0xff])<< 1) | (I(utab[(xy[1]>> 8)&0xff])<<17)
       | (I(utab[(xy[1]>>16)&0xff])<<33) | (I(utab[(xy[1]>>24)&0xff])<<49);
  }
uint64_t morton2block2D_64 (uint64_t v)
  {
  using I=uint64_t;
  I raw1 = v&0x5555555555555555, raw2 = (v>>1)&0x5555555555555555;
  raw1|=raw1>>15;
  raw2|=raw2>>15;
  return I(ctab[ raw1     &0xff])      | (I(ctab[(raw1>> 8)&0xff])<< 4)
      | (I(ctab[(raw1>>32)&0xff])<<16) | (I(ctab[(raw1>>40)&0xff])<<20)
      | (I(ctab[ raw2     &0xff])<<32) | (I(ctab[(raw2>> 8)&0xff])<<36)
      | (I(ctab[(raw2>>32)&0xff])<<48) | (I(ctab[(raw2>>40)&0xff])<<52);
  }
array<uint64_t,2> morton2coord2D_64 (uint64_t v)
  {
  using I=uint64_t;
  I raw1 = v&0x5555555555555555, raw2 = (v>>1)&0x5555555555555555;
  raw1|=raw1>>15;
  raw2|=raw2>>15;
  return { I(ctab[ raw1     &0xff])      | (I(ctab[(raw1>> 8)&0xff])<< 4)
        | (I(ctab[(raw1>>32)&0xff])<<16) | (I(ctab[(raw1>>40)&0xff])<<20),
           I(ctab[ raw2     &0xff])      | (I(ctab[(raw2>> 8)&0xff])<< 4)
        | (I(ctab[(raw2>>32)&0xff])<<16) | (I(ctab[(raw2>>40)&0xff])<<20)};
  }

#else

uint64_t spread_bits_2D_64 (uint64_t v)
  {
  v&=0xffffffffU;
  v = (v|(v<<16)) & 0x0000ffff0000ffffU;
  v = (v|(v<< 8)) & 0x00ff00ff00ff00ffU;
  v = (v|(v<< 4)) & 0x0f0f0f0f0f0f0f0fU;
  v = (v|(v<< 2)) & 0x3333333333333333U;
  v = (v|(v<< 1)) & 0x5555555555555555U;
  return v;
  }
uint32_t spread_bits_2D_32 (uint32_t v)
  {
  v&=0xffffU;
  v = (v|(v<< 8)) & 0x00ff00ff00ff00ffU;
  v = (v|(v<< 4)) & 0x0f0f0f0f0f0f0f0fU;
  v = (v|(v<< 2)) & 0x3333333333333333U;
  v = (v|(v<< 1)) & 0x5555555555555555U;
  return v;
  }

namespace {

inline uint64_t compress_bits_2D_64 (uint64_t v)
  {
  v&=0x5555555555555555U;
  v = (v|(v>> 1)) & 0x3333333333333333U;
  v = (v|(v>> 2)) & 0x0f0f0f0f0f0f0f0fU;
  v = (v|(v>> 4)) & 0x00ff00ff00ff00ffU;
  v = (v|(v>> 8)) & 0x0000ffff0000ffffU;
  v = (v|(v>>16)) & 0x00000000ffffffffU;
  return v;
  }
#if 0  // currently not needed
inline uint32_t compress_bits_2D_32 (uint32_t v)
  {
  v&=0x55555555U;
  v = (v|(v>> 1)) & 0x33333333U;
  v = (v|(v>> 2)) & 0x0f0f0f0fU;
  v = (v|(v>> 4)) & 0x00ff00ffU;
  v = (v|(v>> 8)) & 0x0000ffffU;
  return v;
  }
#endif

}  // unnamed namespace

uint32_t block2morton2D_32 (uint32_t v)
  { uint64_t t=spread_bits_2D_64(v); return (t | (t>>31)) & 0xffffffffU; }
uint32_t coord2morton2D_32 (array<uint32_t,2> xy)
  {
  uint64_t t=spread_bits_2D_64((xy[0]&0xffff)|(xy[1]<<16));
  return (t | (t>>31)) & 0xffffffffU;
  }
uint32_t morton2block2D_32 (uint32_t v)
  { uint64_t t=v; t|=t<<31; t=compress_bits_2D_64(t); return t; }

array<uint32_t,2> morton2coord2D_32 (uint32_t v)
  {
  uint64_t t=v; t|=t<<31; t=compress_bits_2D_64(t);
  return {uint32_t(t&0xffff), uint32_t(t>>16)};
  }
uint64_t block2morton2D_64 (uint64_t v)
  { return spread_bits_2D_64(v) | (spread_bits_2D_64(v>>32)<<1); }
uint64_t coord2morton2D_64 (array<uint64_t,2> xy)
  { return spread_bits_2D_64(xy[0]) | (spread_bits_2D_64(xy[1])<<1); }
uint64_t morton2block2D_64 (uint64_t v)
  { return compress_bits_2D_64(v) | (compress_bits_2D_64(v>>1)<<32); }
array<uint64_t,2> morton2coord2D_64 (uint64_t v)
  { return {compress_bits_2D_64(v), compress_bits_2D_64(v>>1)}; }

#endif

namespace {

inline uint32_t spread_bits_3D_32 (uint32_t v)
  {
  v&=0x3ff;
  v = (v|(v<< 8)|(v<<16)) & 0x0f00f00fU;
  v = (v|(v<< 4)) & 0xc30c30c3U;
  v = (v|(v<< 2)) & 0x49249249U;
  return v;
  }
inline uint32_t compress_bits_3D_32 (uint32_t v)
  {
  v&=0x9249249U;
  v = (v|(v>> 2)) & 0xc30c30c3U;
  v = (v|(v>> 4)) & 0x0f00f00fU;
  v = (v|(v>> 8)|(v>>16)) & 0x3ffU;
  return v;
  }
inline uint64_t spread_bits_3D_64 (uint64_t v)
  {
  v&=0x1fffff;
  v = (v|(v<<16)|(v<<32)) & 0x00ff0000ff0000ffU;
  v = (v|(v<< 8)) & 0xf00f00f00f00f00fU;
  v = (v|(v<< 4)) & 0x30c30c30c30c30c3U;
  v = (v|(v<< 2)) & 0x9249249249249249U;
  return v;
  }
inline uint64_t compress_bits_3D_64 (uint64_t v)
  {
  v&=0x1249249249249249U;
  v=(v|(v>> 2)) & 0x30c30c30c30c30c3U;
  v=(v|(v>> 4)) & 0xf00f00f00f00f00fU;
  v=(v|(v>> 8)) & 0x00ff0000ff0000ffU;
  v=(v|(v>>16)|(v>>32)) & 0x1fffffU;
  return v;
  }

} // unnamed namespace

uint32_t block2morton3D_32 (uint32_t v)
  {
  uint32_t v2=v&0xfffff;
  uint64_t v3=spread_bits_3D_64(v2);
  v3=(v3|(v3>>29))&0x1fffffff;
  return uint32_t(v3)|(spread_bits_3D_32(v>>20)<<2);
  }
uint32_t coord2morton3D_32 (array<uint32_t,3> xyz)
  {
  uint32_t v2=(xyz[0]&0x3ff)|((xyz[1]&0x3ff)<<10);
  uint64_t v3=spread_bits_3D_64(v2);
  v3=(v3|(v3>>29))&0x1fffffff;
  return uint32_t(v3)|(spread_bits_3D_32(xyz[2]&0x3ff)<<2);
  }
uint32_t morton2block3D_32 (uint32_t v)
  {
  return compress_bits_3D_32(v)
      | (compress_bits_3D_32(v>>1)<<10)
      | (compress_bits_3D_32(v>>2)<<20);
  }
array<uint32_t,3> morton2coord3D_32 (uint32_t v)
  {
  return {compress_bits_3D_32(v),
          compress_bits_3D_32(v>>1),
          compress_bits_3D_32(v>>2)};
  }

uint64_t block2morton3D_64 (uint64_t v)
  {
  return spread_bits_3D_64(v)
      | (spread_bits_3D_64(v>>21)<<1)
      | (spread_bits_3D_64(v>>42)<<2);
  }
uint64_t coord2morton3D_64 (array<uint64_t,3> xyz)
  {
  return spread_bits_3D_64(xyz[0])
      | (spread_bits_3D_64(xyz[1])<<1)
      | (spread_bits_3D_64(xyz[2])<<2);
  }
uint64_t morton2block3D_64 (uint64_t v)
  {
  return compress_bits_3D_64(v)
      | (compress_bits_3D_64(v>>1)<<21)
      | (compress_bits_3D_64(v>>2)<<42);
  }
array<uint64_t,3> morton2coord3D_64 (uint64_t v)
  {
  return { compress_bits_3D_64(v),
           compress_bits_3D_64(v>>1),
           compress_bits_3D_64(v>>2)};
  }

#endif

#if 0

namespace {

constexpr const array<array<uint8_t,4>,4> m2p2D_1 = {{
{ 4, 1,11, 2},{ 0,15, 5, 6},{10, 9, 3,12},{14, 7,13, 8}}};

constexpr const array<array<uint8_t,4>,4> p2m2D_1 = {{
{ 4, 1, 3,10},{ 0, 6, 7,13},{15, 9, 8, 2},{11,14,12, 5}}};

template<bool m2p> constexpr const array<array<uint8_t, 64>,4> peano2d_1to3()
  {
  array<array<uint8_t, 64>,4> res;
  for (int rot=0; rot<4; ++rot)
    for (int j=0; j<64; ++j)
      {
      uint8_t x=0;
      uint8_t v=j;
      uint8_t r=rot;
      for (int iter=0; iter<3; ++iter)
        {
        unsigned tab;
        if constexpr(m2p)
          tab = m2p2D_1[r][(v>>(2*(2-iter)))&3];
        else
          tab = p2m2D_1[r][(v>>(2*(2-iter)))&3];
        r = tab>>2;
        x = (x<<2) | (tab&3);
        }
      res[rot][j] = x+(r<<6);
      }
  return res;
  }

constexpr const array<array<uint8_t, 64>,4> m2p2D_3 = peano2d_1to3<true>();
constexpr const array<array<uint8_t, 64>,4> p2m2D_3 = peano2d_1to3<false>();

constexpr const array<array<uint8_t, 8>,24> m2p3D_1={{
{144,119, 97,110, 43, 44, 98,109},
{ 36, 35,101,106,127,152,102,105},
{ 96, 27,135, 28,161,162,182,181},
{170,169,189,190, 19,104, 20,143},
{134,137,151,112,133,138, 12, 11},
{130,141,  3,  4,129,142,120,159},
{174,173,185,186,103, 60,128, 59},
{ 52,111, 51,136,165,166,178,177},
{118,167,117, 92,153,168,154, 91},
{160,145, 83,146,175,126, 84,125},
{114, 75,113,176,157, 76,158,191},
{ 68,149,183,150, 67,122,184,121},
{ 16,139,  1,  2, 55,140, 14, 13},
{132, 63,  5,  6,131, 24, 10,  9},
{ 70,  7, 81, 32, 69,124, 82,123},
{116, 77,115, 90, 15, 78, 40, 89},
{ 38, 37, 23,108, 41, 42, 48,107},
{ 34, 33, 99, 56, 45, 46,100, 31},
{  0, 73, 39, 94,155, 74,156, 93},
{ 66,147, 85,148, 65,  8, 86, 47},
{ 72, 71,187,188, 17, 62, 18, 61},
{ 54, 25, 53, 26, 79, 64,180,179},
{172,171, 95, 80, 21, 58, 22, 57},
{ 50, 29, 49, 30,163,164, 88, 87}}};

constexpr const array<array<uint8_t, 8>,24> p2m3D_1={{
{144, 98,102, 44, 45,111,107,113},
{157,111,107, 33, 32, 98,102,124},
{ 96,164,165, 25, 27,183,182,130},
{109,169,168, 20, 22,186,187,143},
{115,137,141, 15, 14,132,128,146},
{126,132,128,  2,  3,137,141,159},
{134,186,187, 63, 61,169,168,100},
{139,183,182, 50, 48,164,165,105},
{173,156,158, 95, 91,114,112,161},
{160,145,147, 82, 86,127,125,172},
{179,114,112, 73, 77,156,158,191},
{190,127,125, 68, 64,145,147,178},
{ 16,  2,  3,137,141, 15, 14, 52},
{ 29, 15, 14,132,128,  2,  3, 57},
{ 35, 82, 86,127,125, 68, 64,  1},
{ 46, 95, 91,114,112, 73, 77, 12},
{ 54, 44, 45,111,107, 33, 32, 18},
{ 59, 33, 32, 98,102, 44, 45, 31},
{  0, 73, 77,156,158, 95, 91, 34},
{ 13, 68, 64,145,147, 82, 86, 47},
{ 72, 20, 22,186,187, 63, 61, 65},
{ 69, 25, 27,183,182, 50, 48, 76},
{ 83, 63, 61,169,168, 20, 22, 90},
{ 94, 50, 48,164,165, 25, 27, 87}}};

template<bool m2p> constexpr const array<array<uint16_t, 64>,24> peano3D_1to2()
  {
  array<array<uint16_t, 64>,24> res;
  for (int rot=0; rot<24; ++rot)
    for (int j=0; j<64; ++j)
      {
      uint16_t x=0;
      uint16_t v=j;
      uint16_t r=rot;
      for (int iter=0; iter<2; ++iter)
        {
        unsigned tab;
        if constexpr(m2p)
          tab = m2p3D_1[r][(v>>(3*(1-iter)))&7];
        else
          tab = p2m3D_1[r][(v>>(3*(1-iter)))&7];
        r = tab>>3;
        x = (x<<3) | (tab&7);
        }
      res[rot][j] = x+(r<<6);
      }
  return res;
  }

constexpr const array<array<uint16_t, 64>,24> m2p3D_2 = peano3D_1to2<true>();
constexpr const array<array<uint16_t, 64>,24> p2m3D_2 = peano3D_1to2<false>();

template<bool m2p, unsigned rbits, typename T> inline T mp2D_sub(unsigned rot,
  T res, T v)
  {
  constexpr unsigned nbits = 8*sizeof(T);
  static_assert(2*rbits<=nbits, "too many significant bits");
  // if constexpr (rbits==0) return res;
  const auto &tab1 = m2p ? m2p2D_1 : p2m2D_1;
  const auto &tab3 = m2p ? m2p2D_3 : p2m2D_3;
  if constexpr (rbits==1)
    return (res<<2)|(tab1[rot][v>>(nbits-2)]&0x3);
  if constexpr (rbits==2)
    {
    unsigned tab=tab1[rot][v>>(nbits-2)];
    return mp2D_sub<m2p, rbits-1>(tab>>2, (res<<2)|(tab&0x3), v<<2);
    }
  if constexpr (rbits==3)
    return (res<<6)|(tab3[rot][v>>(nbits-6)]&0x3fU);
  if constexpr (rbits>3)
    {
    unsigned tab=tab3[rot][v>>(nbits-6)];
    return mp2D_sub<m2p, rbits-3>(tab>>6, (res<<6)|(tab&0x3fU), v<<6);
    }
  }

template<bool m2p, unsigned rbits, typename T> inline T mp3D_sub(unsigned rot,
  T res, T v)
  {
  constexpr unsigned nbits = 8*sizeof(T);
  static_assert(3*rbits<=nbits, "too many significant bits");
  // if constexpr (rbits==0) return res;
  const auto &tab1 = m2p ? m2p3D_1 : p2m3D_1;
  const auto &tab2 = m2p ? m2p3D_2 : p2m3D_2;
  if constexpr (rbits==1)
    return (res<<3)|(tab1[rot][v>>(nbits-3)]&0x7);
  if constexpr (rbits==2)
    return (res<<6)|(tab2[rot][v>>(nbits-6)]&0x3fU);
  if constexpr (rbits>2)
    {
    unsigned tab=tab2[rot][v>>(nbits-6)];
    return mp3D_sub<m2p, rbits-2>(tab>>6, (res<<6)|(tab&0x3fU), v<<6);
    }
  }

template<bool m2p, unsigned bits> uint64_t tmp2D_64(uint64_t v)
  { return mp2D_sub<m2p, bits, uint64_t>(0, 0, v<<(64-(bits<<1))); }
template<bool m2p, unsigned bits> uint32_t tmp2D_32(uint32_t v)
  { return mp2D_sub<m2p, bits, uint32_t>(0, 0, v<<(32-(bits<<1))); }
template<bool m2p, unsigned bits> uint64_t tmp3D_64(uint64_t v)
  { return mp3D_sub<m2p, bits, uint64_t>(0, 0, v<<(3*(21-bits)+1)); }
template<bool m2p, unsigned bits> uint32_t tmp3D_32(uint32_t v)
  { return mp3D_sub<m2p, bits, uint32_t>(0, 0, v<<(3*(10-bits)+2)); }

template<bool m2p> uint32_t mp2D_32(uint32_t v, unsigned bits)
  {
  switch(bits)
    {
    case  1: return tmp2D_32<m2p,  1>(v);
    case  2: return tmp2D_32<m2p,  2>(v);
    case  3: return tmp2D_32<m2p,  3>(v);
    case  4: return tmp2D_32<m2p,  4>(v);
    case  5: return tmp2D_32<m2p,  5>(v);
    case  6: return tmp2D_32<m2p,  6>(v);
    case  7: return tmp2D_32<m2p,  7>(v);
    case  8: return tmp2D_32<m2p,  8>(v);
    case  9: return tmp2D_32<m2p,  9>(v);
    case 10: return tmp2D_32<m2p, 10>(v);
    case 11: return tmp2D_32<m2p, 11>(v);
    case 12: return tmp2D_32<m2p, 12>(v);
    case 13: return tmp2D_32<m2p, 13>(v);
    case 14: return tmp2D_32<m2p, 14>(v);
    case 15: return tmp2D_32<m2p, 15>(v);
    case 16: return tmp2D_32<m2p, 16>(v);
    default: MR_fail("bad number of requested bits");
    }
  }
template<bool m2p> uint64_t mp2D_64(uint64_t v, unsigned bits)
  {
  if (bits<=16) return uint64_t(mp2D_32<m2p>(uint32_t(v), bits));

  switch(bits)
    {
    case 17: return tmp2D_64<m2p, 17>(v);
    case 18: return tmp2D_64<m2p, 18>(v);
    case 19: return tmp2D_64<m2p, 19>(v);
    case 20: return tmp2D_64<m2p, 20>(v);
    case 21: return tmp2D_64<m2p, 21>(v);
    case 22: return tmp2D_64<m2p, 22>(v);
    case 23: return tmp2D_64<m2p, 23>(v);
    case 24: return tmp2D_64<m2p, 24>(v);
    case 25: return tmp2D_64<m2p, 25>(v);
    case 26: return tmp2D_64<m2p, 26>(v);
    case 27: return tmp2D_64<m2p, 27>(v);
    case 28: return tmp2D_64<m2p, 28>(v);
    case 29: return tmp2D_64<m2p, 29>(v);
    case 30: return tmp2D_64<m2p, 30>(v);
    case 31: return tmp2D_64<m2p, 31>(v);
    case 32: return tmp2D_64<m2p, 32>(v);
    default: MR_fail("bad number of requested bits");
    }
  }

template<bool m2p> uint32_t mp3D_32(uint32_t v, unsigned bits)
  {
  switch(bits)
    {
    case  1: return tmp3D_32<m2p,  1>(v);
    case  2: return tmp3D_32<m2p,  2>(v);
    case  3: return tmp3D_32<m2p,  3>(v);
    case  4: return tmp3D_32<m2p,  4>(v);
    case  5: return tmp3D_32<m2p,  5>(v);
    case  6: return tmp3D_32<m2p,  6>(v);
    case  7: return tmp3D_32<m2p,  7>(v);
    case  8: return tmp3D_32<m2p,  8>(v);
    case  9: return tmp3D_32<m2p,  9>(v);
    case 10: return tmp3D_32<m2p, 10>(v);
    default: MR_fail("bad number of requested bits");
    }
  }

template<bool m2p> uint64_t mp3D_64(uint64_t v, unsigned bits)
  {
  if (bits<=10) return uint64_t(mp3D_32<m2p>(uint32_t(v), bits));

  switch(bits)
    {
    case 11: return tmp3D_64<m2p, 11>(v);
    case 12: return tmp3D_64<m2p, 12>(v);
    case 13: return tmp3D_64<m2p, 13>(v);
    case 14: return tmp3D_64<m2p, 14>(v);
    case 15: return tmp3D_64<m2p, 15>(v);
    case 16: return tmp3D_64<m2p, 16>(v);
    case 17: return tmp3D_64<m2p, 17>(v);
    case 18: return tmp3D_64<m2p, 18>(v);
    case 19: return tmp3D_64<m2p, 19>(v);
    case 20: return tmp3D_64<m2p, 20>(v);
    case 21: return tmp3D_64<m2p, 21>(v);
    default: MR_fail("bad number of requested bits");
    }
  }

}  // unnamed namespace

uint32_t morton2peano3D_32(uint32_t v, unsigned bits)
  { return mp3D_32<true> (v, bits); }
uint32_t peano2morton3D_32(uint32_t v, unsigned bits)
  { return mp3D_32<false> (v, bits); }
uint64_t morton2peano3D_64(uint64_t v, unsigned bits)
  { return mp3D_64<true> (v, bits); }
uint64_t peano2morton3D_64(uint64_t v, unsigned bits)
  { return mp3D_64<false> (v, bits); }
uint32_t morton2peano2D_32(uint32_t v, unsigned bits)
  { return mp2D_32<true> (v, bits); }
uint32_t peano2morton2D_32(uint32_t v, unsigned bits)
  { return mp2D_32<false> (v, bits); }
uint64_t morton2peano2D_64(uint64_t v, unsigned bits)
  { return mp2D_64<true> (v, bits); }
uint64_t peano2morton2D_64(uint64_t v, unsigned bits)
  { return mp2D_64<false> (v, bits); }
#endif
}
