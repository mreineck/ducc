/*
 *  This file is part of the MR utility library.
 *
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

/* Copyright (C) 2019 Max-Planck-Society
   Author: Martin Reinecke */

#ifndef MRUTIL_SIMD_H
#define MRUTIL_SIMD_H

// only enable SIMD support for gcc>=5.0 and clang>=5.0
#ifndef MRUTIL_NO_SIMD
#define MRUTIL_NO_SIMD
#if defined(__INTEL_COMPILER)
// do nothing. This is necessary because this compiler also sets __GNUC__.
#elif defined(__clang__)
#if __clang_major__>=5
#undef MRUTIL_NO_SIMD
#endif
#elif defined(__GNUC__)
#if __GNUC__>=5
#undef MRUTIL_NO_SIMD
#endif
#endif
#endif

#ifndef MRUTIL_NO_SIMD
#include <cstdlib>
#include <cmath>

namespace mr {

namespace detail_simd {

using namespace std;

#if (defined(__AVX512F__))
constexpr size_t vbytes = 64;
#elif (defined(__AVX__))
constexpr size_t vbytes = 32;
#elif (defined(__SSE2__))
constexpr size_t vbytes = 16;
#elif (defined(__VSX__))
constexpr size_t vbytes = 16;
#endif

template<typename T, size_t len=vbytes/sizeof(T)> class vtp
  {
  protected:
    using Tv __attribute__ ((vector_size (len*sizeof(T)))) = T;
    static_assert((len>0) && ((len&(len-1))==0), "bad vector length");
    Tv v;

    void from_scalar(T other)
//      { for (size_t i=0; i<len; ++i) v[i]=other; }
      { v=v*0+other; }

  public:
    static constexpr size_t vlen=len;
    vtp () {}
    vtp(T other)
      { from_scalar(other); }
    vtp(const Tv &other)
      : v(other) {}
    vtp(const vtp &other) = default;
    vtp &operator=(T other)
      { from_scalar(other); return *this; }
    vtp operator-() const { return vtp(-v); }
    vtp operator+(vtp other) const { return vtp(v+other.v); }
    vtp operator-(vtp other) const { return vtp(v-other.v); }
    vtp operator*(vtp other) const { return vtp(v*other.v); }
    vtp operator/(vtp other) const { return vtp(v/other.v); }
    vtp &operator+=(vtp other) { v+=other.v; return *this; }
    vtp &operator-=(vtp other) { v-=other.v; return *this; }
    vtp &operator*=(vtp other) { v*=other.v; return *this; }
    vtp &operator/=(vtp other) { v/=other.v; return *this; }
    inline vtp exp() const
      {
      vtp res;
      for (size_t i=0; i<len; ++i) res.v[i] = std::exp(v[i]);
      return res;
      }
    vtp sqrt() const
      {
      vtp res;
      for (size_t i=0; i<len; ++i) res.v[i] = std::sqrt(v[i]);
      return res;
      }

    template<typename I> void Set (I i, T val) { v[i]=val; }
    template<typename I> T operator[](I i) const { return v[i]; }
  };

}

using detail_simd::vtp;

}

#endif

#endif
