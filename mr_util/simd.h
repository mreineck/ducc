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

#define MRUTIL_HOMEGROWN_SIMD

#ifndef MRUTIL_HOMEGROWN_SIMD

#include <experimental/simd>

#else

// only enable SIMD support for gcc>=5.0 and clang>=5.0
#ifndef MRUTIL_NO_SIMD
#define MRUTIL_NO_SIMD
#if defined(__INTEL_COMPILER)
// do nothing. This is necessary because this compiler also sets __GNUC__.
#elif defined(__clang__)
#ifdef __APPLE__
#  if (__clang_major__ > 9) || (__clang_major__ == 9 && __clang_minor__ >= 3)
#     undef MRUTIL_NO_SIMD
#  endif
#elif __clang_major__ >= 5
#  undef MRUTIL_NO_SIMD
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
#include <x86intrin.h>

namespace std {

namespace experimental {

namespace detail_simd {

#if (defined(__AVX512F__))
constexpr size_t vbytes = 64;
#elif (defined(__AVX__))
constexpr size_t vbytes = 32;
#elif (defined(__SSE2__))
constexpr size_t vbytes = 16;
#elif (defined(__VSX__))
constexpr size_t vbytes = 16;
#endif

template<size_t ibytes> struct Itype {};

template<>struct Itype<4>
  { using type = int32_t; };

template<>struct Itype<8>
  { using type = int64_t; };


template<typename T, size_t len=vbytes/sizeof(T)> class vtp
  {
  public:
    using Tv [[gnu::vector_size (len*sizeof(T))]] = T;
    static_assert((len>0) && ((len&(len-1))==0), "bad vector length");
    Tv v;

    inline void from_scalar(const T &other)
      { v=v*0+other; }

  public:
    using mask_type = vtp<typename Itype<sizeof(T)>::type,len>;
    static constexpr size_t size() { return len; }
    vtp () = default;
    vtp(T other) { from_scalar(other); }
    vtp(const Tv &other)
      : v(other) {}
    vtp(const vtp &other) = default;
    vtp &operator=(const T &other) { from_scalar(other); return *this; }
    vtp operator-() const { return vtp(-v); }
    vtp operator+(vtp other) const { return vtp(v+other.v); }
    vtp operator-(vtp other) const { return vtp(v-other.v); }
    vtp operator*(vtp other) const { return vtp(v*other.v); }
    vtp operator/(vtp other) const { return vtp(v/other.v); }
    vtp operator&(vtp other) const { return vtp(v&other.v); }
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
    inline vtp sqrt() const
      {
      vtp res;
      for (size_t i=0; i<len; ++i) res.v[i] = std::sqrt(v[i]);
      return res;
      }
    vtp max(const vtp &other) const
      { return vtp(v>other.v?v:other.v); }
    mask_type operator>(const vtp &other) const
      { return v>other.v; }
    mask_type operator>=(const vtp &other) const
      { return v>=other.v; }
    mask_type operator<(const vtp &other) const
      { return v<other.v; }
    mask_type operator!=(const vtp &other) const
      { return v!=other.v; }

    class reference
      {
      private:
        vtp &v;
        size_t i;
      public:
        reference (vtp &v_, size_t i_)
          : v(v_), i(i_) {}
        reference &operator= (T other)
          { v.v[i] = other; return *this; }
        reference &operator*= (T other)
          { v.v[i] *= other; return *this; }
        operator T() const { return v.v[i]; }
      };

    class where_expr
      {
      private:
        vtp &v;
        mask_type m;
      public:
        where_expr (mask_type m_, vtp &v_)
          : v(v_), m(m_) {}
        where_expr &operator*= (const vtp &other)
          { v = vtp(m.v ? v.v*other.v : v.v); return *this; }
        where_expr &operator+= (const vtp &other)
          { v = vtp(m.v ? v.v+other.v : v.v); return *this; }
        where_expr &operator-= (const vtp &other)
          { v = vtp(m.v ? v.v-other.v : v.v); return *this; }
      };
    reference operator[](size_t i) { return reference(*this, i); }
    T operator[](size_t i) const { return v[i]; }
    Tv __data() const { return v; }
  };
template<typename T, size_t len> inline typename vtp<T, len>::Tv __data(const vtp<T, len> &v) { return v.__data(); }
template<typename T, size_t len> inline vtp<T, len> abs(vtp<T, len> v)
  {
  vtp<T, len> res;
  for (size_t i=0; i<len; ++i) res.v[i] = std::abs(v.v[i]);
  return res;
  }
template<typename T, size_t len> typename vtp<T, len>::where_expr where(typename vtp<T, len>::mask_type m, vtp<T, len> &v)
  { return typename vtp<T, len>::where_expr(m, v); }
template<typename T0, typename T, size_t len> vtp<T, len> operator*(T0 a, vtp<T, len> b)
  { return b*a; }
template<typename T, size_t len> vtp<T, len> operator+(T a, vtp<T, len> b)
  { return b+a; }
template<typename T, size_t len> vtp<T, len> operator-(T a, vtp<T, len> b)
  { return vtp<T, len>(a) - b; }
template<typename T, size_t len> vtp<T, len> max(vtp<T, len> a, vtp<T, len> b)
  { return a.max(b); }
template<typename T, size_t len> vtp<T, len> sqrt(vtp<T, len> v)
  { return v.sqrt(); }

template<typename T> using native_simd=vtp<T>;
template<typename Op, typename T, size_t len> T reduce(const vtp<T, len> &v, Op op)
  {
  T res=v[0];
  for (size_t i=1; i<len; ++i)
    res = op(res, v[i]);
  return res;
  }

template<typename T, size_t len> inline bool any_of(const vtp<T, len> &mask)
  {
  bool res=false;
  for (size_t i=0; i<mask.size(); ++i)
    res = res || (mask[i]!=0);
  return res;
  }
template<typename T, size_t len> inline bool none_of(const vtp<T, len> & mask)
  {
  return !any_of(mask);
  }
template<typename T, size_t len> inline bool all_of(const vtp<T, len> & mask)
  {
  bool res=true;
  for (size_t i=0; i<mask.size(); ++i)
    res = res && (mask[i]!=0);
  return res;
  }
template<typename T> inline native_simd<T> vload(T v) { return native_simd<T>(v); }
#if defined(__AVX512F__)
#elif defined(__AVX__)
template<> inline void vtp<double, 4>::from_scalar(const double &other)
  { v=_mm256_set1_pd(other); }
template<> inline vtp<double, 4> vtp<double, 4>::sqrt() const
  { return vtp<double, 4>(_mm256_sqrt_pd(v)); }
template<> inline vtp<double, 4> vtp<double, 4>::operator-() const
  { return vtp<double, 4>(_mm256_xor_pd(_mm256_set1_pd(-0.),v)); }
template<> inline vtp<double, 4> abs(vtp<double, 4> v)
  { return vtp<double, 4>(_mm256_andnot_pd(_mm256_set1_pd(-0.),v.v)); }
template<> inline bool any_of(const vtp<int64_t, 4> &mask)
  { return _mm256_movemask_pd(__m256d(mask.v))!=0; }
template<> inline bool none_of(const vtp<int64_t, 4> &mask)
  { return _mm256_movemask_pd(__m256d(mask.v))==0; }
template<> inline bool all_of(const vtp<int64_t, 4> &mask)
  { return _mm256_movemask_pd(__m256d(mask.v))==15; }
#endif
}

using detail_simd::native_simd;
using detail_simd::reduce;
using detail_simd::max;
using detail_simd::abs;
using detail_simd::sqrt;
using detail_simd::any_of;
using detail_simd::none_of;
using detail_simd::all_of;
using detail_simd::vload;
}
}

#endif

#endif

#endif
