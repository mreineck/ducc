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

/* Copyright (C) 2019-2020 Max-Planck-Society
   Author: Martin Reinecke */

#ifndef DUCC0_SIMD_H
#define DUCC0_SIMD_H

// only enable SIMD support for gcc>=5.0 and clang>=5.0
#ifndef DUCC0_NO_SIMD
#define DUCC0_NO_SIMD
#if defined(__clang__)
// AppleClang has their own version numbering
#ifdef __apple_build_version__
#  if (__clang_major__ > 9) || (__clang_major__ == 9 && __clang_minor__ >= 1)
#     undef DUCC0_NO_SIMD
#  endif
#elif __clang_major__ >= 5
#  undef DUCC0_NO_SIMD
#endif
#elif defined(__GNUC__)
#if __GNUC__>=5
#undef DUCC0_NO_SIMD
#endif
#endif
#endif

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#ifndef DUCC0_NO_SIMD
#include <x86intrin.h>
#endif

namespace ducc0 {

namespace detail_simd {

template<typename T> constexpr inline bool vectorizable = false;
template<> constexpr inline bool vectorizable<float> = true;
template<> constexpr inline bool vectorizable<double> = true;
template<> constexpr inline bool vectorizable<int8_t> = true;
template<> constexpr inline bool vectorizable<uint8_t> = true;
template<> constexpr inline bool vectorizable<int16_t> = true;
template<> constexpr inline bool vectorizable<uint16_t> = true;
template<> constexpr inline bool vectorizable<int32_t> = true;
template<> constexpr inline bool vectorizable<uint32_t> = true;
template<> constexpr inline bool vectorizable<int64_t> = true;
template<> constexpr inline bool vectorizable<uint64_t> = true;

template<typename T, size_t len> constexpr inline bool simd_exists = false;

template<typename T, size_t reglen> constexpr size_t vlen
  = vectorizable<T> ? reglen/sizeof(T) : 1;

template<typename T, size_t len> class helper_;
template<typename T, size_t len> struct vmask_
  {
  private:
    using hlp = helper_<T, len>;
    using Tm = typename hlp::Tm;
    Tm v;

  public:
#if defined(_MSC_VER)
    vmask_() {}
    vmask_(const vmask_ &other) : v(other.v) {}
#else
    vmask_() = default;
    vmask_(const vmask_ &other) = default;
#endif
    vmask_(Tm v_): v(v_) {}
    operator Tm() const  { return v; }
    size_t bits() const { return hlp::maskbits(v); }
    vmask_ operator& (const vmask_ &other) const { return hlp::mask_and(v,other.v); }
  };

template<typename T, size_t len> class vtp
  {
  private:
    using hlp = helper_<T, len>;

  public:
    using Ts = T;
    using Tv = typename hlp::Tv;
    using Tm = vmask_<T, len>;
    static constexpr size_t size() { return len; }

  private:
    Tv v;

  public:
#if defined(_MSC_VER)
    vtp() {}
    vtp(const vtp &other): v(other.v) {}
#else
    vtp() = default;
    vtp(const vtp &other) = default;
#endif
    vtp(T other): vtp(hlp::from_scalar(other)) {}
    vtp(const Tv &other) : v(other) {}
    vtp &operator=(const T &other) { v=hlp::from_scalar(other); return *this; }
    operator Tv() const { return v; }

    static vtp loadu(const T *ptr) { return vtp(hlp::loadu(ptr)); }
    void storeu(T *ptr) const { hlp::storeu(ptr, v); }

    vtp operator-() const { return vtp(-v); }
    vtp operator+(vtp other) const { return vtp(v+other.v); }
    vtp operator-(vtp other) const { return vtp(v-other.v); }
    vtp operator*(vtp other) const { return vtp(v*other.v); }
    vtp operator/(vtp other) const { return vtp(v/other.v); }
    vtp &operator+=(vtp other) { v+=other.v; return *this; }
    vtp &operator-=(vtp other) { v-=other.v; return *this; }
    vtp &operator*=(vtp other) { v*=other.v; return *this; }
    vtp &operator/=(vtp other) { v/=other.v; return *this; }
    vtp abs() const { return hlp::abs(v); }
    template<typename Func> vtp apply(Func func) const
      {
      vtp res;
      for (size_t i=0; i<len; ++i)
        res[i] = func(v[i]);
      return res;
      }
    inline vtp sqrt() const
      { return hlp::sqrt(v); }
    vtp max(const vtp &other) const
      { return hlp::max(v, other.v); }
    Tm operator>(const vtp &other) const
      { return hlp::gt(v, other.v); }
    Tm operator>=(const vtp &other) const
      { return hlp::ge(v, other.v); }
    Tm operator<(const vtp &other) const
      { return hlp::lt(v, other.v); }
    Tm operator!=(const vtp &other) const
      { return hlp::ne(v, other.v); }

    class reference
      {
      private:
        vtp &v;
        size_t i;
      public:
        reference (vtp<T, len> &v_, size_t i_)
          : v(v_), i(i_) {}
        reference &operator= (T other)
          { v.v[i] = other; return *this; }
        reference &operator*= (T other)
          { v.v[i] *= other; return *this; }
        operator T() const { return v.v[i]; }
      };

    void Set(size_t i, T val) { v[i] = val; }
    reference operator[](size_t i) { return reference(*this, i); }
    T operator[](size_t i) const { return v[i]; }

    class where_expr
      {
      private:
        vtp &v;
        Tm m;

      public:
        where_expr (Tm m_, vtp &v_)
          : v(v_), m(m_) {}
        where_expr &operator*= (const vtp &other)
          { v=hlp::blend(m, v.v*other.v, v.v); return *this; }
        where_expr &operator+= (const vtp &other)
          { v=hlp::blend(m, v.v+other.v, v.v); return *this; }
        where_expr &operator-= (const vtp &other)
          { v=hlp::blend(m, v.v-other.v, v.v); return *this; }
      };
  };
template<typename T, size_t len> inline vtp<T, len> abs(vtp<T, len> v) { return v.abs(); }
template<typename T, size_t len> typename vtp<T, len>::where_expr where(typename vtp<T, len>::Tm m, vtp<T, len> &v)
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
template<typename T, size_t len> inline bool any_of(const vmask_<T, len> &mask)
  { return mask.bits()!=0; }
template<typename T, size_t len> inline bool none_of(const vmask_<T, len> &mask)
  { return mask.bits()==0; }
template<typename T, size_t len> inline bool all_of(const vmask_<T, len> &mask)
  { return mask.bits()==(size_t(1)<<len)-1; }
template<typename Op, typename T, size_t len> T reduce(const vtp<T, len> &v, Op op)
  {
  T res=v[0];
  for (size_t i=1; i<len; ++i)
    res = op(res, v[i]);
  return res;
  }
template<typename Op, typename T, size_t len> T accumulate(const vtp<T, len> &v, Op op)
  { return reduce(v, op); }
template<typename T> class pseudoscalar
  {
  private:
    T v;

  public:
#if defined(_MSC_VER)
    pseudoscalar() {}
    pseudoscalar(const pseudoscalar &other) :v(other.v) {}
#else
    pseudoscalar() = default;
    pseudoscalar(const pseudoscalar &other) = default;
#endif
    pseudoscalar(T v_):v(v_) {}
    pseudoscalar operator-() const { return pseudoscalar(-v); }
    pseudoscalar operator+(pseudoscalar other) const { return pseudoscalar(v+other.v); }
    pseudoscalar operator-(pseudoscalar other) const { return pseudoscalar(v-other.v); }
    pseudoscalar operator*(pseudoscalar other) const { return pseudoscalar(v*other.v); }
    pseudoscalar operator/(pseudoscalar other) const { return pseudoscalar(v/other.v); }
    pseudoscalar &operator+=(pseudoscalar other) { v+=other.v; return *this; }
    pseudoscalar &operator-=(pseudoscalar other) { v-=other.v; return *this; }
    pseudoscalar &operator*=(pseudoscalar other) { v*=other.v; return *this; }
    pseudoscalar &operator/=(pseudoscalar other) { v/=other.v; return *this; }

    pseudoscalar abs() const { return std::abs(v); }
    inline pseudoscalar sqrt() const { return std::sqrt(v); }
    pseudoscalar max(const pseudoscalar &other) const
      { return std::max(v, other.v); }

    bool operator>(const pseudoscalar &other) const
      { return v>other.v; }
    bool operator>=(const pseudoscalar &other) const
      { return v>=other.v; }
    bool operator<(const pseudoscalar &other) const
      { return v<other.v; }
    bool operator!=(const pseudoscalar &other) const
      { return v!=other.v; }
    const T &operator[] (size_t /*i*/) const { return v; }
    T &operator[](size_t /*i*/) { return v; }
  };
template<typename T> class helper_<T,1>
  {
  private:
    static constexpr size_t len = 1;
  public:
    using Tv = pseudoscalar<T>;
    using Tm = bool;

    static Tv loadu(const T *ptr) { return *ptr; }
    static void storeu(T *ptr, Tv v) { *ptr = v[0]; }

    static Tv from_scalar(T v) { return v; }
    static Tv abs(Tv v) { return v.abs(); }
    static Tv max(Tv v1, Tv v2) { return v1.max(v2); }
    static Tv blend(Tm m, Tv v1, Tv v2) { return m ? v1 : v2; }
    static Tv sqrt(Tv v) { return v.sqrt(); }
    static Tm gt (Tv v1, Tv v2) { return v1>v2; }
    static Tm ge (Tv v1, Tv v2) { return v1>=v2; }
    static Tm lt (Tv v1, Tv v2) { return v1<v2; }
    static Tm ne (Tv v1, Tv v2) { return v1!=v2; }
    static Tm mask_and (Tm v1, Tm v2) { return v1&&v2; }
    static size_t maskbits(Tm v) { return v; }
  };

#ifndef DUCC0_NO_SIMD

#if defined(__AVX512F__)
template<> constexpr inline bool simd_exists<double,8> = true;
template<> class helper_<double,8>
  {
  private:
    using T = double;
    static constexpr size_t len = 8;
  public:
    using Tv = __m512d;
    using Tm = __mmask8;

    static Tv loadu(const T *ptr) { return _mm512_loadu_pd(ptr); }
    static void storeu(T *ptr, Tv v) { _mm512_storeu_pd(ptr, v); }

    static Tv from_scalar(T v) { return _mm512_set1_pd(v); }
    static Tv abs(Tv v) { return __m512d(_mm512_andnot_epi64(__m512i(_mm512_set1_pd(-0.)),__m512i(v))); }
    static Tv max(Tv v1, Tv v2) { return _mm512_max_pd(v1, v2); }
    static Tv blend(Tm m, Tv v1, Tv v2) { return _mm512_mask_blend_pd(m, v2, v1); }
    static Tv sqrt(Tv v) { return _mm512_sqrt_pd(v); }
    static Tm gt (Tv v1, Tv v2) { return _mm512_cmp_pd_mask(v1,v2,_CMP_GT_OQ); }
    static Tm ge (Tv v1, Tv v2) { return _mm512_cmp_pd_mask(v1,v2,_CMP_GE_OQ); }
    static Tm lt (Tv v1, Tv v2) { return _mm512_cmp_pd_mask(v1,v2,_CMP_LT_OQ); }
    static Tm ne (Tv v1, Tv v2) { return _mm512_cmp_pd_mask(v1,v2,_CMP_NEQ_OQ); }
    static Tm mask_and (Tm v1, Tm v2) { return v1&v2; }
    static size_t maskbits(Tm v) { return v; }
  };
template<> constexpr inline bool simd_exists<float,16> = true;
template<> class helper_<float,16>
  {
  private:
    using T = float;
    static constexpr size_t len = 16;
  public:
    using Tv = __m512;
    using Tm = __mmask16;

    static Tv loadu(const T *ptr) { return _mm512_loadu_ps(ptr); }
    static void storeu(T *ptr, Tv v) { _mm512_storeu_ps(ptr, v); }

    static Tv from_scalar(T v) { return _mm512_set1_ps(v); }
    static Tv abs(Tv v) { return __m512(_mm512_andnot_epi32(__m512i(_mm512_set1_ps(-0.)),__m512i(v))); }
    static Tv max(Tv v1, Tv v2) { return _mm512_max_ps(v1, v2); }
    static Tv blend(Tm m, Tv v1, Tv v2) { return _mm512_mask_blend_ps(m, v2, v1); }
    static Tv sqrt(Tv v) { return _mm512_sqrt_ps(v); }
    static Tm gt (Tv v1, Tv v2) { return _mm512_cmp_ps_mask(v1,v2,_CMP_GT_OQ); }
    static Tm ge (Tv v1, Tv v2) { return _mm512_cmp_ps_mask(v1,v2,_CMP_GE_OQ); }
    static Tm lt (Tv v1, Tv v2) { return _mm512_cmp_ps_mask(v1,v2,_CMP_LT_OQ); }
    static Tm ne (Tv v1, Tv v2) { return _mm512_cmp_ps_mask(v1,v2,_CMP_NEQ_OQ); }
    static Tm mask_and (Tm v1, Tm v2) { return v1&v2; }
    static size_t maskbits(Tm v) { return v; }
  };
#endif
#if defined(__AVX__)
template<> constexpr inline bool simd_exists<double,4> = true;
template<> class helper_<double,4>
  {
  private:
    using T = double;
    static constexpr size_t len = 4;
  public:
    using Tv = __m256d;
    using Tm = __m256d;

    static Tv loadu(const T *ptr) { return _mm256_loadu_pd(ptr); }
    static void storeu(T *ptr, Tv v) { _mm256_storeu_pd(ptr, v); }

    static Tv from_scalar(T v) { return _mm256_set1_pd(v); }
    static Tv abs(Tv v) { return _mm256_andnot_pd(_mm256_set1_pd(-0.),v); }
    static Tv max(Tv v1, Tv v2) { return _mm256_max_pd(v1, v2); }
    static Tv blend(Tm m, Tv v1, Tv v2) { return _mm256_blendv_pd(v2, v1, m); }
    static Tv sqrt(Tv v) { return _mm256_sqrt_pd(v); }
    static Tm gt (Tv v1, Tv v2) { return _mm256_cmp_pd(v1,v2,_CMP_GT_OQ); }
    static Tm ge (Tv v1, Tv v2) { return _mm256_cmp_pd(v1,v2,_CMP_GE_OQ); }
    static Tm lt (Tv v1, Tv v2) { return _mm256_cmp_pd(v1,v2,_CMP_LT_OQ); }
    static Tm ne (Tv v1, Tv v2) { return _mm256_cmp_pd(v1,v2,_CMP_NEQ_OQ); }
    static Tm mask_and (Tm v1, Tm v2) { return _mm256_and_pd(v1,v2); }
    static size_t maskbits(Tm v) { return size_t(_mm256_movemask_pd(v)); }
  };
template<> constexpr inline bool simd_exists<float,8> = true;
template<> class helper_<float,8>
  {
  private:
    using T = float;
    static constexpr size_t len = 8;
  public:
    using Tv = __m256;
    using Tm = __m256;

    static Tv loadu(const T *ptr) { return _mm256_loadu_ps(ptr); }
    static void storeu(T *ptr, Tv v) { _mm256_storeu_ps(ptr, v); }

    static Tv from_scalar(T v) { return _mm256_set1_ps(v); }
    static Tv abs(Tv v) { return _mm256_andnot_ps(_mm256_set1_ps(-0.),v); }
    static Tv max(Tv v1, Tv v2) { return _mm256_max_ps(v1, v2); }
    static Tv blend(Tm m, Tv v1, Tv v2) { return _mm256_blendv_ps(v2, v1, m); }
    static Tv sqrt(Tv v) { return _mm256_sqrt_ps(v); }
    static Tm gt (Tv v1, Tv v2) { return _mm256_cmp_ps(v1,v2,_CMP_GT_OQ); }
    static Tm ge (Tv v1, Tv v2) { return _mm256_cmp_ps(v1,v2,_CMP_GE_OQ); }
    static Tm lt (Tv v1, Tv v2) { return _mm256_cmp_ps(v1,v2,_CMP_LT_OQ); }
    static Tm ne (Tv v1, Tv v2) { return _mm256_cmp_ps(v1,v2,_CMP_NEQ_OQ); }
    static Tm mask_and (Tm v1, Tm v2) { return _mm256_and_ps(v1,v2); }
    static size_t maskbits(Tm v) { return size_t(_mm256_movemask_ps(v)); }
  };
#endif
#if defined(__SSE2__)
template<> constexpr inline bool simd_exists<double,2> = true;
template<> class helper_<double,2>
  {
  private:
    using T = double;
    static constexpr size_t len = 2;
  public:
    using Tv = __m128d;
    using Tm = __m128d;

    static Tv loadu(const T *ptr) { return _mm_loadu_pd(ptr); }
    static void storeu(T *ptr, Tv v) { _mm_storeu_pd(ptr, v); }

    static Tv from_scalar(T v) { return _mm_set1_pd(v); }
    static Tv abs(Tv v) { return _mm_andnot_pd(_mm_set1_pd(-0.),v); }
    static Tv max(Tv v1, Tv v2) { return _mm_max_pd(v1, v2); }
    static Tv blend(Tm m, Tv v1, Tv v2)
      {
#if defined(__SSE4_1__)
      return _mm_blendv_pd(v2,v1,m);
#else
      return _mm_or_pd(_mm_and_pd(m,v1),_mm_andnot_pd(m,v2));
#endif
      }
    static Tv sqrt(Tv v) { return _mm_sqrt_pd(v); }
    static Tm gt (Tv v1, Tv v2) { return _mm_cmpgt_pd(v1,v2); }
    static Tm ge (Tv v1, Tv v2) { return _mm_cmpge_pd(v1,v2); }
    static Tm lt (Tv v1, Tv v2) { return _mm_cmplt_pd(v1,v2); }
    static Tm ne (Tv v1, Tv v2) { return _mm_cmpneq_pd(v1,v2); }
    static Tm mask_and (Tm v1, Tm v2) { return _mm_and_pd(v1,v2); }
    static size_t maskbits(Tm v) { return size_t(_mm_movemask_pd(v)); }
  };
template<> constexpr inline bool simd_exists<float,4> = true;
template<> class helper_<float,4>
  {
  private:
    using T = float;
    static constexpr size_t len = 4;
  public:
    using Tv = __m128;
    using Tm = __m128;

    static Tv loadu(const T *ptr) { return _mm_loadu_ps(ptr); }
    static void storeu(T *ptr, Tv v) { _mm_storeu_ps(ptr, v); }

    static Tv from_scalar(T v) { return _mm_set1_ps(v); }
    static Tv abs(Tv v) { return _mm_andnot_ps(_mm_set1_ps(-0.),v); }
    static Tv max(Tv v1, Tv v2) { return _mm_max_ps(v1, v2); }
    static Tv blend(Tm m, Tv v1, Tv v2)
      {
#if defined(__SSE4_1__)
      return _mm_blendv_ps(v2,v1,m);
#else
      return _mm_or_ps(_mm_and_ps(m,v1),_mm_andnot_ps(m,v2));
#endif
      }
    static Tv sqrt(Tv v) { return _mm_sqrt_ps(v); }
    static Tm gt (Tv v1, Tv v2) { return _mm_cmpgt_ps(v1,v2); }
    static Tm ge (Tv v1, Tv v2) { return _mm_cmpge_ps(v1,v2); }
    static Tm lt (Tv v1, Tv v2) { return _mm_cmplt_ps(v1,v2); }
    static Tm ne (Tv v1, Tv v2) { return _mm_cmpneq_ps(v1,v2); }
    static Tm mask_and (Tm v1, Tm v2) { return _mm_and_ps(v1,v2); }
    static size_t maskbits(Tm v) { return size_t(_mm_movemask_ps(v)); }
  };
#endif

#if defined(__AVX512F__)
template<typename T> using native_simd = vtp<T,vlen<T,64>>;
#elif defined(__AVX__)
template<typename T> using native_simd = vtp<T,vlen<T,32>>;
#elif defined(__SSE2__)
template<typename T> using native_simd = vtp<T,vlen<T,16>>;
#else
template<typename T> using native_simd = vtp<T,1>;
#endif

#else // DUCC0_NO_SIMD is defined
template<typename T> using native_simd = vtp<T,1>;
#endif
}

using detail_simd::native_simd;
template<typename T, size_t len> using simd = detail_simd::vtp<T, len>;
using detail_simd::simd_exists;
using detail_simd::reduce;
using detail_simd::max;
using detail_simd::abs;
using detail_simd::sqrt;
using detail_simd::any_of;
using detail_simd::none_of;
using detail_simd::all_of;

// since we are explicitly introducing a few names that are also available in
// std::, we need to import them from std::as well, otherwise name resolution
// can fail in certain circumstances.

using std::abs;
using std::sqrt;
using std::max;

}

#endif
