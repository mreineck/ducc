/*
 *  This file is part of libcxxsupport.
 *
 *  libcxxsupport is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  libcxxsupport is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with libcxxsupport; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/*
 *  libcxxsupport is being developed at the Max-Planck-Institut fuer Astrophysik
 *  and financially supported by the Deutsches Zentrum fuer Luft- und Raumfahrt
 *  (DLR).
 */

/*! \file math_utils.h
 *  Various convenience mathematical functions.
 *
 *  Copyright (C) 2002-2026 Max-Planck-Society
 *  \author Martin Reinecke
 */

#ifndef DUCC0_MATH_UTILS_H
#define DUCC0_MATH_UTILS_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <bit>

#include "ducc0/infra/useful_macros.h"

namespace DUCC0_NAMESPACE {

namespace detail_math_utils {

using namespace std;

/*! Returns \e true if | \a a-b | <= \a epsilon * | \a b |, else \e false. */
template<typename F> inline bool approx (F a, F b, F epsilon=F(1e-5L))
  { return abs(a-b) <= (epsilon*abs(b)); }

/*! Returns \e true if | \a a-b | <= \a epsilon, else \e false. */
template<typename F> inline bool abs_approx (F a, F b, F epsilon=F(1e-5L))
  { return abs(a-b) <= epsilon; }

/*! Returns the largest integer which is smaller than (or equal to) \a arg. */
template<typename I, typename F> inline I ifloor (F arg)
  { return I(floor(arg)); }

/*! Returns the remainder of the division \a v1/v2.
    The result is non-negative.
    \a v1 can be positive or negative; \a v2 must be positive. */
template<typename F> inline F fmodulo (F v1, F v2)
  {
  if (v1>=0)
    return (v1<v2) ? v1 : fmod(v1,v2);
  auto tmp=fmod(v1,v2)+v2;
  return (tmp==v2) ? F(0) : tmp;
  }

/*! Returns the integer \a n, which fulfills \a n*n<=arg<(n+1)*(n+1). */
/* Should produce correct results for nonnegative arguments up to 2**64-1. */
template<typename I> inline uint32_t isqrt (I arg)
  {
  if constexpr (sizeof(I)<=4)
    return uint32_t (sqrt(arg+0.5));

  uint64_t uarg = uint64_t(arg);
  uint64_t res = uint64_t(sqrt(double(uarg)+0.5));
  if (uarg<(uint64_t(1)<<50)) return uint32_t(res);
  if (res >= 0x100000000ULL)
    res = 0xFFFFFFFFULL;
  if (res*res>uarg)
    --res;
  else if ((res<0xFFFFFFFFULL) && ((res+1)*(res+1)<=uarg))
    ++res;
  return uint32_t(res);
  }

/*! Returns the largest integer \a n that fulfills \a 2^n<=arg. */
template<typename I> inline unsigned int ilog2 (I arg)
  {
  if (arg<2) return 0;
  using U = typename make_unsigned<I>::type;
  return bit_width<U>(arg)-1;
  }

/*! Returns \a atan2(y,x) if \a x!=0 or \a y!=0; else returns 0. */
inline double safe_atan2 (double y, double x)
  { return ((x==0.) && (y==0.)) ? 0.0 : atan2(y,x); }

}

using detail_math_utils::approx;
using detail_math_utils::abs_approx;
using detail_math_utils::ifloor;
using detail_math_utils::fmodulo;
using detail_math_utils::isqrt;
using detail_math_utils::ilog2;
using detail_math_utils::safe_atan2;

}

#endif
