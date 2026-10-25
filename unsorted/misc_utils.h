#ifndef MRBASE_MISC_UTILS_H
#define MRBASE_MISC_UTILS_H

#include <type_traits>

/* Unsorted small useful functions needed in many situations */

/*! Returns the smallest multiple of \a sz that is >= \a n. */
template<typename I, typename I2> inline I roundup_to_multiple_of (I n, I2 sz)
  {
  static_assert(std::is_integral<I>::value, "integer type needed");
  static_assert(std::is_integral<I2>::value, "integer type needed");
  return ((n+sz-1)/sz)*sz;
  }

/*! Returns the largest integer \a n that fulfills \a 2^n<=arg. */
template<typename I> inline int ilog2 (I arg)
  {
  static_assert(std::is_integral<I>::value, "integer type needed");
#ifdef __GNUC__
  if (arg==0) return 0;
  if (sizeof(I)<=sizeof(int))
    return 8*sizeof(int)-1-__builtin_clz(arg);
  if (sizeof(I)==sizeof(long))
    return 8*sizeof(long)-1-__builtin_clzl(arg);
  if (sizeof(I)==sizeof(long long))
    return 8*sizeof(long long)-1-__builtin_clzll(arg);
#endif
  int res=0;
  while (arg > 0xFFFF) { res+=16; arg>>=16; }
  if (arg > 0x00FF) { res|=8; arg>>=8; }
  if (arg > 0x000F) { res|=4; arg>>=4; }
  if (arg > 0x0003) { res|=2; arg>>=2; }
  if (arg > 0x0001) { res|=1; }
  return res;
  }

/*! Returns the smallest power of 2 that is >= \a n. */
template<typename I> inline I roundup_to_power_of_2 (I n)
  {
  static_assert(std::is_integral<I>::value, "integer type needed");
  if (n<=1) return 1;
  return I(1)<<(ilog2(n-1)+1);
  }

#endif
