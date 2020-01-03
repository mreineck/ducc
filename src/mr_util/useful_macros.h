#ifndef MRUTIL_USEFUL_MACROS_H
#define MRUTIL_USEFUL_MACROS_H

#if defined(__GNUC__)
#define MRUTIL_NOINLINE __attribute__((noinline))
#define MRUTIL_RESTRICT __restrict__
//#define MRUTIL_ALIGNED(align) __attribute__ ((aligned(align)))
#elif defined(_MSC_VER)
#define MRUTIL_NOINLINE __declspec(noinline)
#define MRUTIL_RESTRICT __restrict
#else
#define MRUTIL_NOINLINE
#define MRUTIL_RESTRICT
#endif


#endif
