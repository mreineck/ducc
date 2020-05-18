#ifndef MRUTIL_USEFUL_MACROS_H
#define MRUTIL_USEFUL_MACROS_H

#if defined(__GNUC__)
#define MRUTIL_NOINLINE __attribute__((noinline))
#define MRUTIL_RESTRICT __restrict__
#define MRUTIL_ALIGNED(align) __attribute__ ((aligned(align)))
#define MRUTIL_PREFETCH_R(addr) __builtin_prefetch(addr);
#define MRUTIL_PREFETCH_W(addr) __builtin_prefetch(addr,1);
#elif defined(_MSC_VER)
#define MRUTIL_NOINLINE __declspec(noinline)
#define MRUTIL_RESTRICT __restrict
#define MRUTIL_ALIGNED(align)
#define MRUTIL_PREFETCH_R(addr)
#define MRUTIL_PREFETCH_W(addr)
#else
#define MRUTIL_NOINLINE
#define MRUTIL_RESTRICT
#define MRUTIL_ALIGNED(align)
#define MRUTIL_PREFETCH_R(addr)
#define MRUTIL_PREFETCH_W(addr)
#endif

#endif
