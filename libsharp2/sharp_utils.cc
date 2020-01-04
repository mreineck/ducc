/*
 *  This file is part of libsharp2.
 *
 *  libsharp2 is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  libsharp2 is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with libsharp2; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/* libsharp2 is being developed at the Max-Planck-Institut fuer Astrophysik */

/*
 *  Convenience functions
 *
 *  Copyright (C) 2008-2019 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#include <stdio.h>
#include "libsharp2/sharp_utils.h"
#include "mr_util/error_handling.h"

#pragma GCC visibility push(hidden)

/* This function tries to avoid allocations with a total size close to a high
   power of two (called the "critical stride" here), by adding a few more bytes
   if necessary. This lowers the probability that two arrays differ by a multiple
   of the critical stride in their starting address, which in turn lowers the
   risk of cache line contention. */
static size_t manipsize(size_t sz)
  {
  const size_t critical_stride=4096, cacheline=64, overhead=32;
  if (sz < (critical_stride/2)) return sz;
  if (((sz+overhead)%critical_stride)>(2*cacheline)) return sz;
  return sz+2*cacheline;
  }
#pragma GCC visibility pop

#ifdef __SSE__
#include <xmmintrin.h>
#pragma GCC visibility push(hidden)
void *sharp_malloc_ (size_t sz)
  {
  void *res;
  if (sz==0) return NULL;
  res = _mm_malloc(manipsize(sz),32);
  MR_assert(res,"_mm_malloc() failed");
  return res;
  }
void sharp_free_ (void *ptr)
  { if ((ptr)!=NULL) _mm_free(ptr); }
#pragma GCC visibility pop
#else
#pragma GCC visibility push(hidden)
void *sharp_malloc_ (size_t sz)
  {
  void *res;
  if (sz==0) return NULL;
  res = malloc(manipsize(sz));
  MR_assert(res,"malloc() failed");
  return res;
  }
void sharp_free_ (void *ptr)
  { if ((ptr)!=NULL) free(ptr); }
#pragma GCC visibility pop
#endif
