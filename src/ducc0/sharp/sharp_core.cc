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

/*! \file sharp_core.cc
 *  Spherical transform library
 *
 *  Copyright (C) 2019-2020 Max-Planck-Society
 *  \author Martin Reinecke
 */

#define ARCH default
#define GENERIC_ARCH
#include "ducc0/sharp/sharp_core_inc.cc"
#undef GENERIC_ARCH
#undef ARCH

namespace ducc0 {

namespace detail_sharp {

using t_inner_loop = void (*) (sharp_job &job, const vector<bool> &ispair,
  const vector<double> &cth_, const vector<double> &sth_, size_t llim, size_t ulim,
  sharp_Ylmgen &gen, size_t mi, const vector<size_t> &mlim);
using t_veclen = size_t (*) (void);
using t_max_nvec = size_t (*) (size_t spin);
using t_architecture = const char *(*) (void);

static t_inner_loop inner_loop_ = nullptr;
static t_veclen veclen_ = nullptr;
static t_max_nvec max_nvec_ = nullptr;
static t_architecture architecture_ = nullptr;

#ifdef MULTIARCH

#if (defined(___AVX512F__) || defined(__FMA4__) || defined(__FMA__) || \
     defined(__AVX2__) || defined(__AVX__))
#error MULTIARCH specified but platform-specific flags detected
#endif

#define DECL(arch) \
static int XCONCATX2(have,arch)(void) \
  { \
  static int res=-1; \
  if (res<0) \
    { \
    __builtin_cpu_init(); \
    res = __builtin_cpu_supports(#arch); \
    } \
  return res; \
  } \
\
void XCONCATX2(inner_loop,arch) (sharp_job &job, const vector<bool> &ispair, \
  const vector<double> &cth_, const vector<double> &sth_, size_t llim, size_t ulim, \
  sharp_Ylmgen &gen, size_t mi, const vector<size_t> &mlim); \
size_t XCONCATX2(sharp_veclen,arch) (void); \
size_t XCONCATX2(sharp_max_nvec,arch) (size_t spin); \
const char *XCONCATX2(sharp_architecture,arch) (void);

#if (!defined(__APPLE__))
DECL(avx512f)
#endif
DECL(fma4)
DECL(fma)
DECL(avx2)
DECL(avx)

#endif

static void assign_funcs(void)
  {
#ifdef MULTIARCH
#define DECL2(arch) \
  if (XCONCATX2(have,arch)()) \
    { \
    inner_loop_ = XCONCATX2(inner_loop,arch); \
    veclen_ = XCONCATX2(sharp_veclen,arch); \
    max_nvec_ = XCONCATX2(sharp_max_nvec,arch); \
    architecture_ = XCONCATX2(sharp_architecture,arch); \
    return; \
    }
#if (!defined(__APPLE__))
DECL2(avx512f)
#endif
DECL2(fma4)
DECL2(fma)
DECL2(avx2)
DECL2(avx)
#endif
  inner_loop_ = inner_loop_default;
  veclen_ = sharp_veclen_default;
  max_nvec_ = sharp_max_nvec_default;
  architecture_ = sharp_architecture_default;
  }

#pragma GCC visibility push(hidden)

void inner_loop (sharp_job &job, const vector<bool> &ispair,
  const vector<double> &cth, const vector<double> &sth,
  size_t llim, size_t ulim, sharp_Ylmgen &gen, size_t mi,
  const vector<size_t> &mlim)
  {
  if (!inner_loop_) assign_funcs();
  inner_loop_(job, ispair, cth, sth, llim, ulim, gen, mi, mlim);
  }

size_t sharp_max_nvec(size_t spin)
  {
  if (!max_nvec_) assign_funcs();
  return max_nvec_(spin);
  }

#pragma GCC visibility pop

size_t sharp_veclen(void)
  {
  if (!veclen_) assign_funcs();
  return veclen_();
  }
const char *sharp_architecture(void)
  {
  if (!architecture_) assign_funcs();
  return architecture_();
  }

}}
