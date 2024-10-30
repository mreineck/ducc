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

/* Copyright (C) 2019-2024 Max-Planck-Society
   Author: Martin Reinecke */

#ifndef DUCC0_NUFFT_COMMON_H
#define DUCC0_NUFFT_COMMON_H

#include <algorithm>
#include "ducc0/infra/simd.h"
#include "ducc0/math/gridding_kernel.h"
#include "ducc0/fft/fft.h"

namespace ducc0 {

namespace detail_nufft {

using namespace std;
// the next line is necessary to address some sloppy name choices in AdaptiveCpp
using std::min, std::max;

// Generally we want to use SIMD types with the largest possible size, but not
// larger than 8; length-16 SIMD types (like full AVX512 float32 vectors) would
// be overkill for typical kernel supports (we don't let float32 kernels have
// a support larger than 8 anyway).
template<typename T> constexpr inline int good_simdlen
  = min<int>(8, native_simd<T>::size());

template<typename T> using mysimd = typename simd_select<T,good_simdlen<T>>::type;

/// convenience function for squaring a number
template<typename T> T sqr(T val) { return val*val; }

/// Function for quickly zeroing a 2D array with arbitrary strides.
template<typename T> void quickzero(const vmav<T,2> &arr, size_t nthreads)
  {
#if 0
  arr.fill(T(0));
#else
  MR_assert((arr.stride(0)>0) && (arr.stride(1)>0), "bad memory ordering");
  MR_assert(arr.stride(0)>=arr.stride(1), "bad memory ordering");
  size_t s0=arr.shape(0), s1=arr.shape(1);
  execParallel(s0, nthreads, [&](size_t lo, size_t hi)
    {
    if (arr.stride(1)==1)
      {
      if (size_t(arr.stride(0))==arr.shape(1))
        memset(reinterpret_cast<char *>(&arr(lo,0)), 0, sizeof(T)*s1*(hi-lo));
      else
        for (auto i=lo; i<hi; ++i)
          memset(reinterpret_cast<char *>(&arr(i,0)), 0, sizeof(T)*s1);
      }
    else
      for (auto i=lo; i<hi; ++i)
        for (size_t j=0; j<s1; ++j)
          arr(i,j) = T(0);
    });
#endif
  }

template<typename T> complex<T> hsum_cmplx(mysimd<T> vr, mysimd<T> vi)
  { return complex<T>(reduce(vr, plus<>()), reduce(vi, plus<>())); }

#if (!defined(DUCC0_NO_SIMD))
#if (!defined(__AVX512F__))
#if (defined(__AVX__))
static_assert(mysimd<float>::size()==8, "must not happen");
#if 1
template<> inline complex<float> hsum_cmplx<float>(mysimd<float> vr, mysimd<float> vi)
  {
  auto t1 = _mm256_hadd_ps(__m256(vr), __m256(vi));
  auto t2 = _mm_hadd_ps(_mm256_extractf128_ps(t1, 0), _mm256_extractf128_ps(t1, 1));
  t2 += _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,0,3,2));
  return complex<float>(t2[0], t2[1]);
  }
#else
// this version may be slightly faster, but this needs more benchmarking
template<> inline complex<float> hsum_cmplx<float>(mysimd<float> vr, mysimd<float> vi)
  {
  auto t1 = _mm256_shuffle_ps(vr, vi, _MM_SHUFFLE(0,2,0,2));
  auto t2 = _mm256_shuffle_ps(vr, vi, _MM_SHUFFLE(1,3,1,3));
  auto t3 = _mm256_add_ps(t1,t2);
  t3 = _mm256_shuffle_ps(t3, t3, _MM_SHUFFLE(3,0,2,1));
  auto t4 = _mm_add_ps(_mm256_extractf128_ps(t3, 1), _mm256_castps256_ps128(t3));
  auto t5 = _mm_add_ps(t4, _mm_movehl_ps(t4, t4));
  return complex<float>(t5[0], t5[1]);
  }
#endif
#elif defined(__SSE3__)
static_assert(mysimd<float>::size()==4, "must not happen");
template<> inline complex<float> hsum_cmplx<float>(mysimd<float> vr, mysimd<float> vi)
  {
  auto t1 = _mm_hadd_ps(__m128(vr), __m128(vi));
  t1 += _mm_shuffle_ps(t1, t1, _MM_SHUFFLE(2,3,0,1));
  return complex<float>(t1[0], t1[2]);
  }
#endif
#endif
#endif

[[gnu::always_inline]] [[gnu::hot]]
inline auto comp_indices(size_t idx, size_t nuni, size_t nbig, bool fft_order)
  {
  int icf = abs(int(nuni/2)-int(idx));
  size_t i1 = fft_order ? nuni-nuni/2+idx : idx;
  if (i1>=nuni) i1-=nuni;
  size_t i2 = nbig-nuni/2+idx;
  if (i2>=nbig) i2-=nbig;
  return make_tuple(icf, i1, i2);
  }

/*! Selects the most efficient combination of gridding kernel and oversampled
    grid size for the provided problem parameters. */
template<typename Tcalc, typename Tacc> auto findNufftParameters(double epsilon,
  double sigma_min, double sigma_max, const vector<size_t> &dims,
  size_t npoints, bool gridding, size_t nthreads)
  {
  auto vlen = gridding ? mysimd<Tacc>::size() : mysimd<Tcalc>::size();
  auto ndim = dims.size();
  auto idx = getAvailableKernels<Tcalc>(epsilon, ndim, sigma_min, sigma_max);
  double mincost = 1e300;
  constexpr double nref_fft=2048;
  constexpr double costref_fft=0.0693;
  vector<size_t> bigdims(ndim, 0);
  size_t minidx=~(size_t(0));
  for (size_t i=0; i<idx.size(); ++i)
    {
    const auto &krn(getKernel(idx[i]));
    auto supp = krn.W;
    auto nvec = (supp+vlen-1)/vlen;
    auto ofactor = krn.ofactor;
    vector<size_t> lbigdims(ndim,0);
    double gridsize=1;
    for (size_t idim=0; idim<ndim; ++idim)
      {
      lbigdims[idim] = 2*good_size_complex(size_t(dims[idim]*ofactor*0.5)+1);
      lbigdims[idim] = max<size_t>(lbigdims[idim], 16);
      gridsize *= lbigdims[idim];
      }
    double logterm = log(gridsize)/log(nref_fft*nref_fft);
    double fftcost = gridsize/(nref_fft*nref_fft)*logterm*costref_fft;
    size_t kernelpoints = nvec*vlen;
    for (size_t idim=0; idim+1<ndim; ++idim)
      kernelpoints*=supp;
    double gridcost = 2.2e-10*npoints*(kernelpoints + (ndim*nvec*(supp+3)*vlen));
    if (gridding) gridcost *= sizeof(Tacc)/sizeof(Tcalc);
    // FIXME: heuristics could be improved
    gridcost /= nthreads;  // assume perfect scaling for now
    constexpr double max_fft_scaling = 6;
    constexpr double scaling_power=2;
    auto sigmoid = [](double x, double m, double s)
      {
      auto x2 = x-1;
      auto m2 = m-1;
      return 1.+x2/pow((1.+pow(x2/m2,s)),1./s);
      };
    fftcost /= sigmoid(nthreads, max_fft_scaling, scaling_power);
    double cost = fftcost+gridcost;
    if (cost<mincost)
      {
      mincost=cost;
      bigdims=lbigdims;
      minidx = idx[i];
      }
    }
  return make_tuple(minidx, bigdims);
  }
template<typename Tcalc, typename Tacc> auto findNufftKernel(double epsilon,
  double sigma_min, double sigma_max, const vector<size_t> &dims,
  size_t npoints, bool gridding, size_t nthreads)
  {
  auto [minidx, bigdims] = findNufftParameters<Tcalc, Tacc>
    (epsilon, sigma_min, sigma_max, dims, npoints, gridding, nthreads);
  return minidx;
  }

/*! Selects the most efficient combination of gridding kernel and oversampled
    grid size for the provided Type 3 problem parameters. */
template<typename Tcalc, typename Tacc> auto findNufftParameters_type3(double epsilon,
  double sigma_min, double sigma_max, const vector<double> &hdelta_in, const vector<double> &hdelta_out,
  size_t npoints, size_t nthreads)
  {
  auto vlen = mysimd<Tacc>::size();
  auto ndim = hdelta_in.size();

vector<double> rawdim(ndim), vssafe(ndim);
for (size_t idim=0; idim<ndim; ++idim)
  {
  double Xsafe = hdelta_in[idim],
         Ssafe = hdelta_out[idim];
  if ((Xsafe==0) && (Ssafe==0))
    Xsafe = Ssafe = 1.0;
  else
    {
    if (Xsafe==0) Xsafe = 1./Ssafe;
    if (Ssafe==0) Ssafe = 1./Xsafe;
    }
  rawdim[idim] = 2*Ssafe*Xsafe/pi;
  vssafe[idim] = Ssafe;
  }


  auto idx = getAvailableKernels<Tcalc>(epsilon, ndim, sigma_min, sigma_max);
  double mincost = 1e300;
  constexpr double nref_fft=2048;
  constexpr double costref_fft=0.0693;
  vector<size_t> bigdims(ndim, 0);
  size_t minidx=~(size_t(0));
  for (size_t i=0; i<idx.size(); ++i)
    {
    const auto &krn(getKernel(idx[i]));
    auto supp = krn.W;
    auto nvec = (supp+vlen-1)/vlen;
    auto ofactor = krn.ofactor;
    vector<size_t> lbigdims(ndim,0);
    double gridsize=1;
    for (size_t idim=0; idim<ndim; ++idim)
      {
// new type3 stuff here
double tmp = rawdim[idim]*ofactor+supp+1;
      lbigdims[idim] = 2*good_size_complex(size_t(tmp*0.5)+1);
lbigdims[idim] = max<size_t>(lbigdims[idim], 16);
lbigdims[idim] = max<size_t>(lbigdims[idim], 2*supp);  // FINUFFT does this ... why exactly?
      gridsize *= lbigdims[idim];
      }
    double logterm = log(gridsize)/log(nref_fft*nref_fft);
    double fftcost = gridsize/(nref_fft*nref_fft)*logterm*costref_fft;
    size_t kernelpoints = nvec*vlen;
    for (size_t idim=0; idim+1<ndim; ++idim)
      kernelpoints*=supp;
    double gridcost = 2.2e-10*npoints*(kernelpoints + (ndim*nvec*(supp+3)*vlen));
    gridcost *= sizeof(Tacc)/sizeof(Tcalc);
    // FIXME: heuristics could be improved
    gridcost /= nthreads;  // assume perfect scaling for now
    constexpr double max_fft_scaling = 6;
    constexpr double scaling_power=2;
    auto sigmoid = [](double x, double m, double s)
      {
      auto x2 = x-1;
      auto m2 = m-1;
      return 1.+x2/pow((1.+pow(x2/m2,s)),1./s);
      };
    fftcost /= sigmoid(nthreads, max_fft_scaling, scaling_power);
    double cost = fftcost+gridcost;
    if (cost<mincost)
      {
      mincost=cost;
      bigdims=lbigdims;
      minidx = idx[i];
      }
    }
  return make_tuple(minidx, bigdims, vssafe);
  }
//#define NEW_DUMP
template<typename Tacc, size_t ndim> constexpr inline int log2tile_=-1;
template<> constexpr inline int log2tile_<long double, 1> = 9;
template<> constexpr inline int log2tile_<double, 1> = 9;
template<> constexpr inline int log2tile_<float , 1> = 9;
template<> constexpr inline int log2tile_<long double, 2> = 4;
template<> constexpr inline int log2tile_<double, 2> = 4;
template<> constexpr inline int log2tile_<float , 2> = 5;
#ifdef NEW_DUMP
template<> constexpr inline int log2tile_<double, 3> = 5;
template<> constexpr inline int log2tile_<float , 3> = 5;
#else
template<> constexpr inline int log2tile_<double, 3> = 4;
template<> constexpr inline int log2tile_<float , 3> = 4;
#endif
template<> constexpr inline int log2tile_<long double, 3> = 4;

template<size_t ndim> constexpr inline size_t max_ntile=-1;
template<> constexpr inline size_t max_ntile<1> = (~uint32_t(0))-10;
template<> constexpr inline size_t max_ntile<2> = (uint32_t(1<<16))-10;
template<> constexpr inline size_t max_ntile<3> = (uint32_t(1<<10))-10;

}} // close namespaces

#endif
