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

/* Copyright (C) 2019-2022 Max-Planck-Society
   Author: Martin Reinecke */

#ifndef DUCC0_NUFFT_H
#define DUCC0_NUFFT_H

#include <cstring>
#include <complex>
#include <cstdint>
#include <functional>
#include <map>
#include <type_traits>
#include <utility>
#include <mutex>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <array>
#include <atomic>
#include <memory>
#if ((!defined(DUCC0_NO_SIMD)) && (defined(__AVX__)||defined(__SSE3__)))
#include <x86intrin.h>
#endif

#include "ducc0/infra/error_handling.h"
#include "ducc0/math/constants.h"
#include "ducc0/fft/fft.h"
#include "ducc0/infra/threading.h"
#include "ducc0/infra/misc_utils.h"
#include "ducc0/infra/useful_macros.h"
#include "ducc0/infra/mav.h"
#include "ducc0/infra/simd.h"
#include "ducc0/infra/timers.h"
#include "ducc0/infra/bucket_sort.h"
#include "ducc0/math/gridding_kernel.h"

namespace ducc0 {

namespace detail_nufft {

using namespace std;
// the next line is necessary to address some sloppy name choices in hipSYCL
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
template<typename T> void quickzero(vmav<T,2> &arr, size_t nthreads)
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
#if (defined(__AVX__))
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
template<> inline complex<float> hsum_cmplx<float>(mysimd<float> vr, mysimd<float> vi)
  {
  auto t1 = _mm_hadd_ps(__m128(vr), __m128(vi));
  t1 += _mm_shuffle_ps(t1, t1, _MM_SHUFFLE(2,3,0,1));
  return complex<float>(t1[0], t1[2]);
  }
#endif
#endif

template<size_t ndim> void checkShape
  (const array<size_t, ndim> &shp1, const array<size_t, ndim> &shp2)
  { MR_assert(shp1==shp2, "shape mismatch"); }

/*! Selects the most efficient combination of gridding kernel and oversampled
    grid size for the provided problem parameters. */
template<typename Tcalc, typename Tacc> auto findNufftParameters(double epsilon,
  double sigma_min, double sigma_max, const vector<size_t> &dims,
  size_t npoints, bool gridding, size_t vlen, size_t nthreads)
  {
  auto ndim = dims.size();
  auto idx = getAvailableKernels<Tcalc>(epsilon, ndim, sigma_min, sigma_max);
  double mincost = 1e300;
  constexpr double nref_fft=2048;
  constexpr double costref_fft=0.0693;
  vector<size_t> bigdims(ndim, 0);
  size_t minidx=KernelDB.size();
  for (size_t i=0; i<idx.size(); ++i)
    {
    const auto &krn(KernelDB[idx[i]]);
    auto supp = krn.W;
    auto nvec = (supp+vlen-1)/vlen;
    auto ofactor = krn.ofactor;
    vector<size_t> lbigdims(ndim,0);
    double gridsize=1;
    for (size_t idim=0; idim<ndim; ++idim)
      {
      lbigdims[idim] = 2*good_size_complex(size_t(dims[idim]*ofactor*0.5)+1);
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

//
// Start of real NUFFT functionality
//

/*! Helper class for carrying out 1D nonuniform FFTs of types 1 and 2.
    Tcalc: the floating-point type in which all kernel-related calculations
           are performed
    Tacc:  the floating-point type used for the grid on which data is
           accumulated in nu2u transforms. Can usually be the same as Tcalc,
           but may be chosen to be more accurate in specific situations.
    Tpoints: the floating-point type used for stoing the values at the
           non-uniform points
    Tgrid: the floating-point type used for storing the values on the uniform
           grid.
    Tcoord: the floating-point type used for storing the coordinates of the
           non-uniform points.
 */
template<typename Tcalc, typename Tacc, typename Tpoints, typename Tgrid, typename Tcoord> class Nufft1d
  {
  private:
    // the base-2 logarithm of the linear dimension of a computational tile.
    constexpr static int log2tile=9;

    // if true, perform nonuniform-to-uniform transform, else uniform-to-nonuniform
    bool gridding;
    // if true, use negative exponent in the FFT, else positive
    bool forward;
    TimerHierarchy timers;
    // reference to nonuniform input data. In u2nu case, points to a 0-sized array.
    const cmav<complex<Tpoints>,1> &points_in;
    // reference to nonuniform output data. In nu2u case, points to a 0-sized array.
    vmav<complex<Tpoints>,1> &points_out;
    // reference to uniform input data. In nu2u case, points to a 0-sized array.
    const cmav<complex<Tgrid>,1> &uniform_in;
    // reference to uniform output data. In u2nu case, points to a 0-sized array.
    vmav<complex<Tgrid>,1> &uniform_out;
    // uniform grid dimensions
    size_t nxuni;
    // requested epsilon value for this transform.
    double epsilon;
    // number of threads to use for this transform.
    size_t nthreads;
    // 0: no output, 1: some diagnostic console output
    size_t verbosity;

    // 1./<periodicity of coordinates>
    double coordfct;

    // if true, start with zero mode
    // if false, start with most negative mode
    bool fft_order;

    // reference to coordinates of the non-uniform points. Shape is (npoints, ndim).
    const cmav<Tcoord,2> &coords;

    // holds the indices of the nonuniform points in the order in which they
    // should be processed
    quick_array<uint32_t> coord_idx;

    // oversampled grid dimensions
    size_t nu;

    shared_ptr<PolynomialKernel> krn;

    size_t supp, nsafe;
    double ushift;
    int maxiu0;

    static_assert(sizeof(Tcalc)<=sizeof(Tacc), "bad type combination");
    static_assert(sizeof(Tpoints)<=sizeof(Tcalc), "bad type combination");
    static_assert(sizeof(Tgrid)<=sizeof(Tcalc), "bad type combination");

    /*! Compute minimum index in the oversampled grid touched by the kernel
        around coordinate \a u. */
    [[gnu::always_inline]] void getpix(double u_in, double &u, int &iu0) const
      {
      // do range reduction in long double when Tcoord is double,
      // to avoid inaccuracies with very large grids
      using Tbig = typename conditional<is_same<Tcoord,double>::value, long double, double>::type;
      u_in *= coordfct;
      auto tmpu = Tbig(u_in-floor(u_in))*nu;
      iu0 = min(int(tmpu+ushift)-int(nu), maxiu0);
      u = double(tmpu-iu0);
      }

    /*! Compute index of the tile into which \a coord falls. */
    [[gnu::always_inline]] uint32_t get_utile(double u_in)
      {
      double udum;
      int iu0;
      getpix(u_in, udum, iu0);
      iu0 = (iu0+nsafe)>>log2tile;
      return uint32_t(iu0);
      }

    template<size_t supp> class HelperNu2u
      {
      public:
        static constexpr size_t vlen = mysimd<Tacc>::size();
        static constexpr size_t nvec = (supp+vlen-1)/vlen;

      private:
        static constexpr int nsafe = (supp+1)/2;
        static constexpr int su = 2*nsafe+(1<<log2tile);
        static constexpr int suvec = su+vlen-1;
        static constexpr double xsupp=2./supp;
        const Nufft1d *parent;
        TemplateKernel<supp, mysimd<Tacc>> tkrn;
        vmav<complex<Tcalc>,1> &grid;
        int iu0; // start index of the current nonuniform point
        int bu0; // start index of the current buffer

        vmav<Tacc,1> bufr, bufi;
        Tacc *px0r, *px0i;
        mutex &mylock;

        // add the acumulated local tile to the global oversampled grid
        DUCC0_NOINLINE void dump()
          {
          int inu = int(parent->nu);
          if (bu0<-nsafe) return; // nothing written into buffer yet

          int idxu = (bu0+inu)%inu;
          {
          lock_guard<mutex> lock(mylock);
          for (int iu=0; iu<su; ++iu)
            {
            grid(idxu) += complex<Tcalc>(Tcalc(bufr(iu)), Tcalc(bufi(iu)));
            bufr(iu) = bufi(iu) = 0;
            if (++idxu>=inu) idxu=0;
            }
          }
          }

      public:
        Tacc * DUCC0_RESTRICT p0r, * DUCC0_RESTRICT p0i;
        union kbuf {
          Tacc scalar[nvec*vlen];
          mysimd<Tacc> simd[nvec];
#if defined(_MSC_VER)
          kbuf() {}
#endif
          };
        kbuf buf;

        HelperNu2u(const Nufft1d *parent_, vmav<complex<Tcalc>,1> &grid_,
          mutex &mylock_)
          : parent(parent_), tkrn(*parent->krn), grid(grid_),
            iu0(-1000000),
            bu0(-1000000),
            bufr({size_t(suvec)}),
            bufi({size_t(suvec)}),
            px0r(bufr.data()), px0i(bufi.data()),
            mylock(mylock_)
          { checkShape(grid.shape(), {parent->nu}); }
        ~HelperNu2u() { dump(); }

        [[gnu::always_inline]] [[gnu::hot]] void prep(double u_in)
          {
          double ufrac;
          auto iu0old = iu0;
          parent->getpix(u_in, ufrac, iu0);
          auto x0 = -ufrac*2+(supp-1);
          tkrn.eval1(Tacc(x0), &buf.simd[0]);
          if (iu0==iu0old) return;
          if ((iu0<bu0) || (iu0+int(supp)>bu0+su))
            {
            dump();
            bu0=((((iu0+nsafe)>>log2tile)<<log2tile))-nsafe;
            }
          auto ofs = iu0-bu0;
          p0r = px0r+ofs;
          p0i = px0i+ofs;
          }
      };

    template<size_t supp> class HelperU2nu
      {
      public:
        static constexpr size_t vlen = mysimd<Tcalc>::size();
        static constexpr size_t nvec = (supp+vlen-1)/vlen;

      private:
        static constexpr int nsafe = (supp+1)/2;
        static constexpr int su = 2*nsafe+(1<<log2tile);
        static constexpr int suvec = su+vlen-1;
        static constexpr double xsupp=2./supp;
        const Nufft1d *parent;

        TemplateKernel<supp, mysimd<Tcalc>> tkrn;
        const cmav<complex<Tcalc>,1> &grid;
        int iu0; // start index of the current nonuniform point
        int bu0; // start index of the current buffer

        vmav<Tcalc,1> bufr, bufi;
        const Tcalc *px0r, *px0i;

        // load a tile from the global oversampled grid into local buffer
        DUCC0_NOINLINE void load()
          {
          int inu = int(parent->nu);
          int idxu = (bu0+inu)%inu;
          for (int iu=0; iu<su; ++iu)
            {
            bufr(iu) = grid(idxu).real();
            bufi(iu) = grid(idxu).imag();
            if (++idxu>=inu) idxu=0;
            }
          }

      public:
        const Tcalc * DUCC0_RESTRICT p0r, * DUCC0_RESTRICT p0i;
        union kbuf {
          Tcalc scalar[nvec*vlen];
          mysimd<Tcalc> simd[nvec];
#if defined(_MSC_VER)
          kbuf() {}
#endif
          };
        kbuf buf;

        HelperU2nu(const Nufft1d *parent_, const cmav<complex<Tcalc>,1> &grid_)
          : parent(parent_), tkrn(*parent->krn), grid(grid_),
            iu0(-1000000),
            bu0(-1000000),
            bufr({size_t(suvec)}),
            bufi({size_t(suvec)}),
            px0r(bufr.data()), px0i(bufi.data())
          { checkShape(grid.shape(), {parent->nu}); }

        [[gnu::always_inline]] [[gnu::hot]] void prep(double u_in)
          {
          double ufrac;
          auto iu0old = iu0;
          parent->getpix(u_in, ufrac, iu0);
          auto x0 = -ufrac*2+(supp-1);
          tkrn.eval1(Tcalc(x0), &buf.simd[0]);
          if (iu0==iu0old) return;
          if ((iu0<bu0) || (iu0+int(supp)>bu0+su))
            {
            bu0=((((iu0+nsafe)>>log2tile)<<log2tile))-nsafe;
            load();
            }
          auto ofs = iu0-bu0;
          p0r = px0r+ofs;
          p0i = px0i+ofs;
          }
      };

    template<size_t SUPP> [[gnu::hot]] void spreading_helper
      (size_t supp, vmav<complex<Tcalc>,1> &grid)
      {
      if constexpr (SUPP>=8)
        if (supp<=SUPP/2) return spreading_helper<SUPP/2>(supp, grid);
      if constexpr (SUPP>4)
        if (supp<SUPP) return spreading_helper<SUPP-1>(supp, grid);
      MR_assert(supp==SUPP, "requested support out of range");

      mutex mylock;

      execDynamic(coord_idx.size(), nthreads, 1000, [&](Scheduler &sched)
        {
        constexpr size_t vlen=mysimd<Tcalc>::size();
        constexpr size_t NVEC((SUPP+vlen-1)/vlen);
        HelperNu2u<SUPP> hlp(this, grid, mylock);
        const auto * DUCC0_RESTRICT ku = hlp.buf.simd;

        constexpr size_t lookahead=10;
        while (auto rng=sched.getNext()) for(auto ix=rng.lo; ix<rng.hi; ++ix)
          {
          if (ix+lookahead<coord_idx.size())
            {
            auto nextidx = coord_idx[ix+lookahead];
            DUCC0_PREFETCH_R(&points_in(nextidx));
            DUCC0_PREFETCH_R(&coords(nextidx,0));
            }
          size_t row = coord_idx[ix];
          hlp.prep(coords(row,0));
          auto v(points_in(row));

          Tacc vr(v.real()), vi(v.imag());
          for (size_t cu=0; cu<NVEC; ++cu)
            {
            auto * DUCC0_RESTRICT pxr = hlp.p0r+cu*hlp.vlen;
            auto * DUCC0_RESTRICT pxi = hlp.p0i+cu*hlp.vlen;
            auto tr = mysimd<Tacc>(pxr,element_aligned_tag());
            tr += vr*ku[cu];
            tr.copy_to(pxr,element_aligned_tag());
            auto ti = mysimd<Tacc>(pxi, element_aligned_tag());
            ti += vi*ku[cu];
            ti.copy_to(pxi,element_aligned_tag());
            }
          }
        });
      }

    template<size_t SUPP> [[gnu::hot]] void interpolation_helper
      (size_t supp, const cmav<complex<Tcalc>,1> &grid)
      {
      if constexpr (SUPP>=8)
        if (supp<=SUPP/2) return interpolation_helper<SUPP/2>(supp, grid);
      if constexpr (SUPP>4)
        if (supp<SUPP) return interpolation_helper<SUPP-1>(supp, grid);
      MR_assert(supp==SUPP, "requested support out of range");

      execDynamic(coord_idx.size(), nthreads, 1000, [&](Scheduler &sched)
        {
        constexpr size_t vlen=mysimd<Tcalc>::size();
        constexpr size_t NVEC((SUPP+vlen-1)/vlen);
        HelperU2nu<SUPP> hlp(this, grid);
        const auto * DUCC0_RESTRICT ku = hlp.buf.simd;

        constexpr size_t lookahead=10;
        while (auto rng=sched.getNext()) for(auto ix=rng.lo; ix<rng.hi; ++ix)
          {
          if (ix+lookahead<coord_idx.size())
            {
            auto nextidx = coord_idx[ix+lookahead];
            DUCC0_PREFETCH_W(&points_out(nextidx));
            DUCC0_PREFETCH_R(&coords(nextidx,0));
            }
          size_t row = coord_idx[ix];
          hlp.prep(coords(row,0));
          mysimd<Tcalc> rr=0, ri=0;
          for (size_t cu=0; cu<NVEC; ++cu)
            {
            const auto * DUCC0_RESTRICT pxr = hlp.p0r + cu*hlp.vlen;
            const auto * DUCC0_RESTRICT pxi = hlp.p0i + cu*hlp.vlen;
            rr += ku[cu]*mysimd<Tcalc>(pxr,element_aligned_tag());
            ri += ku[cu]*mysimd<Tcalc>(pxi,element_aligned_tag());
            }
          points_out(row) = hsum_cmplx<Tcalc>(rr,ri);
          }
        });
      }

    void report()
      {
      if (verbosity==0) return;
      cout << (gridding ? "Nonuniform to uniform:" : "Uniform to nonuniform:") << endl
           << "  nthreads=" << nthreads << ", "
           << "grid=(" << nxuni << "), "
           << "oversampled grid=(" << nu;
      cout << "), supp=" << supp
           << ", eps=" << epsilon
           << endl;
      cout << "  npoints=" << coords.shape(0) << endl;
      cout << "  memory overhead: "
           << coords.shape(0)*sizeof(uint32_t)/double(1<<30) << "GB (index) + "
           << nu*sizeof(complex<Tcalc>)/double(1<<30) << "GB (oversampled grid)" << endl;
      }

    void nonuni2uni()
      {
      timers.push("allocating grid");
      auto grid = vmav<complex<Tcalc>,1>::build_noncritical({nu});
      timers.poppush("spreading");
      constexpr size_t maxsupp = is_same<Tacc, double>::value ? 16 : 8;
      spreading_helper<maxsupp>(supp, grid);
      timers.poppush("FFT");
      vfmav<complex<Tcalc>> fgrid(grid);
      c2c(fgrid, fgrid, {0}, forward, Tcalc(1), nthreads);
      timers.poppush("grid correction");
      checkShape(uniform_out.shape(), {nxuni});
      auto cfu = krn->corfunc(nxuni/2+1, 1./nu, nthreads);
      execParallel(nxuni, nthreads, [&](size_t lo, size_t hi)
        {
        for (auto i=lo; i<hi; ++i)
          {
          int icfu = abs(int(nxuni/2)-int(i));
          size_t iin = nu-nxuni/2+i;
          if (iin>=nu) iin-=nu;
          size_t iout = fft_order ? nxuni-nxuni/2+i : i;
          if (iout>=nxuni) iout-=nxuni;
          uniform_out(iout) = complex<Tgrid>(grid(iin)*Tgrid(cfu[icfu]));
          }
        });
      timers.pop();
      }

    void uni2nonuni()
      {
      timers.push("allocating grid");
      auto grid = vmav<complex<Tcalc>,1>::build_noncritical({nu});
      timers.poppush("zeroing grid");
      mav_apply([](complex<Tcalc> &v){v=complex<Tcalc>(0);},nthreads,grid);
      timers.poppush("grid correction");
      checkShape(uniform_in.shape(), {nxuni});
      auto cfu = krn->corfunc(nxuni/2+1, 1./nu, nthreads);
      execParallel(nxuni, nthreads, [&](size_t lo, size_t hi)
        {
        for (auto i=lo; i<hi; ++i)
          {
          int icfu = abs(int(nxuni/2)-int(i));
          size_t iin = fft_order ? nxuni-nxuni/2+i : i;
          if (iin>=nxuni) iin-=nxuni;
          size_t iout = nu-nxuni/2+i;
          if (iout>=nu) iout-=nu;
          grid(iout) = uniform_in(iin)*Tcalc(cfu[icfu]);
          }
        });
      timers.poppush("FFT");
      vfmav<complex<Tcalc>> fgrid(grid);
      c2c(fgrid, fgrid, {0}, forward, Tcalc(1), nthreads);
      timers.poppush("interpolation");
      constexpr size_t maxsupp = is_same<Tcalc, double>::value ? 16 : 8;
      interpolation_helper<maxsupp>(supp, grid);
      timers.pop();
      }

  public:
    Nufft1d(const cmav<Tcoord,2> &coords_,
           const cmav<complex<Tpoints>,1> &points_in_, vmav<complex<Tpoints>,1> &points_out_,
           const cmav<complex<Tgrid>,1> &uniform_in_, vmav<complex<Tgrid>,1> &uniform_out_,
           double epsilon_, bool forward_,
           size_t nthreads_, size_t verbosity_,
           double sigma_min,
           double sigma_max,
           double periodicity,
           bool fft_order_)
      : gridding(points_out_.size()==0),
        forward(forward_),
        timers(gridding ? "gridding" : "degridding"),
        points_in(points_in_), points_out(points_out_),
        uniform_in(uniform_in_), uniform_out(uniform_out_),
        nxuni(gridding ? uniform_out.shape(0) : uniform_in.shape(0)),
        epsilon(epsilon_),
        nthreads((nthreads_==0) ? get_default_nthreads() : nthreads_),
        verbosity(verbosity_),
        coordfct(1./periodicity),
        fft_order(fft_order_),
        coords(coords_)
      {
      MR_assert(coords.shape(0)<=(~uint32_t(0)), "too many rows in the MS");
      checkShape(points_in.shape(), {coords.shape(0)});
      if (coords.shape(0)==0)
        {
        if (gridding) mav_apply([](complex<Tgrid> &v){v=complex<Tgrid>(0);}, nthreads, uniform_out);
        return;
        }

      timers.push("parameter calculation");
      auto [kidx, dims] = findNufftParameters<Tcalc,Tacc>(epsilon, sigma_min, sigma_max,
        {nxuni}, coords.shape(0), gridding,
        gridding ? mysimd<Tacc>::size() : mysimd<Tcalc>::size(), nthreads);
      nu = dims[0];
      timers.pop();

      MR_assert((nu>>log2tile)<=(~uint32_t(0)), "nu too large");
      krn = selectKernel(kidx);
      supp = krn->support();
      nsafe = (supp+1)/2;
      ushift = supp*(-0.5)+1+nu;
      maxiu0 = (nu+nsafe)-supp;
      MR_assert(nu>=2*nsafe, "nu too small");
      MR_assert((nu&1)==0, "nu must be even");
      MR_assert(epsilon>0, "epsilon must be positive");

      timers.push("building index");
      size_t nrow=coords.shape(0);
      size_t ntiles_u = (nu>>log2tile) + 3;
      coord_idx.resize(nrow);
      quick_array<uint32_t> key(nrow);
      execParallel(nrow, nthreads, [&](size_t lo, size_t hi)
        {
        for (size_t i=lo; i<hi; ++i)
          key[i] = get_utile(coords(i,0));
        });
      bucket_sort2(key, coord_idx, ntiles_u, nthreads);
      timers.pop();

      report();
      gridding ? nonuni2uni() : uni2nonuni();

      if (verbosity>0)
        timers.report(cout);
      }
  };

template<typename Tcalc, typename Tacc, typename Tpoints, typename Tgrid, typename Tcoord> class Nufft2d
  {
  private:
    constexpr static int log2tile=is_same<Tacc,float>::value ? 5 : 4;
    bool gridding;
    bool forward;
    TimerHierarchy timers;
    const cmav<complex<Tpoints>,1> &points_in;
    vmav<complex<Tpoints>,1> &points_out;
    const cmav<complex<Tgrid>,2> &uniform_in;
    vmav<complex<Tgrid>,2> &uniform_out;
    size_t nxuni, nyuni;
    double epsilon;
    size_t nthreads;
    size_t verbosity;
    double coordfct;
    bool fft_order;

    const cmav<Tcoord,2> &coords;

    quick_array<uint32_t> coord_idx;

    size_t nu, nv;

    shared_ptr<PolynomialKernel> krn;

    size_t supp, nsafe;
    double ushift, vshift;
    int maxiu0, maxiv0;

    static_assert(sizeof(Tcalc)<=sizeof(Tacc), "bad type combination");
    static_assert(sizeof(Tpoints)<=sizeof(Tcalc), "bad type combination");
    static_assert(sizeof(Tgrid)<=sizeof(Tcalc), "bad type combination");

    [[gnu::always_inline]] void getpix(double u_in, double v_in, double &u, double &v, int &iu0, int &iv0) const
      {
      // probably no need for long double here, since grid dimensions will never be that large
      using Tbig = double;
        //typename conditional<is_same<Tcoord,double>::value, long double, double>::type;
      u_in *= coordfct;
      auto tmpu = Tbig(u_in-floor(u_in))*nu;
      iu0 = min(int(tmpu+ushift)-int(nu), maxiu0);
      u = double(tmpu-iu0);
      v_in *= coordfct;
      auto tmpv = Tbig(v_in-floor(v_in))*nv;
      iv0 = min(int(tmpv+vshift)-int(nv), maxiv0);
      v = double(tmpv-iv0);
      }

    [[gnu::always_inline]] auto get_uvtile(double u_in, double v_in)
      {
      double udum, vdum;
      int iu0, iv0;
      getpix(u_in, v_in, udum, vdum, iu0, iv0);
      iu0 = (iu0+nsafe)>>log2tile;
      iv0 = (iv0+nsafe)>>log2tile;
      return make_tuple(uint32_t(iu0), uint32_t(iv0));
      }

    template<size_t supp> class HelperNu2u
      {
      public:
        static constexpr size_t vlen = mysimd<Tacc>::size();
        static constexpr size_t nvec = (supp+vlen-1)/vlen;

      private:
        static constexpr int nsafe = (supp+1)/2;
        static constexpr int su = supp+(1<<log2tile);
        static constexpr int sv = supp+(1<<log2tile);
        static constexpr double xsupp=2./supp;
        const Nufft2d *parent;
        TemplateKernel<supp, mysimd<Tacc>> tkrn;
        vmav<complex<Tcalc>,2> &grid;
        int iu0, iv0; // start index of the current nonuniform point
        int bu0, bv0; // start index of the current buffer

        vmav<complex<Tacc>,2> gbuf;
        complex<Tacc> *px0;
        vector<mutex> &locks;

        DUCC0_NOINLINE void dump()
          {
          int inu = int(parent->nu);
          int inv = int(parent->nv);
          if (bu0<-nsafe) return; // nothing written into buffer yet

          int idxu = (bu0+inu)%inu;
          int idxv0 = (bv0+inv)%inv;
          for (int iu=0; iu<su; ++iu)
            {
            int idxv = idxv0;
            {
            lock_guard<mutex> lock(locks[idxu]);
            for (int iv=0; iv<sv; ++iv)
              {
              grid(idxu,idxv) += complex<Tcalc>(gbuf(iu,iv));
              gbuf(iu,iv) = 0;
              if (++idxv>=inv) idxv=0;
              }
            }
            if (++idxu>=inu) idxu=0;
            }
          }

      public:
        complex<Tacc> * DUCC0_RESTRICT p0;
        union kbuf {
          Tacc scalar[2*nvec*vlen];
          mysimd<Tacc> simd[2*nvec];
#if defined(_MSC_VER)
          kbuf() {}
#endif
          };
        kbuf buf;

        HelperNu2u(const Nufft2d *parent_, vmav<complex<Tcalc>,2> &grid_,
          vector<mutex> &locks_)
          : parent(parent_), tkrn(*parent->krn), grid(grid_),
            iu0(-1000000), iv0(-1000000),
            bu0(-1000000), bv0(-1000000),
            gbuf({size_t(su+1),size_t(sv)}),
            px0(gbuf.data()),
            locks(locks_)
          { checkShape(grid.shape(), {parent->nu,parent->nv}); }
        ~HelperNu2u() { dump(); }

        constexpr int lineJump() const { return sv; }

        [[gnu::always_inline]] [[gnu::hot]] void prep(double u_in, double v_in)
          {
          double ufrac, vfrac;
          auto iu0old = iu0;
          auto iv0old = iv0;
          parent->getpix(u_in, v_in, ufrac, vfrac, iu0, iv0);
          auto x0 = -ufrac*2+(supp-1);
          auto y0 = -vfrac*2+(supp-1);
          tkrn.eval2(Tacc(x0), Tacc(y0), &buf.simd[0]);
          if ((iu0==iu0old) && (iv0==iv0old)) return;
          if ((iu0<bu0) || (iv0<bv0) || (iu0+int(supp)>bu0+su) || (iv0+int(supp)>bv0+sv))
            {
            dump();
            bu0=((((iu0+nsafe)>>log2tile)<<log2tile))-nsafe;
            bv0=((((iv0+nsafe)>>log2tile)<<log2tile))-nsafe;
            }
          p0 = px0 + (iu0-bu0)*sv + iv0-bv0;
          }
      };

    template<size_t supp> class HelperU2nu
      {
      public:
        static constexpr size_t vlen = mysimd<Tcalc>::size();
        static constexpr size_t nvec = (supp+vlen-1)/vlen;

      private:
        static constexpr int nsafe = (supp+1)/2;
        static constexpr int su = supp+(1<<log2tile);
        static constexpr int sv = supp+(1<<log2tile);
        static constexpr int svvec = max<size_t>(sv, ((supp+2*vlen-2)/vlen)*vlen);
        static constexpr double xsupp=2./supp;
        const Nufft2d *parent;

        TemplateKernel<supp, mysimd<Tcalc>> tkrn;
        const cmav<complex<Tcalc>,2> &grid;
        int iu0, iv0; // start index of the current nonuniform point
        int bu0, bv0; // start index of the current buffer

        vmav<Tcalc,2> bufri;
        const Tcalc *px0r, *px0i;

        DUCC0_NOINLINE void load()
          {
          int inu = int(parent->nu);
          int inv = int(parent->nv);
          int idxu = (bu0+inu)%inu;
          int idxv0 = (bv0+inv)%inv;
          for (int iu=0; iu<su; ++iu)
            {
            int idxv = idxv0;
            for (int iv=0; iv<sv; ++iv)
              {
              bufri(2*iu,iv) = grid(idxu, idxv).real();
              bufri(2*iu+1,iv) = grid(idxu, idxv).imag();
              if (++idxv>=inv) idxv=0;
              }
            if (++idxu>=inu) idxu=0;
            }
          }

      public:
        const Tcalc * DUCC0_RESTRICT p0r, * DUCC0_RESTRICT p0i;
        union kbuf {
          Tcalc scalar[2*nvec*vlen];
          mysimd<Tcalc> simd[2*nvec];
#if defined(_MSC_VER)
          kbuf() {}
#endif
          };
        kbuf buf;

        HelperU2nu(const Nufft2d *parent_, const cmav<complex<Tcalc>,2> &grid_)
          : parent(parent_), tkrn(*parent->krn), grid(grid_),
            iu0(-1000000), iv0(-1000000),
            bu0(-1000000), bv0(-1000000),
            bufri({size_t(2*su+1),size_t(svvec)}),
            px0r(bufri.data()), px0i(bufri.data()+svvec)
          { checkShape(grid.shape(), {parent->nu,parent->nv}); }

        constexpr int lineJump() const { return 2*svvec; }

        [[gnu::always_inline]] [[gnu::hot]] void prep(double u_in, double v_in)
          {
          double ufrac, vfrac;
          auto iu0old = iu0;
          auto iv0old = iv0;
          parent->getpix(u_in, v_in, ufrac, vfrac, iu0, iv0);
          auto x0 = -ufrac*2+(supp-1);
          auto y0 = -vfrac*2+(supp-1);
            tkrn.eval2(Tcalc(x0), Tcalc(y0), &buf.simd[0]);
          if ((iu0==iu0old) && (iv0==iv0old)) return;
          if ((iu0<bu0) || (iv0<bv0) || (iu0+int(supp)>bu0+su) || (iv0+int(supp)>bv0+sv))
            {
            bu0=((((iu0+nsafe)>>log2tile)<<log2tile))-nsafe;
            bv0=((((iv0+nsafe)>>log2tile)<<log2tile))-nsafe;
            load();
            }
          auto ofs = (iu0-bu0)*2*svvec + iv0-bv0;
          p0r = px0r+ofs;
          p0i = px0i+ofs;
          }
      };

    template<size_t SUPP> [[gnu::hot]] void spreading_helper
      (size_t supp, vmav<complex<Tcalc>,2> &grid)
      {
      if constexpr (SUPP>=8)
        if (supp<=SUPP/2) return spreading_helper<SUPP/2>(supp, grid);
      if constexpr (SUPP>4)
        if (supp<SUPP) return spreading_helper<SUPP-1>(supp, grid);
      MR_assert(supp==SUPP, "requested support out of range");

      vector<mutex> locks(nu);

      execDynamic(coord_idx.size(), nthreads, 1000, [&](Scheduler &sched)
        {
        constexpr size_t vlen=mysimd<Tcalc>::size();
        constexpr size_t NVEC((SUPP+vlen-1)/vlen);
        HelperNu2u<SUPP> hlp(this, grid, locks);
        constexpr auto jump = hlp.lineJump();
        const auto * DUCC0_RESTRICT ku = hlp.buf.scalar;
        const auto * DUCC0_RESTRICT kv = hlp.buf.scalar+NVEC*vlen;
        constexpr size_t NVEC2 = (2*SUPP+vlen-1)/vlen;
        union Txdata{
          array<complex<Tacc>,SUPP> c;
          array<mysimd<Tacc>,NVEC2> v;
          Txdata(){for (size_t i=0; i<v.size(); ++i) v[i]=0;}
          };
        Txdata xdata;

        constexpr size_t lookahead=3;
        while (auto rng=sched.getNext()) for(auto ix=rng.lo; ix<rng.hi; ++ix)
          {
          if (ix+lookahead<coord_idx.size())
            {
            auto nextidx = coord_idx[ix+lookahead];
            DUCC0_PREFETCH_R(&points_in(nextidx));
            DUCC0_PREFETCH_R(&coords(nextidx,0));
            DUCC0_PREFETCH_R(&coords(nextidx,1));
            }
          size_t row = coord_idx[ix];
          hlp.prep(coords(row,0), coords(row,1));
          auto v(points_in(row));

          for (size_t cv=0; cv<SUPP; ++cv)
            xdata.c[cv] = kv[cv]*v;

          Tacc * DUCC0_RESTRICT xpx = reinterpret_cast<Tacc *>(hlp.p0);
          for (size_t cu=0; cu<SUPP; ++cu)
            {
            Tacc tmpx=ku[cu];
            for (size_t cv=0; cv<NVEC2; ++cv)
              {
              auto * DUCC0_RESTRICT px = xpx+cu*2*jump+cv*hlp.vlen;
              auto tval = mysimd<Tacc>(px,element_aligned_tag());
              tval += tmpx*xdata.v[cv];
              tval.copy_to(px,element_aligned_tag());
              }
            }
          }
        });
      }

    template<size_t SUPP> [[gnu::hot]] void interpolation_helper
      (size_t supp, const cmav<complex<Tcalc>,2> &grid)
      {
      if constexpr (SUPP>=8)
        if (supp<=SUPP/2) return interpolation_helper<SUPP/2>(supp, grid);
      if constexpr (SUPP>4)
        if (supp<SUPP) return interpolation_helper<SUPP-1>(supp, grid);
      MR_assert(supp==SUPP, "requested support out of range");

      execDynamic(coord_idx.size(), nthreads, 1000, [&](Scheduler &sched)
        {
        constexpr size_t vlen=mysimd<Tcalc>::size();
        constexpr size_t NVEC((SUPP+vlen-1)/vlen);
        HelperU2nu<SUPP> hlp(this, grid);
        constexpr int jump = hlp.lineJump();
        const auto * DUCC0_RESTRICT ku = hlp.buf.scalar;
        const auto * DUCC0_RESTRICT kv = hlp.buf.simd+NVEC;

        constexpr size_t lookahead=3;
        while (auto rng=sched.getNext()) for(auto ix=rng.lo; ix<rng.hi; ++ix)
          {
          if (ix+lookahead<coord_idx.size())
            {
            auto nextidx = coord_idx[ix+lookahead];
            DUCC0_PREFETCH_W(&points_out(nextidx));
            DUCC0_PREFETCH_R(&coords(nextidx,0));
            DUCC0_PREFETCH_R(&coords(nextidx,1));
            }
          size_t row = coord_idx[ix];
          hlp.prep(coords(row,0), coords(row,1));
          mysimd<Tcalc> rr=0, ri=0;
          if constexpr (NVEC==1)
            {
            for (size_t cu=0; cu<SUPP; ++cu)
              {
              const auto * DUCC0_RESTRICT pxr = hlp.p0r + cu*jump;
              const auto * DUCC0_RESTRICT pxi = hlp.p0i + cu*jump;
              rr += mysimd<Tcalc>(pxr,element_aligned_tag())*ku[cu];
              ri += mysimd<Tcalc>(pxi,element_aligned_tag())*ku[cu];
              }
            rr *= kv[0];
            ri *= kv[0];
            }
          else
            {
            for (size_t cu=0; cu<SUPP; ++cu)
              {
              mysimd<Tcalc> tmpr(0), tmpi(0);
              for (size_t cv=0; cv<NVEC; ++cv)
                {
                const auto * DUCC0_RESTRICT pxr = hlp.p0r + cu*jump + hlp.vlen*cv;
                const auto * DUCC0_RESTRICT pxi = hlp.p0i + cu*jump + hlp.vlen*cv;
                tmpr += kv[cv]*mysimd<Tcalc>(pxr,element_aligned_tag());
                tmpi += kv[cv]*mysimd<Tcalc>(pxi,element_aligned_tag());
                }
              rr += ku[cu]*tmpr;
              ri += ku[cu]*tmpi;
              }
            }
          points_out(row) = hsum_cmplx<Tcalc>(rr,ri);
          }
        });
      }

    void report()
      {
      if (verbosity==0) return;
      cout << (gridding ? "Gridding:" : "Degridding:") << endl
           << "  nthreads=" << nthreads << ", "
           << "grid=(" << nxuni << "x" << nyuni << "), "
           << "oversampled grid=(" << nu << "x" << nv;
      cout << "), supp=" << supp
           << ", eps=" << epsilon
           << endl;
      cout << "  npoints=" << coords.shape(0) << endl;
      cout << "  memory overhead: "
           << coords.shape(0)*sizeof(uint32_t)/double(1<<30) << "GB (index) + "
           << nu*nv*sizeof(complex<Tcalc>)/double(1<<30) << "GB (oversampled grid)" << endl;
      }

    void nonuni2uni()
      {
      timers.push("allocating grid");
      auto grid = vmav<complex<Tcalc>,2>::build_noncritical({nu,nv});
      timers.poppush("spreading");
      constexpr size_t maxsupp = is_same<Tacc, double>::value ? 16 : 8;
      spreading_helper<maxsupp>(supp, grid);

      timers.poppush("FFT");
      checkShape(grid.shape(), {nu,nv});
      {
      vfmav<complex<Tcalc>> fgrid(grid);
      c2c(fgrid, fgrid, {1}, forward, Tcalc(1), nthreads);
      auto fgridl=fgrid.subarray({{},{0,(nyuni+1)/2}});
      c2c(fgridl, fgridl, {0}, forward, Tcalc(1), nthreads);
      auto fgridh=fgrid.subarray({{},{fgrid.shape(1)-nyuni/2,MAXIDX}});
      c2c(fgridh, fgridh, {0}, forward, Tcalc(1), nthreads);
      }
      timers.poppush("grid correction");
      checkShape(uniform_out.shape(), {nxuni, nyuni});
      auto cfu = krn->corfunc(nxuni/2+1, 1./nu, nthreads);
      auto cfv = krn->corfunc(nyuni/2+1, 1./nv, nthreads);
      execParallel(nxuni, nthreads, [&](size_t lo, size_t hi)
        {
        for (auto i=lo; i<hi; ++i)
          {
          int icfu = abs(int(nxuni/2)-int(i));
          size_t iin = nu-nxuni/2+i;
          if (iin>=nu) iin-=nu;
          size_t iout = fft_order ? nxuni-nxuni/2+i : i;
          if (iout>=nxuni) iout-=nxuni;
          for (size_t j=0; j<nyuni; ++j)
            {
            int icfv = abs(int(nyuni/2)-int(j));
            size_t jin = nv-nyuni/2+j;
            if (jin>=nv) jin-=nv;
            size_t jout = fft_order ? nyuni-nyuni/2+j : j;
            if (jout>=nyuni) jout-=nyuni;
            uniform_out(iout,jout) = complex<Tgrid>(grid(iin,jin)*Tgrid(cfu[icfu]*cfv[icfv]));
            }
          }
        });
      timers.pop();
      }

    void uni2nonuni()
      {
      timers.push("allocating grid");
      auto grid = vmav<complex<Tcalc>,2>::build_noncritical({nu,nv});
      timers.poppush("zeroing grid");
      checkShape(uniform_in.shape(), {nxuni, nyuni});

      // only zero the parts of the grid that are not filled afterwards anyway
      { auto a0 = subarray<2>(grid, {{0,nxuni/2}, {nyuni/2,nv-nyuni/2+1}}); quickzero(a0, nthreads); }
      { auto a0 = subarray<2>(grid, {{nxuni/2, nu-nxuni/2+1}, {}}); quickzero(a0, nthreads); }
      { auto a0 = subarray<2>(grid, {{nu-nxuni/2+1,MAXIDX}, {nyuni/2, nv-nyuni/2+1}}); quickzero(a0, nthreads); }
      timers.poppush("grid correction");
      auto cfu = krn->corfunc(nxuni/2+1, 1./nu, nthreads);
      auto cfv = krn->corfunc(nyuni/2+1, 1./nv, nthreads);
      execParallel(nxuni, nthreads, [&](size_t lo, size_t hi)
        {
        for (auto i=lo; i<hi; ++i)
          {
          int icfu = abs(int(nxuni/2)-int(i));
          size_t iin = fft_order ? nxuni-nxuni/2+i : i;
          if (iin>=nxuni) iin-=nxuni;
          size_t iout = nu-nxuni/2+i;
          if (iout>=nu) iout-=nu;
          for (size_t j=0; j<nyuni; ++j)
            {
            int icfv = abs(int(nyuni/2)-int(j));
            size_t jin = fft_order ? nyuni-nyuni/2+j : j;
            if (jin>=nyuni) jin-=nyuni;
            size_t jout = nv-nyuni/2+j;
            if (jout>=nv) jout-=nv;
            grid(iout,jout) = uniform_in(iin,jin)*Tcalc(cfu[icfu]*cfv[icfv]);
            }
          }
        });
      timers.poppush("FFT");
      {
      vfmav<complex<Tcalc>> fgrid(grid);
      auto fgridl=fgrid.subarray({{},{0,(nyuni+1)/2}});
      c2c(fgridl, fgridl, {0}, forward, Tcalc(1), nthreads);
      auto fgridh=fgrid.subarray({{},{fgrid.shape(1)-nyuni/2,MAXIDX}});
      c2c(fgridh, fgridh, {0}, forward, Tcalc(1), nthreads);
      c2c(fgrid, fgrid, {1}, forward, Tcalc(1), nthreads);
      }
      timers.poppush("interpolation");
      constexpr size_t maxsupp = is_same<Tcalc, double>::value ? 16 : 8;
      interpolation_helper<maxsupp>(supp, grid);
      timers.pop();
      }

  public:
    Nufft2d(const cmav<Tcoord,2> &coords_,
           const cmav<complex<Tpoints>,1> &points_in_, vmav<complex<Tpoints>,1> &points_out_,
           const cmav<complex<Tgrid>,2> &uniform_in_, vmav<complex<Tgrid>,2> &uniform_out_,
           double epsilon_, bool forward_,
           size_t nthreads_, size_t verbosity_,
           double sigma_min,
           double sigma_max,
           double periodicity,
           bool fft_order_)
      : gridding(points_out_.size()==0),
        forward(forward_),
        timers(gridding ? "gridding" : "degridding"),
        points_in(points_in_), points_out(points_out_),
        uniform_in(uniform_in_), uniform_out(uniform_out_),
        nxuni(gridding ? uniform_out.shape(0) : uniform_in.shape(0)),
        nyuni(gridding ? uniform_out.shape(1) : uniform_in.shape(1)),
        epsilon(epsilon_),
        nthreads((nthreads_==0) ? get_default_nthreads() : nthreads_),
        verbosity(verbosity_),
        coordfct(1./periodicity),
        fft_order(fft_order_),
        coords(coords_)
      {
      MR_assert(coords.shape(0)<=(~uint32_t(0)), "too many rows in the MS");
      checkShape(points_in.shape(), {coords.shape(0)});
      if (coords.shape(0)==0)
        {
        if (gridding) mav_apply([](complex<Tgrid> &v){v=complex<Tgrid>(0);}, nthreads, uniform_out);
        return;
        }

      timers.push("parameter calculation");
      auto [kidx, dims] = findNufftParameters<Tcalc,Tacc>(epsilon, sigma_min, sigma_max,
        {nxuni, nyuni}, coords.shape(0), gridding,
        gridding ? mysimd<Tacc>::size() : mysimd<Tcalc>::size(), nthreads);
      nu = dims[0];
      nv = dims[1];
      timers.pop();

      MR_assert((nu>>log2tile)<(size_t(1)<<16), "nu too large");
      MR_assert((nv>>log2tile)<(size_t(1)<<16), "nv too large");
      krn = selectKernel(kidx);
      supp = krn->support();
      nsafe = (supp+1)/2;
      ushift = supp*(-0.5)+1+nu;
      vshift = supp*(-0.5)+1+nv;
      maxiu0 = (nu+nsafe)-supp;
      maxiv0 = (nv+nsafe)-supp;
      MR_assert(nu>=2*nsafe, "nu too small");
      MR_assert(nv>=2*nsafe, "nv too small");
      MR_assert((nu&1)==0, "nu must be even");
      MR_assert((nv&1)==0, "nv must be even");
      MR_assert(epsilon>0, "epsilon must be positive");

      timers.push("building index");
      size_t ntiles_u = (nu>>log2tile) + 3;
      size_t ntiles_v = (nv>>log2tile) + 3;
      coord_idx.resize(coords.shape(0));
      quick_array<uint32_t> key(coords.shape(0));
      execParallel(coords.shape(0), nthreads, [&](size_t lo, size_t hi)
        {
        for (size_t i=lo; i<hi; ++i)
          {
          auto [tile_u, tile_v] = get_uvtile(coords(i,0), coords(i,1));
          key[i] = tile_u*ntiles_v + tile_v;
          }
        });
      bucket_sort2(key, coord_idx, ntiles_u*ntiles_v, nthreads);
      timers.pop();

      report();
      gridding ? nonuni2uni() : uni2nonuni();

      if (verbosity>0)
        timers.report(cout);
      }
  };

template<typename Tcalc, typename Tacc, typename Tpoints, typename Tgrid, typename Tcoord> class Nufft3d
  {
  private:
    constexpr static int log2tile=4;
    bool gridding;
    bool forward;
    TimerHierarchy timers;
    const cmav<complex<Tpoints>,1> &points_in;
    vmav<complex<Tpoints>,1> &points_out;
    const cmav<complex<Tgrid>,3> &uniform_in;
    vmav<complex<Tgrid>,3> &uniform_out;
    size_t nxuni, nyuni, nzuni;
    double epsilon;
    size_t nthreads;
    size_t verbosity;
    double coordfct;
    bool fft_order;

    const cmav<Tcoord,2> &coords;

    quick_array<uint32_t> coord_idx;

    size_t nu, nv, nw;

    shared_ptr<PolynomialKernel> krn;

    size_t supp, nsafe;
    double ushift, vshift, wshift;
    int maxiu0, maxiv0,maxiw0;

    static_assert(sizeof(Tcalc)<=sizeof(Tacc), "bad type combination");
    static_assert(sizeof(Tpoints)<=sizeof(Tcalc), "bad type combination");
    static_assert(sizeof(Tgrid)<=sizeof(Tcalc), "bad type combination");

    [[gnu::always_inline]] void getpix(double u_in, double v_in, double w_in, double &u, double &v, double &w, int &iu0, int &iv0, int &iw0) const
      {
      // probably no need for long double here, since grid dimensions will never be that large
      using Tbig = double;
        //typename conditional<is_same<Tcoord,double>::value, long double, double>::type;
      u_in *= coordfct;
      auto tmpu = Tbig(u_in-floor(u_in))*nu;
      iu0 = min(int(tmpu+ushift)-int(nu), maxiu0);
      u = double(tmpu-iu0);
      v_in *= coordfct;
      auto tmpv = Tbig(v_in-floor(v_in))*nv;
      iv0 = min(int(tmpv+vshift)-int(nv), maxiv0);
      v = double(tmpv-iv0);
      w_in *= coordfct;
      auto tmpw = Tbig(w_in-floor(w_in))*nw;
      iw0 = min(int(tmpw+wshift)-int(nw), maxiw0);
      w = double(tmpw-iw0);
      }

    [[gnu::always_inline]] auto get_uvwtile(double u_in, double v_in, double w_in, size_t lsq2)
      {
      double udum, vdum, wdum;
      int iu0, iv0, iw0;
      getpix(u_in, v_in, w_in, udum, vdum, wdum, iu0, iv0, iw0);
      iu0 = (iu0+nsafe)>>lsq2;
      iv0 = (iv0+nsafe)>>lsq2;
      iw0 = (iw0+nsafe)>>lsq2;
      return make_tuple(uint32_t(iu0), uint32_t(iv0), uint32_t(iw0));
      }

    template<size_t supp> class HelperNu2u
      {
      public:
        static constexpr size_t vlen = mysimd<Tacc>::size();
        static constexpr size_t nvec = (supp+vlen-1)/vlen;

      private:
        static constexpr int nsafe = (supp+1)/2;
        static constexpr int su = supp+(1<<log2tile);
        static constexpr int sv = supp+(1<<log2tile);
        static constexpr int sw = supp+(1<<log2tile);
        static constexpr double xsupp=2./supp;
        const Nufft3d *parent;
        TemplateKernel<supp, mysimd<Tacc>> tkrn;
        vmav<complex<Tcalc>,3> &grid;
        int iu0, iv0, iw0; // start index of the current nonuniform point
        int bu0, bv0, bw0; // start index of the current buffer

        vmav<complex<Tacc>,3> gbuf;
        complex<Tacc> *px0;
        vector<mutex> &locks;

        DUCC0_NOINLINE void dump()
          {
          int inu = int(parent->nu);
          int inv = int(parent->nv);
          int inw = int(parent->nw);
          if (bu0<-nsafe) return; // nothing written into buffer yet

          int idxu = (bu0+inu)%inu;
          int idxv0 = (bv0+inv)%inv;
          int idxw0 = (bw0+inw)%inw;
          for (int iu=0; iu<su; ++iu)
            {
            int idxv = idxv0;
            {
            lock_guard<mutex> lock(locks[idxu]);
            for (int iv=0; iv<sv; ++iv)
              {
              int idxw = idxw0;
              for (int iw=0; iw<sw; ++iw)
                {
                auto t=gbuf(iu,iv,iw);
                grid(idxu,idxv,idxw) += complex<Tcalc>(t);
                gbuf(iu,iv,iw) = 0;
                if (++idxw>=inw) idxw=0;
                }
              if (++idxv>=inv) idxv=0;
              }
            }
            if (++idxu>=inu) idxu=0;
            }
          }

      public:
        complex<Tacc> * DUCC0_RESTRICT p0;
        union kbuf {
          Tacc scalar[3*nvec*vlen];
          mysimd<Tacc> simd[3*nvec];
#if defined(_MSC_VER)
          kbuf() {}
#endif
          };
        kbuf buf;

        HelperNu2u(const Nufft3d *parent_, vmav<complex<Tcalc>,3> &grid_,
          vector<mutex> &locks_)
          : parent(parent_), tkrn(*parent->krn), grid(grid_),
            iu0(-1000000), iv0(-1000000), iw0(-1000000),
            bu0(-1000000), bv0(-1000000), bw0(-1000000),
            gbuf({size_t(su),size_t(sv),size_t(sw)}),
            px0(gbuf.data()),
            locks(locks_)
          { checkShape(grid.shape(), {parent->nu,parent->nv,parent->nw}); }
        ~HelperNu2u() { dump(); }

        constexpr int lineJump() const { return sw; }
        constexpr int planeJump() const { return sv*sw; }

        [[gnu::always_inline]] [[gnu::hot]] void prep(double u_in, double v_in, double w_in)
          {
          double ufrac, vfrac, wfrac;
          auto iu0old = iu0;
          auto iv0old = iv0;
          auto iw0old = iw0;
          parent->getpix(u_in, v_in, w_in, ufrac, vfrac, wfrac, iu0, iv0, iw0);
          auto x0 = -ufrac*2+(supp-1);
          auto y0 = -vfrac*2+(supp-1);
          auto z0 = -wfrac*2+(supp-1);
          tkrn.eval3(Tacc(x0), Tacc(y0), Tacc(z0), &buf.simd[0]);
          if ((iu0==iu0old) && (iv0==iv0old) && (iw0==iw0old)) return;
          if ((iu0<bu0) || (iv0<bv0) || (iw0<bw0)
           || (iu0+int(supp)>bu0+su) || (iv0+int(supp)>bv0+sv) || (iw0+int(supp)>bw0+sw))
            {
            dump();
            bu0=((((iu0+nsafe)>>log2tile)<<log2tile))-nsafe;
            bv0=((((iv0+nsafe)>>log2tile)<<log2tile))-nsafe;
            bw0=((((iw0+nsafe)>>log2tile)<<log2tile))-nsafe;
            }
          p0 = px0 + (iu0-bu0)*sv*sw + (iv0-bv0)*sw + (iw0-bw0);
          }
      };

    template<size_t supp> class HelperU2nu
      {
      public:
        static constexpr size_t vlen = mysimd<Tcalc>::size();
        static constexpr size_t nvec = (supp+vlen-1)/vlen;

      private:
        static constexpr int nsafe = (supp+1)/2;
        static constexpr int su = 2*nsafe+(1<<log2tile);
        static constexpr int sv = 2*nsafe+(1<<log2tile);
        static constexpr int sw = 2*nsafe+(1<<log2tile);
        static constexpr int swvec = max<size_t>(sw, ((supp+2*nvec-2)/nvec)*nvec);
        static constexpr double xsupp=2./supp;
        const Nufft3d *parent;

        TemplateKernel<supp, mysimd<Tcalc>> tkrn;
        const cmav<complex<Tcalc>,3> &grid;
        int iu0, iv0, iw0; // start index of the nonuniform point
        int bu0, bv0, bw0; // start index of the current buffer

        vmav<Tcalc,3> bufri;
        const Tcalc *px0r, *px0i;

        DUCC0_NOINLINE void load()
          {
          int inu = int(parent->nu);
          int inv = int(parent->nv);
          int inw = int(parent->nw);
          int idxu = (bu0+inu)%inu;
          int idxv0 = (bv0+inv)%inv;
          int idxw0 = (bw0+inw)%inw;
          for (int iu=0; iu<su; ++iu)
            {
            int idxv = idxv0;
            for (int iv=0; iv<sv; ++iv)
              {
              int idxw = idxw0;
              for (int iw=0; iw<sw; ++iw)
                {
                bufri(iu,2*iv,iw) = grid(idxu, idxv, idxw).real();
                bufri(iu,2*iv+1,iw) = grid(idxu, idxv, idxw).imag();
                if (++idxw>=inw) idxw=0;
                }
              if (++idxv>=inv) idxv=0;
              }
            if (++idxu>=inu) idxu=0;
            }
          }

      public:
        const Tcalc * DUCC0_RESTRICT p0r, * DUCC0_RESTRICT p0i;
        union kbuf {
          Tcalc scalar[3*nvec*vlen];
          mysimd<Tcalc> simd[3*nvec];
#if defined(_MSC_VER)
          kbuf() {}
#endif
          };
        kbuf buf;

        HelperU2nu(const Nufft3d *parent_, const cmav<complex<Tcalc>,3> &grid_)
          : parent(parent_), tkrn(*parent->krn), grid(grid_),
            iu0(-1000000), iv0(-1000000), iw0(-1000000),
            bu0(-1000000), bv0(-1000000), bw0(-1000000),
            bufri({size_t(su+1),size_t(2*sv),size_t(swvec)}),
            px0r(bufri.data()), px0i(bufri.data()+swvec)
          { checkShape(grid.shape(), {parent->nu,parent->nv,parent->nw}); }

        constexpr int lineJump() const { return 2*swvec; }
        constexpr int planeJump() const { return 2*sv*swvec; }

        [[gnu::always_inline]] [[gnu::hot]] void prep(double u_in, double v_in, double w_in)
          {
          double ufrac, vfrac, wfrac;
          auto iu0old = iu0;
          auto iv0old = iv0;
          auto iw0old = iw0;
          parent->getpix(u_in, v_in, w_in, ufrac, vfrac, wfrac, iu0, iv0, iw0);
          auto x0 = -ufrac*2+(supp-1);
          auto y0 = -vfrac*2+(supp-1);
          auto z0 = -wfrac*2+(supp-1);
          tkrn.eval3(Tcalc(x0), Tcalc(y0), Tcalc(z0), &buf.simd[0]);
          if ((iu0==iu0old) && (iv0==iv0old) && (iw0==iw0old)) return;
          if ((iu0<bu0) || (iv0<bv0) || (iw0<bw0)
           || (iu0+int(supp)>bu0+su) || (iv0+int(supp)>bv0+sv) || (iw0+int(supp)>bw0+sw))
            {
            bu0=((((iu0+nsafe)>>log2tile)<<log2tile))-nsafe;
            bv0=((((iv0+nsafe)>>log2tile)<<log2tile))-nsafe;
            bw0=((((iw0+nsafe)>>log2tile)<<log2tile))-nsafe;
            load();
            }
          auto ofs = (iu0-bu0)*2*sv*swvec + (iv0-bv0)*2*swvec + (iw0-bw0);
          p0r = px0r+ofs;
          p0i = px0i+ofs;
          }
      };

    template<size_t SUPP> [[gnu::hot]] void spreading_helper
      (size_t supp, vmav<complex<Tcalc>,3> &grid)
      {
      if constexpr (SUPP>=8)
        if (supp<=SUPP/2) return spreading_helper<SUPP/2>(supp, grid);
      if constexpr (SUPP>4)
        if (supp<SUPP) return spreading_helper<SUPP-1>(supp, grid);
      MR_assert(supp==SUPP, "requested support out of range");

      vector<mutex> locks(nu);

      execDynamic(coord_idx.size(), nthreads, 10000, [&](Scheduler &sched)
        {
        constexpr size_t vlen=mysimd<Tcalc>::size();
        constexpr size_t NVEC((SUPP+vlen-1)/vlen);
        HelperNu2u<SUPP> hlp(this, grid, locks);
        constexpr auto ljump = hlp.lineJump();
        constexpr auto pjump = hlp.planeJump();
        const auto * DUCC0_RESTRICT ku = hlp.buf.scalar;
        const auto * DUCC0_RESTRICT kv = hlp.buf.scalar+vlen*NVEC;
        const auto * DUCC0_RESTRICT kw = hlp.buf.scalar+2*vlen*NVEC;
        union Txdata{
          array<complex<Tacc>,SUPP> c;
          array<Tacc,2*SUPP> f;
          Txdata(){for (size_t i=0; i<f.size(); ++i) f[i]=0;}
          };
        Txdata xdata;

        while (auto rng=sched.getNext()) for(auto ix=rng.lo; ix<rng.hi; ++ix)
          {
          constexpr size_t lookahead=3;
          if (ix+lookahead<coord_idx.size())
            {
            auto nextidx = coord_idx[ix+lookahead];
            DUCC0_PREFETCH_R(&points_in(nextidx));
            DUCC0_PREFETCH_R(&coords(nextidx,0));
            DUCC0_PREFETCH_R(&coords(nextidx,1));
            DUCC0_PREFETCH_R(&coords(nextidx,2));
            }
          size_t row = coord_idx[ix];
          hlp.prep(coords(row,0), coords(row,1), coords(row,2));
          auto v(points_in(row));

          for (size_t cw=0; cw<SUPP; ++cw)
            xdata.c[cw]=kw[cw]*v;
          const Tacc * DUCC0_RESTRICT fptr1=xdata.f.data();
          Tacc * DUCC0_RESTRICT fptr2=reinterpret_cast<Tacc *>(hlp.p0);
          const auto j1 = 2*ljump;
          const auto j2 = 2*(pjump-SUPP*ljump);
          for (size_t cu=0; cu<SUPP; ++cu, fptr2+=j2)
            for (size_t cv=0; cv<SUPP; ++cv, fptr2+=j1)
              {
              Tacc tmp2x=ku[cu]*kv[cv];
              for (size_t cw=0; cw<2*SUPP; ++cw)
                fptr2[cw] += tmp2x*fptr1[cw];
              }
          }
        });
      }

    template<size_t SUPP> [[gnu::hot]] void interpolation_helper
      (size_t supp, const cmav<complex<Tcalc>,3> &grid)
      {
      if constexpr (SUPP>=8)
        if (supp<=SUPP/2) return interpolation_helper<SUPP/2>(supp, grid);
      if constexpr (SUPP>4)
        if (supp<SUPP) return interpolation_helper<SUPP-1>(supp, grid);
      MR_assert(supp==SUPP, "requested support out of range");

      execDynamic(coord_idx.size(), nthreads, 1000, [&](Scheduler &sched)
        {
        constexpr size_t vlen=mysimd<Tcalc>::size();
        constexpr size_t NVEC((SUPP+vlen-1)/vlen);
        HelperU2nu<SUPP> hlp(this, grid);
        constexpr auto ljump = hlp.lineJump();
        constexpr auto pjump = hlp.planeJump();
        const auto * DUCC0_RESTRICT ku = hlp.buf.scalar;
        const auto * DUCC0_RESTRICT kv = hlp.buf.scalar+vlen*NVEC;
        const auto * DUCC0_RESTRICT kw = hlp.buf.simd+2*NVEC;

        while (auto rng=sched.getNext()) for(auto ix=rng.lo; ix<rng.hi; ++ix)
          {
          constexpr size_t lookahead=3;
          if (ix+lookahead<coord_idx.size())
            {
            auto nextidx = coord_idx[ix+lookahead];
            DUCC0_PREFETCH_W(&points_out(nextidx));
            DUCC0_PREFETCH_R(&coords(nextidx,0));
            DUCC0_PREFETCH_R(&coords(nextidx,1));
            DUCC0_PREFETCH_R(&coords(nextidx,2));
            }
          size_t row = coord_idx[ix];
          hlp.prep(coords(row,0), coords(row,1), coords(row,2));
          mysimd<Tcalc> rr=0, ri=0;
          if constexpr (NVEC==1)
            {
            for (size_t cu=0; cu<SUPP; ++cu)
              {
              mysimd<Tcalc> r2r=0, r2i=0;
              for (size_t cv=0; cv<SUPP; ++cv)
                {
                const auto * DUCC0_RESTRICT pxr = hlp.p0r + cu*pjump + cv*ljump;
                const auto * DUCC0_RESTRICT pxi = hlp.p0i + cu*pjump + cv*ljump;
                r2r += mysimd<Tcalc>(pxr,element_aligned_tag())*kv[cv];
                r2i += mysimd<Tcalc>(pxi,element_aligned_tag())*kv[cv];
                }
              rr += r2r*ku[cu];
              ri += r2i*ku[cu];
              }
            rr *= kw[0];
            ri *= kw[0];
            }
          else
            {
            for (size_t cu=0; cu<SUPP; ++cu)
              {
              mysimd<Tcalc> tmpr(0), tmpi(0);
              for (size_t cv=0; cv<SUPP; ++cv)
                {
                mysimd<Tcalc> tmp2r(0), tmp2i(0);
                for (size_t cw=0; cw<NVEC; ++cw)
                  {
                  const auto * DUCC0_RESTRICT pxr = hlp.p0r + cu*pjump + cv*ljump + hlp.vlen*cw;
                  const auto * DUCC0_RESTRICT pxi = hlp.p0i + cu*pjump + cv*ljump + hlp.vlen*cw;
                  tmp2r += kw[cw]*mysimd<Tcalc>(pxr,element_aligned_tag());
                  tmp2i += kw[cw]*mysimd<Tcalc>(pxi,element_aligned_tag());
                  }
                tmpr += kv[cv]*tmp2r;
                tmpi += kv[cv]*tmp2i;
                }
              rr += ku[cu]*tmpr;
              ri += ku[cu]*tmpi;
              }
            }
          points_out(row) = hsum_cmplx<Tcalc>(rr,ri);
          }
        });
      }

    void report()
      {
      if (verbosity==0) return;
      cout << (gridding ? "Gridding:" : "Degridding:") << endl
           << "  nthreads=" << nthreads << ", "
           << "grid=(" << nxuni << "x" << nyuni << "x" << nzuni << "), "
           << "oversampled grid=(" << nu << "x" << nv << "x" << nw;
      cout << "), supp=" << supp
           << ", eps=" << epsilon
           << endl;
      cout << "  npoints=" << coords.shape(0) << endl;
      cout << "  memory overhead: "
           << coords.shape(0)*sizeof(uint32_t)/double(1<<30) << "GB (index) + "
           << nu*nv*nw*sizeof(complex<Tcalc>)/double(1<<30) << "GB (oversampled grid)" << endl;
      }

    void nonuni2uni()
      {
      timers.push("allocating grid");
      auto grid = vmav<complex<Tcalc>,3>::build_noncritical({nu,nv,nw});
      timers.poppush("spreading");
      constexpr size_t maxsupp = is_same<Tacc, double>::value ? 16 : 8;
      spreading_helper<maxsupp>(supp, grid);
      timers.poppush("FFT");
      {
      vfmav<complex<Tcalc>> fgrid(grid);
      slice slz{0,(nzuni+1)/2}, shz{fgrid.shape(2)-nzuni/2,MAXIDX};
      slice sly{0,(nyuni+1)/2}, shy{fgrid.shape(1)-nyuni/2,MAXIDX};
      c2c(fgrid, fgrid, {2}, forward, Tcalc(1), nthreads);
      auto fgridl=fgrid.subarray({{},{},slz});
      c2c(fgridl, fgridl, {1}, forward, Tcalc(1), nthreads);
      auto fgridh=fgrid.subarray({{},{},shz});
      c2c(fgridh, fgridh, {1}, forward, Tcalc(1), nthreads);
      auto fgridll=fgrid.subarray({{},sly,slz});
      c2c(fgridll, fgridll, {0}, forward, Tcalc(1), nthreads);
      auto fgridlh=fgrid.subarray({{},sly,shz});
      c2c(fgridlh, fgridlh, {0}, forward, Tcalc(1), nthreads);
      auto fgridhl=fgrid.subarray({{},shy,slz});
      c2c(fgridhl, fgridhl, {0}, forward, Tcalc(1), nthreads);
      auto fgridhh=fgrid.subarray({{},shy,shz});
      c2c(fgridhh, fgridhh, {0}, forward, Tcalc(1), nthreads);
      }
      timers.poppush("grid correction");
      checkShape(uniform_out.shape(), {nxuni, nyuni, nzuni});
      auto cfu = krn->corfunc(nxuni/2+1, 1./nu, nthreads);
      auto cfv = krn->corfunc(nyuni/2+1, 1./nv, nthreads);
      auto cfw = krn->corfunc(nzuni/2+1, 1./nw, nthreads);
      execParallel(nxuni, nthreads, [&](size_t lo, size_t hi)
        {
        for (auto i=lo; i<hi; ++i)
          {
          int icfu = abs(int(nxuni/2)-int(i));
          size_t iin = nu-nxuni/2+i;
          if (iin>=nu) iin-=nu;
          size_t iout = fft_order ? nxuni-nxuni/2+i : i;
          if (iout>=nxuni) iout-=nxuni;
          for (size_t j=0; j<nyuni; ++j)
            {
            int icfv = abs(int(nyuni/2)-int(j));
            size_t jin = nv-nyuni/2+j;
            if (jin>=nv) jin-=nv;
            size_t jout = fft_order ? nyuni-nyuni/2+j : j;
            if (jout>=nyuni) jout-=nyuni;
            for (size_t k=0; k<nzuni; ++k)
              {
              int icfw = abs(int(nzuni/2)-int(k));
              size_t kin = nw-nzuni/2+k;
              if (kin>=nw) kin-=nw;
              size_t kout = fft_order ? nzuni-nzuni/2+k : k;
              if (kout>=nzuni) kout-=nzuni;
              uniform_out(iout,jout,kout) = complex<Tgrid>(grid(iin,jin,kin)*Tgrid(cfu[icfu]*cfv[icfv]*cfw[icfw]));
              }
            }
          }
        });
      timers.pop();
      }

    void uni2nonuni()
      {
      timers.push("allocating grid");
      auto grid = vmav<complex<Tcalc>,3>::build_noncritical({nu,nv,nw});
      timers.poppush("zeroing grid");
      checkShape(uniform_in.shape(), {nxuni, nyuni, nzuni});
      // TODO: not all entries need to be zeroed, perhaps some time can be saved here
      mav_apply([](complex<Tcalc> &v){v=complex<Tcalc>(0);},nthreads,grid);
      timers.poppush("grid correction");
      auto cfu = krn->corfunc(nxuni/2+1, 1./nu, nthreads);
      auto cfv = krn->corfunc(nyuni/2+1, 1./nv, nthreads);
      auto cfw = krn->corfunc(nzuni/2+1, 1./nw, nthreads);
      execParallel(nxuni, nthreads, [&](size_t lo, size_t hi)
        {
        for (auto i=lo; i<hi; ++i)
          {
          int icfu = abs(int(nxuni/2)-int(i));
          size_t iin = fft_order ? nxuni-nxuni/2+i : i;
          if (iin>=nxuni) iin-=nxuni;
          size_t iout = nu-nxuni/2+i;
          if (iout>=nu) iout-=nu;
          for (size_t j=0; j<nyuni; ++j)
            {
            int icfv = abs(int(nyuni/2)-int(j));
            size_t jin = fft_order ? nyuni-nyuni/2+j : j;
            if (jin>=nyuni) jin-=nyuni;
            size_t jout = nv-nyuni/2+j;
            if (jout>=nv) jout-=nv;
            for (size_t k=0; k<nzuni; ++k)
              {
              int icfw = abs(int(nzuni/2)-int(k));
              size_t kin = fft_order ? nzuni-nzuni/2+k : k;
              if (kin>=nzuni) kin-=nzuni;
              size_t kout = nw-nzuni/2+k;
              if (kout>=nw) kout-=nw;
              grid(iout,jout,kout) = uniform_in(iin,jin,kin)*Tcalc(cfu[icfu]*cfv[icfv]*cfw[icfw]);
              }
            }
          }
        });
      timers.poppush("FFT");
      {
      vfmav<complex<Tcalc>> fgrid(grid);
      slice slz{0,(nzuni+1)/2}, shz{fgrid.shape(2)-nzuni/2,MAXIDX};
      slice sly{0,(nyuni+1)/2}, shy{fgrid.shape(1)-nyuni/2,MAXIDX};
      auto fgridll=fgrid.subarray({{},sly,slz});
      c2c(fgridll, fgridll, {0}, forward, Tcalc(1), nthreads);
      auto fgridlh=fgrid.subarray({{},sly,shz});
      c2c(fgridlh, fgridlh, {0}, forward, Tcalc(1), nthreads);
      auto fgridhl=fgrid.subarray({{},shy,slz});
      c2c(fgridhl, fgridhl, {0}, forward, Tcalc(1), nthreads);
      auto fgridhh=fgrid.subarray({{},shy,shz});
      c2c(fgridhh, fgridhh, {0}, forward, Tcalc(1), nthreads);
      auto fgridl=fgrid.subarray({{},{},slz});
      c2c(fgridl, fgridl, {1}, forward, Tcalc(1), nthreads);
      auto fgridh=fgrid.subarray({{},{},shz});
      c2c(fgridh, fgridh, {1}, forward, Tcalc(1), nthreads);
      c2c(fgrid, fgrid, {2}, forward, Tcalc(1), nthreads);
      }
      timers.poppush("interpolation");
      constexpr size_t maxsupp = is_same<Tcalc, double>::value ? 16 : 8;
      interpolation_helper<maxsupp>(supp, grid);
      timers.pop();
      }

  public:
    Nufft3d(const cmav<Tcoord,2> &coords_,
           const cmav<complex<Tpoints>,1> &points_in_, vmav<complex<Tpoints>,1> &points_out_,
           const cmav<complex<Tgrid>,3> &uniform_in_, vmav<complex<Tgrid>,3> &uniform_out_,
           double epsilon_, bool forward_,
           size_t nthreads_, size_t verbosity_,
           double sigma_min,
           double sigma_max,
           double periodicity,
           bool fft_order_)
      : gridding(points_out_.size()==0),
        forward(forward_),
        timers(gridding ? "gridding" : "degridding"),
        points_in(points_in_), points_out(points_out_),
        uniform_in(uniform_in_), uniform_out(uniform_out_),
        nxuni(gridding ? uniform_out.shape(0) : uniform_in.shape(0)),
        nyuni(gridding ? uniform_out.shape(1) : uniform_in.shape(1)),
        nzuni(gridding ? uniform_out.shape(2) : uniform_in.shape(2)),
        epsilon(epsilon_),
        nthreads((nthreads_==0) ? get_default_nthreads() : nthreads_),
        verbosity(verbosity_),
        coordfct(1./periodicity),
        fft_order(fft_order_),
        coords(coords_)
      {
      MR_assert(coords.shape(0)<=(~uint32_t(0)), "too many rows in the MS");
      checkShape(points_in.shape(), {coords.shape(0)});
      if (coords.shape(0)==0)
        {
        if (gridding) mav_apply([](complex<Tgrid> &v){v=complex<Tgrid>(0);}, nthreads, uniform_out);
        return;
        }

      timers.push("parameter calculation");
      auto [kidx, dims] = findNufftParameters<Tcalc,Tacc>(epsilon, sigma_min, sigma_max,
        {nxuni, nyuni, nzuni}, coords.shape(0), gridding,
        gridding ? mysimd<Tacc>::size() : mysimd<Tcalc>::size(), nthreads);
      nu = dims[0];
      nv = dims[1];
      nw = dims[2];
      timers.pop();

      MR_assert((nu>>log2tile)<(uint32_t(1)<<10), "nu too large");
      MR_assert((nv>>log2tile)<(uint32_t(1)<<10), "nv too large");
      MR_assert((nw>>log2tile)<(uint32_t(1)<<10), "nw too large");
      krn = selectKernel(kidx);
      supp = krn->support();
      nsafe = (supp+1)/2;
      ushift = supp*(-0.5)+1+nu;
      vshift = supp*(-0.5)+1+nv;
      wshift = supp*(-0.5)+1+nw;
      maxiu0 = (nu+nsafe)-supp;
      maxiv0 = (nv+nsafe)-supp;
      maxiw0 = (nw+nsafe)-supp;
      MR_assert(nu>=2*nsafe, "nu too small");
      MR_assert(nv>=2*nsafe, "nv too small");
      MR_assert(nw>=2*nsafe, "nw too small");
      MR_assert((nu&1)==0, "nu must be even");
      MR_assert((nv&1)==0, "nv must be even");
      MR_assert((nw&1)==0, "nw must be even");
      MR_assert(epsilon>0, "epsilon must be positive");

      timers.push("building index");
      size_t ntiles_u = (nu>>log2tile) + 3;
      size_t ntiles_v = (nv>>log2tile) + 3;
      size_t ntiles_w = (nw>>log2tile) + 3;
      size_t lsq2 = log2tile;
      while ((lsq2>=1) && (((ntiles_u*ntiles_v*ntiles_w)<<(3*(log2tile-lsq2)))<(size_t(1)<<28)))
        --lsq2;
      auto ssmall = log2tile-lsq2;
      auto msmall = (size_t(1)<<ssmall) - 1;

      coord_idx.resize(coords.shape(0));
      quick_array<uint32_t> key(coords.shape(0));
      execParallel(coords.shape(0), nthreads, [&](size_t lo, size_t hi)
        {
        for (size_t i=lo; i<hi; ++i)
          {
          auto [tile_u, tile_v, tile_w] = get_uvwtile(coords(i,0),coords(i,1),coords(i,2),lsq2);
          auto lowkey = ((tile_u&msmall)<<(2*ssmall))
                      | ((tile_v&msmall)<<   ssmall)
                      |  (tile_w&msmall);
          auto hikey = ((tile_u>>ssmall)*ntiles_v*ntiles_w)
                     + ((tile_v>>ssmall)*ntiles_w)
                     +  (tile_w>>ssmall);
          key[i] = (hikey<<(3*ssmall)) | lowkey;
          }
        });
      bucket_sort2(key, coord_idx, (ntiles_u*ntiles_v*ntiles_w)<<(3*ssmall), nthreads);
      timers.pop();

      report();
      gridding ? nonuni2uni() : uni2nonuni();

      if (verbosity>0)
        timers.report(cout);
      }
  };

template<typename Tcalc, typename Tacc, typename Tpoints, typename Tgrid, typename Tcoord> void nu2u(const cmav<Tcoord,2> &coord,
  const cmav<complex<Tpoints>,1> &ms, bool forward,
  double epsilon,
  size_t nthreads, vfmav<complex<Tgrid>> &uniform, size_t verbosity,
  double sigma_min=1.1,
  double sigma_max=2.6,
  double periodicity=2*pi, bool fft_order=false)
  {
  auto ndim = uniform.ndim();
  MR_assert((ndim>=1) && (ndim<=3), "transform must be 1D/2D/3D");
  MR_assert(ndim==coord.shape(1), "dimensionality mismatch");
  auto points_out(vmav<complex<Tpoints>,1>::build_empty());
  if (ndim==1)
    {
    auto uniform_in(vmav<complex<Tgrid>,1>::build_empty());
    vmav<complex<Tpoints>,1> uniform2(uniform);
    Nufft1d<Tcalc, Tacc, Tpoints, Tgrid, Tcoord> par(coord, ms, points_out, uniform_in, uniform2,
      epsilon, forward, nthreads, verbosity, sigma_min, sigma_max, periodicity, fft_order);
    }
  else if (ndim==2)
    {
    auto uniform_in(vmav<complex<Tgrid>,2>::build_empty());
    vmav<complex<Tpoints>,2> uniform2(uniform);
    Nufft2d<Tcalc, Tacc, Tpoints, Tgrid, Tcoord> par(coord, ms, points_out, uniform_in, uniform2,
      epsilon, forward, nthreads, verbosity, sigma_min, sigma_max, periodicity, fft_order);
    }
  else if (ndim==3)
    {
    auto uniform_in(vmav<complex<Tgrid>,3>::build_empty());
    vmav<complex<Tpoints>,3> uniform2(uniform);
    Nufft3d<Tcalc, Tacc, Tpoints, Tgrid, Tcoord> par(coord, ms, points_out, uniform_in, uniform2,
      epsilon, forward, nthreads, verbosity, sigma_min, sigma_max, periodicity, fft_order);
    }
  }
template<typename Tcalc, typename Tacc, typename Tpoints, typename Tgrid, typename Tcoord> void u2nu(const cmav<Tcoord,2> &coord,
  const cfmav<complex<Tgrid>> &uniform, bool forward,
  double epsilon, size_t nthreads, vmav<complex<Tpoints>,1> &ms,
  size_t verbosity,
  double sigma_min=1.1, double sigma_max=2.6,
  double periodicity=2*pi, bool fft_order=false)
  {
  auto ndim = uniform.ndim();
  MR_assert((ndim>=1) && (ndim<=3), "transform must be 1D/2D/3D");
  MR_assert(ndim==coord.shape(1), "dimensionality mismatch");
  auto points_in(ms.build_uniform(ms.shape(),complex<Tpoints>(1.)));
  if (ndim==1)
    {
    auto uniform_out(vmav<complex<Tgrid>,1>::build_empty());
    Nufft1d<Tcalc, Tacc, Tpoints, Tgrid, Tcoord> par(coord, points_in, ms, uniform, uniform_out,
      epsilon, forward, nthreads, verbosity, sigma_min, sigma_max, periodicity, fft_order);
    }
  else if (ndim==2)
    {
    auto uniform_out(vmav<complex<Tgrid>,2>::build_empty());
    Nufft2d<Tcalc, Tacc, Tpoints, Tgrid, Tcoord> par(coord, points_in, ms, uniform, uniform_out,
      epsilon, forward, nthreads, verbosity, sigma_min, sigma_max, periodicity, fft_order);
    }
  else if (ndim==3)
    {
    auto uniform_out(vmav<complex<Tgrid>,3>::build_empty());
    Nufft3d<Tcalc, Tacc, Tpoints, Tgrid, Tcoord> par(coord, points_in, ms, uniform, uniform_out,
      epsilon, forward, nthreads, verbosity, sigma_min, sigma_max, periodicity, fft_order);
    }
  }
} // namespace detail_nufft

// public names
using detail_nufft::u2nu;
using detail_nufft::nu2u;

} // namespace ducc0

#endif
