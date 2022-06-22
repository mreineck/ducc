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

template<typename T> constexpr inline int mysimdlen
  = min<int>(8, native_simd<T>::size());

template<typename T> using mysimd = typename simd_select<T,mysimdlen<T>>::type;

template<typename T> T sqr(T val) { return val*val; }

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

//
// Start of real gridder functionality
//

template<typename Tcalc, typename Tacc, typename Tms, typename Timg, typename Tcoord> class Params1d
  {
  private:
    struct U
      { double u; };

    constexpr static int logsquare=9;
    static constexpr double coordfct=0.5/pi;

    bool gridding;
    bool forward;
    TimerHierarchy timers;
    const cmav<complex<Tms>,1> &ms_in;
    vmav<complex<Tms>,1> &ms_out;
    const cmav<complex<Timg>,1> &dirty_in;
    vmav<complex<Timg>,1> &dirty_out;
    size_t nxdirty;
    double epsilon;
    size_t nthreads;
    size_t verbosity;
    double sigma_min, sigma_max;

    const cmav<Tcoord,2> &coord;

    quick_array<uint32_t> coord_idx;

    size_t nvis;

    size_t nu;
    double ofactor;

    shared_ptr<HornerKernel> krn;

    size_t supp, nsafe;
    double ushift;
    int maxiu0;

    static_assert(sizeof(Tcalc)<=sizeof(Tacc), "bad type combination");
    static_assert(sizeof(Tms)<=sizeof(Tcalc), "bad type combination");
    static_assert(sizeof(Timg)<=sizeof(Tcalc), "bad type combination");

    [[gnu::always_inline]] void getpix(double u_in, double &u, int &iu0) const
      {
      u = (u_in-floor(u_in))*nu;
      iu0 = min(int(u+ushift)-int(nu), maxiu0);
      u -= iu0;
      }

    [[gnu::always_inline]] uint32_t get_uvwidx(const U &u)
      {
      double udum;
      int iu0;
      getpix(u.u, udum, iu0);
      iu0 = (iu0+nsafe)>>logsquare;
      return uint32_t(iu0);
      }

    template<size_t supp> class HelperX2g2
      {
      public:
        static constexpr size_t vlen = mysimd<Tacc>::size();
        static constexpr size_t nvec = (supp+vlen-1)/vlen;

      private:
        static constexpr int nsafe = (supp+1)/2;
        static constexpr int su = 2*nsafe+(1<<logsquare);
        static constexpr int suvec = su+vlen-1;
        static constexpr double xsupp=2./supp;
        const Params1d *parent;
        TemplateKernel<supp, mysimd<Tacc>> tkrn;
        vmav<complex<Tcalc>,1> &grid;
        int iu0; // start index of the current visibility
        int bu0; // start index of the current buffer

        vmav<Tacc,1> bufr, bufi;
        Tacc *px0r, *px0i;
        mutex &mylock;

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

        HelperX2g2(const Params1d *parent_, vmav<complex<Tcalc>,1> &grid_,
          mutex &mylock_)
          : parent(parent_), tkrn(*parent->krn), grid(grid_),
            iu0(-1000000),
            bu0(-1000000),
            bufr({size_t(suvec)}),
            bufi({size_t(suvec)}),
            px0r(bufr.data()), px0i(bufi.data()),
            mylock(mylock_)
          { checkShape(grid.shape(), {parent->nu}); }
        ~HelperX2g2() { dump(); }

        [[gnu::always_inline]] [[gnu::hot]] void prep(const U &in)
          {
          double ufrac;
          auto iu0old = iu0;
          parent->getpix(in.u, ufrac, iu0);
          auto x0 = -ufrac*2+(supp-1);
          tkrn.eval1(Tacc(x0), &buf.simd[0]);
          if (iu0==iu0old) return;
          if ((iu0<bu0) || (iu0+int(supp)>bu0+su))
            {
            dump();
            bu0=((((iu0+nsafe)>>logsquare)<<logsquare))-nsafe;
            }
          auto ofs = iu0-bu0;
          p0r = px0r+ofs;
          p0i = px0i+ofs;
          }
      };

    template<size_t supp> class HelperG2x2
      {
      public:
        static constexpr size_t vlen = mysimd<Tcalc>::size();
        static constexpr size_t nvec = (supp+vlen-1)/vlen;

      private:
        static constexpr int nsafe = (supp+1)/2;
        static constexpr int su = 2*nsafe+(1<<logsquare);
        static constexpr int suvec = su+vlen-1;
        static constexpr double xsupp=2./supp;
        const Params1d *parent;

        TemplateKernel<supp, mysimd<Tcalc>> tkrn;
        const cmav<complex<Tcalc>,1> &grid;
        int iu0; // start index of the current visibility
        int bu0; // start index of the current buffer

        vmav<Tcalc,1> bufr, bufi;
        const Tcalc *px0r, *px0i;

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

        HelperG2x2(const Params1d *parent_, const cmav<complex<Tcalc>,1> &grid_)
          : parent(parent_), tkrn(*parent->krn), grid(grid_),
            iu0(-1000000),
            bu0(-1000000),
            bufr({size_t(suvec)}),
            bufi({size_t(suvec)}),
            px0r(bufr.data()), px0i(bufi.data())
          { checkShape(grid.shape(), {parent->nu}); }

        [[gnu::always_inline]] [[gnu::hot]] void prep(const U &in)
          {
          double ufrac;
          auto iu0old = iu0;
          parent->getpix(in.u, ufrac, iu0);
          auto x0 = -ufrac*2+(supp-1);
          tkrn.eval1(Tcalc(x0), &buf.simd[0]);
          if (iu0==iu0old) return;
          if ((iu0<bu0) || (iu0+int(supp)>bu0+su))
            {
            bu0=((((iu0+nsafe)>>logsquare)<<logsquare))-nsafe;
            load();
            }
          auto ofs = iu0-bu0;
          p0r = px0r+ofs;
          p0i = px0i+ofs;
          }
      };

    template<size_t SUPP> [[gnu::hot]] void x2grid_c_helper
      (size_t supp, vmav<complex<Tcalc>,1> &grid)
      {
      if constexpr (SUPP>=8)
        if (supp<=SUPP/2) return x2grid_c_helper<SUPP/2>(supp, grid);
      if constexpr (SUPP>4)
        if (supp<SUPP) return x2grid_c_helper<SUPP-1>(supp, grid);
      MR_assert(supp==SUPP, "requested support out of range");

      mutex mylock;

      execDynamic(coord_idx.size(), nthreads, 1000, [&](Scheduler &sched)
        {
        constexpr size_t vlen=mysimd<Tcalc>::size();
        constexpr size_t NVEC((SUPP+vlen-1)/vlen);
        HelperX2g2<SUPP> hlp(this, grid, mylock);
        const auto * DUCC0_RESTRICT ku = hlp.buf.simd;

        constexpr size_t lookahead=10;
        while (auto rng=sched.getNext()) for(auto ix=rng.lo; ix<rng.hi; ++ix)
          {
          if (ix+lookahead<coord_idx.size())
            {
            auto nextidx = coord_idx[ix+lookahead];
            DUCC0_PREFETCH_R(&ms_in(nextidx));
            DUCC0_PREFETCH_R(&coord(nextidx,0));
            }
          size_t row = coord_idx[ix];
          auto lcoord = U{coord(row,0)*coordfct};
          hlp.prep(lcoord);
          auto v(ms_in(row));

          mysimd<Tacc> vr(v.real()), vi(v.imag());
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

    template<size_t SUPP> [[gnu::hot]] void grid2x_c_helper
      (size_t supp, const cmav<complex<Tcalc>,1> &grid)
      {
      if constexpr (SUPP>=8)
        if (supp<=SUPP/2) return grid2x_c_helper<SUPP/2>(supp, grid);
      if constexpr (SUPP>4)
        if (supp<SUPP) return grid2x_c_helper<SUPP-1>(supp, grid);
      MR_assert(supp==SUPP, "requested support out of range");

      execDynamic(coord_idx.size(), nthreads, 1000, [&](Scheduler &sched)
        {
        constexpr size_t vlen=mysimd<Tcalc>::size();
        constexpr size_t NVEC((SUPP+vlen-1)/vlen);
        HelperG2x2<SUPP> hlp(this, grid);
        const auto * DUCC0_RESTRICT ku = hlp.buf.simd;

        constexpr size_t lookahead=10;
        while (auto rng=sched.getNext()) for(auto ix=rng.lo; ix<rng.hi; ++ix)
          {
          if (ix+lookahead<coord_idx.size())
            {
            auto nextidx = coord_idx[ix+lookahead];
            DUCC0_PREFETCH_W(&ms_out(nextidx));
            DUCC0_PREFETCH_R(&coord(nextidx,0));
            }
          size_t row = coord_idx[ix];
          auto lcoord = U{coord(row,0)*coordfct};
          hlp.prep(lcoord);
          mysimd<Tcalc> rr=0, ri=0;
          for (size_t cu=0; cu<NVEC; ++cu)
            {
            const auto * DUCC0_RESTRICT pxr = hlp.p0r + cu*hlp.vlen;
            const auto * DUCC0_RESTRICT pxi = hlp.p0i + cu*hlp.vlen;
            rr += ku[cu]*mysimd<Tcalc>(pxr,element_aligned_tag());
            ri += ku[cu]*mysimd<Tcalc>(pxi,element_aligned_tag());
            }
          ms_out(row) = hsum_cmplx<Tcalc>(rr,ri);
          }
        });
      }

    void report()
      {
      if (verbosity==0) return;
      cout << (gridding ? "Nonuniform to uniform:" : "Uniform to nonuniform:") << endl
           << "  nthreads=" << nthreads << ", "
           << "grid=(" << nxdirty << "), "
           << "oversampled grid=(" << nu;
      cout << "), supp=" << supp
           << ", eps=" << epsilon
           << endl;
      cout << "  npoints=" << coord.shape(0) << endl;
      size_t ovh0 = coord.shape(0)*sizeof(uint32_t);
      size_t ovh1 = nu*sizeof(complex<Tcalc>);             // grid
      if (!gridding)
        ovh1 += nxdirty*sizeof(Timg);                 // tdirty
      cout << "  memory overhead: "
           << ovh0/double(1<<30) << "GB (index) + "
           << ovh1/double(1<<30) << "GB (1D arrays)" << endl;
      }

    void x2dirty()
      {
      timers.push("allocating grid");
      auto grid = vmav<complex<Tcalc>,1>::build_noncritical({nu});
      timers.poppush("gridding proper");
      constexpr size_t maxsupp = is_same<Tacc, double>::value ? 16 : 8;
      x2grid_c_helper<maxsupp>(supp, grid);
      timers.poppush("FFT");
      vfmav<complex<Tcalc>> fgrid(grid);
      c2c(fgrid, fgrid, {0}, forward, Tcalc(1), nthreads);
      timers.poppush("grid correction");
      checkShape(dirty_out.shape(), {nxdirty});
      auto cfu = krn->corfunc(nxdirty/2+1, 1./nu, nthreads);
      execParallel(nxdirty, nthreads, [&](size_t lo, size_t hi)
        {
        for (auto i=lo; i<hi; ++i)
          {
          int icfu = abs(int(nxdirty/2)-int(i));
          size_t i2 = nu-nxdirty/2+i;
          if (i2>=nu) i2-=nu;
          dirty_out(i) = complex<Timg>(grid(i2)*Timg(cfu[icfu]));
          }
        });
      timers.pop();
      }

    void dirty2x()
      {
      timers.push("allocating grid");
      auto grid = vmav<complex<Tcalc>,1>::build_noncritical({nu});
      timers.poppush("zeroing grid");
      mav_apply([](complex<Tcalc> &v){v=complex<Tcalc>(0);},nthreads,grid); // quickzero(grid, nthreads);
      timers.poppush("grid correction");
      checkShape(dirty_in.shape(), {nxdirty});
      auto cfu = krn->corfunc(nxdirty/2+1, 1./nu, nthreads);
      execParallel(nxdirty, nthreads, [&](size_t lo, size_t hi)
        {
        for (auto i=lo; i<hi; ++i)
          {
          int icfu = abs(int(nxdirty/2)-int(i));
          size_t i2 = nu-nxdirty/2+i;
          if (i2>=nu) i2-=nu;
          grid(i2) = dirty_in(i)*Tcalc(cfu[icfu]);
          }
        });
      timers.poppush("FFT");
      vfmav<complex<Tcalc>> fgrid(grid);
      c2c(fgrid, fgrid, {0}, forward, Tcalc(1), nthreads);
      timers.poppush("degridding proper");
      constexpr size_t maxsupp = is_same<Tcalc, double>::value ? 16 : 8;
      grid2x_c_helper<maxsupp>(supp, grid);
      timers.pop();
      }

    auto getNu()
      {
      timers.push("parameter calculation");

      auto idx = getAvailableKernels<Tcalc>(epsilon, 1, sigma_min, sigma_max);
      double mincost = 1e300;
      constexpr double nref_fft=2048;
      constexpr double costref_fft=0.0693;
      size_t minnu=0, minidx=KernelDB.size();
      size_t vlen = gridding ? mysimd<Tacc>::size() : mysimd<Tcalc>::size();
      for (size_t i=0; i<idx.size(); ++i)
        {
        const auto &krn(KernelDB[idx[i]]);
        auto supp = krn.W;
        auto nvec = (supp+vlen-1)/vlen;
        auto ofactor = krn.ofactor;
        size_t nu=2*good_size_complex(size_t(nxdirty*ofactor*0.5)+1);
        double logterm = log(nu)/log(nref_fft*nref_fft);
        double fftcost = nu/(nref_fft*nref_fft)*logterm*costref_fft;
        double gridcost = 2.2e-10*nvis*(nvec*vlen + (nvec*(supp+3)*vlen));
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
          minnu=nu;
          minidx = idx[i];
          }
        }
      timers.pop();
      nu = minnu;
      return minidx;
      }

  public:
    Params1d(const cmav<Tcoord,2> &coord_,
           const cmav<complex<Tms>,1> &ms_in_, vmav<complex<Tms>,1> &ms_out_,
           const cmav<complex<Timg>,1> &dirty_in_, vmav<complex<Timg>,1> &dirty_out_,
           double epsilon_, bool forward_,
           size_t nthreads_, size_t verbosity_,
           double sigma_min_,
           double sigma_max_)
      : gridding(ms_out_.size()==0),
        forward(forward_),
        timers(gridding ? "gridding" : "degridding"),
        ms_in(ms_in_), ms_out(ms_out_),
        dirty_in(dirty_in_), dirty_out(dirty_out_),
        nxdirty(gridding ? dirty_out.shape(0) : dirty_in.shape(0)),
        epsilon(epsilon_),
        nthreads((nthreads_==0) ? get_default_nthreads() : nthreads_),
        verbosity(verbosity_),
        sigma_min(sigma_min_), sigma_max(sigma_max_),
        coord(coord_)
      {
      MR_assert(coord.shape(0)<(uint64_t(1)<<32), "too many rows in the MS");
      checkShape(ms_in.shape(), {coord.shape(0)});
      nvis=coord.shape(0);
      if (nvis==0)
        {
        if (gridding) mav_apply([](complex<Timg> &v){v=complex<Timg>(0);}, nthreads, dirty_out);
        return;
        }
      auto kidx = getNu();
      MR_assert((nu>>logsquare)<(uint64_t(1)<<32), "nu too large");
      ofactor = double(nu)/nxdirty;
      krn = selectKernel(kidx);
      supp = krn->support();
      nsafe = (supp+1)/2;
      ushift = supp*(-0.5)+1+nu;
      maxiu0 = (nu+nsafe)-supp;
      MR_assert(nu>=2*nsafe, "nu too small");
      MR_assert((nu&1)==0, "nu must be even");
      MR_assert(epsilon>0, "epsilon must be positive");

      timers.push("building index");
      size_t nrow=coord.shape(0);
      size_t ntiles_u = (nu>>logsquare) + 3;
      coord_idx.resize(nrow);
      quick_array<uint32_t> key(nrow);
      execParallel(nrow, nthreads, [&](size_t lo, size_t hi)
        {
        for (size_t i=lo; i<hi; ++i)
          key[i] = get_uvwidx(U{coord(i,0)*coordfct});
        });
      bucket_sort2(key, coord_idx, ntiles_u, nthreads);
      timers.pop();

      report();
      gridding ? x2dirty() : dirty2x();

      if (verbosity>0)
        timers.report(cout);
      }
  };

template<typename Tcalc, typename Tacc, typename Tms, typename Timg, typename Tcoord> class Params2d
  {
  private:
    struct Uvwidx
      { uint16_t tile_u, tile_v; };

    struct UV
      { double u, v; };

    class Baselines
      {
      protected:
        const cmav<Tcoord,2> &coord;
        static constexpr double fct=0.5/pi;

      public:
        Baselines(const cmav<Tcoord,2> &coord_)
          : coord(coord_)
          {
          MR_assert(coord_.shape(1)==2, "dimension mismatch");
          }

        UV baseCoord(size_t row) const
          { return UV{coord(row,0)*fct, coord(row,1)*fct}; }
        void prefetchRow(size_t row) const
          {
          DUCC0_PREFETCH_R(&coord(row,0));
          DUCC0_PREFETCH_R(&coord(row,1));
          }
        size_t Nrows() const { return coord.shape(0); }
      };

    constexpr static int logsquare=is_same<Tacc,float>::value ? 5 : 4;
    bool gridding;
    bool forward;
    TimerHierarchy timers;
    const cmav<complex<Tms>,1> &ms_in;
    vmav<complex<Tms>,1> &ms_out;
    const cmav<complex<Timg>,2> &dirty_in;
    vmav<complex<Timg>,2> &dirty_out;
    size_t nxdirty, nydirty;
    double epsilon;
    size_t nthreads;
    size_t verbosity;
    double sigma_min, sigma_max;

    Baselines bl;

    quick_array<uint32_t> coord_idx;

    size_t nvis;

    size_t nu, nv;
    double ofactor;

    shared_ptr<HornerKernel> krn;

    size_t supp, nsafe;
    double ushift, vshift;
    int maxiu0, maxiv0;

    static_assert(sizeof(Tcalc)<=sizeof(Tacc), "bad type combination");
    static_assert(sizeof(Tms)<=sizeof(Tcalc), "bad type combination");
    static_assert(sizeof(Timg)<=sizeof(Tcalc), "bad type combination");

    void grid2dirty(vmav<complex<Tcalc>,2> &grid, vmav<complex<Timg>,2> &dirty)
      {
      timers.push("FFT");
      checkShape(grid.shape(), {nu,nv});
      vfmav<complex<Tcalc>> fgrid(grid);
      c2c(fgrid, fgrid, {0,1}, forward, Tcalc(1), nthreads);
      timers.poppush("grid correction");
      checkShape(dirty.shape(), {nxdirty, nydirty});
      auto cfu = krn->corfunc(nxdirty/2+1, 1./nu, nthreads);
      auto cfv = krn->corfunc(nydirty/2+1, 1./nv, nthreads);
      execParallel(nxdirty, nthreads, [&](size_t lo, size_t hi)
        {
        for (auto i=lo; i<hi; ++i)
          {
          int icfu = abs(int(nxdirty/2)-int(i));
          for (size_t j=0; j<nydirty; ++j)
            {
            int icfv = abs(int(nydirty/2)-int(j));
            size_t i2 = nu-nxdirty/2+i;
            if (i2>=nu) i2-=nu;
            size_t j2 = nv-nydirty/2+j;
            if (j2>=nv) j2-=nv;
            dirty(i,j) = complex<Timg>(grid(i2,j2)*Timg(cfu[icfu]*cfv[icfv]));
            }
          }
        });
      timers.pop();
      }

    void dirty2grid(const cmav<complex<Timg>,2> &dirty, vmav<complex<Tcalc>,2> &grid)
      {
      timers.push("zeroing grid");
      checkShape(dirty.shape(), {nxdirty, nydirty});
      checkShape(grid.shape(), {nu, nv});
      // only zero the parts of the grid that are not filled afterwards anyway
      { auto a0 = subarray<2>(grid, {{0,nxdirty/2}, {nydirty/2,nv-nydirty/2+1}}); quickzero(a0, nthreads); }
      { auto a0 = subarray<2>(grid, {{nxdirty/2, nu-nxdirty/2+1}, {}}); quickzero(a0, nthreads); }
      { auto a0 = subarray<2>(grid, {{nu-nxdirty/2+1,MAXIDX}, {nydirty/2, nv-nydirty/2+1}}); quickzero(a0, nthreads); }
      timers.poppush("grid correction");
      auto cfu = krn->corfunc(nxdirty/2+1, 1./nu, nthreads);
      auto cfv = krn->corfunc(nydirty/2+1, 1./nv, nthreads);
      execParallel(nxdirty, nthreads, [&](size_t lo, size_t hi)
        {
        for (auto i=lo; i<hi; ++i)
          {
          int icfu = abs(int(nxdirty/2)-int(i));
          for (size_t j=0; j<nydirty; ++j)
            {
            int icfv = abs(int(nydirty/2)-int(j));
            size_t i2 = nu-nxdirty/2+i;
            if (i2>=nu) i2-=nu;
            size_t j2 = nv-nydirty/2+j;
            if (j2>=nv) j2-=nv;
            grid(i2,j2) = dirty(i,j)*Tcalc(cfu[icfu]*cfv[icfv]);
            }
          }
        });
      timers.poppush("FFT");
      vfmav<complex<Tcalc>> fgrid(grid);
      c2c(fgrid, fgrid, {0,1}, forward, Tcalc(1), nthreads);
      timers.pop();
      }

    [[gnu::always_inline]] void getpix(double u_in, double v_in, double &u, double &v, int &iu0, int &iv0) const
      {
      u = (u_in-floor(u_in))*nu;
      iu0 = min(int(u+ushift)-int(nu), maxiu0);
      u -= iu0;
      v = (v_in-floor(v_in))*nv;
      iv0 = min(int(v+vshift)-int(nv), maxiv0);
      v -= iv0;
      }

    [[gnu::always_inline]] Uvwidx get_uvwidx(const UV &uv)
      {
      double udum, vdum;
      int iu0, iv0;
      getpix(uv.u, uv.v, udum, vdum, iu0, iv0);
      iu0 = (iu0+nsafe)>>logsquare;
      iv0 = (iv0+nsafe)>>logsquare;
      return Uvwidx{uint16_t(iu0), uint16_t(iv0)};
      }

    template<size_t supp> class HelperX2g2
      {
      public:
        static constexpr size_t vlen = mysimd<Tacc>::size();
        static constexpr size_t nvec = (supp+vlen-1)/vlen;

      private:
        static constexpr int nsafe = (supp+1)/2;
        static constexpr int su = 2*nsafe+(1<<logsquare);
        static constexpr int sv = 2*nsafe+(1<<logsquare);
        static constexpr int svvec = sv+vlen-1;
        static constexpr double xsupp=2./supp;
        const Params2d *parent;
        TemplateKernel<supp, mysimd<Tacc>> tkrn;
        vmav<complex<Tcalc>,2> &grid;
        int iu0, iv0; // start index of the current visibility
        int bu0, bv0; // start index of the current buffer

        vmav<Tacc,2> bufr, bufi;
        Tacc *px0r, *px0i;
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
              grid(idxu,idxv) += complex<Tcalc>(Tcalc(bufr(iu,iv)), Tcalc(bufi(iu,iv)));
              bufr(iu,iv) = bufi(iu,iv) = 0;
              if (++idxv>=inv) idxv=0;
              }
            }
            if (++idxu>=inu) idxu=0;
            }
          }

      public:
        Tacc * DUCC0_RESTRICT p0r, * DUCC0_RESTRICT p0i;
        union kbuf {
          Tacc scalar[2*nvec*vlen];
          mysimd<Tacc> simd[2*nvec];
#if defined(_MSC_VER)
          kbuf() {}
#endif
          };
        kbuf buf;

        HelperX2g2(const Params2d *parent_, vmav<complex<Tcalc>,2> &grid_,
          vector<mutex> &locks_)
          : parent(parent_), tkrn(*parent->krn), grid(grid_),
            iu0(-1000000), iv0(-1000000),
            bu0(-1000000), bv0(-1000000),
            bufr({size_t(su),size_t(svvec)}),
            bufi({size_t(su),size_t(svvec)}),
            px0r(bufr.data()), px0i(bufi.data()),
            locks(locks_)
          { checkShape(grid.shape(), {parent->nu,parent->nv}); }
        ~HelperX2g2() { dump(); }

        constexpr int lineJump() const { return svvec; }

        [[gnu::always_inline]] [[gnu::hot]] void prep(const UV &in)
          {
          double ufrac, vfrac;
          auto iu0old = iu0;
          auto iv0old = iv0;
          parent->getpix(in.u, in.v, ufrac, vfrac, iu0, iv0);
          auto x0 = -ufrac*2+(supp-1);
          auto y0 = -vfrac*2+(supp-1);
          tkrn.eval2(Tacc(x0), Tacc(y0), &buf.simd[0]);
          if ((iu0==iu0old) && (iv0==iv0old)) return;
          if ((iu0<bu0) || (iv0<bv0) || (iu0+int(supp)>bu0+su) || (iv0+int(supp)>bv0+sv))
            {
            dump();
            bu0=((((iu0+nsafe)>>logsquare)<<logsquare))-nsafe;
            bv0=((((iv0+nsafe)>>logsquare)<<logsquare))-nsafe;
            }
          auto ofs = (iu0-bu0)*svvec + iv0-bv0;
          p0r = px0r+ofs;
          p0i = px0i+ofs;
          }
      };

    template<size_t supp> class HelperG2x2
      {
      public:
        static constexpr size_t vlen = mysimd<Tcalc>::size();
        static constexpr size_t nvec = (supp+vlen-1)/vlen;

      private:
        static constexpr int nsafe = (supp+1)/2;
        static constexpr int su = 2*nsafe+(1<<logsquare);
        static constexpr int sv = 2*nsafe+(1<<logsquare);
        static constexpr int svvec = sv+vlen-1;
        static constexpr double xsupp=2./supp;
        const Params2d *parent;

        TemplateKernel<supp, mysimd<Tcalc>> tkrn;
        const cmav<complex<Tcalc>,2> &grid;
        int iu0, iv0; // start index of the current visibility
        int bu0, bv0; // start index of the current buffer

        vmav<Tcalc,2> bufr, bufi;
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
              bufr(iu,iv) = grid(idxu, idxv).real();
              bufi(iu,iv) = grid(idxu, idxv).imag();
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

        HelperG2x2(const Params2d *parent_, const cmav<complex<Tcalc>,2> &grid_)
          : parent(parent_), tkrn(*parent->krn), grid(grid_),
            iu0(-1000000), iv0(-1000000),
            bu0(-1000000), bv0(-1000000),
            bufr({size_t(su),size_t(svvec)}),
            bufi({size_t(su),size_t(svvec)}),
            px0r(bufr.data()), px0i(bufi.data())
          { checkShape(grid.shape(), {parent->nu,parent->nv}); }

        constexpr int lineJump() const { return svvec; }

        [[gnu::always_inline]] [[gnu::hot]] void prep(const UV &in)
          {
          double ufrac, vfrac;
          auto iu0old = iu0;
          auto iv0old = iv0;
          parent->getpix(in.u, in.v, ufrac, vfrac, iu0, iv0);
          auto x0 = -ufrac*2+(supp-1);
          auto y0 = -vfrac*2+(supp-1);
            tkrn.eval2(Tcalc(x0), Tcalc(y0), &buf.simd[0]);
          if ((iu0==iu0old) && (iv0==iv0old)) return;
          if ((iu0<bu0) || (iv0<bv0) || (iu0+int(supp)>bu0+su) || (iv0+int(supp)>bv0+sv))
            {
            bu0=((((iu0+nsafe)>>logsquare)<<logsquare))-nsafe;
            bv0=((((iv0+nsafe)>>logsquare)<<logsquare))-nsafe;
            load();
            }
          auto ofs = (iu0-bu0)*svvec + iv0-bv0;
          p0r = px0r+ofs;
          p0i = px0i+ofs;
          }
      };

    template<size_t SUPP> [[gnu::hot]] void x2grid_c_helper
      (size_t supp, vmav<complex<Tcalc>,2> &grid)
      {
      if constexpr (SUPP>=8)
        if (supp<=SUPP/2) return x2grid_c_helper<SUPP/2>(supp, grid);
      if constexpr (SUPP>4)
        if (supp<SUPP) return x2grid_c_helper<SUPP-1>(supp, grid);
      MR_assert(supp==SUPP, "requested support out of range");

      vector<mutex> locks(nu);

      execDynamic(coord_idx.size(), nthreads, 1000, [&](Scheduler &sched)
        {
        constexpr size_t vlen=mysimd<Tcalc>::size();
        constexpr size_t NVEC((SUPP+vlen-1)/vlen);
        HelperX2g2<SUPP> hlp(this, grid, locks);
        constexpr auto jump = hlp.lineJump();
        const auto * DUCC0_RESTRICT ku = hlp.buf.scalar;
        const auto * DUCC0_RESTRICT kv = hlp.buf.simd+NVEC;

        constexpr size_t lookahead=3;
        while (auto rng=sched.getNext()) for(auto ix=rng.lo; ix<rng.hi; ++ix)
          {
          if (ix+lookahead<coord_idx.size())
            {
            auto nextidx = coord_idx[ix+lookahead];
            DUCC0_PREFETCH_R(&ms_in(nextidx));
            bl.prefetchRow(nextidx);
            }
          size_t row = coord_idx[ix];
          auto coord = bl.baseCoord(row);
          hlp.prep(coord);
          auto v(ms_in(row));

          if constexpr (NVEC==1)
            {
            mysimd<Tacc> vr=v.real()*kv[0], vi=v.imag()*kv[0];
            for (size_t cu=0; cu<SUPP; ++cu)
              {
              auto * DUCC0_RESTRICT pxr = hlp.p0r+cu*jump;
              auto * DUCC0_RESTRICT pxi = hlp.p0i+cu*jump;
              auto tr = mysimd<Tacc>(pxr,element_aligned_tag());
              auto ti = mysimd<Tacc>(pxi,element_aligned_tag());
              tr += vr*ku[cu];
              ti += vi*ku[cu];
              tr.copy_to(pxr,element_aligned_tag());
              ti.copy_to(pxi,element_aligned_tag());
              }
            }
          else
            {
            mysimd<Tacc> vr(v.real()), vi(v.imag());
            for (size_t cu=0; cu<SUPP; ++cu)
              {
              mysimd<Tacc> tmpr=vr*ku[cu], tmpi=vi*ku[cu];
              for (size_t cv=0; cv<NVEC; ++cv)
                {
                auto * DUCC0_RESTRICT pxr = hlp.p0r+cu*jump+cv*hlp.vlen;
                auto * DUCC0_RESTRICT pxi = hlp.p0i+cu*jump+cv*hlp.vlen;
                auto tr = mysimd<Tacc>(pxr,element_aligned_tag());
                tr += tmpr*kv[cv];
                tr.copy_to(pxr,element_aligned_tag());
                auto ti = mysimd<Tacc>(pxi, element_aligned_tag());
                ti += tmpi*kv[cv];
                ti.copy_to(pxi,element_aligned_tag());
                }
              }
            }
          }
        });
      }

    template<size_t SUPP> [[gnu::hot]] void grid2x_c_helper
      (size_t supp, const cmav<complex<Tcalc>,2> &grid)
      {
      if constexpr (SUPP>=8)
        if (supp<=SUPP/2) return grid2x_c_helper<SUPP/2>(supp, grid);
      if constexpr (SUPP>4)
        if (supp<SUPP) return grid2x_c_helper<SUPP-1>(supp, grid);
      MR_assert(supp==SUPP, "requested support out of range");

      execDynamic(coord_idx.size(), nthreads, 1000, [&](Scheduler &sched)
        {
        constexpr size_t vlen=mysimd<Tcalc>::size();
        constexpr size_t NVEC((SUPP+vlen-1)/vlen);
        HelperG2x2<SUPP> hlp(this, grid);
        constexpr int jump = hlp.lineJump();
        const auto * DUCC0_RESTRICT ku = hlp.buf.scalar;
        const auto * DUCC0_RESTRICT kv = hlp.buf.simd+NVEC;

        constexpr size_t lookahead=3;
        while (auto rng=sched.getNext()) for(auto ix=rng.lo; ix<rng.hi; ++ix)
          {
          if (ix+lookahead<coord_idx.size())
            {
            auto nextidx = coord_idx[ix+lookahead];
            DUCC0_PREFETCH_W(&ms_out(nextidx));
            bl.prefetchRow(nextidx);
            }
          size_t row = coord_idx[ix];
          auto coord = bl.baseCoord(row);
          hlp.prep(coord);
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
          ms_out(row) = hsum_cmplx<Tcalc>(rr,ri);
          }
        });
      }

    void report()
      {
      if (verbosity==0) return;
      cout << (gridding ? "Gridding:" : "Degridding:") << endl
           << "  nthreads=" << nthreads << ", "
           << "dirty=(" << nxdirty << "x" << nydirty << "), "
           << "grid=(" << nu << "x" << nv;
      cout << "), supp=" << supp
           << ", eps=" << epsilon
           << endl;
      cout << "  npoints=" << bl.Nrows() << endl;
      size_t ovh0 = bl.Nrows()*sizeof(uint32_t);
      size_t ovh1 = nu*nv*sizeof(complex<Tcalc>);             // grid
      if (!gridding)
        ovh1 += nxdirty*nydirty*sizeof(Timg);                 // tdirty
      cout << "  memory overhead: "
           << ovh0/double(1<<30) << "GB (index) + "
           << ovh1/double(1<<30) << "GB (2D arrays)" << endl;
      }

    void x2dirty()
      {
      timers.push("allocating grid");
      auto grid = vmav<complex<Tcalc>,2>::build_noncritical({nu,nv});
      timers.poppush("gridding proper");
      constexpr size_t maxsupp = is_same<Tacc, double>::value ? 16 : 8;
      x2grid_c_helper<maxsupp>(supp, grid);
      timers.pop();
      grid2dirty(grid, dirty_out);
      }

    void dirty2x()
      {
      timers.push("allocating grid");
      auto grid = vmav<complex<Tcalc>,2>::build_noncritical({nu,nv});
      timers.pop();
      dirty2grid(dirty_in, grid);
      timers.push("degridding proper");
      constexpr size_t maxsupp = is_same<Tcalc, double>::value ? 16 : 8;
      grid2x_c_helper<maxsupp>(supp, grid);
      timers.pop();
      }

    auto getNuNv()
      {
      timers.push("parameter calculation");

      auto idx = getAvailableKernels<Tcalc>(epsilon, 2, sigma_min, sigma_max);
      double mincost = 1e300;
      constexpr double nref_fft=2048;
      constexpr double costref_fft=0.0693;
      size_t minnu=0, minnv=0, minidx=KernelDB.size();
      size_t vlen = gridding ? mysimd<Tacc>::size() : mysimd<Tcalc>::size();
      for (size_t i=0; i<idx.size(); ++i)
        {
        const auto &krn(KernelDB[idx[i]]);
        auto supp = krn.W;
        auto nvec = (supp+vlen-1)/vlen;
        auto ofactor = krn.ofactor;
        size_t nu=2*good_size_complex(size_t(nxdirty*ofactor*0.5)+1);
        size_t nv=2*good_size_complex(size_t(nydirty*ofactor*0.5)+1);
        double logterm = log(nu*nv)/log(nref_fft*nref_fft);
        double fftcost = nu/nref_fft*nv/nref_fft*logterm*costref_fft;
        double gridcost = 2.2e-10*nvis*(supp*nvec*vlen + (2*nvec*(supp+3)*vlen));
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
          minnu=nu;
          minnv=nv;
          minidx = idx[i];
          }
        }
      timers.pop();
      nu = minnu;
      nv = minnv;
      return minidx;
      }

  public:
    Params2d(const cmav<Tcoord,2> &uv,
           const cmav<complex<Tms>,1> &ms_in_, vmav<complex<Tms>,1> &ms_out_,
           const cmav<complex<Timg>,2> &dirty_in_, vmav<complex<Timg>,2> &dirty_out_,
           double epsilon_, bool forward_,
           size_t nthreads_, size_t verbosity_,
           double sigma_min_,
           double sigma_max_)
      : gridding(ms_out_.size()==0),
        forward(forward_),
        timers(gridding ? "gridding" : "degridding"),
        ms_in(ms_in_), ms_out(ms_out_),
        dirty_in(dirty_in_), dirty_out(dirty_out_),
        nxdirty(gridding ? dirty_out.shape(0) : dirty_in.shape(0)),
        nydirty(gridding ? dirty_out.shape(1) : dirty_in.shape(1)),
        epsilon(epsilon_),
        nthreads((nthreads_==0) ? get_default_nthreads() : nthreads_),
        verbosity(verbosity_),
        sigma_min(sigma_min_), sigma_max(sigma_max_),
        bl(uv)
      {
      MR_assert(bl.Nrows()<(uint64_t(1)<<32), "too many rows in the MS");
      checkShape(ms_in.shape(), {bl.Nrows()});
      nvis=bl.Nrows();
      if (nvis==0)
        {
        if (gridding) mav_apply([](complex<Timg> &v){v=complex<Timg>(0);}, nthreads, dirty_out);
        return;
        }
      auto kidx = getNuNv();
      MR_assert((nu>>logsquare)<(size_t(1)<<16), "nu too large");
      MR_assert((nv>>logsquare)<(size_t(1)<<16), "nv too large");
      ofactor = min(double(nu)/nxdirty, double(nv)/nydirty);
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
      size_t nrow=bl.Nrows();
      size_t ntiles_u = (nu>>logsquare) + 3;
      size_t ntiles_v = (nv>>logsquare) + 3;
      coord_idx.resize(nrow);
      quick_array<uint32_t> key(nrow);
      execParallel(nrow, nthreads, [&](size_t lo, size_t hi)
        {
        for (size_t i=lo; i<hi; ++i)
          {
          auto tmp = get_uvwidx(bl.baseCoord(i));
          key[i] = tmp.tile_u*ntiles_v + tmp.tile_v;
          }
        });
      bucket_sort2(key, coord_idx, ntiles_u*ntiles_v, nthreads);
      timers.pop();

      report();
      gridding ? x2dirty() : dirty2x();

      if (verbosity>0)
        timers.report(cout);
      }
  };

template<typename Tcalc, typename Tacc, typename Tms, typename Timg, typename Tcoord> class Params3d
  {
  private:
    struct Uvwidx
      { uint16_t tile_u, tile_v, tile_w; };

    struct UVW
      { double u, v, w; };

    class Baselines
      {
      protected:
        const cmav<Tcoord,2> &coord;
        static constexpr double fct=0.5/pi;

      public:
        Baselines(const cmav<Tcoord,2> &coord_)
          : coord(coord_)
          {
          MR_assert(coord_.shape(1)==3, "dimension mismatch");
          }

        UVW baseCoord(size_t row) const
          { return UVW{coord(row,0)*fct, coord(row,1)*fct, coord(row,2)*fct}; }
        void prefetchRow(size_t row) const
          {
          DUCC0_PREFETCH_R(&coord(row,0));
          DUCC0_PREFETCH_R(&coord(row,1));
          DUCC0_PREFETCH_R(&coord(row,2));
          }
        size_t Nrows() const { return coord.shape(0); }
      };

    constexpr static int logsquare=3;
    bool gridding;
    bool forward;
    TimerHierarchy timers;
    const cmav<complex<Tms>,1> &ms_in;
    vmav<complex<Tms>,1> &ms_out;
    const cmav<complex<Timg>,3> &dirty_in;
    vmav<complex<Timg>,3> &dirty_out;
    size_t nxdirty, nydirty, nzdirty;
    double epsilon;
    size_t nthreads;
    size_t verbosity;
    double sigma_min, sigma_max;

    Baselines bl;

    quick_array<uint32_t> coord_idx;

    size_t nvis;

    size_t nu, nv, nw;
    double ofactor;

    shared_ptr<HornerKernel> krn;

    size_t supp, nsafe;
    double ushift, vshift, wshift;
    int maxiu0, maxiv0,maxiw0;

    static_assert(sizeof(Tcalc)<=sizeof(Tacc), "bad type combination");
    static_assert(sizeof(Tms)<=sizeof(Tcalc), "bad type combination");
    static_assert(sizeof(Timg)<=sizeof(Tcalc), "bad type combination");

    void grid2dirty(vmav<complex<Tcalc>,3> &grid, vmav<complex<Timg>,3> &dirty)
      {
      timers.push("FFT");
      checkShape(grid.shape(), {nu,nv,nw});
      vfmav<complex<Tcalc>> fgrid(grid);
      c2c(fgrid, fgrid, {0,1,2}, forward, Tcalc(1), nthreads);
      timers.poppush("grid correction");
      checkShape(dirty.shape(), {nxdirty, nydirty, nzdirty});
      auto cfu = krn->corfunc(nxdirty/2+1, 1./nu, nthreads);
      auto cfv = krn->corfunc(nydirty/2+1, 1./nv, nthreads);
      auto cfw = krn->corfunc(nzdirty/2+1, 1./nw, nthreads);
      execParallel(nxdirty, nthreads, [&](size_t lo, size_t hi)
        {
        for (auto i=lo; i<hi; ++i)
          {
          int icfu = abs(int(nxdirty/2)-int(i));
          size_t i2 = nu-nxdirty/2+i;
          if (i2>=nu) i2-=nu;
          for (size_t j=0; j<nydirty; ++j)
            {
            int icfv = abs(int(nydirty/2)-int(j));
            size_t j2 = nv-nydirty/2+j;
            if (j2>=nv) j2-=nv;
            for (size_t k=0; k<nzdirty; ++k)
              {
              int icfw = abs(int(nzdirty/2)-int(k));
              size_t k2 = nw-nzdirty/2+k;
              if (k2>=nw) k2-=nw;
              dirty(i,j,k) = complex<Timg>(grid(i2,j2,k2)*Timg(cfu[icfu]*cfv[icfv]*cfw[icfw]));
              }
            }
          }
        });
      timers.pop();
      }

    void dirty2grid(const cmav<complex<Timg>,3> &dirty, vmav<complex<Tcalc>,3> &grid)
      {
      timers.push("zeroing grid");
      checkShape(dirty.shape(), {nxdirty, nydirty, nzdirty});
      checkShape(grid.shape(), {nu, nv, nw});
      mav_apply([](complex<Tcalc> &v){v=complex<Tcalc>(0);},nthreads,grid); // quickzero(grid, nthreads);
      timers.poppush("grid correction");
      auto cfu = krn->corfunc(nxdirty/2+1, 1./nu, nthreads);
      auto cfv = krn->corfunc(nydirty/2+1, 1./nv, nthreads);
      auto cfw = krn->corfunc(nzdirty/2+1, 1./nw, nthreads);
      execParallel(nxdirty, nthreads, [&](size_t lo, size_t hi)
        {
        for (auto i=lo; i<hi; ++i)
          {
          int icfu = abs(int(nxdirty/2)-int(i));
          size_t i2 = nu-nxdirty/2+i;
          if (i2>=nu) i2-=nu;
          for (size_t j=0; j<nydirty; ++j)
            {
            int icfv = abs(int(nydirty/2)-int(j));
            size_t j2 = nv-nydirty/2+j;
            if (j2>=nv) j2-=nv;
            for (size_t k=0; k<nzdirty; ++k)
              {
              int icfw = abs(int(nzdirty/2)-int(k));
              size_t k2 = nw-nzdirty/2+k;
              if (k2>=nw) k2-=nw;
              grid(i2,j2,k2) = dirty(i,j,k)*Tcalc(cfu[icfu]*cfv[icfv]*cfw[icfw]);
              }
            }
          }
        });
      timers.poppush("FFT");
      vfmav<complex<Tcalc>> fgrid(grid);
      c2c(fgrid, fgrid, {0,1,2}, forward, Tcalc(1), nthreads);
      timers.pop();
      }

    [[gnu::always_inline]] void getpix(double u_in, double v_in, double w_in, double &u, double &v, double &w, int &iu0, int &iv0, int &iw0) const
      {
      u = (u_in-floor(u_in))*nu;
      iu0 = min(int(u+ushift)-int(nu), maxiu0);
      u -= iu0;
      v = (v_in-floor(v_in))*nv;
      iv0 = min(int(v+vshift)-int(nv), maxiv0);
      v -= iv0;
      w = (w_in-floor(w_in))*nw;
      iw0 = min(int(w+wshift)-int(nw), maxiw0);
      w -= iw0;
      }

    [[gnu::always_inline]] Uvwidx get_uvwidx(const UVW &uvw, size_t lsq2)
      {
      double udum, vdum, wdum;
      int iu0, iv0, iw0;
      getpix(uvw.u, uvw.v, uvw.w, udum, vdum, wdum, iu0, iv0, iw0);
      iu0 = (iu0+nsafe)>>lsq2;
      iv0 = (iv0+nsafe)>>lsq2;
      iw0 = (iw0+nsafe)>>lsq2;
      return Uvwidx{uint16_t(iu0), uint16_t(iv0), uint16_t(iw0)};
      }

    template<size_t supp> class HelperX2g2
      {
      public:
        static constexpr size_t vlen = mysimd<Tacc>::size();
        static constexpr size_t nvec = (supp+vlen-1)/vlen;

      private:
        static constexpr int nsafe = (supp+1)/2;
        static constexpr int su = supp+(1<<logsquare);
        static constexpr int sv = supp+(1<<logsquare);
        static constexpr int sw = supp+(1<<logsquare);
        static constexpr double xsupp=2./supp;
        const Params3d *parent;
        TemplateKernel<supp, mysimd<Tacc>> tkrn;
        vmav<complex<Tcalc>,3> &grid;
        int iu0, iv0, iw0; // start index of the current visibility
        int bu0, bv0, bw0; // start index of the current buffer

        vmav<Tacc,3> bufri;
        Tacc *px0r, *px0i;
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
                grid(idxu,idxv,idxw) += complex<Tcalc>(Tcalc(bufri(iu,2*iv,iw)), Tcalc(bufri(iu,2*iv+1,iw)));
                bufri(iu,2*iv,iw) = bufri(iu,2*iv+1,iw) = 0;
                if (++idxw>=inw) idxw=0;
                }
              if (++idxv>=inv) idxv=0;
              }
            }
            if (++idxu>=inu) idxu=0;
            }
          }

      public:
        Tacc * DUCC0_RESTRICT p0r, * DUCC0_RESTRICT p0i;
        union kbuf {
          Tacc scalar[3*nvec*vlen];
          mysimd<Tacc> simd[3*nvec];
#if defined(_MSC_VER)
          kbuf() {}
#endif
          };
        kbuf buf;

        HelperX2g2(const Params3d *parent_, vmav<complex<Tcalc>,3> &grid_,
          vector<mutex> &locks_)
          : parent(parent_), tkrn(*parent->krn), grid(grid_),
            iu0(-1000000), iv0(-1000000), iw0(-1000000),
            bu0(-1000000), bv0(-1000000), bw0(-1000000),
            bufri({size_t(su+1),size_t(2*sv),size_t(sw)}),
            px0r(bufri.data()), px0i(bufri.data()+sw),
            locks(locks_)
          { checkShape(grid.shape(), {parent->nu,parent->nv,parent->nw}); }
        ~HelperX2g2() { dump(); }

        constexpr int lineJump() const { return 2*sw; }
        constexpr int planeJump() const { return 2*sv*sw; }

        [[gnu::always_inline]] [[gnu::hot]] void prep(const UVW &in)
          {
          double ufrac, vfrac, wfrac;
          auto iu0old = iu0;
          auto iv0old = iv0;
          auto iw0old = iw0;
          parent->getpix(in.u, in.v, in.w, ufrac, vfrac, wfrac, iu0, iv0, iw0);
          auto x0 = -ufrac*2+(supp-1);
          auto y0 = -vfrac*2+(supp-1);
          auto z0 = -wfrac*2+(supp-1);
          tkrn.eval3(Tacc(x0), Tacc(y0), Tacc(z0), &buf.simd[0]);
          if ((iu0==iu0old) && (iv0==iv0old) && (iw0==iw0old)) return;
          if ((iu0<bu0) || (iv0<bv0) || (iw0<bw0)
           || (iu0+int(supp)>bu0+su) || (iv0+int(supp)>bv0+sv) || (iw0+int(supp)>bw0+sw))
            {
            dump();
            bu0=((((iu0+nsafe)>>logsquare)<<logsquare))-nsafe;
            bv0=((((iv0+nsafe)>>logsquare)<<logsquare))-nsafe;
            bw0=((((iw0+nsafe)>>logsquare)<<logsquare))-nsafe;
            }
          auto ofs = (iu0-bu0)*2*sv*sw + (iv0-bv0)*2*sw + (iw0-bw0);
          p0r = px0r+ofs;
          p0i = px0i+ofs;
          }
      };

    template<size_t supp> class HelperG2x2
      {
      public:
        static constexpr size_t vlen = mysimd<Tcalc>::size();
        static constexpr size_t nvec = (supp+vlen-1)/vlen;

      private:
        static constexpr int nsafe = (supp+1)/2;
        static constexpr int su = 2*nsafe+(1<<logsquare);
        static constexpr int sv = 2*nsafe+(1<<logsquare);
        static constexpr int sw = 2*nsafe+(1<<logsquare);
        static constexpr int swvec = sw+vlen-1;
        static constexpr double xsupp=2./supp;
        const Params3d *parent;

        TemplateKernel<supp, mysimd<Tcalc>> tkrn;
        const cmav<complex<Tcalc>,3> &grid;
        int iu0, iv0, iw0; // start index of the current visibility
        int bu0, bv0, bw0; // start index of the current buffer

        vmav<Tcalc,3> bufr, bufi;
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
                bufr(iu,iv,iw) = grid(idxu, idxv, idxw).real();
                bufi(iu,iv,iw) = grid(idxu, idxv, idxw).imag();
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

        HelperG2x2(const Params3d *parent_, const cmav<complex<Tcalc>,3> &grid_)
          : parent(parent_), tkrn(*parent->krn), grid(grid_),
            iu0(-1000000), iv0(-1000000), iw0(-1000000),
            bu0(-1000000), bv0(-1000000), bw0(-1000000),
            bufr({size_t(su),size_t(sv),size_t(swvec)}),
            bufi({size_t(su),size_t(sv),size_t(swvec)}),
            px0r(bufr.data()), px0i(bufi.data())
          { checkShape(grid.shape(), {parent->nu,parent->nv,parent->nw}); }

        constexpr int lineJump() const { return swvec; }
        constexpr int planeJump() const { return sv*swvec; }

        [[gnu::always_inline]] [[gnu::hot]] void prep(const UVW &in)
          {
          double ufrac, vfrac, wfrac;
          auto iu0old = iu0;
          auto iv0old = iv0;
          auto iw0old = iw0;
          parent->getpix(in.u, in.v, in.w, ufrac, vfrac, wfrac, iu0, iv0, iw0);
          auto x0 = -ufrac*2+(supp-1);
          auto y0 = -vfrac*2+(supp-1);
          auto z0 = -wfrac*2+(supp-1);
          tkrn.eval3(Tcalc(x0), Tcalc(y0), Tcalc(z0), &buf.simd[0]);
          if ((iu0==iu0old) && (iv0==iv0old) && (iw0==iw0old)) return;
          if ((iu0<bu0) || (iv0<bv0) || (iw0<bw0)
           || (iu0+int(supp)>bu0+su) || (iv0+int(supp)>bv0+sv) || (iw0+int(supp)>bw0+sw))
            {
            bu0=((((iu0+nsafe)>>logsquare)<<logsquare))-nsafe;
            bv0=((((iv0+nsafe)>>logsquare)<<logsquare))-nsafe;
            bw0=((((iw0+nsafe)>>logsquare)<<logsquare))-nsafe;
            load();
            }
          auto ofs = (iu0-bu0)*sv*swvec + (iv0-bv0)*swvec + (iw0-bw0);
          p0r = px0r+ofs;
          p0i = px0i+ofs;
          }
      };

    template<size_t SUPP> [[gnu::hot]] void x2grid_c_helper
      (size_t supp, vmav<complex<Tcalc>,3> &grid)
      {
      if constexpr (SUPP>=8)
        if (supp<=SUPP/2) return x2grid_c_helper<SUPP/2>(supp, grid);
      if constexpr (SUPP>4)
        if (supp<SUPP) return x2grid_c_helper<SUPP-1>(supp, grid);
      MR_assert(supp==SUPP, "requested support out of range");

      vector<mutex> locks(nu);

      execDynamic(coord_idx.size(), nthreads, 1000, [&](Scheduler &sched)
        {
        constexpr size_t vlen=mysimd<Tcalc>::size();
        constexpr size_t NVEC((SUPP+vlen-1)/vlen);
        HelperX2g2<SUPP> hlp(this, grid, locks);
        constexpr auto ljump = hlp.lineJump();
        constexpr auto pjump = hlp.planeJump();
        const auto * DUCC0_RESTRICT ku = hlp.buf.scalar;
        const auto * DUCC0_RESTRICT kv = hlp.buf.scalar+vlen*NVEC;
        const auto * DUCC0_RESTRICT kw = hlp.buf.simd+2*NVEC;

        constexpr size_t du=16384/(SUPP*ljump*sizeof(Tacc));
        while (auto rng=sched.getNext())
{
for (size_t ustart=0; ustart<SUPP; ustart+=du)
{
size_t ustop = min(SUPP, ustart+du);

        for(auto ix=rng.lo; ix<rng.hi; ++ix)
          {
          constexpr size_t lookahead=3;
          if (ix+lookahead<coord_idx.size())
            {
            auto nextidx = coord_idx[ix+lookahead];
            DUCC0_PREFETCH_R(&ms_in(nextidx));
            bl.prefetchRow(nextidx);
            }
          size_t row = coord_idx[ix];
          auto coord = bl.baseCoord(row);
          hlp.prep(coord);
          auto v(ms_in(row));

          if constexpr (NVEC==1)
            {
            mysimd<Tacc> vr=v.real()*kw[0], vi=v.imag()*kw[0];
            for (size_t cu=ustart; cu<ustop; ++cu)
              {
              mysimd<Tacc> v2r=vr*ku[cu], v2i=vi*ku[cu];
              for (size_t cv=0; cv<SUPP; ++cv)
                {
                auto * DUCC0_RESTRICT pxr = hlp.p0r+cu*pjump+cv*ljump;
                auto * DUCC0_RESTRICT pxi = hlp.p0i+cu*pjump+cv*ljump;
                auto tr = mysimd<Tacc>(pxr,element_aligned_tag());
                auto ti = mysimd<Tacc>(pxi,element_aligned_tag());
                tr += v2r*kv[cv];
                ti += v2i*kv[cv];
                tr.copy_to(pxr,element_aligned_tag());
                ti.copy_to(pxi,element_aligned_tag());
                }
              }
            }
          else
            {
            Tacc vr(v.real()), vi(v.imag());
            auto * DUCC0_RESTRICT pxr = hlp.p0r+ustart*pjump;
            auto * DUCC0_RESTRICT pxi = hlp.p0i+ustart*pjump;
            for (size_t cu=ustart; cu<ustop; ++cu)
              {
              Tacc tmpr=vr*ku[cu], tmpi=vi*ku[cu];
              for (size_t cv=0; cv<SUPP; ++cv)
                {
                Tacc tmp2r=tmpr*kv[cv], tmp2i=tmpi*kv[cv];
                for (size_t cw=0; cw<NVEC; ++cw)
                  {
                  auto tr = mysimd<Tacc>(pxr,element_aligned_tag());
                  tr += tmp2r*kw[cw];
                  tr.copy_to(pxr,element_aligned_tag());
                  auto ti = mysimd<Tacc>(pxi, element_aligned_tag());
                  ti += tmp2i*kw[cw];
                  ti.copy_to(pxi,element_aligned_tag());
                  pxr += hlp.vlen;
                  pxi += hlp.vlen;
                  }
                pxr += ljump-NVEC*hlp.vlen;
                pxi += ljump-NVEC*hlp.vlen;
                }
              pxr += pjump-SUPP*ljump;
              pxi += pjump-SUPP*ljump;
              }
            }
          }
}
}
        });
      }

    template<size_t SUPP> [[gnu::hot]] void grid2x_c_helper
      (size_t supp, const cmav<complex<Tcalc>,3> &grid)
      {
      if constexpr (SUPP>=8)
        if (supp<=SUPP/2) return grid2x_c_helper<SUPP/2>(supp, grid);
      if constexpr (SUPP>4)
        if (supp<SUPP) return grid2x_c_helper<SUPP-1>(supp, grid);
      MR_assert(supp==SUPP, "requested support out of range");

      execDynamic(coord_idx.size(), nthreads, 1000, [&](Scheduler &sched)
        {
        constexpr size_t vlen=mysimd<Tcalc>::size();
        constexpr size_t NVEC((SUPP+vlen-1)/vlen);
        HelperG2x2<SUPP> hlp(this, grid);
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
            DUCC0_PREFETCH_W(&ms_out(nextidx));
            bl.prefetchRow(nextidx);
            }
          size_t row = coord_idx[ix];
          auto coord = bl.baseCoord(row);
          hlp.prep(coord);
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
          ms_out(row) = hsum_cmplx<Tcalc>(rr,ri);
          }
        });
      }

    void report()
      {
      if (verbosity==0) return;
      cout << (gridding ? "Gridding:" : "Degridding:") << endl
           << "  nthreads=" << nthreads << ", "
           << "dirty=(" << nxdirty << "x" << nydirty << "x" << nzdirty << "), "
           << "grid=(" << nu << "x" << nv << "x" << nw;
      cout << "), supp=" << supp
           << ", eps=" << epsilon
           << endl;
      cout << "  npoints=" << bl.Nrows() << endl;
      size_t ovh0 = bl.Nrows()*sizeof(uint32_t);
      size_t ovh1 = nu*nv*nw*sizeof(complex<Tcalc>);             // grid
      if (!gridding)
        ovh1 += nxdirty*nydirty*nzdirty*sizeof(Timg);                 // tdirty
      cout << "  memory overhead: "
           << ovh0/double(1<<30) << "GB (index) + "
           << ovh1/double(1<<30) << "GB (3D arrays)" << endl;
      }

    void x2dirty()
      {
      timers.push("allocating grid");
      auto grid = vmav<complex<Tcalc>,3>::build_noncritical({nu,nv,nw});
      timers.poppush("gridding proper");
      constexpr size_t maxsupp = is_same<Tacc, double>::value ? 16 : 8;
      x2grid_c_helper<maxsupp>(supp, grid);
      timers.pop();
      grid2dirty(grid, dirty_out);
      }

    void dirty2x()
      {
      timers.push("allocating grid");
      auto grid = vmav<complex<Tcalc>,3>::build_noncritical({nu,nv,nw});
      timers.pop();
      dirty2grid(dirty_in, grid);
      timers.push("degridding proper");
      constexpr size_t maxsupp = is_same<Tcalc, double>::value ? 16 : 8;
      grid2x_c_helper<maxsupp>(supp, grid);
      timers.pop();
      }

    auto getNuNvNw()
      {
      timers.push("parameter calculation");

      auto idx = getAvailableKernels<Tcalc>(epsilon, 3, sigma_min, sigma_max);
      double mincost = 1e300;
      constexpr double nref_fft=2048;
      constexpr double costref_fft=0.0693;
      size_t minnu=0, minnv=0, minnw=0, minidx=KernelDB.size();
      size_t vlen = gridding ? mysimd<Tacc>::size() : mysimd<Tcalc>::size();
      for (size_t i=0; i<idx.size(); ++i)
        {
        const auto &krn(KernelDB[idx[i]]);
        auto supp = krn.W;
        auto nvec = (supp+vlen-1)/vlen;
        auto ofactor = krn.ofactor;
        size_t nu=2*good_size_complex(size_t(nxdirty*ofactor*0.5)+1);
        size_t nv=2*good_size_complex(size_t(nydirty*ofactor*0.5)+1);
        size_t nw=2*good_size_complex(size_t(nzdirty*ofactor*0.5)+1);
        double logterm = log(nu*nv*nw)/log(nref_fft*nref_fft);
        double fftcost = nu*nv*nw/(nref_fft*nref_fft)*logterm*costref_fft;
        double gridcost = 2.2e-10*nvis*(supp*supp*nvec*vlen + (3*nvec*(supp+3)*vlen));
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
          minnu=nu;
          minnv=nv;
          minnw=nw;
          minidx = idx[i];
          }
        }
      timers.pop();
      nu = minnu;
      nv = minnv;
      nw = minnw;
      return minidx;
      }

  public:
    Params3d(const cmav<Tcoord,2> &uvw,
           const cmav<complex<Tms>,1> &ms_in_, vmav<complex<Tms>,1> &ms_out_,
           const cmav<complex<Timg>,3> &dirty_in_, vmav<complex<Timg>,3> &dirty_out_,
           double epsilon_, bool forward_,
           size_t nthreads_, size_t verbosity_,
           double sigma_min_,
           double sigma_max_)
      : gridding(ms_out_.size()==0),
        forward(forward_),
        timers(gridding ? "gridding" : "degridding"),
        ms_in(ms_in_), ms_out(ms_out_),
        dirty_in(dirty_in_), dirty_out(dirty_out_),
        nxdirty(gridding ? dirty_out.shape(0) : dirty_in.shape(0)),
        nydirty(gridding ? dirty_out.shape(1) : dirty_in.shape(1)),
        nzdirty(gridding ? dirty_out.shape(2) : dirty_in.shape(2)),
        epsilon(epsilon_),
        nthreads((nthreads_==0) ? get_default_nthreads() : nthreads_),
        verbosity(verbosity_),
        sigma_min(sigma_min_), sigma_max(sigma_max_),
        bl(uvw)
      {
      MR_assert(bl.Nrows()<(uint64_t(1)<<32), "too many rows in the MS");
      checkShape(ms_in.shape(), {bl.Nrows()});
      nvis=bl.Nrows();
      if (nvis==0)
        {
        if (gridding) mav_apply([](complex<Timg> &v){v=complex<Timg>(0);}, nthreads, dirty_out);
        return;
        }
      auto kidx = getNuNvNw();
      MR_assert((nu>>logsquare)<(size_t(1)<<10), "nu too large");
      MR_assert((nv>>logsquare)<(size_t(1)<<10), "nv too large");
      MR_assert((nw>>logsquare)<(size_t(1)<<10), "nw too large");
      ofactor = min(double(nu)/nxdirty, double(nv)/nydirty);
      ofactor = min(ofactor, double(nw)/nzdirty);
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
      size_t nrow=bl.Nrows();
      size_t ntiles_u = (nu>>logsquare) + 3;
      size_t ntiles_v = (nv>>logsquare) + 3;
      size_t ntiles_w = (nw>>logsquare) + 3;
      size_t lsq2 = logsquare;
      while ((lsq2>=1) && (((ntiles_u*ntiles_v*ntiles_w)<<3*(logsquare-lsq2))<(size_t(1)<<28)))
        --lsq2;
      auto ssmall = logsquare-lsq2;
      auto msmall = (size_t(1)<<ssmall) - 1;

      coord_idx.resize(nrow);
      quick_array<uint32_t> key(nrow);
      execParallel(nrow, nthreads, [&](size_t lo, size_t hi)
        {
        for (size_t i=lo; i<hi; ++i)
          {
          auto tmp = get_uvwidx(bl.baseCoord(i),lsq2);
          auto lowkey = ((tmp.tile_u&msmall)<<(2*ssmall))
                      | ((tmp.tile_v&msmall)<<   ssmall)
                      |  (tmp.tile_w&msmall);
          auto hikey = ((tmp.tile_u>>ssmall)*ntiles_v*ntiles_w)
                     + ((tmp.tile_v>>ssmall)*ntiles_w)
                     +  (tmp.tile_w>>ssmall);
          key[i] = (hikey<<(3*ssmall)) | lowkey;
          }
        });
      bucket_sort2(key, coord_idx, (ntiles_u*ntiles_v*ntiles_w)<<(3*ssmall), nthreads);
      timers.pop();

      report();
      gridding ? x2dirty() : dirty2x();

      if (verbosity>0)
        timers.report(cout);
      }
  };

template<typename Tcalc, typename Tacc, typename Tms, typename Timg, typename Tcoord> void nu2u(const cmav<Tcoord,2> &coord,
  const cmav<complex<Tms>,1> &ms, bool forward,
  double epsilon,
  size_t nthreads, vfmav<complex<Timg>> &dirty, size_t verbosity,
  double sigma_min=1.1,
  double sigma_max=2.6)
  {
  auto ndim = dirty.ndim();
  MR_assert((ndim>=1) && (ndim<=3), "transform must be 1D/2D/3D");
  MR_assert(ndim==coord.shape(1), "dimensionality mismatch");
  auto ms_out(vmav<complex<Tms>,1>::build_empty());
  if (ndim==1)
    {
    auto dirty_in(vmav<complex<Timg>,1>::build_empty());
    vmav<complex<Tms>,1> dirty2(dirty);
    Params1d<Tcalc, Tacc, Tms, Timg, Tcoord> par(coord, ms, ms_out, dirty_in, dirty2,
      epsilon, forward, nthreads, verbosity, sigma_min, sigma_max);
    }
  else if (ndim==2)
    {
    auto dirty_in(vmav<complex<Timg>,2>::build_empty());
    vmav<complex<Tms>,2> dirty2(dirty);
    Params2d<Tcalc, Tacc, Tms, Timg, Tcoord> par(coord, ms, ms_out, dirty_in, dirty2,
      epsilon, forward, nthreads, verbosity, sigma_min, sigma_max);
    }
  else if (ndim==3)
    {
    auto dirty_in(vmav<complex<Timg>,3>::build_empty());
    vmav<complex<Tms>,3> dirty2(dirty);
    Params3d<Tcalc, Tacc, Tms, Timg, Tcoord> par(coord, ms, ms_out, dirty_in, dirty2,
      epsilon, forward, nthreads, verbosity, sigma_min, sigma_max);
    }
  }
template<typename Tcalc, typename Tacc, typename Tms, typename Timg, typename Tcoord> void u2nu(const cmav<Tcoord,2> &coord,
  const cfmav<complex<Timg>> &dirty, bool forward,
  double epsilon, size_t nthreads, vmav<complex<Tms>,1> &ms,
  size_t verbosity,
  double sigma_min=1.1, double sigma_max=2.6)
  {
  auto ndim = dirty.ndim();
  MR_assert((ndim>=1) && (ndim<=3), "transform must be 1D/2D/3D");
  MR_assert(ndim==coord.shape(1), "dimensionality mismatch");
  auto ms_in(ms.build_uniform(ms.shape(),complex<Tms>(1.)));
  if (ndim==1)
    {
    auto dirty_out(vmav<complex<Timg>,1>::build_empty());
    Params1d<Tcalc, Tacc, Tms, Timg, Tcoord> par(coord, ms_in, ms, dirty, dirty_out,
      epsilon, forward, nthreads, verbosity, sigma_min, sigma_max);
    }
  else if (ndim==2)
    {
    auto dirty_out(vmav<complex<Timg>,2>::build_empty());
    Params2d<Tcalc, Tacc, Tms, Timg, Tcoord> par(coord, ms_in, ms, dirty, dirty_out,
      epsilon, forward, nthreads, verbosity, sigma_min, sigma_max);
    }
  else if (ndim==3)
    {
    auto dirty_out(vmav<complex<Timg>,3>::build_empty());
    Params3d<Tcalc, Tacc, Tms, Timg, Tcoord> par(coord, ms_in, ms, dirty, dirty_out,
      epsilon, forward, nthreads, verbosity, sigma_min, sigma_max);
    }
  }
} // namespace detail_nufft

// public names
using detail_nufft::u2nu;
using detail_nufft::nu2u;

} // namespace ducc0

#endif
