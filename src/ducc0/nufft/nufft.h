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
#include "ducc0/math/rangeset.h"

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

template<typename T> void complex2hartley
  (const cmav<complex<T>, 2> &grid, vmav<T,2> &grid2, size_t nthreads)
  {
  MR_assert(grid.conformable(grid2), "shape mismatch");
  size_t nu=grid.shape(0), nv=grid.shape(1);

  execParallel(nu, nthreads, [&](size_t lo, size_t hi)
    {
    for(auto u=lo, xu=(u==0) ? 0 : nu-u; u<hi; ++u, xu=nu-u)
      for (size_t v=0, xv=0; v<nv; ++v, xv=nv-v)
#ifdef DUCC0_USE_PROPER_HARTLEY_CONVENTION
        grid2(u,v) = T(0.5)*(grid( u, v).real()-grid( u, v).imag()+
                             grid(xu,xv).real()+grid(xu,xv).imag());
#else
        grid2(u,v) = T(0.5)*(grid( u, v).real()+grid( u, v).imag()+
                             grid(xu,xv).real()-grid(xu,xv).imag());
#endif
    });
  }

template<typename T> void hartley2complex
  (const cmav<T,2> &grid, vmav<complex<T>,2> &grid2, size_t nthreads)
  {
  MR_assert(grid.conformable(grid2), "shape mismatch");
  size_t nu=grid.shape(0), nv=grid.shape(1);

  execParallel(nu, nthreads, [&](size_t lo, size_t hi)
    {
    for(size_t u=lo, xu=(u==0) ? 0 : nu-u; u<hi; ++u, xu=nu-u)
      for (size_t v=0, xv=0; v<nv; ++v, xv=nv-v)
#ifdef DUCC0_USE_PROPER_HARTLEY_CONVENTION
        grid2(u,v) = complex<T>(T(.5)*(grid(u,v)+grid(xu,xv)),
                                T(.5)*(grid(xu,xv)-grid(u,v)));
#else
        grid2(u,v) = complex<T>(T(.5)*(grid(u,v)+grid(xu,xv)),
                                T(.5)*(grid(u,v)-grid(xu,xv)));
#endif
    });
  }

template<typename T> void hartley2_2D(vmav<T,2> &arr, size_t nthreads)
  {
  size_t nu=arr.shape(0), nv=arr.shape(1);
  vfmav<T> farr(arr);
  r2r_separable_hartley(farr, farr, {0,1}, T(1), nthreads);

  execParallel((nu+1)/2-1, nthreads, [&](size_t lo, size_t hi)
    {
    for(auto i=lo+1; i<hi+1; ++i)
      for(size_t j=1; j<(nv+1)/2; ++j)
        {
        T ll = arr(i   ,j   );
        T hl = arr(nu-i,j   );
        T lh = arr(i   ,nv-j);
        T hh = arr(nu-i,nv-j);
        T v = T(0.5)*(ll+lh+hl+hh);
        arr(i   ,j   ) = v-hh;
        arr(nu-i,j   ) = v-lh;
        arr(i   ,nv-j) = v-hl;
        arr(nu-i,nv-j) = v-ll;
        }
     });
  }

class Uvwidx
  {
  public:
    uint16_t tile_u, tile_v, minplane;

    Uvwidx() {}
    Uvwidx(uint16_t tile_u_, uint16_t tile_v_, uint16_t minplane_)
      : tile_u(tile_u_), tile_v(tile_v_), minplane(minplane_) {}

    uint64_t idx() const
      { return (uint64_t(tile_u)<<32) + (uint64_t(tile_v)<<16) + minplane; }
    bool operator!=(const Uvwidx &other) const
      { return idx()!=other.idx(); }
    bool operator<(const Uvwidx &other) const
      { return idx()<other.idx(); }
  };

class RowchanRange
  {
  public:
    uint32_t row;
    uint16_t ch_begin, ch_end;

    RowchanRange() = default;
    RowchanRange(const RowchanRange &) = default;
    RowchanRange &operator=(const RowchanRange &) = default;
    RowchanRange(uint32_t row_, uint16_t ch_begin_, uint16_t ch_end_)
      : row(row_), ch_begin(ch_begin_), ch_end(ch_end_) {}
    uint16_t nchan() const { return ch_end-ch_begin; }
  };

struct UV
  {
  double u, v;
  UV() {}
  UV(double u_, double v_) : u(u_), v(v_) {}
  UV operator* (double fct) const
    { return UV(u*fct, v*fct); }
  };

class Baselines
  {
  protected:
    vector<UV> coord;
    size_t nrows;

  public:
    Baselines() = default;
    template<typename T> Baselines(const cmav<T,2> &coord_, size_t nxdirty, size_t nydirty)
      {
      MR_assert(coord_.shape(1)==2, "dimension mismatch");
      nrows = coord_.shape(0);
      coord.resize(nrows);
      for (size_t i=0; i<coord.size(); ++i)
        coord[i] = UV(coord_(i,0)*nxdirty/(4*pi*pi), coord_(i,1)*nydirty/(4*pi*pi));
      }

    UV effectiveCoord(size_t row, size_t /*chan*/) const
      { return coord[row]; }
    UV baseCoord(size_t row) const
      { return coord[row]; }
    void prefetchRow(size_t row) const
      { DUCC0_PREFETCH_R(&coord[row]); }
    size_t Nrows() const { return nrows; }
  };


template<typename Tcalc, typename Tacc, typename Tms, typename Timg> class Params
  {
  private:
    constexpr static int logsquare=is_same<Tacc,float>::value ? 5 : 4;
    bool gridding;
    TimerHierarchy timers;
    const cmav<complex<Tms>,1> &ms_in;
    vmav<complex<Tms>,1> &ms_out;
    const cmav<Timg,2> &dirty_in;
    vmav<Timg,2> &dirty_out;
    size_t nxdirty, nydirty;
    double pixsize_x, pixsize_y;
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
    vector<rangeset<int>> uranges, vranges;

    static_assert(sizeof(Tcalc)<=sizeof(Tacc), "bad type combination");
    static_assert(sizeof(Tms)<=sizeof(Tcalc), "bad type combination");
    static_assert(sizeof(Timg)<=sizeof(Tcalc), "bad type combination");

    void grid2dirty_post(vmav<Tcalc,2> &tmav, vmav<Timg,2> &dirty) const
      {
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
            dirty(i,j) = Timg(tmav(i2,j2)*cfu[icfu]*cfv[icfv]);
            }
          }
        });
      }

    void grid2dirty_overwrite(vmav<Tcalc,2> &grid, vmav<Timg,2> &dirty)
      {
      timers.push("FFT");
      checkShape(grid.shape(), {nu,nv});
      hartley2_2D(grid, nthreads);
      timers.poppush("grid correction");
      grid2dirty_post(grid, dirty);
      timers.pop();
      }

    void dirty2grid_pre(const cmav<Timg,2> &dirty, vmav<Tcalc,2> &grid)
      {
      timers.push("zeroing grid");
      checkShape(dirty.shape(), {nxdirty, nydirty});
      checkShape(grid.shape(), {nu, nv});
      auto cfu = krn->corfunc(nxdirty/2+1, 1./nu, nthreads);
      auto cfv = krn->corfunc(nydirty/2+1, 1./nv, nthreads);
      // only zero the parts of the grid that are not filled afterwards anyway
      { auto a0 = subarray<2>(grid, {{0,nxdirty/2}, {nydirty/2,nv-nydirty/2+1}}); quickzero(a0, nthreads); }
      { auto a0 = subarray<2>(grid, {{nxdirty/2, nu-nxdirty/2+1}, {}}); quickzero(a0, nthreads); }
      { auto a0 = subarray<2>(grid, {{nu-nxdirty/2+1,MAXIDX}, {nydirty/2, nv-nydirty/2+1}}); quickzero(a0, nthreads); }
      timers.poppush("grid correction");
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
      timers.pop();
      }

    void dirty2grid(const cmav<Timg,2> &dirty, vmav<Tcalc,2> &grid)
      {
      dirty2grid_pre(dirty, grid);
      timers.push("FFT");
      hartley2_2D(grid, nthreads);
      timers.pop();
      }

    [[gnu::always_inline]] void getpix(double u_in, double v_in, double &u, double &v, int &iu0, int &iv0) const
      {
      u = u_in*pixsize_x;
      u = (u-floor(u))*nu;
      iu0 = min(int(u+ushift)-int(nu), maxiu0);
      u -= iu0;
      v = v_in*pixsize_y;
      v = (v-floor(v))*nv;
      iv0 = min(int(v+vshift)-int(nv), maxiv0);
      v -= iv0;
      }

    [[gnu::always_inline]] Uvwidx get_uvwidx(const UV &uv, uint32_t /*ch*/)
      {
      double udum, vdum;
      int iu0, iv0;
      getpix(uv.u, uv.v, udum, vdum, iu0, iv0);
      iu0 = (iu0+nsafe)>>logsquare;
      iv0 = (iv0+nsafe)>>logsquare;
      return Uvwidx(iu0, iv0, 0);
      }

    void countRanges()
      {
      timers.push("building index new");
      size_t nrow=bl.Nrows();
      size_t ntiles_u = (nu>>logsquare) + 3;
      size_t ntiles_v = (nv>>logsquare) + 3;
      coord_idx.resize(nrow);
      quick_array<uint32_t> key(nrow);
      execParallel(nrow, nthreads, [&](size_t lo, size_t hi)
        {
        for (size_t i=lo; i<hi; ++i)
          {
          auto tmp = get_uvwidx(bl.baseCoord(i), 0);
          key[i] = tmp.tile_u*ntiles_v + tmp.tile_v;
          }
        });
      bucket_sort2(key, coord_idx, ntiles_u*ntiles_v, nthreads);
      timers.pop();
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
        const Params *parent;
        TemplateKernel<supp, mysimd<Tacc>> tkrn;
        vmav<complex<Tcalc>,2> &grid;
        int iu0, iv0; // start index of the current visibility
        int bu0, bv0; // start index of the current buffer

        vmav<Tacc,2> bufr, bufi;
        Tacc *px0r, *px0i;
        double w0, xdw;
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

        HelperX2g2(const Params *parent_, vmav<complex<Tcalc>,2> &grid_,
          vector<mutex> &locks_, double w0_=-1, double dw_=-1)
          : parent(parent_), tkrn(*parent->krn), grid(grid_),
            iu0(-1000000), iv0(-1000000),
            bu0(-1000000), bv0(-1000000),
            bufr({size_t(su),size_t(svvec)}),
            bufi({size_t(su),size_t(svvec)}),
            px0r(bufr.data()), px0i(bufi.data()),
            w0(w0_),
            xdw(1./dw_),
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
        const Params *parent;

        TemplateKernel<supp, mysimd<Tcalc>> tkrn;
        const cmav<complex<Tcalc>,2> &grid;
        int iu0, iv0; // start index of the current visibility
        int bu0, bv0; // start index of the current buffer

        vmav<Tcalc,2> bufr, bufi;
        const Tcalc *px0r, *px0i;
        double w0, xdw;

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

        HelperG2x2(const Params *parent_, const cmav<complex<Tcalc>,2> &grid_,
          double w0_=-1, double dw_=-1)
          : parent(parent_), tkrn(*parent->krn), grid(grid_),
            iu0(-1000000), iv0(-1000000),
            bu0(-1000000), bv0(-1000000),
            bufr({size_t(su),size_t(svvec)}),
            bufi({size_t(su),size_t(svvec)}),
            px0r(bufr.data()), px0i(bufi.data()),
            w0(w0_),
            xdw(1./dw_)
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
      MR_assert(supp==SUPP, "requested support ou of range");

      vector<mutex> locks(nu);

      execDynamic(coord_idx.size(), nthreads, 1000, [&](Scheduler &sched)
        {
        constexpr size_t vlen=mysimd<Tcalc>::size();
        constexpr size_t NVEC((SUPP+vlen-1)/vlen);
        HelperX2g2<SUPP> hlp(this, grid, locks);
        constexpr auto jump = hlp.lineJump();
        const auto * DUCC0_RESTRICT ku = hlp.buf.scalar;
        const auto * DUCC0_RESTRICT kv = hlp.buf.simd+NVEC;

        while (auto rng=sched.getNext()) for(auto ix=rng.lo; ix<rng.hi; ++ix)
          {
          if (ix+1<coord_idx.size())
            {
            auto nextidx = coord_idx[ix+1];
            DUCC0_PREFETCH_R(&ms_in(nextidx));
            bl.prefetchRow(nextidx);
            }
          size_t row = coord_idx[ix];
          auto bcoord = bl.baseCoord(row);
          auto imflip = Tcalc(1);
          {
          auto coord = bcoord;
          hlp.prep(coord);
          auto v(ms_in(row));

          if constexpr (NVEC==1)
            {
            mysimd<Tacc> vr=v.real()*kv[0], vi=v.imag()*imflip*kv[0];
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
            mysimd<Tacc> vr(v.real()), vi(v.imag()*imflip);
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
          }
        });
      }

    void x2grid_c(vmav<complex<Tcalc>,2> &grid)
      {
      checkShape(grid.shape(), {nu, nv});
      constexpr size_t maxsupp = is_same<Tacc, double>::value ? 16 : 8;
      x2grid_c_helper<maxsupp>(supp, grid);
      }

    template<size_t SUPP> [[gnu::hot]] void grid2x_c_helper
      (size_t supp, const cmav<complex<Tcalc>,2> &grid)
      {
      if constexpr (SUPP>=8)
        if (supp<=SUPP/2) return grid2x_c_helper<SUPP/2>(supp, grid);
      if constexpr (SUPP>4)
        if (supp<SUPP) return grid2x_c_helper<SUPP-1>(supp, grid);
      MR_assert(supp==SUPP, "requested support ou of range");

      execDynamic(coord_idx.size(), nthreads, 1000, [&](Scheduler &sched)
        {
        constexpr size_t vlen=mysimd<Tcalc>::size();
        constexpr size_t NVEC((SUPP+vlen-1)/vlen);
        HelperG2x2<SUPP> hlp(this, grid);
        constexpr int jump = hlp.lineJump();
        const auto * DUCC0_RESTRICT ku = hlp.buf.scalar;
        const auto * DUCC0_RESTRICT kv = hlp.buf.simd+NVEC;

        while (auto rng=sched.getNext()) for(auto ix=rng.lo; ix<rng.hi; ++ix)
          {
          if (ix+1<coord_idx.size())
            {
            auto nextidx = coord_idx[ix+1];
            DUCC0_PREFETCH_R(&ms_out(nextidx));
            DUCC0_PREFETCH_W(&ms_out(nextidx));
            bl.prefetchRow(nextidx);
            }
          size_t row = coord_idx[ix];
          auto bcoord = bl.baseCoord(row);
          auto imflip = Tcalc(1);
          {
          auto coord = bcoord;
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
          ri *= imflip;
          ms_out(row) = hsum_cmplx<Tcalc>(rr,ri);
          }
          }
        });
      }

    void grid2x_c(const cmav<complex<Tcalc>,2> &grid)
      {
      checkShape(grid.shape(), {nu, nv});
      constexpr size_t maxsupp = is_same<Tcalc, double>::value ? 16 : 8;
      grid2x_c_helper<maxsupp>(supp, grid);
      }

    void report()
      {
      if (verbosity==0) return;
      cout << (gridding ? "Gridding:" : "Degridding:") << endl
           << "  nthreads=" << nthreads << ", "
           << "dirty=(" << nxdirty << "x" << nydirty << "), "
           << "grid=(" << nu << "x" << nv;
      cout << "), supp=" << supp
           << ", eps=" << (epsilon * 2)
           << endl;
      cout << "  nrow=" << bl.Nrows()
           << ", nvis=" << nvis << "/" << bl.Nrows() << endl;
      size_t ovh0 = bl.Nrows()*sizeof(uint32_t);
      size_t ovh1 = nu*nv*sizeof(complex<Tcalc>);             // grid
      ovh1 += nu*nv*sizeof(Tcalc);                            // rgrid
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
      x2grid_c(grid);
      timers.poppush("allocating rgrid");
      auto rgrid = vmav<Tcalc,2>::build_noncritical(grid.shape(), UNINITIALIZED);
      timers.poppush("complex2hartley");
      complex2hartley(grid, rgrid, nthreads);
      timers.pop();
      grid2dirty_overwrite(rgrid, dirty_out);
      }

    void dirty2x()
      {
      timers.push("allocating grid");
      auto rgrid = vmav<Tcalc,2>::build_noncritical({nu,nv}, UNINITIALIZED);
      timers.pop();
      dirty2grid(dirty_in, rgrid);
      timers.push("allocating grid");
      auto grid = vmav<complex<Tcalc>,2>::build_noncritical(rgrid.shape());
      timers.poppush("hartley2complex");
      hartley2complex(rgrid, grid, nthreads);
      timers.poppush("degridding proper");
      grid2x_c(grid);
      timers.pop();
      }

    auto getNuNv()
      {
      timers.push("parameter calculation");

      auto idx = getAvailableKernels<Tcalc>(epsilon, sigma_min, sigma_max);
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
        double gridcost = 2.2e-10*nvis*(supp*nvec*vlen + ((2*nvec+1)*(supp+3)*vlen));
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

    void scanData()
      {
      timers.push("Initial scan");
      checkShape(ms_in.shape(), {bl.Nrows()});
      nvis=bl.Nrows();
      timers.pop();
      }

  public:
    Params(const cmav<double,2> &uv,
           const cmav<complex<Tms>,1> &ms_in_, vmav<complex<Tms>,1> &ms_out_,
           const cmav<Timg,2> &dirty_in_, vmav<Timg,2> &dirty_out_,
           double epsilon_,
           size_t nthreads_, size_t verbosity_,
           double sigma_min_,
           double sigma_max_)
      : gridding(ms_out_.size()==0),
        timers(gridding ? "gridding" : "degridding"),
        ms_in(ms_in_), ms_out(ms_out_),
        dirty_in(dirty_in_), dirty_out(dirty_out_),
        nxdirty(gridding ? dirty_out.shape(0) : dirty_in.shape(0)),
        nydirty(gridding ? dirty_out.shape(1) : dirty_in.shape(1)),
        pixsize_x(2*pi/nxdirty), pixsize_y(2*pi/nydirty),
        epsilon(epsilon_),
        nthreads((nthreads_==0) ? get_default_nthreads() : nthreads_),
        verbosity(verbosity_),
        sigma_min(sigma_min_), sigma_max(sigma_max_)
      {
      timers.push("Baseline construction");
      bl = Baselines(uv, nxdirty, nydirty);
      MR_assert(bl.Nrows()<(uint64_t(1)<<32), "too many rows in the MS");
      timers.pop();
      // adjust for increased error when gridding in 2 dimensions
      epsilon /= 2;
      scanData();
      if (nvis==0)
        {
        if (gridding) mav_apply([](Timg &v){v=Timg(0);}, nthreads, dirty_out);
        return;
        }
      auto kidx = getNuNv();
      MR_assert((nu>>logsquare)<(size_t(1)<<16), "nu too large");
      MR_assert((nv>>logsquare)<(size_t(1)<<16), "nv too large");
      ofactor = min(double(nu)/nxdirty, double(nv)/nydirty);
      krn = selectKernel<Tcalc>(ofactor, epsilon, kidx);
      supp = krn->support();
      nsafe = (supp+1)/2;
      ushift = supp*(-0.5)+1+nu;
      vshift = supp*(-0.5)+1+nv;
      maxiu0 = (nu+nsafe)-supp;
      maxiv0 = (nv+nsafe)-supp;
      MR_assert(nu>=2*nsafe, "nu too small");
      MR_assert(nv>=2*nsafe, "nv too small");
      MR_assert((nxdirty&1)==0, "nx_dirty must be even");
      MR_assert((nydirty&1)==0, "ny_dirty must be even");
      MR_assert((nu&1)==0, "nu must be even");
      MR_assert((nv&1)==0, "nv must be even");
      MR_assert(epsilon>0, "epsilon must be positive");
      MR_assert(pixsize_x>0, "pixsize_x must be positive");
      MR_assert(pixsize_y>0, "pixsize_y must be positive");
      countRanges();
      report();
      gridding ? x2dirty() : dirty2x();

      if (verbosity>0)
        timers.report(cout);
      }
  };

template<typename Tcalc, typename Tacc, typename Tms, typename Timg> void ms2dirty_nufft(const cmav<double,2> &uv,
  const cmav<complex<Tms>,1> &ms,
  double epsilon,
  size_t nthreads, vmav<Timg,2> &dirty, size_t verbosity,
  double sigma_min=1.1,
  double sigma_max=2.6)
  {
  auto ms_out(vmav<complex<Tms>,1>::build_empty());
  auto dirty_in(vmav<Timg,2>::build_empty());
  Params<Tcalc, Tacc, Tms, Timg> par(uv, ms, ms_out, dirty_in, dirty,
    epsilon, nthreads, verbosity, sigma_min, sigma_max);
  }

template<typename Tcalc, typename Tacc, typename Tms, typename Timg> void dirty2ms_nufft(const cmav<double,2> &uv,
  const cmav<Timg,2> &dirty,
  double epsilon, size_t nthreads, vmav<complex<Tms>,1> &ms,
  size_t verbosity,
  double sigma_min=1.1, double sigma_max=2.6)
  {
  if (ms.size()==0) return;  // nothing to do
  auto ms_in(ms.build_uniform(ms.shape(),1.));
  auto dirty_out(vmav<Timg,2>::build_empty());
  Params<Tcalc, Tacc, Tms, Timg> par(uv, ms_in, ms, dirty, dirty_out,
    epsilon, nthreads, verbosity, sigma_min, sigma_max);
  }

} // namespace detail_nufft

// public names
using detail_nufft::ms2dirty_nufft;
using detail_nufft::dirty2ms_nufft;

} // namespace ducc0

#endif
