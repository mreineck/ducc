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

/* Copyright (C) 2019-2026 Max-Planck-Society
   Author: Martin Reinecke */

#ifndef DUCC0_WGRIDDER_IMPL_H
#define DUCC0_WGRIDDER_IMPL_H

#include <cstring>
#include <complex>
#include <cstdint>
#include <functional>
#include <map>
#include <type_traits>
#include <utility>
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
#include "ducc0/math/gridding_kernel.h"
#include "ducc0/math/rangeset.h"
#include "ducc0/wgridder/wgridder.h"

namespace ducc0 {

namespace detail_gridder {

using namespace std;

template<typename T> constexpr inline int mysimdlen
  = min<int>(8, native_simd<T>::size());

template<typename T> using mysimd = typename simd_select<T,mysimdlen<T>>::type;

template<typename T> T sqr(T val) { return val*val; }

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

template<typename T, typename F> [[gnu::hot]] void expi(vector<complex<T>> &res, vector<T> &buf, F getang)
  {
  using Tsimd = native_simd<T>;
  static constexpr auto vlen = Tsimd::size();
  auto n=res.size();
  for (size_t j=0; j<n; ++j)
    buf[j] = getang(j);
  size_t i=0;
  for (; i+vlen-1<n; i+=vlen)
    {
    auto vang = loadu<Tsimd>(&buf[i]);
    auto vcos = cos(vang);
    auto vsin = sin(vang);
    for (size_t ii=0; ii<vlen; ++ii)
      res[i+ii] = complex<T>(vcos[ii], vsin[ii]);
    }
  for (; i<n; ++i)
    res[i] = complex<T>(cos(buf[i]), sin(buf[i]));
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
  (const cmav<complex<T>, 2> &grid, const vmav<T,2> &grid2, size_t nthreads)
  {
  MR_assert(grid.conformable(grid2), "shape mismatch");
  size_t nu=grid.shape(0), nv=grid.shape(1);

  execParallel(nu, nthreads, [&](size_t lo, size_t hi)
    {
    for(auto u=lo, xu=(u==0) ? 0 : nu-u; u<hi; ++u, xu=nu-u)
      for (size_t v=0, xv=0; v<nv; ++v, xv=nv-v)
        grid2(u,v) = T(0.5)*(grid( u, v).real()-grid( u, v).imag()+
                             grid(xu,xv).real()+grid(xu,xv).imag());
    });
  }

template<typename T> void hartley2complex
  (const cmav<T,2> &grid, const vmav<complex<T>,2> &grid2, size_t nthreads)
  {
  MR_assert(grid.conformable(grid2), "shape mismatch");
  size_t nu=grid.shape(0), nv=grid.shape(1);

  execParallel(nu, nthreads, [&](size_t lo, size_t hi)
    {
    for(size_t u=lo, xu=(u==0) ? 0 : nu-u; u<hi; ++u, xu=nu-u)
      for (size_t v=0, xv=0; v<nv; ++v, xv=nv-v)
        grid2(u,v) = complex<T>(T(.5)*(grid(u,v)+grid(xu,xv)),
                                T(.5)*(grid(xu,xv)-grid(u,v)));
    });
  }

template<typename T> void hartley2_2D(const vmav<T,2> &arr, size_t vlim,
  bool first_fast, size_t nthreads)
  {
  size_t nu=arr.shape(0), nv=arr.shape(1);
  vfmav<T> farr(arr);
  if (2*vlim<nv)
    {
    if (!first_fast)
      r2r_separable_fht(farr, farr, {1}, T(1), nthreads);
    auto flo = subarray(farr, {{}, {0,vlim}});
    r2r_separable_fht(flo, flo, {0}, T(1), nthreads);
    auto fhi = subarray(farr, {{}, {farr.shape(1)-vlim, MAXIDX}});
    r2r_separable_fht(fhi, fhi, {0}, T(1), nthreads);
    if (first_fast)
      r2r_separable_fht(farr, farr, {1}, T(1), nthreads);
    }
  else
    r2r_separable_fht(farr, farr, {0,1}, T(1), nthreads);

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

struct UVW
  {
  double u, v, w;
  UVW() {}
  UVW(double u_, double v_, double w_) : u(u_), v(v_), w(w_) {}
  UVW operator* (double fct) const
    { return UVW(u*fct, v*fct, w*fct); }
  void Flip() { u=-u; v=-v; w=-w; }
  double FixW()
    {
    double res=1.-2.*(w<0);
    u*=res; v*=res; w*=res;
    return res;
    }
  };

class Baselines
  {
  protected:
    vector<UVW> coord;
    size_t nfreqs = 0;   // if 0, data set is "ragged", and we need
                         // more complicated address calculations
    vector<size_t> id;        // only allocated if nfreqs == 0
    vector<size_t> ms_ofs;    // only allocated if nfreqs == 0
    vector<size_t> freq_ofs;
    vector<double> f_over_c;
    double umax, vmax;
public:
  double r_l, r_m;

  public:
    Baselines() = default;
    template<typename T> Baselines(
      const cmav<T,2> &coord_,
      const cmav<size_t,1> &freqlist_id,
      const cmav<size_t,1> &freqlist_nfreqs,
      const cmav<double,1> &freqlist_freqs,
      bool flip_u=false, bool flip_v=false, bool flip_w=false, double r_l_=0, double r_m_=0)
      : r_l(r_l_), r_m(r_m_)
      {
      size_t nrows = coord_.shape(0);
      constexpr double speedOfLight = 299792458.;
      MR_assert(coord_.shape(1)==3, "dimension mismatch");
      MR_assert(freqlist_id.shape(0)==nrows, "freqlist_id dimension mismatch");
      size_t max_id = 0;
      for (size_t i=0; i<nrows; ++i)
        max_id = max(max_id, freqlist_id(i));
      MR_assert(max_id+1<=freqlist_nfreqs.shape(0), "freqlist_nfreqs array is too small");
      freq_ofs.resize(max_id+2);
      freq_ofs[0] = 0;
      for (size_t i=0; i<=max_id; ++i)
        freq_ofs[i+1] = freq_ofs[i] + freqlist_nfreqs(i);
      MR_assert(freqlist_freqs.shape(0)>=freq_ofs.back(), "freqlist_freqs array is too small");
      if (max_id==0)  // simple case, we only have one frequency list
        {
        nfreqs = freqlist_nfreqs(0);
        }
      else
        {
        id.resize(nrows);
        for (size_t i=0; i<nrows; ++i)
          id[i] = freqlist_id(i);
        ms_ofs.resize(nrows+1);
        ms_ofs[0] = 0;
        for (size_t i=0; i<nrows; ++i)
          ms_ofs[i+1] = ms_ofs[i] + freqlist_nfreqs(id[i]);
        }

      f_over_c.resize(freq_ofs.back());
      double fcmax = 0;
      for (size_t i=0; i<=max_id; ++i)
        {
        MR_assert(freqlist_freqs(freq_ofs[i])>0, "negative channel frequency encountered");
        for (size_t j=freq_ofs[i]; j<freq_ofs[i+1]; ++j)
          {
          if (j>freq_ofs[i])
            MR_assert(freqlist_freqs(j)>=freqlist_freqs(j-1),
              "channel frequencies must be sorted in ascending order");
          f_over_c[j] = freqlist_freqs(j)/speedOfLight;
          fcmax = max(fcmax, f_over_c[j]);
          }
        }
      coord.resize(nrows);
      double ufac = flip_u ? -1 : 1;
      double vfac = flip_v ? -1 : 1;
      double wfac = flip_w ? -1 : 1;
      umax=vmax=0;
      for (size_t i=0; i<coord.size(); ++i)
        {
double tu = ufac*coord_(i,0);
double tv = vfac*coord_(i,1);
double tw = wfac*coord_(i,2);
        coord[i] = UVW(tu-r_l*tw, tv-r_m*tw, tw);
        umax = max(umax, abs(coord_(i,0)));
        vmax = max(vmax, abs(coord_(i,1)));
        }
      umax *= fcmax;
      vmax *= fcmax;
      }

    size_t ofs_ms(size_t irow) const
      { return nfreqs==0 ? ms_ofs[irow] : irow*nfreqs; }
    size_t ofs_ms(size_t irow, size_t ichan) const
      { return ofs_ms(irow) + ichan; }
  private:
    size_t ofs_freq(size_t irow) const
      { return nfreqs==0 ? freq_ofs[id[irow]] : 0; }
    size_t ofs_freq(size_t irow, size_t ichan) const
      { return ofs_freq(irow) + ichan; }

  public:
    double ffact(size_t irow, size_t ichan) const
      { return f_over_c[ofs_freq(irow, ichan)]; }
    UVW effectiveCoord(size_t irow, size_t ichan) const
      { return coord[irow]*ffact(irow,ichan); }
    double absEffectiveW(size_t irow, size_t ichan) const
      { return abs(coord[irow].w*ffact(irow,ichan)); }
    UVW baseCoord(size_t irow) const
      { return coord[irow]; }
    void prefetchRow(size_t irow) const
      {
      DUCC0_PREFETCH_R(&coord[irow]);
      if (nfreqs==0) DUCC0_PREFETCH_R(&id[irow]);
      } // FIXME: prefetch channels?
    size_t Nrows() const { return coord.size(); }
    size_t Nchannels(size_t irow) const { return nfreqs==0 ? freq_ofs[id[irow]+1]-freq_ofs[id[irow]] : nfreqs; }
    double Umax() const { return umax; }
    double Vmax() const { return vmax; }
    size_t Nvis() const { return nfreqs==0 ? ms_ofs.back() : Nrows()*nfreqs; }
    bool BDA() const { return nfreqs==0; }
  };


template<typename Tcalc, typename Tacc, typename Tms, typename Timg,
  typename Tms_in=cmav<complex<Tms>,1>, typename Tms2d_in=cmav<complex<Tms>,2>> class Wgridder
  {
  private:
    constexpr static int log2tile=is_same<Tacc,float>::value ? 5 : 4;
    bool gridding;
    bool bda;
    TimerHierarchy timers;
    const Tms_in *ms_in=nullptr;
    const Tms2d_in *ms2d_in=nullptr;
    const vmav<complex<Tms>,1> *ms_out=nullptr;
    const vmav<complex<Tms>,2> *ms2d_out=nullptr;
    const cmav<Timg,2> &dirty_in;
    const vmav<Timg,2> &dirty_out;
    const cmav<Tms,1> *wgt=nullptr;
    const cmav<Tms,2> *wgt2d=nullptr;
    const cmav<uint8_t,1> *mask=nullptr;
    const cmav<uint8_t,2> *mask2d=nullptr;
    vmav<uint8_t,1> lmask;
    double pixsize_x, pixsize_y;
    size_t nxdirty, nydirty;
    double epsilon;
    bool do_wgridding;
    size_t nthreads;
    size_t verbosity;
    bool divide_by_n;
    double sigma_min, sigma_max;

    Baselines bl;
    vector<RowchanRange> ranges;
    vector<pair<Uvwidx, size_t>> blockstart;

    double wmin_d, wmax_d;
    size_t nvis;
    double wmin, dw, xdw, wshift;
    size_t nplanes;
    double nm1min, nm1max;

    double lshift, mshift, nshift;
    bool shifting, lmshift, no_nshift;

    size_t nu, nv;
    double ofactor;

    shared_ptr<PolynomialKernel> krn;

    size_t supp, nsafe;
    double ushift, vshift;
    int maxiu0, maxiv0;
    size_t vlim;
    bool uv_side_fast;
    vector<rangeset<int>> uranges, vranges;

    static_assert(sizeof(Tcalc)<=sizeof(Tacc), "bad type combination");
    static_assert(sizeof(Tms)<=sizeof(Tcalc), "bad type combination");
    static_assert(sizeof(Timg)<=sizeof(Tcalc), "bad type combination");

    double phase(double xsq, double ysq, double w, bool adjoint, double nshift, double x, double y)
      {
      double tmp = 1.-xsq-ysq;
      // more accurate form of sqrt(1-xsq-ysq)-1 for nm1 close to zero
      double nm1 = (tmp>=0) ? (-xsq-ysq)/(sqrt(tmp)+1) : -sqrt(-tmp)-1;
// re-centering
nm1 += bl.r_l*x + bl.r_m*y;
      double phs = w*(nm1+nshift);

      if (adjoint) phs *= -1;
      if constexpr (is_same<Tcalc, double>::value)
        return twopi*phs;
      // we are reducing accuracy, so let's better do range reduction first
      return twopi*(phs-floor(phs));
      }

    void grid2dirty_post(const vmav<Tcalc,2> &tmav, const vmav<Timg,2> &dirty) const
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
    void grid2dirty_post2(const vmav<complex<Tcalc>,2> &tmav, const vmav<Timg,2> &dirty, double w)
      {
      timers.push("wscreen+grid correction");
      checkShape(dirty.shape(), {nxdirty,nydirty});
      double x0 = lshift-0.5*nxdirty*pixsize_x,
             y0 = mshift-0.5*nydirty*pixsize_y;
      size_t nxd = lmshift ? nxdirty : (nxdirty/2+1);
      execParallel(nxd, nthreads, [&](size_t lo, size_t hi)
        {
        vector<complex<Tcalc>> phases(lmshift ? nydirty : (nydirty/2+1));
        vector<Tcalc> buf(lmshift ? nydirty : (nydirty/2+1));
        for (auto i=lo; i<hi; ++i)
          {
double x = x0+i*pixsize_x;
          double xsq = sqr(x0+i*pixsize_x);
          size_t ix = nu-nxdirty/2+i;
          if (ix>=nu) ix-=nu;
          expi(phases, buf, [&](size_t i)
            { return Tcalc(phase(xsq, sqr(y0+i*pixsize_y), w, true, nshift, x, y0+i*pixsize_y)); });
          if (lmshift)
            for (size_t j=0, jx=nv-nydirty/2; j<nydirty; ++j, jx=(jx+1>=nv)? jx+1-nv : jx+1)
              {
              dirty(i,j) += Timg(tmav(ix,jx).real()*phases[j].real()
                               - tmav(ix,jx).imag()*phases[j].imag());
              tmav(ix,jx) = complex<Tcalc>(0);
              }
          else
            {
            size_t i2 = nxdirty-i;
            size_t ix2 = nu-nxdirty/2+i2;
            if (ix2>=nu) ix2-=nu;
            if ((i>0)&&(i<i2))
              for (size_t j=0, jx=nv-nydirty/2; j<nydirty; ++j, jx=(jx+1>=nv)? jx+1-nv : jx+1)
                {
                size_t j2 = min(j, nydirty-j);
                Tcalc re = phases[j2].real(), im = phases[j2].imag();
                dirty(i ,j) += Timg(tmav(ix ,jx).real()*re - tmav(ix ,jx).imag()*im);
                dirty(i2,j) += Timg(tmav(ix2,jx).real()*re - tmav(ix2,jx).imag()*im);
                tmav(ix,jx) = tmav(ix2,jx) = complex<Tcalc>(0);
                }
            else
              for (size_t j=0, jx=nv-nydirty/2; j<nydirty; ++j, jx=(jx+1>=nv)? jx+1-nv : jx+1)
                {
                size_t j2 = min(j, nydirty-j);
                Tcalc re = phases[j2].real(), im = phases[j2].imag();
                dirty(i,j) += Timg(tmav(ix,jx).real()*re - tmav(ix,jx).imag()*im); // lower left
                tmav(ix,jx) = complex<Tcalc>(0);
                }
            }
          }
        });
      timers.poppush("zeroing grid");
      // only zero the parts of the grid that have not been zeroed before
      { auto a0 = subarray<2>(tmav, {{0,nxdirty/2}, {nydirty/2,nv-nydirty/2}}); quickzero(a0, nthreads); }
      { auto a0 = subarray<2>(tmav, {{nxdirty/2, nu-nxdirty/2}, {}}); quickzero(a0, nthreads); }
      { auto a0 = subarray<2>(tmav, {{nu-nxdirty/2,MAXIDX}, {nydirty/2, nv-nydirty/2}}); quickzero(a0, nthreads); }
      timers.pop();
      }

    void grid2dirty_overwrite(const vmav<Tcalc,2> &grid, const vmav<Timg,2> &dirty)
      {
      timers.push("FFT");
      checkShape(grid.shape(), {nu,nv});
      hartley2_2D(grid, vlim, uv_side_fast, nthreads);
      timers.poppush("grid correction");
      grid2dirty_post(grid, dirty);
      timers.pop();
      }

    void grid2dirty_c_overwrite_wscreen_add
      (const vmav<complex<Tcalc>,2> &grid, const vmav<Timg,2> &dirty, double w, size_t iplane)
      {
      timers.push("FFT");
      checkShape(grid.shape(), {nu,nv});
      vfmav<complex<Tcalc>> inout(grid);

      const auto &rsu(uranges[iplane]);
      const auto &rsv(vranges[iplane]);
      auto cost_ufirst = nxdirty*log(nv)*nv + rsv.nval()*log(nu)*nu;
      auto cost_vfirst = nydirty*log(nu)*nu + rsu.nval()*log(nv)*nv;
      if (cost_ufirst<cost_vfirst)
        {
        for (size_t i=0; i<rsv.nranges(); ++i)
          {
          auto inout_tmp = inout.subarray({{},{size_t(rsv.ivbegin(i)), size_t(rsv.ivend(i))}});
          c2c(inout_tmp, inout_tmp, {0}, BACKWARD, Tcalc(1), nthreads);
          }
        auto inout_lo = inout.subarray({{0,nxdirty/2},{}});
        c2c(inout_lo, inout_lo, {1}, BACKWARD, Tcalc(1), nthreads);
        auto inout_hi = inout.subarray({{inout.shape(0)-nxdirty/2, MAXIDX},{}});
        c2c(inout_hi, inout_hi, {1}, BACKWARD, Tcalc(1), nthreads);
        }
      else
        {
        for (size_t i=0; i<rsu.nranges(); ++i)
          {
          auto inout_tmp = inout.subarray({{size_t(rsu.ivbegin(i)), size_t(rsu.ivend(i))}, {}});
          c2c(inout_tmp, inout_tmp, {1}, BACKWARD, Tcalc(1), nthreads);
          }
        auto inout_lo = inout.subarray({{}, {0,nydirty/2}});
        c2c(inout_lo, inout_lo, {0}, BACKWARD, Tcalc(1), nthreads);
        auto inout_hi = inout.subarray({{},{inout.shape(1)-nydirty/2, MAXIDX}});
        c2c(inout_hi, inout_hi, {0}, BACKWARD, Tcalc(1), nthreads);
        }

      timers.pop();
      grid2dirty_post2(grid, dirty, w);
      }

    void dirty2grid_pre(const cmav<Timg,2> &dirty, const vmav<Tcalc,2> &grid)
      {
      timers.push("zeroing grid");
      checkShape(grid.shape(), {nu, nv});
      // only zero the parts of the grid that are not filled afterwards anyway
      { auto a0 = subarray<2>(grid, {{0,nxdirty/2}, {nydirty/2,nv-nydirty/2}}); quickzero(a0, nthreads); }
      { auto a0 = subarray<2>(grid, {{nxdirty/2, nu-nxdirty/2}, {}}); quickzero(a0, nthreads); }
      { auto a0 = subarray<2>(grid, {{nu-nxdirty/2,MAXIDX}, {nydirty/2, nv-nydirty/2}}); quickzero(a0, nthreads); }
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
            grid(i2,j2) = dirty(i,j)*Tcalc(cfu[icfu]*cfv[icfv]);
            }
          }
        });
      timers.pop();
      }
    void dirty2grid_pre2(const cmav<Timg,2> &dirty, const vmav<complex<Tcalc>,2> &grid, double w)
      {
      timers.push("zeroing grid");
      checkShape(dirty.shape(), {nxdirty, nydirty});
      checkShape(grid.shape(), {nu, nv});
      // only zero the parts of the grid that are not filled afterwards anyway
      { auto a0 = subarray<2>(grid, {{0,nxdirty/2}, {nydirty/2, nv-nydirty/2}}); quickzero(a0, nthreads); }
      { auto a0 = subarray<2>(grid, {{nxdirty/2,nu-nxdirty/2}, {}}); quickzero(a0, nthreads); }
      { auto a0 = subarray<2>(grid, {{nu-nxdirty/2,MAXIDX}, {nydirty/2,nv-nydirty/2}}); quickzero(a0, nthreads); }
      timers.poppush("wscreen+grid correction");
      double x0 = lshift-0.5*nxdirty*pixsize_x,
             y0 = mshift-0.5*nydirty*pixsize_y;
      size_t nxd = lmshift ? nxdirty : (nxdirty/2+1);
      execParallel(nxd, nthreads, [&](size_t lo, size_t hi)
        {
        vector<complex<Tcalc>> phases(lmshift ? nydirty : (nydirty/2+1));
        vector<Tcalc> buf(lmshift ? nydirty : (nydirty/2+1));
        for(auto i=lo; i<hi; ++i)
          {
double x = x0+i*pixsize_x;
          double xsq = sqr(x0+i*pixsize_x);
          size_t ix = nu-nxdirty/2+i;
          if (ix>=nu) ix-=nu;
          expi(phases, buf, [&](size_t i)
            { return Tcalc(phase(xsq, sqr(y0+i*pixsize_y), w, false, nshift, x, y0+i*pixsize_y)); });
          if (lmshift)
            for (size_t j=0, jx=nv-nydirty/2; j<nydirty; ++j, jx=(jx+1>=nv)? jx+1-nv : jx+1)
              grid(ix,jx) = Tcalc(dirty(i,j))*phases[j];
          else
            {
            size_t i2 = nxdirty-i;
            size_t ix2 = nu-nxdirty/2+i2;
            if (ix2>=nu) ix2-=nu;
            if ((i>0)&&(i<i2))
              for (size_t j=0, jx=nv-nydirty/2; j<nydirty; ++j, jx=(jx+1>=nv)? jx+1-nv : jx+1)
                {
                size_t j2 = min(j, nydirty-j);
                grid(ix ,jx) = Tcalc(dirty(i ,j))*phases[j2]; // lower left
                grid(ix2,jx) = Tcalc(dirty(i2,j))*phases[j2]; // lower right
                }
            else
              for (size_t j=0, jx=nv-nydirty/2; j<nydirty; ++j, jx=(jx+1>=nv)? jx+1-nv : jx+1)
                grid(ix,jx) = Tcalc(dirty(i,j))*phases[min(j, nydirty-j)]; // lower left
            }
          }
        });
      timers.pop();
      }

    void dirty2grid(const cmav<Timg,2> &dirty, const vmav<Tcalc,2> &grid)
      {
      dirty2grid_pre(dirty, grid);
      timers.push("FFT");
      hartley2_2D(grid, vlim, !uv_side_fast, nthreads);
      timers.pop();
      }

    void dirty2grid_c_wscreen(const cmav<Timg,2> &dirty,
      const vmav<complex<Tcalc>,2> &grid, double w, size_t iplane)
      {
      dirty2grid_pre2(dirty, grid, w);
      timers.push("FFT");
      vfmav<complex<Tcalc>> inout(grid);

      const auto &rsu(uranges[iplane]);
      const auto &rsv(vranges[iplane]);
      auto cost_ufirst = nydirty*log(nu)*nu + rsu.nval()*log(nv)*nv;
      auto cost_vfirst = nxdirty*log(nv)*nv + rsv.nval()*log(nu)*nu;
      if (cost_ufirst<cost_vfirst)
        {
        auto inout_lo = inout.subarray({{}, {0,nydirty/2}});
        c2c(inout_lo, inout_lo, {0}, FORWARD, Tcalc(1), nthreads);
        auto inout_hi = inout.subarray({{},{inout.shape(1)-nydirty/2, MAXIDX}});
        c2c(inout_hi, inout_hi, {0}, FORWARD, Tcalc(1), nthreads);
        for (size_t i=0; i<rsu.nranges(); ++i)
          {
          auto inout_tmp = inout.subarray({{size_t(rsu.ivbegin(i)), size_t(rsu.ivend(i))}, {}});
          c2c(inout_tmp, inout_tmp, {1}, FORWARD, Tcalc(1), nthreads);
          }
        }
      else
        {
        auto inout_lo = inout.subarray({{0,nxdirty/2},{}});
        c2c(inout_lo, inout_lo, {1}, FORWARD, Tcalc(1), nthreads);
        auto inout_hi = inout.subarray({{inout.shape(0)-nxdirty/2, MAXIDX},{}});
        c2c(inout_hi, inout_hi, {1}, FORWARD, Tcalc(1), nthreads);
        for (size_t i=0; i<rsv.nranges(); ++i)
          {
          auto inout_tmp = inout.subarray({{},{size_t(rsv.ivbegin(i)), size_t(rsv.ivend(i))}});
          c2c(inout_tmp, inout_tmp, {0}, FORWARD, Tcalc(1), nthreads);
          }
        }
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

    [[gnu::always_inline]] Uvwidx get_uvwidx(const UVW &uvwbase, uint32_t row, uint32_t ch)
      {
      auto uvw = uvwbase*bl.ffact(row, ch);
      double udum, vdum;
      int iu0, iv0, iw;
      getpix(uvw.u, uvw.v, udum, vdum, iu0, iv0);
      iu0 = (iu0+nsafe)>>log2tile;
      iv0 = (iv0+nsafe)>>log2tile;
      iw = do_wgridding ? max(0,int((uvw.w+wshift)*xdw)) : 0;
      return Uvwidx(iu0, iv0, iw);
      }

    void countRanges()
      {
      timers.push("building index");
      size_t nrow=bl.Nrows();

      if (do_wgridding)
        {
        dw = 0.5/ofactor/max(abs(nm1max+nshift), abs(nm1min+nshift));
        xdw = 1./dw;
        nplanes = size_t((wmax_d-wmin_d)/dw+supp);
        MR_assert(nplanes<(size_t(1)<<16), "too many w planes");
        wmin = (wmin_d+wmax_d)*0.5 - 0.5*(nplanes-1)*dw;
        wshift = dw-(0.5*supp*dw)-wmin;
        }
      else
        dw = wmin  = xdw = wshift = nplanes = 0;
      size_t nbunch = do_wgridding ? supp : 1;
      // we want a maximum deviation of 1% in gridding time between threads
      constexpr double max_asymm = 0.01;
      size_t max_allowed = size_t(nvis/double(nbunch*nthreads)*max_asymm);

      if (!bda)
        {
        size_t nchan = bl.Nchannels(0);
        checkShape(wgt2d->shape(),{nrow,nchan});
        checkShape((gridding?ms2d_in:ms2d_out)->shape(), {nrow,nchan});
        checkShape(mask2d->shape(), {nrow,nchan});
        }
      else
        {
        checkShape(wgt->shape(),{bl.Nvis()});
        checkShape((gridding?ms_in:ms_out)->shape(), {bl.Nvis()});
        checkShape(mask->shape(), {bl.Nvis()});
        }

      size_t ntiles_u = (nu>>log2tile) + 3;
      size_t ntiles_v = (nv>>log2tile) + 3;
      size_t nwmin = do_wgridding ? nplanes-supp+3 : 1;
timers.push("counting");
      // align members with cache lines
      struct alignas(64) spaced_size_t { atomic<size_t> v; };
      vector<spaced_size_t> buf(ntiles_u*ntiles_v*nwmin+1);
      auto chunk = max<size_t>(1, nrow/(20*nthreads));
      execDynamic(nrow, nthreads, chunk, [&](Scheduler &sched)
        {
        while (auto rng=sched.getNext())
        for(auto irow=rng.lo; irow<rng.hi; ++irow)
          {
          auto uvwbase = bl.baseCoord(irow);
          uvwbase.FixW();

          uint32_t nchan = bl.Nchannels(irow);
          uint32_t ch0=0;
          while(ch0<nchan)
            {
            auto idx0 = bl.ofs_ms(irow);
            while((ch0<nchan) && (!lmask(idx0+ch0))) ++ch0;
            uint32_t ch1=min<uint32_t>(nchan,ch0+1);
            while( (ch1<nchan) && (lmask(idx0+ch1))) ++ch1;
            // now [ch0;ch1[ contains an active range or we are at end
            auto inc0 = [&](Uvwidx idx)
              {
              ++buf[idx.tile_u*ntiles_v*nwmin + idx.tile_v*nwmin + idx.minplane].v;
              };
            auto inc = [&](Uvwidx idx, uint32_t ch)
              {
              inc0(idx);
              lmask(idx0+ch)=2;
              };
            auto recurse=[&](uint32_t ch_lo, uint32_t ch_hi, Uvwidx uvw_lo, Uvwidx uvw_hi, auto &&recurse) -> void
              {
              if (ch_lo+1==ch_hi)
                {
                if (uvw_lo!=uvw_hi)
                  inc(uvw_hi,ch_hi);
                }
              else
                {
                auto ch_mid = ch_lo+(ch_hi-ch_lo)/2;
                auto uvw_mid = get_uvwidx(uvwbase, irow, ch_mid);
                if (uvw_lo!=uvw_mid)
                  recurse(ch_lo, ch_mid, uvw_lo, uvw_mid, recurse);
                if (uvw_mid!=uvw_hi)
                  recurse(ch_mid, ch_hi, uvw_mid, uvw_hi, recurse);
                }
              };

            if (ch0!=ch1)
              {
              auto uvw0 = get_uvwidx(uvwbase,irow,ch0);
              inc0(uvw0);
              if (ch0+1<ch1)
                recurse(ch0,ch1-1,uvw0,get_uvwidx(uvwbase,irow,ch1-1),recurse);
              }
            ch0 = ch1;
            }
          }
        });

timers.poppush("allocation");
// accumulate
      {
      blockstart.clear(); // for now
      size_t acc=0;
      for (size_t tu=0; tu<ntiles_u; ++tu)
        for (size_t tv=0; tv<ntiles_v; ++tv)
          for (size_t mp=0; mp<nwmin; ++mp)
            {
            size_t i = tu*ntiles_v*nwmin + tv*nwmin + mp;
            size_t tmp = buf[i].v;
            if (tmp>0) blockstart.push_back({Uvwidx(tu,tv,mp),acc});
            buf[i].v = acc;
            acc += tmp;
            }
      buf.back().v=acc;
      }
timers.poppush("filling");
      ranges.resize(buf.back().v);
      execDynamic(nrow, nthreads, chunk, [&](Scheduler &sched)
        {
        vector<pair<uint16_t, uint16_t>> interbuf;
        while (auto rng=sched.getNext())
        for(auto irow=rng.lo; irow<rng.hi; ++irow)
          {
          bool on=false;
          Uvwidx uvwlast(0,0,0);
          uint32_t nchan = bl.Nchannels(irow);
          size_t chan0=0;

          auto flush=[&]()
            {
            if (interbuf.empty()) return;
            auto bufidx = uvwlast.tile_u*ntiles_v*nwmin + uvwlast.tile_v*nwmin + uvwlast.minplane;
            auto bufpos = (buf[bufidx].v+=interbuf.size()) - interbuf.size();
            for (size_t i=0; i<interbuf.size(); ++i)
              ranges[bufpos+i] = RowchanRange(irow,interbuf[i].first,interbuf[i].second);
            interbuf.clear();
            };
          auto add=[&](uint16_t cb, uint16_t ce)
            { interbuf.emplace_back(cb, ce); };

          auto uvwbase = bl.baseCoord(irow);
          uvwbase.FixW();
          for (size_t ichan=0; ichan<nchan; ++ichan)
            {
            auto idx = bl.ofs_ms(irow, ichan);
            auto xmask = lmask(idx);
            if (xmask)
              {
              if ((!on)||(xmask==2))
                {
                auto uvwcur = get_uvwidx(uvwbase, irow, ichan);
                if (!on) // new active region
                  {
                  on=true;
                  if (uvwlast!=uvwcur) flush();
                  uvwlast=uvwcur; chan0=ichan;
                  }
                else if (uvwlast!=uvwcur) // change of active region
                  {
                  add(chan0, ichan);
                  flush();
                  uvwlast=uvwcur; chan0=ichan;
                  }
                }
              }
            else if (on) // end of active region
              {
              add(chan0, ichan);
              on=false;
              }
            }
          if (on) // end of active region at last channel
            add(chan0, nchan);
          flush();
          }
        });
timers.poppush("building blockstart");
      vector<size_t> vissum;
      vissum.reserve(ranges.size()+1);
      size_t visacc=0;
      for (size_t i=0; i<ranges.size(); ++i)
        {
        vissum.push_back(visacc);
        visacc += ranges[i].nchan();
        }
      vissum.push_back(visacc);
      vector<pair<Uvwidx, size_t>> bs2;
      swap(blockstart, bs2);
      for (size_t i=0; i<bs2.size(); ++i)
        {
        blockstart.push_back(bs2[i]);
        size_t i1 = bs2[i].second;
        size_t i2 = vissum.size();
        if (i+1<bs2.size()) i2 = bs2[i+1].second;
        size_t acc=0;
        for (size_t j=i1+1; j<i2; ++j)
          {
          acc += vissum[j]-vissum[j-1];
          if (acc>max_allowed)
            {
            blockstart.push_back({bs2[i].first, j});
            acc=0;
            }
          }
        }
      lmask.unassign();
timers.pop();

      // compute which grid regions are required
      if (do_wgridding)
        {
        timers.poppush("grid regions");
        vmav<unsigned char, 2> tmpu({nplanes,(nu>>log2tile)+1}),
                               tmpv({nplanes,(nv>>log2tile)+1});
        for (const auto &rng: blockstart)
          for (size_t i=0; i<supp; ++i)
            {
            tmpu(rng.first.minplane+i, rng.first.tile_u) = 1;
            tmpv(rng.first.minplane+i, rng.first.tile_v) = 1;
            }
        uranges.resize(nplanes);
        vranges.resize(nplanes);
        constexpr int tilesize = 1<<log2tile;
        for (size_t i=0; i<nplanes; ++i)
          {
          auto &rsu(uranges[i]);
          auto &rsv(vranges[i]);
          for (size_t j=0; j<tmpu.shape(1); ++j)
            if (tmpu(i,j))
              rsu.add(j*tilesize-int(supp/2)-1, (j+1)*tilesize+int(supp/2)+1);
          // handle wraparound
          if (!rsu.empty() && rsu.ivbegin(0)<0)
            {
            int tmp = rsu.ivbegin(0);
            rsu.remove(tmp,0);
            rsu.add(nu+tmp, nu);
            }
          if (!rsu.empty() && rsu.ivend(rsu.size()-1)>int(nu))
            {
            int tmp = rsu.ivend(rsu.size()-1);
            rsu.remove(nu,tmp);
            rsu.add(0, tmp-nu);
            }
          for (size_t j=0; j<tmpv.shape(1); ++j)
            if (tmpv(i,j))
              rsv.add(j*tilesize-int(supp/2)-1, (j+1)*tilesize+int(supp/2)+1);
          // handle wraparound
          if (!rsv.empty() && rsv.ivbegin(0)<0)
            {
            int tmp = rsv.ivbegin(0);
            rsv.remove(tmp,0);
            rsv.add(nv+tmp, nv);
            }
          if (!rsv.empty() && rsv.ivend(rsv.size()-1)>int(nv))
            {
            int tmp = rsv.ivend(rsv.size()-1);
            rsv.remove(nv,tmp);
            rsv.add(0, tmp-nv);
            }
          }
        }
      timers.pop();
      }

    template<size_t supp, bool wgrid> class HelperX2g2
      {
      public:
        static constexpr size_t vlen = mysimd<Tacc>::size();
        static constexpr size_t nvec = (supp+vlen-1)/vlen;

      private:
        static constexpr int nsafe = (supp+1)/2;
        static constexpr int su = 2*nsafe+(1<<log2tile);
        static constexpr int sv = 2*nsafe+(1<<log2tile);
        static constexpr int svvec = sv+vlen-1;
        static constexpr double xsupp=2./supp;
        const Wgridder *parent;
        TemplateKernel<supp, mysimd<Tacc>> tkrn;
        const vmav<complex<Tcalc>,2> &grid;
        int iu0, iv0; // start index of the current visibility
        int bu0, bv0; // start index of the current buffer

        vmav<Tacc,2> bufr, bufi;
        Tacc *px0r, *px0i;
        double w0, xdw;
        vector<Mutex> &locks;

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
            LockGuard lock(locks[idxu]);
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
          };
        kbuf buf;

        HelperX2g2(const Wgridder *parent_, const vmav<complex<Tcalc>,2> &grid_,
          vector<Mutex> &locks_, double w0_=-1, double dw_=-1)
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

        [[gnu::always_inline]] [[gnu::hot]] void prep(const UVW &in,
          [[maybe_unused]] size_t nth=0)
          {
          double ufrac, vfrac;
          auto iu0old = iu0;
          auto iv0old = iv0;
          parent->getpix(in.u, in.v, ufrac, vfrac, iu0, iv0);
          auto x0 = -ufrac*2+(supp-1);
          auto y0 = -vfrac*2+(supp-1);
          if constexpr(wgrid)
            tkrn.eval2s(Tacc(x0), Tacc(y0), Tacc(xdw*(w0-in.w)), nth, &buf.simd[0]);
          else
            tkrn.eval2(Tacc(x0), Tacc(y0), &buf.simd[0]);
          if ((iu0==iu0old) && (iv0==iv0old)) return;
          if ((iu0<bu0) || (iv0<bv0) || (iu0+int(supp)>bu0+su) || (iv0+int(supp)>bv0+sv))
            {
            dump();
            bu0=((((iu0+nsafe)>>log2tile)<<log2tile))-nsafe;
            bv0=((((iv0+nsafe)>>log2tile)<<log2tile))-nsafe;
            }
          auto ofs = (iu0-bu0)*svvec + iv0-bv0;
          p0r = px0r+ofs;
          p0i = px0i+ofs;
          }
      };


    template<size_t supp, bool wgrid> class HelperG2x2
      {
      public:
        static constexpr size_t vlen = mysimd<Tcalc>::size();
        static constexpr size_t nvec = (supp+vlen-1)/vlen;

      private:
        static constexpr int nsafe = (supp+1)/2;
        static constexpr int su = 2*nsafe+(1<<log2tile);
        static constexpr int sv = 2*nsafe+(1<<log2tile);
        static constexpr int svvec = sv+vlen-1;
        static constexpr double xsupp=2./supp;
        const Wgridder *parent;

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
          };
        kbuf buf;

        HelperG2x2(const Wgridder *parent_, const cmav<complex<Tcalc>,2> &grid_,
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

        [[gnu::always_inline]] [[gnu::hot]] void prep(const UVW &in,
          [[maybe_unused]] size_t nth=0)
          {
          double ufrac, vfrac;
          auto iu0old = iu0;
          auto iv0old = iv0;
          parent->getpix(in.u, in.v, ufrac, vfrac, iu0, iv0);
          auto x0 = -ufrac*2+(supp-1);
          auto y0 = -vfrac*2+(supp-1);
          if constexpr(wgrid)
            tkrn.eval2s(Tcalc(x0), Tcalc(y0), Tcalc(xdw*(w0-in.w)), nth, &buf.simd[0]);
          else
            tkrn.eval2(Tcalc(x0), Tcalc(y0), &buf.simd[0]);
          if ((iu0==iu0old) && (iv0==iv0old)) return;
          if ((iu0<bu0) || (iv0<bv0) || (iu0+int(supp)>bu0+su) || (iv0+int(supp)>bv0+sv))
            {
            bu0=((((iu0+nsafe)>>log2tile)<<log2tile))-nsafe;
            bv0=((((iv0+nsafe)>>log2tile)<<log2tile))-nsafe;
            load();
            }
          auto ofs = (iu0-bu0)*svvec + iv0-bv0;
          p0r = px0r+ofs;
          p0i = px0i+ofs;
          }
      };

    void compute_phases(vector<complex<Tcalc>> &phases, vector<Tcalc> &buf,
      Tcalc imflip, const UVW &bcoord, const RowchanRange &rcr)
      {
      phases.resize(rcr.ch_end-rcr.ch_begin);
      buf.resize(rcr.ch_end-rcr.ch_begin);
      double fct = imflip*(bcoord.u*lshift + bcoord.v*mshift + bcoord.w*nshift);
      expi(phases, buf, [&](size_t i) {
                      auto tmp = fct*bl.ffact(rcr.row, rcr.ch_begin+i);
                      if constexpr (is_same<double, Tcalc>::value)
                        return Tcalc(twopi*tmp);
                      // we are reducing accuracy,
                      // so let's better do range reduction first
                      return Tcalc(twopi*(tmp-floor(tmp)));
                      });
      }

    template<size_t SUPP, bool wgrid> [[gnu::hot]] void x2grid_c_helper
      (size_t supp, const vmav<complex<Tcalc>,2> &grid, size_t p0, double w0)
      {
      if constexpr (SUPP>=8)
        if (supp<=SUPP/2) return x2grid_c_helper<SUPP/2, wgrid>(supp, grid, p0, w0);
      if constexpr (SUPP>4)
        if (supp<SUPP) return x2grid_c_helper<SUPP-1, wgrid>(supp, grid, p0, w0);
      MR_assert(supp==SUPP, "requested support out of range");

      vector<Mutex> locks(nu);

      execDynamic(blockstart.size(), nthreads, wgrid ? SUPP : 1, [&](Scheduler &sched)
        {
        constexpr auto vlen=mysimd<Tacc>::size();
        constexpr auto NVEC((SUPP+vlen-1)/vlen);
        HelperX2g2<SUPP,wgrid> hlp(this, grid, locks, w0, dw);
        constexpr auto jump = hlp.lineJump();
        const auto * DUCC0_RESTRICT ku = hlp.buf.scalar;
        const auto * DUCC0_RESTRICT kv = hlp.buf.simd+NVEC;
        vector<complex<Tcalc>> phases;
        vector<Tcalc> buf;

        while (auto rng=sched.getNext()) for(auto ix=rng.lo; ix<rng.hi; ++ix)
          {
//auto ix = ix_+ranges.size()/2; if (ix>=ranges.size()) ix -=ranges.size();
          const auto &uvwidx(blockstart[ix].first);
          if ((!wgrid) || ((uvwidx.minplane+SUPP>p0)&&(uvwidx.minplane<=p0)))
            {
//bool lastplane = (!wgrid) || (uvwidx.minplane+SUPP-1==p0);
            size_t nth = p0-uvwidx.minplane;
            size_t iend = (ix+1<blockstart.size()) ? blockstart[ix+1].second : ranges.size();
            for (size_t cnt=blockstart[ix].second; cnt<iend; ++cnt)
              {
              const auto &rcr(ranges[cnt]);
              if (cnt+1<iend)
                {
                const auto &nextrcr(ranges[cnt+1]);
                if (bda)
                  {
                  wgt->prefetch_r(bl.ofs_ms(nextrcr.row, nextrcr.ch_begin));
                  ms_in->prefetch_r(bl.ofs_ms(nextrcr.row, nextrcr.ch_begin));
                  }
                else
                  {
                  wgt2d->prefetch_r(nextrcr.row, nextrcr.ch_begin);
                  ms2d_in->prefetch_r(nextrcr.row, nextrcr.ch_begin);
                  }
                bl.prefetchRow(nextrcr.row);
                }
              size_t row = rcr.row;
              auto bcoord = bl.baseCoord(row);
              auto imflip = Tcalc(bcoord.FixW());
              if (shifting)
                compute_phases(phases, buf, imflip, bcoord, rcr);
              auto idx0 = bl.ofs_ms(row);
              for (size_t ch=rcr.ch_begin; ch<rcr.ch_end; ++ch)
                {
                auto coord = bcoord*bl.ffact(rcr.row, ch);
                hlp.prep(coord, nth);
                auto v = bda ? (*ms_in)(idx0+ch) : (*ms2d_in)(row,ch);
                if (shifting)
                  v*=phases[ch-rcr.ch_begin];
                v *= bda ? (*wgt)(idx0+ch) : (*wgt2d)(row,ch);

                if constexpr (NVEC==1)
                  {
                  mysimd<Tacc> vr=v.real()*kv[0], vi=v.imag()*imflip*kv[0];
                  for (size_t cu=0; cu<SUPP; ++cu)
                    {
                    auto * DUCC0_RESTRICT pxr = hlp.p0r+cu*jump;
                    auto * DUCC0_RESTRICT pxi = hlp.p0i+cu*jump;
                    unaligned_add(pxr, vr*ku[cu]);
                    unaligned_add(pxi, vi*ku[cu]);
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
                      unaligned_add(pxr, tmpr*kv[cv]);
                      unaligned_add(pxi, tmpi*kv[cv]);
                      }
                    }
                  }
                }
              }
            }
          }
        });
      }

    template<bool wgrid> void x2grid_c(const vmav<complex<Tcalc>,2> &grid,
      size_t p0, double w0=-1)
      {
      checkShape(grid.shape(), {nu, nv});
      constexpr size_t maxsupp = is_same<Tacc, double>::value ? 16 : 8;
      x2grid_c_helper<maxsupp, wgrid>(supp, grid, p0, w0);
      }

    template<size_t SUPP, bool wgrid> [[gnu::hot]] void grid2x_c_helper
      (size_t supp, const cmav<complex<Tcalc>,2> &grid, size_t p0, double w0)
      {
      if constexpr (SUPP>=8)
        if (supp<=SUPP/2) return grid2x_c_helper<SUPP/2, wgrid>(supp, grid, p0, w0);
      if constexpr (SUPP>4)
        if (supp<SUPP) return grid2x_c_helper<SUPP-1, wgrid>(supp, grid, p0, w0);
      MR_assert(supp==SUPP, "requested support out of range");

      // Loop over sampling points
      execDynamic(blockstart.size(), nthreads, wgrid ? SUPP : 1, [&](Scheduler &sched)
        {
        constexpr size_t vlen=mysimd<Tcalc>::size();
        constexpr size_t NVEC((SUPP+vlen-1)/vlen);
        HelperG2x2<SUPP,wgrid> hlp(this, grid, w0, dw);
        constexpr int jump = hlp.lineJump();
        const auto * DUCC0_RESTRICT ku = hlp.buf.scalar;
        const auto * DUCC0_RESTRICT kv = hlp.buf.simd+NVEC;
        vector<complex<Tcalc>> phases;
        vector<Tcalc> buf;

        while (auto rng=sched.getNext()) for(auto ix=rng.lo; ix<rng.hi; ++ix)
          {
          const auto &uvwidx(blockstart[ix].first);
          if ((!wgrid) || ((uvwidx.minplane+SUPP>p0)&&(uvwidx.minplane<=p0)))
            {
            bool firstplane = (!wgrid) || (uvwidx.minplane==p0);
            bool lastplane = (!wgrid) || (uvwidx.minplane+SUPP-1==p0);
            size_t nth = p0-uvwidx.minplane;
            size_t iend = (ix+1<blockstart.size()) ? blockstart[ix+1].second : ranges.size();
            for (size_t cnt=blockstart[ix].second; cnt<iend; ++cnt)
              {
              const auto &rcr(ranges[cnt]);
              if (cnt+1<iend)
                {
                const auto &nextrcr(ranges[cnt+1]);
                if (bda)
                  {
                  auto idx = bl.ofs_ms(nextrcr.row, nextrcr.ch_begin);
                  wgt->prefetch_r(idx);
                  ms_out->prefetch_r(idx);
                  ms_out->prefetch_w(idx);
                  }
                else
                  {
                  wgt2d->prefetch_r(nextrcr.row, nextrcr.ch_begin);
                  ms2d_out->prefetch_r(nextrcr.row, nextrcr.ch_begin);
                  ms2d_out->prefetch_w(nextrcr.row, nextrcr.ch_begin);
                  }
                bl.prefetchRow(nextrcr.row);
                }
              size_t row = rcr.row;
              auto bcoord = bl.baseCoord(row);
              auto imflip = Tcalc(bcoord.FixW());
              if (shifting&&lastplane)
                compute_phases(phases, buf, -imflip, bcoord, rcr);
              auto idx0 = bl.ofs_ms(rcr.row);
              for (size_t ch=rcr.ch_begin; ch<rcr.ch_end; ++ch)
                {
                auto coord = bcoord*bl.ffact(rcr.row, ch);
                hlp.prep(coord, nth);
                mysimd<Tcalc> rr=0, ri=0;
                if constexpr (NVEC==1)
                  {
                  for (size_t cu=0; cu<SUPP; ++cu)
                    {
                    const auto * DUCC0_RESTRICT pxr = hlp.p0r + cu*jump;
                    const auto * DUCC0_RESTRICT pxi = hlp.p0i + cu*jump;
                    rr += loadu<mysimd<Tcalc>>(pxr)*ku[cu];
                    ri += loadu<mysimd<Tcalc>>(pxi)*ku[cu];
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
                      tmpr += kv[cv]*loadu<mysimd<Tcalc>>(pxr);
                      tmpi += kv[cv]*loadu<mysimd<Tcalc>>(pxi);
                      }
                    rr += ku[cu]*tmpr;
                    ri += ku[cu]*tmpi;
                    }
                  }
                ri *= imflip;
                auto r = hsum_cmplx<Tcalc>(rr,ri);
                if (!firstplane) r += bda ? (*ms_out)(idx0+ch) : (*ms2d_out)(row,ch);
                if (lastplane)
                  {
                  auto twgt = bda ? (*wgt)(idx0+ch) : (*wgt2d)(row,ch);
                  r *= shifting ?
                    complex<Tms>(phases[ch-rcr.ch_begin]*Tcalc(twgt)) : twgt;
                  }
                bda ? (*ms_out)(idx0+ch) = r : (*ms2d_out)(row,ch) = r;
                }
              }
            }
          }
        });
      }

    template<bool wgrid> void grid2x_c(const cmav<complex<Tcalc>,2> &grid,
      size_t p0, double w0=-1)
      {
      checkShape(grid.shape(), {nu, nv});
      constexpr size_t maxsupp = is_same<Tcalc, double>::value ? 16 : 8;
      grid2x_c_helper<maxsupp, wgrid>(supp, grid, p0, w0);
      }

    void apply_global_corrections(const vmav<Timg,2> &dirty)
      {
      timers.push("global corrections");
      double x0 = lshift-0.5*nxdirty*pixsize_x,
             y0 = mshift-0.5*nydirty*pixsize_y;
      auto cfu = krn->corfunc(nxdirty/2+1, 1./nu, nthreads);
      auto cfv = krn->corfunc(nydirty/2+1, 1./nv, nthreads);
      size_t nxd = lmshift ? nxdirty : (nxdirty/2+1);
      size_t nyd = lmshift ? nydirty : (nydirty/2+1);
      execParallel(nxd, nthreads, [&](size_t lo, size_t hi)
        {
        for(auto i=lo; i<hi; ++i)
          {
          double xsq = sqr(x0+i*pixsize_x);
          for (size_t j=0; j<nyd; ++j)
            {
            double ysq = sqr(y0+j*pixsize_y);
            double fct = 0;
            auto tmp = 1-xsq-ysq;
            if (tmp>=0)
              {
              // accurate form of sqrt(1-xsq-ysq)-1 for nm1 close to zero
              auto nm1 = (-xsq-ysq)/(sqrt(tmp)+1);
nm1 += bl.r_l*(x0+i*pixsize_x) + bl.r_m*(y0+j*pixsize_y);
              fct = krn->corfunc((nm1+nshift)*dw);
              if (divide_by_n)
                fct /= nm1+1;
              }
            else // beyond the horizon, don't really know what to do here
              fct = divide_by_n ? 0 : krn->corfunc((-sqrt(-tmp)-1.+nshift)*dw);
            if (lmshift)
              {
              auto i2=min(i, nxdirty-i), j2=min(j, nydirty-j);
              fct *= cfu[nxdirty/2-i2]*cfv[nydirty/2-j2];
              dirty(i,j)*=Timg(fct);
              }
            else
              {
              fct *= cfu[nxdirty/2-i]*cfv[nydirty/2-j];
              size_t i2 = nxdirty-i, j2 = nydirty-j;
              dirty(i,j)*=Timg(fct);
              if ((i>0)&&(i<i2))
                {
                dirty(i2,j)*=Timg(fct);
                if ((j>0)&&(j<j2))
                  dirty(i2,j2)*=Timg(fct);
                }
              if ((j>0)&&(j<j2))
                dirty(i,j2)*=Timg(fct);
              }
            }
          }
        });
      timers.pop();
      }

    void report()
      {
      if (verbosity==0) return;
      cout << (gridding ? "Gridding:" : "Degridding:") << endl
           << "  nthreads=" << nthreads << ", "
           << "dirty=(" << nxdirty << "x" << nydirty << "), "
           << "grid=(" << nu << "x" << nv;
      if (do_wgridding) cout << "x" << nplanes;
      cout << "), supp=" << supp
           << ", eps=" << epsilon
           << endl;
      cout << "  nrow=" << bl.Nrows();
      if (!bl.BDA())
        cout << ", nchan=" << bl.Nchannels(0);
      cout << ", nvis=" << nvis << "/" << bl.Nvis() << endl;
      if (do_wgridding)
        cout << "  w=[" << wmin_d << "; " << wmax_d << "], min(n-1)=" << nm1min
             << ", dw=" << dw << ", (wmax-wmin)/dw=" << (wmax_d-wmin_d)/dw << endl;
      size_t ovh0 = ranges.size()*sizeof(ranges[0]);
      ovh0 += blockstart.size()*sizeof(blockstart[0]);
      size_t ovh1 = nu*nv*sizeof(complex<Tcalc>);             // grid
      if (!do_wgridding)
        ovh1 += nu*nv*sizeof(Tcalc);                          // rgrid
      if (!gridding)
        ovh1 += nxdirty*nydirty*sizeof(Timg);                 // tdirty
      cout << "  memory overhead: "
           << ovh0/double(1<<30) << "GB (index) + "
           << ovh1/double(1<<30) << "GB (2D arrays)" << endl;
      }

    void x2dirty()
      {
      if (do_wgridding)
        {
        timers.push("zeroing dirty image");
        mav_apply([](Timg &v){v=Timg(0);}, nthreads, dirty_out);
        timers.poppush("allocating grid");
        auto grid = vmav<complex<Tcalc>,2>::build_noncritical({nu,nv});
        timers.pop();
        for (size_t pl=0; pl<nplanes; ++pl)
          {
          double w = wmin+pl*dw;
          timers.push("gridding proper");
          x2grid_c<true>(grid, pl, w);
          timers.pop();
          grid2dirty_c_overwrite_wscreen_add(grid, dirty_out, w, pl);
          }
        // correct for w gridding etc.
        apply_global_corrections(dirty_out);
        }
      else
        {
        timers.push("allocating grid");
        auto grid = vmav<complex<Tcalc>,2>::build_noncritical({nu,nv});
        timers.poppush("gridding proper");
        x2grid_c<false>(grid, 0);
        timers.poppush("allocating rgrid");
        auto rgrid = vmav<Tcalc,2>::build_noncritical(grid.shape(), PAGE_IN(nthreads));
        timers.poppush("complex2hartley");
        complex2hartley(grid, rgrid, nthreads);
        timers.pop();
        grid2dirty_overwrite(rgrid, dirty_out);
        }
      }

    void dirty2x()
      {
      if (do_wgridding)
        {
        timers.push("copying dirty image");
        vmav<Timg,2> tdirty({nxdirty,nydirty}, PAGE_IN(nthreads));
        mav_apply([](Timg &a, const Timg &b) {a=b;}, nthreads, tdirty, dirty_in);
        timers.pop();
        // correct for w gridding etc.
        apply_global_corrections(tdirty);
        timers.push("allocating grid");
        auto grid = vmav<complex<Tcalc>,2>::build_noncritical({nu,nv}, PAGE_IN(nthreads));
        timers.pop();
        for (size_t pl=0; pl<nplanes; ++pl)
          {
          double w = wmin+pl*dw;
          dirty2grid_c_wscreen(tdirty, grid, w, pl);
          timers.push("degridding proper");
          grid2x_c<true>(grid, pl, w);
          timers.pop();
          }
        }
      else
        {
        timers.push("allocating grid");
        auto rgrid = vmav<Tcalc,2>::build_noncritical({nu,nv}, PAGE_IN(nthreads));
        timers.pop();
        dirty2grid(dirty_in, rgrid);
        timers.push("allocating grid");
        auto grid = vmav<complex<Tcalc>,2>::build_noncritical(rgrid.shape());
        timers.poppush("hartley2complex");
        hartley2complex(rgrid, grid, nthreads);
        timers.poppush("degridding proper");
        grid2x_c<false>(grid, 0);
        timers.pop();
        }
      }

    auto getNuNv()
      {
      timers.push("parameter calculation");

      double xmin = lshift - 0.5*nxdirty*pixsize_x,
             xmax = xmin + (nxdirty-1)*pixsize_x,
             ymin = mshift - 0.5*nydirty*pixsize_y,
             ymax = ymin + (nydirty-1)*pixsize_y;
      vector<double> xext{xmin, xmax},
                     yext{ymin, ymax};
      if (xmin*xmax<0) xext.push_back(0);
      if (ymin*ymax<0) yext.push_back(0);
      nm1min = 1e300, nm1max = -1e300;
      for (auto xc: xext)
        for (auto yc: yext)
          {
          double tmp = xc*xc+yc*yc;
          double nval = (tmp<=1.) ?  (sqrt(1.-tmp)-1.) : (-sqrt(tmp-1.)-1.);
//nval += bl.r_l*xc + bl.r_m*yc;
          nm1min = min(nm1min, nval);
          nm1max = max(nm1max, nval);
          }
double nm1minb = 1e300, nm1maxb = -1e300;
xext.push_back(0.5*(xmin+xmax));
yext.push_back(0.5*(ymin+ymax));
      for (auto xc: xext)
        for (auto yc: yext)
          {
          double tmp = xc*xc+yc*yc;
          double nval = (tmp<=1.) ?  (sqrt(1.-tmp)-1.) : (-sqrt(tmp-1.)-1.);
nval += bl.r_l*xc + bl.r_m*yc;
          nm1minb = min(nm1minb, nval);
          nm1maxb = max(nm1maxb, nval);
          }
cout << "bla" << endl;
cout << xmin <<" " << xmax << endl;
cout << ymin <<" " << ymax << endl;
cout << nm1min <<" " << nm1max << endl;
cout << nm1minb <<" " << nm1maxb << endl;
nm1min = nm1minb;
nm1max = nm1maxb;

      nshift = (no_nshift||(!do_wgridding)) ? 0. : -0.5*(nm1max+nm1min);
      shifting = lmshift || (nshift!=0);

      auto idx = getAvailableKernels<Tcalc>(epsilon, do_wgridding ? 3 : 2, sigma_min, sigma_max);
      double mincost = 1e300;
      constexpr double nref_fft=2048;
      constexpr double costref_fft=0.0693;
      size_t minnu=0, minnv=0, minidx=~(size_t(0));
      size_t vlen;
      // Avoid duplicated-branches warning when the sizes are equal.
      if constexpr (mysimd<Tacc>::size() == mysimd<Tcalc>::size())
        vlen = mysimd<Tacc>::size();
      else
        vlen = gridding ? mysimd<Tacc>::size() : mysimd<Tcalc>::size();
      for (size_t i=0; i<idx.size(); ++i)
        {
        const auto &krn(getKernel(idx[i]));
        auto supp = krn.W;
        auto nvec = (supp+vlen-1)/vlen;
        auto ofactor = krn.ofactor;
        size_t nu=2*good_size_complex(size_t(nxdirty*ofactor*0.5)+1);
        size_t nv=2*good_size_complex(size_t(nydirty*ofactor*0.5)+1);
        nu = max<size_t>(nu,16);
        nv = max<size_t>(nv,16);
        double logterm = log(nu*nv)/log(nref_fft*nref_fft);
        double fftcost = nu/nref_fft*nv/nref_fft*logterm*costref_fft;
        double gridcost = 2.2e-10*nvis*(supp*nvec*vlen + ((2*nvec+1)*(supp+3)*vlen));
        if (gridding) gridcost *= sizeof(Tacc)/sizeof(Tcalc);
        if (do_wgridding)
          {
          double dw = 0.5/ofactor/max(abs(nm1max+nshift), abs(nm1min+nshift));
          size_t nplanes = size_t((wmax_d-wmin_d)/dw+supp);
          fftcost *= nplanes;
          gridcost *= supp;
          }
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
      size_t nrow=bl.Nrows();
      if (!bda)
        {
        size_t nchan = bl.Nchannels(0);
        checkShape(wgt2d->shape(), {nrow,nchan});
        checkShape((gridding?ms2d_in:ms2d_out)->shape(), {nrow,nchan});
        checkShape(mask2d->shape(), {nrow,nchan});
        }
      else
        {
        checkShape(wgt->shape(),{bl.Nvis()});
        checkShape((gridding?ms_in:ms_out)->shape(), {bl.Nvis()});
        checkShape(mask->shape(), {bl.Nvis()});
        }

      nvis=0;
      wmin_d=1e300;
      wmax_d=-1e300;
      Mutex mut;
      execParallel(nrow, nthreads, [&](size_t lo, size_t hi)
        {
        double lwmin_d=1e300, lwmax_d=-1e300;
        size_t lnvis=0;
        for(auto irow=lo; irow<hi; ++irow)
          for (size_t ichan=0; ichan<bl.Nchannels(irow); ++ichan)
{
//            if (mask(irow,ichan) && (wgt(irow, ichan)!=0) && (norm(ms_in(irow,ichan)!=0)))
            auto idx = bl.ofs_ms(irow,ichan);
            auto tnorm = gridding ? norm(bda ? (*ms_in)(idx) : (*ms2d_in)(irow,ichan)) : 1.;
            auto twgt = bda ? (*wgt)(idx) : (*wgt2d)(irow,ichan);
            auto tmask = bda ? (*mask)(idx) : (*mask2d)(irow,ichan);
            if (tnorm*twgt*tmask != 0)
              {
              lmask(idx)=1;
              ++lnvis;
              double w = bl.absEffectiveW(irow, ichan);
              lwmin_d = min(lwmin_d, w);
              lwmax_d = max(lwmax_d, w);
              }
            else
              {
              if (!gridding) bda ? (*ms_out)(idx)=0 : (*ms2d_out)(irow, ichan)=0;
              }
}
        {
        LockGuard lock(mut);
        wmin_d = min(wmin_d, lwmin_d);
        wmax_d = max(wmax_d, lwmax_d);
        nvis += lnvis;
        }
        });
      timers.pop();
      }

  public:
    Wgridder(const cmav<double,2> &uvw,
           const cmav<size_t,1> &freqlist_id,                // (nrows),
           const cmav<size_t,1> &freqlist_nfreqs,            // (max(freqlist_id)+1)
           const cmav<double,1> &freqlist_freqs,             // (sum(freqlist_nfreqs), concatenated frequency lists for all freqlist_ids
           const Tms_in *ms_in_, const Tms2d_in *ms2d_in_,
           const vmav<complex<Tms>,1> *ms_out_,
           const vmav<complex<Tms>,2> *ms2d_out_,
           const cmav<Timg,2> &dirty_in_, const vmav<Timg,2> &dirty_out_,
           const cmav<Tms,1> *wgt_, const cmav<Tms,2> *wgt2d_,
           const cmav<uint8_t,1> *mask_, const cmav<uint8_t,2> *mask2d_,
           double pixsize_x_, double pixsize_y_, double epsilon_,
           bool do_wgridding_, size_t nthreads_, size_t verbosity_,
           bool flip_u, bool flip_v, bool flip_w, bool divide_by_n_,
           double sigma_min_, double sigma_max_,
           double center_x, double center_y, bool allow_nshift)
      : gridding(ms_in_||ms2d_in_),
        bda(ms_out_||ms_in_),
        timers(gridding ? "gridding" : "degridding"),
        ms_in(ms_in_), ms2d_in(ms2d_in_), ms_out(ms_out_), ms2d_out(ms2d_out_),
        dirty_in(dirty_in_), dirty_out(dirty_out_),
        wgt(wgt_), wgt2d(wgt2d_), mask(mask_), mask2d(mask2d_),
        lmask(bda ? wgt->shape() : typename cmav<Tms,1>::shape_t({wgt2d->size()})),
        pixsize_x(pixsize_x_), pixsize_y(pixsize_y_),
        nxdirty(gridding ? dirty_out.shape(0) : dirty_in.shape(0)),
        nydirty(gridding ? dirty_out.shape(1) : dirty_in.shape(1)),
        epsilon(epsilon_),
        do_wgridding(do_wgridding_),
        nthreads(adjust_nthreads(nthreads_)),
        verbosity(verbosity_),
        divide_by_n(divide_by_n_),
        sigma_min(sigma_min_), sigma_max(sigma_max_),
        lshift(flip_u ? -center_x : center_x), mshift(flip_v ? -center_y : center_y),
        lmshift((lshift!=0) || (mshift!=0)),
        no_nshift(!allow_nshift)
      {
      timers.push("Baseline construction");
double r_l = lshift / sqrt(1. - lshift*lshift - mshift*mshift);
double r_m = mshift / sqrt(1. - lshift*lshift - mshift*mshift);
//r_l=r_m=0;
      bl = Baselines(uvw, freqlist_id, freqlist_nfreqs, freqlist_freqs, flip_u, flip_v, flip_w, r_l, r_m);
      MR_assert(bl.Nrows()<(uint64_t(1)<<32), "too many rows in the MS");
 //     MR_assert(bl.Nchannels()<(uint64_t(1)<<16), "too many channels in the MS");
      timers.pop();
      scanData();
      if (nvis==0)
        {
        if (gridding) mav_apply([](Timg &v){v=Timg(0);}, nthreads, dirty_out);
        return;
        }
      auto kidx = getNuNv();
      MR_assert((nu>>log2tile)<(size_t(1)<<16), "nu too large");
      MR_assert((nv>>log2tile)<(size_t(1)<<16), "nv too large");
      ofactor = min(double(nu)/nxdirty, double(nv)/nydirty);
      krn = selectKernel(kidx);
      supp = krn->support();
      nsafe = (supp+1)/2;
      ushift = supp*(-0.5)+1+nu;
      vshift = supp*(-0.5)+1+nv;
      maxiu0 = (nu+nsafe)-supp;
      maxiv0 = (nv+nsafe)-supp;
      vlim = min(nv/2, size_t(nv*bl.Vmax()*pixsize_y+0.5*supp+1));
      uv_side_fast = true;
      size_t vlim2 = (nydirty+1)/2+(supp+1)/2;
      if (vlim2<vlim)
        {
        vlim = vlim2;
        uv_side_fast = false;
        }
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

template<typename Tcalc, typename Tacc, typename Tms, typename Tms_in, typename Timg>
  void ms2dirty_bda(
    const cmav<double,2> &uvw,                        // (nrows,3)
    const cmav<size_t,1> &freqlist_id,                // (nrows),
    const cmav<size_t,1> &freqlist_nfreqs,            // (max(freqlist_id)+1)
    const cmav<double,1> &freqlist_freqs,             // (sum(freqlist_nfreqs), concatenated frequency lists for all freqlist_ids
    const Tms_in &ms,                                 // concatenated array of visibilities, shape (sum_i(freqlist_nfreq[freqlist_id[i]))
    const cmav<Tms,1> &wgt_,                          // same shape as above
    const cmav<uint8_t,1> &mask_,                     // same shape as above
    double pixsize_x, double pixsize_y, double epsilon,
    bool do_wgridding, size_t nthreads, const vmav<Timg,2> &dirty, size_t verbosity,
    bool flip_u, bool flip_v, bool flip_w, bool divide_by_n, double sigma_min,
    double sigma_max, double center_x, double center_y, bool allow_nshift)
  {
  auto dirty_in(vmav<Timg,2>::build_empty());
  auto wgt(wgt_.size()!=0 ? wgt_ : wgt_.build_uniform(ms.shape(), 1.));
  auto mask(mask_.size()!=0 ? mask_ : mask_.build_uniform(ms.shape(), 1));
  Wgridder<Tcalc, Tacc, Tms, Timg, Tms_in, cmav<complex<Tms>,2>> par(uvw, freqlist_id, freqlist_nfreqs, freqlist_freqs, &ms, nullptr, nullptr, nullptr, dirty_in, dirty, &wgt, nullptr, &mask, nullptr, pixsize_x,
    pixsize_y, epsilon, do_wgridding, nthreads, verbosity, flip_u, flip_v, flip_w,
    divide_by_n, sigma_min, sigma_max, center_x, center_y, allow_nshift);
  }
template<typename Tcalc, typename Tacc, typename Tms, typename Tms_in, typename Timg> void ms2dirty(const cmav<double,2> &uvw,
  const cmav<double,1> &freq, const Tms_in &ms,
  const cmav<Tms,2> &wgt_, const cmav<uint8_t,2> &mask_, double pixsize_x, double pixsize_y, double epsilon,
  bool do_wgridding, size_t nthreads, const vmav<Timg,2> &dirty, size_t verbosity,
  bool flip_u, bool flip_v, bool flip_w, bool divide_by_n, double sigma_min,
  double sigma_max, double center_x, double center_y, bool allow_nshift)
  {
  auto dirty_in(vmav<Timg,2>::build_empty());
  auto freqlist_id = cmav<size_t,1>::build_uniform({ms.shape(0)},0);
  auto freqlist_nfreqs = cmav<size_t,1>::build_uniform({1},freq.shape(0));
  auto wgt(wgt_.size()!=0 ? wgt_ : wgt_.build_uniform(ms.shape(), 1.));
  auto mask(mask_.size()!=0 ? mask_ : mask_.build_uniform(ms.shape(), 1));
  Wgridder<Tcalc, Tacc, Tms, Timg, cmav<complex<Tms>,1>, Tms_in> par(uvw, freqlist_id, freqlist_nfreqs, freq, nullptr, &ms, nullptr, nullptr, dirty_in, dirty, nullptr, &wgt, nullptr, &mask, pixsize_x,
    pixsize_y, epsilon, do_wgridding, nthreads, verbosity, flip_u, flip_v, flip_w,
    divide_by_n, sigma_min, sigma_max, center_x, center_y, allow_nshift);
  }

template<typename Tcalc, typename Tacc, typename Tms, typename Timg>
  void dirty2ms_bda(
    const cmav<double,2> &uvw,
    const cmav<size_t,1> &freqlist_id,                // (nrows),
    const cmav<size_t,1> &freqlist_nfreqs,            // (max(freqlist_id)+1)
    const cmav<double,1> &freqlist_freqs,             // (sum(freqlist_nfreqs), concatenated frequency lists for all freqlist_ids
    const cmav<Timg,2> &dirty,
    const cmav<Tms,1> &wgt_, const cmav<uint8_t,1> &mask_, double pixsize_x, double pixsize_y,
    double epsilon, bool do_wgridding, size_t nthreads, const vmav<complex<Tms>,1> &ms,
    size_t verbosity, bool flip_u, bool flip_v, bool flip_w, bool divide_by_n,
    double sigma_min, double sigma_max, double center_x, double center_y, bool allow_nshift)
  {
  if (ms.size()==0) return;  // nothing to do
  auto dirty_out(vmav<Timg,2>::build_empty());
  auto wgt(wgt_.size()!=0 ? wgt_ : wgt_.build_uniform(ms.shape(), 1.));
  auto mask(mask_.size()!=0 ? mask_ : mask_.build_uniform(ms.shape(), 1));
  Wgridder<Tcalc, Tacc, Tms, Timg> par(uvw, freqlist_id, freqlist_nfreqs, freqlist_freqs, nullptr, nullptr, &ms, nullptr, dirty, dirty_out, &wgt, nullptr, &mask, nullptr, pixsize_x,
    pixsize_y, epsilon, do_wgridding, nthreads, verbosity, flip_u, flip_v, flip_w,
    divide_by_n, sigma_min, sigma_max, center_x, center_y, allow_nshift);
  }
template<typename Tcalc, typename Tacc, typename Tms, typename Timg> void dirty2ms(const cmav<double,2> &uvw,
  const cmav<double,1> &freq, const cmav<Timg,2> &dirty,
  const cmav<Tms,2> &wgt_, const cmav<uint8_t,2> &mask_, double pixsize_x, double pixsize_y,
  double epsilon, bool do_wgridding, size_t nthreads, const vmav<complex<Tms>,2> &ms,
  size_t verbosity, bool flip_u, bool flip_v, bool flip_w, bool divide_by_n,
  double sigma_min, double sigma_max, double center_x, double center_y, bool allow_nshift)
  {
  if (ms.size()==0) return;  // nothing to do
  auto dirty_out(vmav<Timg,2>::build_empty());
  auto freqlist_id = cmav<size_t,1>::build_uniform({ms.shape(0)},0);
  auto freqlist_nfreqs = cmav<size_t,1>::build_uniform({1},freq.shape(0));
  auto wgt(wgt_.size()!=0 ? wgt_ : wgt_.build_uniform(ms.shape(), 1.));
  auto mask(mask_.size()!=0 ? mask_ : mask_.build_uniform(ms.shape(), 1));
  Wgridder<Tcalc, Tacc, Tms, Timg> par(uvw, freqlist_id, freqlist_nfreqs, freq, nullptr, nullptr, nullptr, &ms, dirty, dirty_out, nullptr, &wgt, nullptr, &mask, pixsize_x,
    pixsize_y, epsilon, do_wgridding, nthreads, verbosity, flip_u, flip_v, flip_w,
    divide_by_n, sigma_min, sigma_max, center_x, center_y, allow_nshift);
  }

} // namespace detail_gridder

// public names
using detail_gridder::ms2dirty;
using detail_gridder::dirty2ms;

} // namespace ducc0

#endif
