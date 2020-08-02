#ifndef GRIDDER_CXX_H
#define GRIDDER_CXX_H

/*
 *  This file is part of nifty_gridder.
 *
 *  nifty_gridder is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  nifty_gridder is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with nifty_gridder; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/* Copyright (C) 2019-2020 Max-Planck-Society
   Author: Martin Reinecke */

#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <array>
#include <memory>

#include "ducc0/infra/error_handling.h"
#include "ducc0/math/fft.h"
#include "ducc0/infra/threading.h"
#include "ducc0/infra/useful_macros.h"
#include "ducc0/infra/mav.h"
#include "ducc0/infra/simd.h"
#include "ducc0/math/gridding_kernel.h"

namespace ducc0 {

namespace detail_gridder {

using namespace std;

template<size_t ndim> void checkShape
  (const array<size_t, ndim> &shp1, const array<size_t, ndim> &shp2)
  { MR_assert(shp1==shp2, "shape mismatch"); }

template<typename T> inline T fmod1 (T v)
  { return v-floor(v); }

//
// Start of real gridder functionality
//

template<typename T> void complex2hartley
  (const mav<complex<T>, 2> &grid, mav<T,2> &grid2, size_t nthreads)
  {
  MR_assert(grid.conformable(grid2), "shape mismatch");
  size_t nu=grid.shape(0), nv=grid.shape(1);

  execStatic(nu, nthreads, 0, [&](Scheduler &sched)
    {
    while (auto rng=sched.getNext()) for(auto u=rng.lo; u<rng.hi; ++u)
      {
      size_t xu = (u==0) ? 0 : nu-u;
      for (size_t v=0; v<nv; ++v)
        {
        size_t xv = (v==0) ? 0 : nv-v;
        grid2.v(u,v) = T(0.5)*(grid( u, v).real()+grid( u, v).imag()+
                               grid(xu,xv).real()-grid(xu,xv).imag());
        }
      }
    });
  }

template<typename T> void hartley2complex
  (const mav<T,2> &grid, mav<complex<T>,2> &grid2, size_t nthreads)
  {
  MR_assert(grid.conformable(grid2), "shape mismatch");
  size_t nu=grid.shape(0), nv=grid.shape(1);

  execStatic(nu, nthreads, 0, [&](Scheduler &sched)
    {
    while (auto rng=sched.getNext()) for(auto u=rng.lo; u<rng.hi; ++u)
      {
      size_t xu = (u==0) ? 0 : nu-u;
      for (size_t v=0; v<nv; ++v)
        {
        size_t xv = (v==0) ? 0 : nv-v;
        T v1 = T(0.5)*grid( u, v);
        T v2 = T(0.5)*grid(xu,xv);
        grid2.v(u,v) = std::complex<T>(v1+v2, v1-v2);
        }
      }
    });
  }

template<typename T> void hartley2_2D(mav<T,2> &arr, size_t vlim,
  bool first_fast, size_t nthreads)
  {
  size_t nu=arr.shape(0), nv=arr.shape(1);
  fmav<T> farr(arr);
  if (2*vlim<nv)
    {
    if (!first_fast)
      r2r_separable_hartley(farr, farr, {1}, T(1), nthreads);
    auto flo = farr.subarray({0,0},{farr.shape(0),vlim});
    r2r_separable_hartley(flo, flo, {0}, T(1), nthreads);
    auto fhi = farr.subarray({0,farr.shape(1)-vlim},{farr.shape(0),vlim});
    r2r_separable_hartley(fhi, fhi, {0}, T(1), nthreads);
    if (first_fast)
      r2r_separable_hartley(farr, farr, {1}, T(1), nthreads);
    }
  else
    r2r_separable_hartley(farr, farr, {0,1}, T(1), nthreads);
  execStatic((nu+1)/2-1, nthreads, 0, [&](Scheduler &sched)
    {
    while (auto rng=sched.getNext()) for(auto i=rng.lo+1; i<rng.hi+1; ++i)
      for(size_t j=1; j<(nv+1)/2; ++j)
         {
         T a = arr(i,j);
         T b = arr(nu-i,j);
         T c = arr(i,nv-j);
         T d = arr(nu-i,nv-j);
         arr.v(i,j) = T(0.5)*(a+b+c-d);
         arr.v(nu-i,j) = T(0.5)*(a+b+d-c);
         arr.v(i,nv-j) = T(0.5)*(a+c+d-b);
         arr.v(nu-i,nv-j) = T(0.5)*(b+c+d-a);
         }
     });
  }

using idx_t = uint32_t;

struct RowChan
  {
  idx_t row, chan;
  };

struct UVW
  {
  double u, v, w;
  UVW() {}
  UVW(double u_, double v_, double w_) : u(u_), v(v_), w(w_) {}
  UVW operator* (double fct) const
    { return UVW(u*fct, v*fct, w*fct); }
  void Flip() { u=-u; v=-v; w=-w; }
  bool FixW()
    {
    bool flip = w<0;
    if (flip) Flip();
    return flip;
    }
  };

class Baselines
  {
  protected:
    vector<UVW> coord;
    vector<double> f_over_c;
    idx_t nrows, nchan;
    idx_t shift, mask;
    double umax, vmax;

  public:
    template<typename T> Baselines(const mav<T,2> &coord_,
      const mav<T,1> &freq, bool negate_v=false)
      {
      constexpr double speedOfLight = 299792458.;
      MR_assert(coord_.shape(1)==3, "dimension mismatch");
      auto hugeval = size_t(~(idx_t(0)));
      MR_assert(coord_.shape(0)<hugeval, "too many entries in MS");
      MR_assert(coord_.shape(1)<hugeval, "too many entries in MS");
      MR_assert(coord_.size()<hugeval, "too many entries in MS");
      nrows = coord_.shape(0);
      nchan = freq.shape(0);
      shift=0;
      while((idx_t(1)<<shift)<nchan) ++shift;
      mask=(idx_t(1)<<shift)-1;
      MR_assert(nrows*(mask+1)<hugeval, "too many entries in MS");
      f_over_c.resize(nchan);
      double fcmax = 0;
      for (size_t i=0; i<nchan; ++i)
        {
        MR_assert(freq(i)>0, "negative channel frequency encountered");
        f_over_c[i] = freq(i)/speedOfLight;
        fcmax = max(fcmax, abs(f_over_c[i]));
        }
      coord.resize(nrows);
      double vfac = negate_v ? -1 : 1;
      vmax=0;
      for (size_t i=0; i<coord.size(); ++i)
        {
        coord[i] = UVW(coord_(i,0), vfac*coord_(i,1), coord_(i,2));
        umax = max(umax, abs(coord_(i,0)));
        vmax = max(vmax, abs(coord_(i,1)));
        }
      umax *= fcmax;
      vmax *= fcmax;
      }

    RowChan getRowChan(idx_t index) const
      { return RowChan{index>>shift, index&mask}; }

    UVW effectiveCoord(const RowChan &rc) const
      { return coord[rc.row]*f_over_c[rc.chan]; }
    UVW effectiveCoord(idx_t index) const
      { return effectiveCoord(getRowChan(index)); }
    size_t Nrows() const { return nrows; }
    size_t Nchannels() const { return nchan; }
    idx_t getIdx(idx_t irow, idx_t ichan) const
      { return ichan+(irow<<shift); }
    double Umax() const { return umax; }
    double Vmax() const { return vmax; }
  };

template<typename T> class GridderConfig
  {
  protected:
    size_t nx_dirty, ny_dirty, nu, nv;
    double ofactor;

  // FIXME: this should probably be done more cleanly
  public:
    shared_ptr<GriddingKernel<T>> krn;

  protected:
    double psx, psy;
    size_t supp, nsafe;
    double beta;
    size_t nthreads;
    double ushift, vshift;
    int maxiu0, maxiv0;
    size_t vlim;
    bool uv_side_fast;

    T phase (T x, T y, T w, bool adjoint) const
      {
      constexpr T pi = T(3.141592653589793238462643383279502884197);
      T tmp = 1-x-y;
      if (tmp<=0) return 1; // no phase factor beyond the horizon
      T nm1 = (-x-y)/(sqrt(tmp)+1); // more accurate form of sqrt(1-x-y)-1
      T phs = 2*pi*w*nm1;
      if (adjoint) phs *= -1;
      return phs;
      }

  public:
    GridderConfig(size_t nxdirty, size_t nydirty, size_t nu_, size_t nv_,
      double epsilon, double pixsize_x, double pixsize_y,
      const Baselines &baselines, size_t nthreads_)
      : nx_dirty(nxdirty), ny_dirty(nydirty), nu(nu_), nv(nv_),
        ofactor(min(double(nu)/nxdirty, double(nv)/nydirty)),
        krn(selectKernel<T>(ofactor, epsilon)),
        psx(pixsize_x), psy(pixsize_y),
        supp(krn->support()), nsafe((supp+1)/2),
        nthreads(nthreads_),
        ushift(supp*(-0.5)+1+nu), vshift(supp*(-0.5)+1+nv),
        maxiu0((nu+nsafe)-supp), maxiv0((nv+nsafe)-supp),
        vlim(min(nv/2, size_t(nv*baselines.Vmax()*psy+0.5*supp+1))),
        uv_side_fast(true)
      {
      size_t vlim2 = (nydirty+1)/2+(supp+1)/2;
      if (vlim2<vlim)
        {
        vlim = vlim2;
        uv_side_fast = false;
        }
      MR_assert(nu>=2*nsafe, "nu too small");
      MR_assert(nv>=2*nsafe, "nv too small");
      MR_assert((nx_dirty&1)==0, "nx_dirty must be even");
      MR_assert((ny_dirty&1)==0, "ny_dirty must be even");
      MR_assert((nu&1)==0, "nu must be even");
      MR_assert((nv&1)==0, "nv must be even");
      MR_assert(epsilon>0, "epsilon must be positive");
      MR_assert(pixsize_x>0, "pixsize_x must be positive");
      MR_assert(pixsize_y>0, "pixsize_y must be positive");
      }
    size_t Nxdirty() const { return nx_dirty; }
    size_t Nydirty() const { return ny_dirty; }
    double Pixsize_x() const { return psx; }
    double Pixsize_y() const { return psy; }
    size_t Nu() const { return nu; }
    size_t Nv() const { return nv; }
    size_t Supp() const { return supp; }
    size_t Nsafe() const { return nsafe; }
    size_t Nthreads() const { return nthreads; }

    void grid2dirty_post(mav<T,2> &tmav,
      mav<T,2> &dirty) const
      {
      checkShape(dirty.shape(), {nx_dirty, ny_dirty});
      auto cfu = krn->corfunc(nx_dirty/2+1, 1./nu, nthreads);
      auto cfv = krn->corfunc(ny_dirty/2+1, 1./nv, nthreads);
      execStatic(nx_dirty, nthreads, 0, [&](Scheduler &sched)
        {
        while (auto rng=sched.getNext()) for(auto i=rng.lo; i<rng.hi; ++i)
          {
          int icfu = abs(int(nx_dirty/2)-int(i));
          for (size_t j=0; j<ny_dirty; ++j)
            {
            int icfv = abs(int(ny_dirty/2)-int(j));
            size_t i2 = nu-nx_dirty/2+i;
            if (i2>=nu) i2-=nu;
            size_t j2 = nv-ny_dirty/2+j;
            if (j2>=nv) j2-=nv;
            dirty.v(i,j) = tmav(i2,j2)*T(cfu[icfu]*cfv[icfv]);
            }
          }
        });
      }
    void grid2dirty_post2(
      mav<complex<T>,2> &tmav, mav<T,2> &dirty, T w) const
      {
      checkShape(dirty.shape(), {nx_dirty,ny_dirty});
      double x0 = -0.5*nx_dirty*psx,
             y0 = -0.5*ny_dirty*psy;
      execStatic(nx_dirty/2+1, nthreads, 0, [&](Scheduler &sched)
        {
        using vtype = native_simd<T>;
        constexpr size_t vlen=vtype::size();
        size_t nvec = (ny_dirty/2+1+(vlen-1))/vlen;
        vector<vtype> ph(nvec), sp(nvec), cp(nvec);
        while (auto rng=sched.getNext()) for(auto i=rng.lo; i<rng.hi; ++i)
          {
          T fx = T(x0+i*psx);
          fx *= fx;
          size_t ix = nu-nx_dirty/2+i;
          if (ix>=nu) ix-=nu;
          size_t i2 = nx_dirty-i;
          size_t ix2 = nu-nx_dirty/2+i2;
          if (ix2>=nu) ix2-=nu;
          for (size_t j=0; j<=ny_dirty/2; ++j)
            {
            T fy = T(y0+j*psy);
            ph[j/vlen][j%vlen] = phase(fx, fy*fy, w, true);
            }
          for (size_t j=0; j<nvec; ++j)
            for (size_t k=0; k<vlen; ++k)
               sp[j][k]=sin(ph[j][k]);
          for (size_t j=0; j<nvec; ++j)
            for (size_t k=0; k<vlen; ++k)
              cp[j][k]=cos(ph[j][k]);
          if ((i>0)&&(i<i2))
            for (size_t j=0, jx=nv-ny_dirty/2; j<ny_dirty; ++j, jx=(jx+1>=nv)? jx+1-nv : jx+1)
              {
              size_t j2 = min(j, ny_dirty-j);
              T re = cp[j2/vlen][j2%vlen], im = sp[j2/vlen][j2%vlen];
              dirty.v(i,j) += tmav(ix,jx).real()*re - tmav(ix,jx).imag()*im;
              dirty.v(i2,j) += tmav(ix2,jx).real()*re - tmav(ix2,jx).imag()*im;
              }
          else
            for (size_t j=0, jx=nv-ny_dirty/2; j<ny_dirty; ++j, jx=(jx+1>=nv)? jx+1-nv : jx+1)
              {
              size_t j2 = min(j, ny_dirty-j);
              T re = cp[j2/vlen][j2%vlen], im = sp[j2/vlen][j2%vlen];
              dirty.v(i,j) += tmav(ix,jx).real()*re - tmav(ix,jx).imag()*im; // lower left
              }
          }
        });
      }

    void grid2dirty(const mav<T,2> &grid,
      mav<T,2> &dirty) const
      {
      checkShape(grid.shape(), {nu,nv});
      mav<T,2> tmav({nu,nv});
      tmav.apply(grid, [](T&a, T b) {a=b;});
      hartley2_2D<T>(tmav, vlim, uv_side_fast, nthreads);
      grid2dirty_post(tmav, dirty);
      }

    void grid2dirty_c_overwrite_wscreen_add
      (mav<complex<T>,2> &grid, mav<T,2> &dirty, T w) const
      {
      checkShape(grid.shape(), {nu,nv});
      fmav<complex<T>> inout(grid);
      if (2*vlim<nv)
        {
        if (!uv_side_fast)
          c2c(inout, inout, {1}, BACKWARD, T(1), nthreads);
        auto inout_lo = inout.subarray({0,0},{inout.shape(0),vlim});
        c2c(inout_lo, inout_lo, {0}, BACKWARD, T(1), nthreads);
        auto inout_hi = inout.subarray({0,inout.shape(1)-vlim},{inout.shape(0),vlim});
        c2c(inout_hi, inout_hi, {0}, BACKWARD, T(1), nthreads);
        if (uv_side_fast)
          c2c(inout, inout, {1}, BACKWARD, T(1), nthreads);
        }
      else
        c2c(inout, inout, {0,1}, BACKWARD, T(1), nthreads);
      grid2dirty_post2(grid, dirty, w);
      }

    void dirty2grid_pre(const mav<T,2> &dirty,
      mav<T,2> &grid) const
      {
      checkShape(dirty.shape(), {nx_dirty, ny_dirty});
      checkShape(grid.shape(), {nu, nv});
      auto cfu = krn->corfunc(nx_dirty/2+1, 1./nu, nthreads);
      auto cfv = krn->corfunc(ny_dirty/2+1, 1./nv, nthreads);
      grid.fill(0);
      execStatic(nx_dirty, nthreads, 0, [&](Scheduler &sched)
        {
        while (auto rng=sched.getNext()) for(auto i=rng.lo; i<rng.hi; ++i)
          {
          int icfu = abs(int(nx_dirty/2)-int(i));
          for (size_t j=0; j<ny_dirty; ++j)
            {
            int icfv = abs(int(ny_dirty/2)-int(j));
            size_t i2 = nu-nx_dirty/2+i;
            if (i2>=nu) i2-=nu;
            size_t j2 = nv-ny_dirty/2+j;
            if (j2>=nv) j2-=nv;
            grid.v(i2,j2) = dirty(i,j)*T(cfu[icfu]*cfv[icfv]);
            }
          }
        });
      }
    void dirty2grid_pre2(const mav<T,2> &dirty,
      mav<complex<T>,2> &grid, T w) const
      {
      checkShape(dirty.shape(), {nx_dirty, ny_dirty});
      checkShape(grid.shape(), {nu, nv});
      grid.fill(0);

      double x0 = -0.5*nx_dirty*psx,
             y0 = -0.5*ny_dirty*psy;
      execStatic(nx_dirty/2+1, nthreads, 0, [&](Scheduler &sched)
        {
        using vtype = native_simd<T>;
        constexpr size_t vlen=vtype::size();
        size_t nvec = (ny_dirty/2+1+(vlen-1))/vlen;
        vector<vtype> ph(nvec), sp(nvec), cp(nvec);
        while (auto rng=sched.getNext()) for(auto i=rng.lo; i<rng.hi; ++i)
          {
          T fx = T(x0+i*psx);
          fx *= fx;
          size_t ix = nu-nx_dirty/2+i;
          if (ix>=nu) ix-=nu;
          size_t i2 = nx_dirty-i;
          size_t ix2 = nu-nx_dirty/2+i2;
          if (ix2>=nu) ix2-=nu;
          for (size_t j=0; j<=ny_dirty/2; ++j)
            {
            T fy = T(y0+j*psy);
            ph[j/vlen][j%vlen] = phase(fx, fy*fy, w, false);
            }
          for (size_t j=0; j<nvec; ++j)
            for (size_t k=0; k<vlen; ++k)
               sp[j][k]=sin(ph[j][k]);
          for (size_t j=0; j<nvec; ++j)
            for (size_t k=0; k<vlen; ++k)
              cp[j][k]=cos(ph[j][k]);
          if ((i>0)&&(i<i2))
            for (size_t j=0, jx=nv-ny_dirty/2; j<ny_dirty; ++j, jx=(jx+1>=nv)? jx+1-nv : jx+1)
              {
              size_t j2 = min(j, ny_dirty-j);
              auto ws = complex<T>(cp[j2/vlen][j2%vlen],sp[j2/vlen][j2%vlen]);
              grid.v(ix,jx) = dirty(i,j)*ws; // lower left
              grid.v(ix2,jx) = dirty(i2,j)*ws; // lower right
              }
          else
            for (size_t j=0, jx=nv-ny_dirty/2; j<ny_dirty; ++j, jx=(jx+1>=nv)? jx+1-nv : jx+1)
              {
              size_t j2 = min(j, ny_dirty-j);
              auto ws = complex<T>(cp[j2/vlen][j2%vlen],sp[j2/vlen][j2%vlen]);
              grid.v(ix,jx) = dirty(i,j)*ws; // lower left
              }
          }
        });
      }

    void dirty2grid(const mav<T,2> &dirty,
      mav<T,2> &grid) const
      {
      dirty2grid_pre(dirty, grid);
      hartley2_2D<T>(grid, vlim, !uv_side_fast, nthreads);
      }

    void dirty2grid_c_wscreen(const mav<T,2> &dirty,
      mav<complex<T>,2> &grid, T w) const
      {
      dirty2grid_pre2(dirty, grid, w);
      fmav<complex<T>> inout(grid);
      if (2*vlim<nv)
        {
        if (uv_side_fast)
          c2c(inout, inout, {1}, FORWARD, T(1), nthreads);
        auto inout_lo = inout.subarray({0,0},{inout.shape(0),vlim});
        c2c(inout_lo, inout_lo, {0}, FORWARD, T(1), nthreads);
        auto inout_hi = inout.subarray({0,inout.shape(1)-vlim},{inout.shape(0),vlim});
        c2c(inout_hi, inout_hi, {0}, FORWARD, T(1), nthreads);
        if (!uv_side_fast)
          c2c(inout, inout, {1}, FORWARD, T(1), nthreads);
        }
      else
        c2c(inout, inout, {0,1}, FORWARD, T(1), nthreads);
      }

    void getpix(double u_in, double v_in, double &u, double &v, int &iu0, int &iv0) const
      {
      u=fmod1(u_in*psx)*nu;
      iu0 = min(int(u+ushift)-int(nu), maxiu0);
      v=fmod1(v_in*psy)*nv;
      iv0 = min(int(v+vshift)-int(nv), maxiv0);
      }
  };

constexpr int logsquare=4;

template<typename T> class Helper
  {
  private:
    using T2=complex<T>;
    const GridderConfig<T> &gconf;
    int nu, nv, nsafe, supp;
    const T2 *grid_r;
    T2 *grid_w;
    int su, sv, svvec;
    int iu0, iv0; // start index of the current visibility
    int bu0, bv0; // start index of the current buffer
    T wfac;

    mav<T,2> rbufr, rbufi, wbufr, wbufi;
    bool do_w_gridding;
    double w0, xdw;
    vector<std::mutex> &locks;

    void dump() const
      {
      if (bu0<-nsafe) return; // nothing written into buffer yet

      int idxu = (bu0+nu)%nu;
      int idxv0 = (bv0+nv)%nv;
      for (int iu=0; iu<su; ++iu)
        {
        int idxv = idxv0;
        {
        std::lock_guard<std::mutex> lock(locks[idxu]);
        for (int iv=0; iv<sv; ++iv)
          {
          grid_w[idxu*nv + idxv] += complex<T>(wbufr(iu,iv), wbufi(iu,iv));
          if (++idxv>=nv) idxv=0;
          }
        }
        if (++idxu>=nu) idxu=0;
        }
      }

    void load()
      {
      int idxu = (bu0+nu)%nu;
      int idxv0 = (bv0+nv)%nv;
      for (int iu=0; iu<su; ++iu)
        {
        int idxv = idxv0;
        for (int iv=0; iv<sv; ++iv)
          {
          rbufr.v(iu,iv) = grid_r[idxu*nv + idxv].real();
          rbufi.v(iu,iv) = grid_r[idxu*nv + idxv].imag();
          if (++idxv>=nv) idxv=0;
          }
        if (++idxu>=nu) idxu=0;
        }
      }

  public:
    size_t nvec;
    const T *p0rr, *p0ri;
    T *p0wr, *p0wi;
    static constexpr size_t vlen=native_simd<T>::size();
    union kbuf {
      T scalar[64];
      native_simd<T> simd[64/vlen];
      };
    kbuf buf;

    Helper(const GridderConfig<T> &gconf_, const T2 *grid_r_, T2 *grid_w_,
      vector<std::mutex> &locks_, double w0_=-1, double dw_=-1)
      : gconf(gconf_), nu(gconf.Nu()), nv(gconf.Nv()), nsafe(gconf.Nsafe()),
        supp(gconf.Supp()), grid_r(grid_r_),
        grid_w(grid_w_),
        su(2*nsafe+(1<<logsquare)),
        sv(2*nsafe+(1<<logsquare)),
        svvec(((sv+vlen-1)/vlen)*vlen),
        bu0(-1000000), bv0(-1000000),
        rbufr({size_t(su),size_t(svvec)*(grid_r!=nullptr)}),
        rbufi({size_t(su),size_t(svvec)*(grid_r!=nullptr)}),
        wbufr({size_t(su),size_t(svvec)*(grid_w!=nullptr)}),
        wbufi({size_t(su),size_t(svvec)*(grid_w!=nullptr)}),
        do_w_gridding(dw_>0),
        w0(w0_),
        xdw(T(1)/dw_),
        locks(locks_),
        nvec((supp+vlen-1)/vlen)
      {
      MR_assert(supp<=32, "support too large");
      }
    ~Helper() { if (grid_w) dump(); }

    int lineJump() const { return svvec; }
    T Wfac() const { return wfac; }
    void prep(const UVW &in)
      {
      const auto &krn(*(gconf.krn));
      double u, v;
      gconf.getpix(in.u, in.v, u, v, iu0, iv0);
      double xsupp=2./supp;
      double x0 = xsupp*(iu0-u);
      double y0 = xsupp*(iv0-v);
      krn.eval(T(x0), &buf.simd[0]);
      krn.eval(T(y0), &buf.simd[nvec]);
      if (do_w_gridding)
        wfac = krn.eval_single(T(xdw*xsupp*abs(w0-in.w)));
      if ((iu0<bu0) || (iv0<bv0) || (iu0+supp>bu0+su) || (iv0+supp>bv0+sv))
        {
        if (grid_w)
          {
          dump();
          wbufr.apply([](T &v){v=0;});
          wbufi.apply([](T &v){v=0;});
          }
        bu0=((((iu0+nsafe)>>logsquare)<<logsquare))-nsafe;
        bv0=((((iv0+nsafe)>>logsquare)<<logsquare))-nsafe;
        if (grid_r) load();
        }
      p0rr = grid_r ? &rbufr(iu0-bu0, iv0-bv0) : nullptr;
      p0ri = grid_r ? &rbufi(iu0-bu0, iv0-bv0) : nullptr;
      p0wr = grid_w ? &wbufr.v(iu0-bu0, iv0-bv0) : nullptr;
      p0wi = grid_w ? &wbufi.v(iu0-bu0, iv0-bv0) : nullptr;
      }
  };

template<class T, class Serv> class SubServ
  {
  private:
    Serv &srv;
    mav<idx_t,1> subidx;

  public:
    SubServ(Serv &orig, const mav<idx_t,1> &subidx_)
      : srv(orig), subidx(subidx_){}
    size_t Nvis() const { return subidx.size(); }
    const Baselines &getBaselines() const { return srv.getBaselines(); }
    UVW getCoord(size_t i) const
      { return srv.getCoord(subidx(i)); }
    complex<T> getVis(size_t i) const
      { return srv.getVis(subidx(i)); }
    idx_t getIdx(size_t i) const { return srv.getIdx(subidx(i)); }
    void setVis (size_t i, const complex<T> &v)
      { srv.setVis(subidx(i), v); }
    void addVis (size_t i, const complex<T> &v)
      { srv.addVis(subidx(i), v); }
  };

template<class T, class T2> class MsServ
  {
  private:
    const Baselines &baselines;
    mav<idx_t,1> idx;
    T2 ms;
    mav<T,2> wgt;
    size_t nvis;
    bool have_wgt;

  public:
    using Tsub = SubServ<T, MsServ>;

    MsServ(const Baselines &baselines_,
    const mav<idx_t,1> &idx_, T2 ms_, const mav<T,2> &wgt_)
      : baselines(baselines_), idx(idx_), ms(ms_), wgt(wgt_),
        nvis(idx.shape(0)), have_wgt(wgt.size()!=0)
      {
      checkShape(ms.shape(), {baselines.Nrows(), baselines.Nchannels()});
      if (have_wgt) checkShape(wgt.shape(), ms.shape());
      }
    Tsub getSubserv(const mav<idx_t,1> &subidx)
      { return Tsub(*this, subidx); }
    size_t Nvis() const { return nvis; }
    const Baselines &getBaselines() const { return baselines; }
    UVW getCoord(size_t i) const
      { return baselines.effectiveCoord(idx(i)); }
    complex<T> getVis(size_t i) const
      {
      auto rc = baselines.getRowChan(idx(i));
      return have_wgt ? ms(rc.row, rc.chan)*wgt(rc.row, rc.chan)
                      : ms(rc.row, rc.chan);
      }
    idx_t getIdx(size_t i) const { return idx(i); }
    void setVis (size_t i, const complex<T> &v)
      {
      auto rc = baselines.getRowChan(idx(i));
      ms.w()(rc.row, rc.chan) = have_wgt ? v*wgt(rc.row, rc.chan) : v;
      }
    void addVis (size_t i, const complex<T> &v)
      {
      auto rc = baselines.getRowChan(idx(i));
      ms.v(rc.row, rc.chan) += have_wgt ? v*wgt(rc.row, rc.chan) : v;
      }
  };
template<class T, class T2> MsServ<T, T2> makeMsServ
  (const Baselines &baselines,
   const mav<idx_t,1> &idx, T2 &ms, const mav<T,2> &wgt)
  { return MsServ<T, T2>(baselines, idx, ms, wgt); }


template<size_t NVEC, typename T, typename Serv> void x2grid_c_helper
  (const GridderConfig<T> &gconf, Serv &srv, mav<complex<T>,2> &grid,
  double w0=-1, double dw=-1)
  {
  size_t supp = gconf.Supp();
  size_t nthreads = gconf.Nthreads();
  bool do_w_gridding = dw>0;
  vector<std::mutex> locks(gconf.Nu());
  size_t np = srv.Nvis();
  execGuided(np, nthreads, 100, 0.2, [&](Scheduler &sched)
    {
    Helper<T> hlp(gconf, nullptr, grid.vdata(), locks, w0, dw);
    int jump = hlp.lineJump();
    const T * DUCC0_RESTRICT ku = hlp.buf.scalar;
    const auto * DUCC0_RESTRICT kv = hlp.buf.simd+NVEC;

    while (auto rng=sched.getNext()) for(auto ipart=rng.lo; ipart<rng.hi; ++ipart)
      {
      UVW coord = srv.getCoord(ipart);
      auto flip = coord.FixW();
      hlp.prep(coord);
      auto * DUCC0_RESTRICT ptrr = hlp.p0wr;
      auto * DUCC0_RESTRICT ptri = hlp.p0wi;
      auto v(srv.getVis(ipart));
      if (do_w_gridding) v*=hlp.Wfac();
      if (flip) v=conj(v);
      for (size_t cu=0; cu<supp; ++cu)
        {
        complex<T> tmp(v*ku[cu]);
        for (size_t cv=0; cv<NVEC; ++cv)
          {
          auto tr = native_simd<T>::loadu(ptrr+cv*hlp.vlen);
          tr += tmp.real()*kv[cv];
          tr.storeu(ptrr+cv*hlp.vlen);
          auto ti = native_simd<T>::loadu(ptri+cv*hlp.vlen);
          ti += tmp.imag()*kv[cv];
          ti.storeu(ptri+cv*hlp.vlen);
          }
        ptrr+=jump;
        ptri+=jump;
        }
      }
    });
  }

template<typename T, typename Serv> void x2grid_c_helper_general
  (const GridderConfig<T> &gconf, Serv &srv, mav<complex<T>,2> &grid,
  double w0=-1, double dw=-1)
  {
  size_t supp = gconf.Supp();
  size_t nthreads = gconf.Nthreads();
  bool do_w_gridding = dw>0;
  vector<std::mutex> locks(gconf.Nu());
  size_t np = srv.Nvis();
  execGuided(np, nthreads, 100, 0.2, [&](Scheduler &sched)
    {
    Helper<T> hlp(gconf, nullptr, grid.vdata(), locks, w0, dw);
    int jump = hlp.lineJump();
    const T * DUCC0_RESTRICT ku = hlp.buf.scalar;
    const auto * DUCC0_RESTRICT kv = hlp.buf.simd+hlp.nvec;

    while (auto rng=sched.getNext()) for(auto ipart=rng.lo; ipart<rng.hi; ++ipart)
      {
      UVW coord = srv.getCoord(ipart);
      auto flip = coord.FixW();
      hlp.prep(coord);
      auto * DUCC0_RESTRICT ptrr = hlp.p0wr;
      auto * DUCC0_RESTRICT ptri = hlp.p0wi;
      auto v(srv.getVis(ipart));
      if (do_w_gridding) v*=hlp.Wfac();
      if (flip) v=conj(v);
      for (size_t cu=0; cu<supp; ++cu)
        {
        complex<T> tmp(v*ku[cu]);
        for (size_t cv=0; cv<hlp.nvec; ++cv)
          {
          auto tr = native_simd<T>::loadu(ptrr+cv*hlp.vlen);
          tr += tmp.real()*kv[cv];
          tr.storeu(ptrr+cv*hlp.vlen);
          auto ti = native_simd<T>::loadu(ptri+cv*hlp.vlen);
          ti += tmp.imag()*kv[cv];
          ti.storeu(ptri+cv*hlp.vlen);
          }
        ptrr+=jump;
        ptri+=jump;
        }
      }
    });
  }

template<typename T, typename Serv> void x2grid_c
  (const GridderConfig<T> &gconf, Serv &srv, mav<complex<T>,2> &grid,
  double w0=-1, double dw=-1)
  {
  checkShape(grid.shape(), {gconf.Nu(), gconf.Nv()});
  MR_assert(grid.contiguous(), "grid is not contiguous");
  constexpr size_t vlen=native_simd<T>::size();
  size_t nvec((gconf.Supp()+vlen-1)/vlen);

  if (nvec==1)
    x2grid_c_helper<1>(gconf, srv, grid, w0, dw);
  else if (nvec==2)
    x2grid_c_helper<2>(gconf, srv, grid, w0, dw);
  else if (nvec==3)
    x2grid_c_helper<3>(gconf, srv, grid, w0, dw);
  else if (nvec==4)
    x2grid_c_helper<4>(gconf, srv, grid, w0, dw);
  else
    x2grid_c_helper_general(gconf, srv, grid, w0, dw);
  }

template<size_t NVEC, typename T, typename Serv> void grid2x_c_helper
  (const GridderConfig<T> &gconf, const mav<complex<T>,2> &grid,
  Serv &srv, double w0=-1, double dw=-1)
  {
  size_t supp = gconf.Supp();
  size_t nthreads = gconf.Nthreads();
  bool do_w_gridding = dw>0;
  vector<std::mutex> locks(gconf.Nu());

  // Loop over sampling points
  size_t np = srv.Nvis();
  execGuided(np, nthreads, 1000, 0.5, [&](Scheduler &sched)
    {
    Helper<T> hlp(gconf, grid.data(), nullptr, locks, w0, dw);
    int jump = hlp.lineJump();
    const T * DUCC0_RESTRICT ku = hlp.buf.scalar;
    const auto * DUCC0_RESTRICT kv = hlp.buf.simd+NVEC;

    while (auto rng=sched.getNext()) for(auto ipart=rng.lo; ipart<rng.hi; ++ipart)
      {
      UVW coord = srv.getCoord(ipart);
      auto flip = coord.FixW();
      hlp.prep(coord);
      native_simd<T> rr=0, ri=0;
      const auto * DUCC0_RESTRICT ptrr = hlp.p0rr;
      const auto * DUCC0_RESTRICT ptri = hlp.p0ri;
      for (size_t cu=0; cu<supp; ++cu)
        {
        native_simd<T> tmpr(0), tmpi(0);
        for (size_t cv=0; cv<NVEC; ++cv)
          {
          tmpr += kv[cv]*native_simd<T>::loadu(ptrr+hlp.vlen*cv);
          tmpi += kv[cv]*native_simd<T>::loadu(ptri+hlp.vlen*cv);
          }
        rr += ku[cu]*tmpr;
        ri += ku[cu]*tmpi;
        ptrr += jump;
        ptri += jump;
        }
      auto r = complex<T>(reduce(rr, std::plus<>()), reduce(ri, std::plus<>()));
      if (flip) r=conj(r);
      if (do_w_gridding) r*=hlp.Wfac();
      srv.addVis(ipart, r);
      }
    });
  }

template<typename T, typename Serv> void grid2x_c_helper_general
  (const GridderConfig<T> &gconf, const mav<complex<T>,2> &grid,
  Serv &srv, double w0=-1, double dw=-1)
  {
  size_t supp = gconf.Supp();
  size_t nthreads = gconf.Nthreads();
  bool do_w_gridding = dw>0;
  vector<std::mutex> locks(gconf.Nu());

  // Loop over sampling points
  size_t np = srv.Nvis();
  execGuided(np, nthreads, 1000, 0.5, [&](Scheduler &sched)
    {
    Helper<T> hlp(gconf, grid.data(), nullptr, locks, w0, dw);
    int jump = hlp.lineJump();
    const T * DUCC0_RESTRICT ku = hlp.buf.scalar;
    const auto * DUCC0_RESTRICT kv = hlp.buf.simd+hlp.nvec;

    while (auto rng=sched.getNext()) for(auto ipart=rng.lo; ipart<rng.hi; ++ipart)
      {
      UVW coord = srv.getCoord(ipart);
      auto flip = coord.FixW();
      hlp.prep(coord);
      native_simd<T> rr=0, ri=0;
      const auto * DUCC0_RESTRICT ptrr = hlp.p0rr;
      const auto * DUCC0_RESTRICT ptri = hlp.p0ri;
      for (size_t cu=0; cu<supp; ++cu)
        {
        native_simd<T> tmpr(0), tmpi(0);
        for (size_t cv=0; cv<hlp.nvec; ++cv)
          {
          tmpr += kv[cv]*native_simd<T>::loadu(ptrr+hlp.vlen*cv);
          tmpi += kv[cv]*native_simd<T>::loadu(ptri+hlp.vlen*cv);
          }
        rr += ku[cu]*tmpr;
        ri += ku[cu]*tmpi;
        ptrr += jump;
        ptri += jump;
        }
      auto r = complex<T>(reduce(rr, std::plus<>()), reduce(ri, std::plus<>()));
      if (flip) r=conj(r);
      if (do_w_gridding) r*=hlp.Wfac();
      srv.addVis(ipart, r);
      }
    });
  }

template<typename T, typename Serv> void grid2x_c
  (const GridderConfig<T> &gconf, const mav<complex<T>,2> &grid,
  Serv &srv, double w0=-1, double dw=-1)
  {
  checkShape(grid.shape(), {gconf.Nu(), gconf.Nv()});
  MR_assert(grid.contiguous(), "grid is not contiguous");
  constexpr size_t vlen=native_simd<T>::size();
  size_t nvec((gconf.Supp()+vlen-1)/vlen);

  if (nvec==1)
    grid2x_c_helper<1>(gconf, grid, srv, w0, dw);
  else if (nvec==2)
    grid2x_c_helper<2>(gconf, grid, srv, w0, dw);
  else if (nvec==3)
    grid2x_c_helper<3>(gconf, grid, srv, w0, dw);
  else if (nvec==4)
    grid2x_c_helper<4>(gconf, grid, srv, w0, dw);
  else
    grid2x_c_helper_general(gconf, grid, srv, w0, dw);
  }

template<typename T> void apply_global_corrections(const GridderConfig<T> &gconf,
  mav<T,2> &dirty, double dw, bool divide_by_n)
  {
  auto nx_dirty=gconf.Nxdirty();
  auto ny_dirty=gconf.Nydirty();
  size_t nthreads = gconf.Nthreads();
  auto psx=gconf.Pixsize_x();
  auto psy=gconf.Pixsize_y();
  double x0 = -0.5*nx_dirty*psx,
         y0 = -0.5*ny_dirty*psy;
  auto cfu = gconf.krn->corfunc(nx_dirty/2+1, 1./gconf.Nu(), nthreads);
  auto cfv = gconf.krn->corfunc(ny_dirty/2+1, 1./gconf.Nv(), nthreads);
  execStatic(nx_dirty/2+1, nthreads, 0, [&](Scheduler &sched)
    {
    while (auto rng=sched.getNext()) for(auto i=rng.lo; i<rng.hi; ++i)
      {
      auto fx = T(x0+i*psx);
      fx *= fx;
      for (size_t j=0; j<=ny_dirty/2; ++j)
        {
        auto fy = T(y0+j*psy);
        fy*=fy;
        T fct = 0;
        auto tmp = 1-fx-fy;
        if (tmp>=0)
          {
          auto nm1 = (-fx-fy)/(sqrt(tmp)+1); // accurate form of sqrt(1-x-y)-1
          fct = T(gconf.krn->corfunc(nm1*dw));
          if (divide_by_n)
            fct /= nm1+1;
          }
        else // beyond the horizon, don't really know what to do here
          {
          if (divide_by_n)
            fct=0;
          else
            {
            auto nm1 = sqrt(-tmp)-1;
            fct = T(gconf.krn->corfunc(nm1*dw));
            }
          }
        fct *= T(cfu[nx_dirty/2-i]*cfv[ny_dirty/2-j]);
        size_t i2 = nx_dirty-i, j2 = ny_dirty-j;
        dirty.v(i,j)*=fct;
        if ((i>0)&&(i<i2))
          {
          dirty.v(i2,j)*=fct;
          if ((j>0)&&(j<j2))
            dirty.v(i2,j2)*=fct;
          }
        if ((j>0)&&(j<j2))
          dirty.v(i,j2)*=fct;
        }
      }
    });
  }

auto calc_share(size_t nshares, size_t myshare, size_t nwork)
  {
  size_t nbase = nwork/nshares;
  size_t additional = nwork%nshares;
  size_t lo = myshare*nbase + ((myshare<additional) ? myshare : additional);
  size_t hi = lo+nbase+(myshare<additional);
  return make_tuple(lo, hi);
  }

template<typename T, typename Serv> class WgridHelper
  {
  private:
    Serv &srv;
    double wmin, dw;
    size_t nplanes, supp, nthreads;
    vector<vector<idx_t>> minplane;
    size_t verbosity;

    int curplane;
    vector<idx_t> subidx;

    static void wminmax(const Serv &srv, double &wmin, double &wmax)
      {
      size_t nvis = srv.Nvis();

      wmin= 1e38;
      wmax=-1e38;
      // FIXME maybe this can be done more intelligently
      for (size_t ipart=0; ipart<nvis; ++ipart)
        {
        auto wval = abs(srv.getCoord(ipart).w);
        wmin = min(wmin,wval);
        wmax = max(wmax,wval);
        }
      }

    template<typename T2> static void update_idx(vector<T2> &v, const vector<T2> &add,
      const vector<T2> &del, size_t nthreads)
      {
      MR_assert(v.size()>=del.size(), "must not happen");
      vector<T2> res;
      res.reserve((v.size()+add.size())-del.size());
#if 0
      auto iin=v.begin(), ein=v.end();
      auto iadd=add.begin(), eadd=add.end();
      auto irem=del.begin(), erem=del.end();

      while(iin!=ein)
        {
        if ((irem!=erem) && (*iin==*irem))
          {  ++irem; ++iin; } // skip removed entry
        else if ((iadd!=eadd) && (*iadd<*iin))
           res.push_back(*(iadd++)); // add new entry
        else
          res.push_back(*(iin++));
        }
      MR_assert(irem==erem, "must not happen");
      while(iadd!=eadd)
        res.push_back(*(iadd++));
#else
      if (v.empty()) //special case
        {
        MR_assert(del.empty(), "must not happen");
        for (auto x: add)
          res.push_back(x);
        }
      else
        {
        res.resize((v.size()+add.size())-del.size());
        execParallel(nthreads, [&](Scheduler &sched) {
          auto tid = sched.thread_num();
          auto [lo, hi] = calc_share(nthreads, tid, v.size());
          if (lo==hi) return; // if interval is empty, do nothing
          auto iin=v.begin()+lo, ein=v.begin()+hi;
          auto iadd = (iin==v.begin()) ? add.begin() : lower_bound(add.begin(), add.end(), *iin);
          auto eadd = (ein==v.end()) ? add.end() : lower_bound(add.begin(), add.end(), *ein);
          auto irem = (iin==v.begin()) ? del.begin() : lower_bound(del.begin(), del.end(), *iin);
          auto erem = (ein==v.end()) ? del.end() : lower_bound(del.begin(), del.end(), *ein);
          auto iout = res.begin()+lo-(irem-del.begin())+(iadd-add.begin());
          while(iin!=ein)
            {
            if ((irem!=erem) && (*iin==*irem))
              { ++irem; ++iin; } // skip removed entry
            else if ((iadd!=eadd) && (*iadd<*iin))
              *(iout++) = *(iadd++); // add new entry
            else
              *(iout++) = *(iin++);
            }
          MR_assert(irem==erem, "must not happen");
          while(iadd!=eadd)
            *(iout++) = *(iadd++);
          });
        }
#endif
      MR_assert(res.size()==(v.size()+add.size())-del.size(), "must not happen");
      v.swap(res);
      }

  public:
    WgridHelper(const GridderConfig<T> &gconf, Serv &srv_, size_t verbosity_)
      : srv(srv_), supp(gconf.Supp()), nthreads(gconf.Nthreads()),
        verbosity(verbosity_), curplane(-1)
      {
      size_t nvis = srv.Nvis();
      double wmax;

      wminmax(srv, wmin, wmax);
      if (verbosity>0) cout << "Using " << nthreads << " thread"
                            << ((nthreads!=1) ? "s" : "") << endl;
      if (verbosity>0) cout << "W range: " << wmin << " to " << wmax << endl;

      double x0 = -0.5*gconf.Nxdirty()*gconf.Pixsize_x(),
             y0 = -0.5*gconf.Nydirty()*gconf.Pixsize_y();
      double nmin = sqrt(max(1.-x0*x0-y0*y0,0.))-1.;
      if (x0*x0+y0*y0>1.)
        nmin = -sqrt(abs(1.-x0*x0-y0*y0))-1.;
      dw = 0.25/abs(nmin);
      nplanes = size_t((wmax-wmin)/dw+supp);
      wmin = (wmin+wmax)*0.5 - 0.5*(nplanes-1)*dw;
      if (verbosity>0) cout << "Kernel support: " << supp << endl;
      if (verbosity>0) cout << "nplanes: " << nplanes << endl;

      minplane.resize(nplanes);
#if 0
      // extra short, but potentially inefficient version:
      for (size_t ipart=0; ipart<nvis; ++ipart)
        {
        int plane0 = max(0,int(1+(abs(srv.getCoord(ipart).w)-(0.5*supp*dw)-wmin)/dw));
        minplane[plane0].push_back(idx_t(ipart));
        }
#else
      // more efficient: precalculate final vector sizes and avoid reallocations
      vector<int> p0(nvis);
      mav<size_t,2> cnt({nthreads, nplanes+16}); // safety distance against false sharing
      execStatic(nvis, nthreads, 0, [&](Scheduler &sched)
        {
        auto tid=sched.thread_num();
        while (auto rng=sched.getNext()) for(auto i=rng.lo; i<rng.hi; ++i)
          {
          p0[i] = max(0,int(1+(abs(srv.getCoord(i).w)-(0.5*supp*dw)-wmin)/dw));
          ++cnt.v(tid, p0[i]);
          }
        });

      for (size_t p=0; p<nplanes; ++p)
        {
        size_t offset=0;
        for (idx_t tid=0; tid<nthreads; ++tid)
          {
          auto tmp = cnt(tid, p);
          cnt.v(tid, p) = offset;
          offset += tmp;
          }
        minplane[p].resize(offset);
        }

      // fill minplane
      execStatic(nvis, nthreads, 0, [&](Scheduler &sched)
        {
        auto tid=sched.thread_num();
        while (auto rng=sched.getNext()) for(auto i=rng.lo; i<rng.hi; ++i)
          minplane[p0[i]][cnt.v(tid,p0[i])++]=idx_t(i);
        });
#endif
      }

    typename Serv::Tsub getSubserv() const
      {
      auto subidx2 = mav<idx_t, 1>(subidx.data(), {subidx.size()});
      return srv.getSubserv(subidx2);
      }
    double W() const { return wmin+curplane*dw; }
    size_t Nvis() const { return subidx.size(); }
    double DW() const { return dw; }
    bool advance()
      {
      if (++curplane>=int(nplanes)) return false;
      update_idx(subidx, minplane[curplane], curplane>=int(supp) ? minplane[curplane-supp] : vector<idx_t>(), nthreads);
      if (verbosity>1)
        cout << "Working on plane " << curplane << " containing " << subidx.size()
             << " visibilities" << endl;
      return true;
      }
  };

template<typename T, typename Serv> void x2dirty(
  const GridderConfig<T> &gconf, Serv &srv, mav<T,2> &dirty,
  bool do_wstacking, size_t verbosity)
  {
  if (do_wstacking)
    {
    if (verbosity>0) cout << "Gridding using improved w-stacking" << endl;
    WgridHelper<T, Serv> hlp(gconf, srv, verbosity);
    double dw = hlp.DW();
    dirty.fill(0);
    mav<complex<T>,2> grid({gconf.Nu(),gconf.Nv()});
    while(hlp.advance())  // iterate over w planes
      {
      if (hlp.Nvis()==0) continue;
      grid.fill(0);
      auto serv = hlp.getSubserv();
      x2grid_c(gconf, serv, grid, hlp.W(), dw);
      gconf.grid2dirty_c_overwrite_wscreen_add(grid, dirty, T(hlp.W()));
      }
    // correct for w gridding etc.
    apply_global_corrections(gconf, dirty, dw, true);
    }
  else
    {
    if (verbosity>0)
      cout << "Gridding without w-stacking: " << srv.Nvis()
           << " visibilities" << endl;
    if (verbosity>0) cout << "Using " << gconf.Nthreads() << " threads" << endl;

    mav<complex<T>,2> grid({gconf.Nu(), gconf.Nv()});
    grid.fill(0.);
    x2grid_c(gconf, srv, grid);
    mav<T,2> rgrid(grid.shape());
    complex2hartley(grid, rgrid, gconf.Nthreads());
    gconf.grid2dirty(rgrid, dirty);
    }
  }

template<typename T, typename Serv> void dirty2x(
  const GridderConfig<T> &gconf,  const mav<T,2> &dirty,
  Serv &srv, bool do_wstacking, size_t verbosity)
  {
  if (do_wstacking)
    {
    size_t nx_dirty=gconf.Nxdirty(), ny_dirty=gconf.Nydirty();
    if (verbosity>0) cout << "Degridding using improved w-stacking" << endl;
    WgridHelper<T, Serv> hlp(gconf, srv, verbosity);
    double dw = hlp.DW();
    mav<T,2> tdirty({nx_dirty,ny_dirty});
    for (size_t i=0; i<nx_dirty; ++i)
      for (size_t j=0; j<ny_dirty; ++j)
        tdirty.v(i,j) = dirty(i,j);
    // correct for w gridding etc.
    apply_global_corrections(gconf, tdirty, dw, true);
    mav<complex<T>,2> grid({gconf.Nu(),gconf.Nv()});
    while(hlp.advance())  // iterate over w planes
      {
      if (hlp.Nvis()==0) continue;
      gconf.dirty2grid_c_wscreen(tdirty, grid, T(hlp.W()));
      auto serv = hlp.getSubserv();
      grid2x_c(gconf, grid, serv, hlp.W(), dw);
      }
    }
  else
    {
    if (verbosity>0)
      cout << "Degridding without w-stacking: " << srv.Nvis()
           << " visibilities" << endl;
    if (verbosity>0) cout << "Using " << gconf.Nthreads() << " threads" << endl;

    mav<T,2> grid({gconf.Nu(), gconf.Nv()});
    gconf.dirty2grid(dirty, grid);
    mav<complex<T>,2> grid2(grid.shape());
    hartley2complex(grid, grid2, gconf.Nthreads());
    grid2x_c(gconf, grid2, srv);
    }
  }

void calc_share(size_t nshares, size_t myshare, size_t nwork, size_t &lo,
  size_t &hi)
  {
  size_t nbase = nwork/nshares;
  size_t additional = nwork%nshares;
  lo = myshare*nbase + ((myshare<additional) ? myshare : additional);
  hi = lo+nbase+(myshare<additional);
  }


template<typename T> vector<idx_t> getWgtIndices(const Baselines &baselines,
  const GridderConfig<T> &gconf, const mav<T,2> &wgt,
  const mav<complex<T>,2> &ms)
  {
  size_t nrow=baselines.Nrows(),
         nchan=baselines.Nchannels(),
         nsafe=gconf.Nsafe(),
         nthreads=gconf.Nthreads();
  bool have_wgt=wgt.size()!=0;
  if (have_wgt) checkShape(wgt.shape(),{nrow,nchan});
  bool have_ms=ms.size()!=0;
  if (have_ms) checkShape(ms.shape(), {nrow,nchan});
  constexpr int side=1<<logsquare;
  size_t nbu = (gconf.Nu()+1+side-1) >> logsquare,
         nbv = (gconf.Nv()+1+side-1) >> logsquare;
  mav<idx_t,2> acc({nthreads, (nbu*nbv+16)}); // the 16 is safety distance to avoid false sharing
  vector<idx_t> tmp(nrow*nchan);

  execStatic(nrow, nthreads, 0, [&](Scheduler &sched)
    {
    idx_t tid = sched.thread_num();
    while (auto rng=sched.getNext()) for(auto irow=idx_t(rng.lo); irow<idx_t(rng.hi); ++irow)
      {
      for (idx_t ichan=0, idx=irow*nchan; ichan<nchan; ++ichan, ++idx)
        if (((!have_ms ) || (norm(ms(irow,ichan))!=0)) &&
            ((!have_wgt) || (wgt(irow,ichan)!=0)))
        {
        auto uvw = baselines.effectiveCoord(RowChan{irow,idx_t(ichan)});
        if (uvw.w<0) uvw.Flip();
        double u, v;
        int iu0, iv0;
        gconf.getpix(uvw.u, uvw.v, u, v, iu0, iv0);
        iu0 = (iu0+nsafe)>>logsquare;
        iv0 = (iv0+nsafe)>>logsquare;
        tmp[idx] = nbv*iu0 + iv0;
        ++acc.v(tid, tmp[idx]);
        }
      else
        tmp[idx] = ~idx_t(0);
      }
    });

  idx_t offset=0;
  for (idx_t idx=0; idx<nbu*nbv; ++idx)
    for (idx_t tid=0; tid<nthreads; ++tid)
      {
      auto tmp = acc(tid, idx);
      acc.v(tid, idx) = offset;
      offset += tmp;
      }

  vector<idx_t> res(offset);
  execStatic(nrow, nthreads, 0, [&](Scheduler &sched)
    {
    idx_t tid = sched.thread_num();
    while (auto rng=sched.getNext()) for(auto irow=idx_t(rng.lo); irow<idx_t(rng.hi); ++irow)
      for (size_t ichan=0, idx=irow*nchan; ichan<nchan; ++ichan, ++idx)
        if (tmp[idx]!=(~idx_t(0)))
          res[acc.v(tid, tmp[idx])++] = baselines.getIdx(irow, ichan);
    });
  return res;
  }

template<typename T> void ms2dirty(const mav<double,2> &uvw,
  const mav<double,1> &freq, const mav<complex<T>,2> &ms,
  const mav<T,2> &wgt, double pixsize_x, double pixsize_y, size_t nu, size_t nv, double epsilon,
  bool do_wstacking, size_t nthreads, mav<T,2> &dirty, size_t verbosity,
  bool negate_v=false)
  {
  Baselines baselines(uvw, freq, negate_v);
  // adjust for increased error when gridding in 2 or 3 dimensions
  epsilon /= do_wstacking ? 3 : 2;
  GridderConfig<T> gconf(dirty.shape(0), dirty.shape(1), nu, nv, epsilon, pixsize_x, pixsize_y, baselines, nthreads);
  auto idx = getWgtIndices(baselines, gconf, wgt, ms);
  auto idx2 = mav<idx_t,1>(idx.data(),{idx.size()});
  auto serv = makeMsServ(baselines,idx2,ms,wgt);
  x2dirty(gconf, serv, dirty, do_wstacking, verbosity);
  }

template<typename T> void dirty2ms(const mav<double,2> &uvw,
  const mav<double,1> &freq, const mav<T,2> &dirty,
  const mav<T,2> &wgt, double pixsize_x, double pixsize_y, size_t nu, size_t nv,double epsilon,
  bool do_wstacking, size_t nthreads, mav<complex<T>,2> &ms,
  size_t verbosity, bool negate_v=false)
  {
  Baselines baselines(uvw, freq, negate_v);
  // adjust for increased error when gridding in 2 or 3 dimensions
  epsilon /= do_wstacking ? 3 : 2;
  GridderConfig<T> gconf(dirty.shape(0), dirty.shape(1), nu, nv, epsilon, pixsize_x, pixsize_y, baselines, nthreads);
  mav<complex<T>,2> null_ms(nullptr, {0,0}, true);
  auto idx = getWgtIndices(baselines, gconf, wgt, null_ms);
  auto idx2 = mav<idx_t,1>(idx.data(),{idx.size()});
  ms.fill(0);
  auto serv = makeMsServ(baselines,idx2,ms,wgt);
  dirty2x(gconf, dirty, serv, do_wstacking, verbosity);
  }

} // namespace detail_gridder

// public names
using detail_gridder::ms2dirty;
using detail_gridder::dirty2ms;

} // namespace ducc0

#endif
