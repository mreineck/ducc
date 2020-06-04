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

#include "mr_util/infra/error_handling.h"
#include "mr_util/math/fft.h"
#include "mr_util/infra/threading.h"
#include "mr_util/infra/useful_macros.h"
#include "mr_util/infra/mav.h"
#include "mr_util/math/es_kernel.h"

namespace gridder {

namespace detail {

using namespace std;
using namespace mr;

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

template<typename T> void hartley2_2D(const mav<T,2> &in,
  mav<T,2> &out, size_t nthreads)
  {
  MR_assert(in.conformable(out), "shape mismatch");
  size_t nu=in.shape(0), nv=in.shape(1);
  fmav<T> fin(in), fout(out);
  r2r_separable_hartley(fin, fout, {0,1}, T(1), nthreads);
  execStatic((nu+1)/2-1, nthreads, 0, [&](Scheduler &sched)
    {
    while (auto rng=sched.getNext()) for(auto i=rng.lo+1; i<rng.hi+1; ++i)
      for(size_t j=1; j<(nv+1)/2; ++j)
         {
         T a = out(i,j);
         T b = out(nu-i,j);
         T c = out(i,nv-j);
         T d = out(nu-i,nv-j);
         out.v(i,j) = T(0.5)*(a+b+c-d);
         out.v(nu-i,j) = T(0.5)*(a+b+d-c);
         out.v(i,nv-j) = T(0.5)*(a+c+d-b);
         out.v(nu-i,nv-j) = T(0.5)*(b+c+d-a);
         }
     });
  }

/* Compute correction factors for the ES gridding kernel
   This implementation follows eqs. (3.8) to (3.10) of Barnett et al. 2018 */
vector<double> correction_factors(size_t n, double ofactor, size_t nval, size_t supp,
  size_t nthreads)
  {
  ES_Kernel kernel(supp, ofactor, nthreads);
  return kernel.correction_factors(n, nval, nthreads);
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
      for (size_t i=0; i<nchan; ++i)
        {
        MR_assert(freq(i)>0, "negative channel frequency encountered");
        f_over_c[i] = freq(i)/speedOfLight;
        }
      coord.resize(nrows);
      if (negate_v)
        for (size_t i=0; i<coord.size(); ++i)
          coord[i] = UVW(coord_(i,0), -coord_(i,1), coord_(i,2));
      else
        for (size_t i=0; i<coord.size(); ++i)
          coord[i] = UVW(coord_(i,0), coord_(i,1), coord_(i,2));
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
  };

class GridderConfig
  {
  protected:
    size_t nx_dirty, ny_dirty, nu, nv;
    double ofactor, eps, psx, psy;
    size_t supp, nsafe;
    double beta;
    size_t nthreads;
    double ushift, vshift;
    int maxiu0, maxiv0;

    template<typename T> complex<T> wscreen(T x, T y, T w, bool adjoint) const
      {
      constexpr T pi = T(3.141592653589793238462643383279502884197);
      T tmp = 1-x-y;
      if (tmp<=0) return 1; // no phase factor beyond the horizon
      T nm1 = (-x-y)/(sqrt(tmp)+1); // more accurate form of sqrt(1-x-y)-1
      T phase = 2*pi*w*nm1;
      if (adjoint) phase *= -1;
      return complex<T>(cos(phase), sin(phase));
      }

  public:
    GridderConfig(size_t nxdirty, size_t nydirty, size_t nu_, size_t nv_,
      double epsilon, double pixsize_x, double pixsize_y, size_t nthreads_)
      : nx_dirty(nxdirty), ny_dirty(nydirty), nu(nu_), nv(nv_),
        ofactor(min(double(nu)/nxdirty, double(nv)/nydirty)),
        eps(epsilon),
        psx(pixsize_x), psy(pixsize_y),
        supp(ES_Kernel::get_supp(epsilon, ofactor)), nsafe((supp+1)/2),
        beta(ES_Kernel::get_beta(supp, ofactor)*supp),
        nthreads(nthreads_),
        ushift(supp*(-0.5)+1+nu), vshift(supp*(-0.5)+1+nv),
        maxiu0((nu+nsafe)-supp), maxiv0((nv+nsafe)-supp)
      {
      MR_assert(nu>=2*nsafe, "nu too small");
      MR_assert(nv>=2*nsafe, "nv too small");
      MR_assert((nx_dirty&1)==0, "nx_dirty must be even");
      MR_assert((ny_dirty&1)==0, "ny_dirty must be even");
      MR_assert((nu&1)==0, "nu must be even");
      MR_assert((nv&1)==0, "nv must be even");
      MR_assert(epsilon>0, "epsilon must be positive");
      MR_assert(pixsize_x>0, "pixsize_x must be positive");
      MR_assert(pixsize_y>0, "pixsize_y must be positive");
      MR_assert(ofactor>=1.175,
        "oversampling factor too small (>=1.2 recommended)");
      }
    GridderConfig(size_t nxdirty, size_t nydirty,
      double epsilon, double pixsize_x, double pixsize_y, size_t nthreads_)
      : GridderConfig(nxdirty, nydirty, max<size_t>(30,2*nxdirty),
                      max<size_t>(30,2*nydirty), epsilon, pixsize_x,
                      pixsize_y, nthreads_) {}
    size_t Nxdirty() const { return nx_dirty; }
    size_t Nydirty() const { return ny_dirty; }
    double Epsilon() const { return eps; }
    double Pixsize_x() const { return psx; }
    double Pixsize_y() const { return psy; }
    size_t Nu() const { return nu; }
    size_t Nv() const { return nv; }
    size_t Supp() const { return supp; }
    size_t Nsafe() const { return nsafe; }
    double Beta() const { return beta; }
    size_t Nthreads() const { return nthreads; }
    double Ofactor() const{ return ofactor; }

    template<typename T> void grid2dirty_post(mav<T,2> &tmav,
      mav<T,2> &dirty) const
      {
      checkShape(dirty.shape(), {nx_dirty, ny_dirty});
      auto cfu = correction_factors(nu, ofactor, nx_dirty/2+1, supp, nthreads);
      auto cfv = correction_factors(nv, ofactor, ny_dirty/2+1, supp, nthreads);
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
            // FIXME: for some reason g++ warns about double-to-float conversion
            // here, even though there is an explicit cast...
            dirty.v(i,j) = tmav(i2,j2)*T(cfu[icfu]*cfv[icfv]);
            }
          }
        });
      }
    template<typename T> void grid2dirty_post2(
      mav<complex<T>,2> &tmav, mav<T,2> &dirty, T w) const
      {
      checkShape(dirty.shape(), {nx_dirty,ny_dirty});
      double x0 = -0.5*nx_dirty*psx,
             y0 = -0.5*ny_dirty*psy;
      execStatic(nx_dirty/2+1, nthreads, 0, [&](Scheduler &sched)
        {
        while (auto rng=sched.getNext()) for(auto i=rng.lo; i<rng.hi; ++i)
          {
          T fx = T(x0+i*psx);
          fx *= fx;
          for (size_t j=0; j<=ny_dirty/2; ++j)
            {
            T fy = T(y0+j*psy);
            auto ws = wscreen(fx, fy*fy, w, true);
            size_t ix = nu-nx_dirty/2+i;
            if (ix>=nu) ix-=nu;
            size_t jx = nv-ny_dirty/2+j;
            if (jx>=nv) jx-=nv;
            dirty.v(i,j) += (tmav(ix,jx)*ws).real(); // lower left
            size_t i2 = nx_dirty-i, j2 = ny_dirty-j;
            size_t ix2 = nu-nx_dirty/2+i2;
            if (ix2>=nu) ix2-=nu;
            size_t jx2 = nv-ny_dirty/2+j2;
            if (jx2>=nv) jx2-=nv;
            if ((i>0)&&(i<i2))
              {
              dirty.v(i2,j) += (tmav(ix2,jx)*ws).real(); // lower right
              if ((j>0)&&(j<j2))
                dirty.v(i2,j2) += (tmav(ix2,jx2)*ws).real(); // upper right
              }
            if ((j>0)&&(j<j2))
              dirty.v(i,j2) += (tmav(ix,jx2)*ws).real(); // upper left
            }
          }
        });
      }

    template<typename T> void grid2dirty(const mav<T,2> &grid,
      mav<T,2> &dirty) const
      {
      checkShape(grid.shape(), {nu,nv});
      mav<T,2> tmav({nu,nv});
      hartley2_2D<T>(grid, tmav, nthreads);
      grid2dirty_post(tmav, dirty);
      }

    template<typename T> void grid2dirty_c_overwrite_wscreen_add
      (mav<complex<T>,2> &grid, mav<T,2> &dirty, T w) const
      {
      checkShape(grid.shape(), {nu,nv});
      fmav<complex<T>> inout(grid);
      c2c(inout, inout, {0,1}, BACKWARD, T(1), nthreads);
      grid2dirty_post2(grid, dirty, w);
      }

    template<typename T> void dirty2grid_pre(const mav<T,2> &dirty,
      mav<T,2> &grid) const
      {
      checkShape(dirty.shape(), {nx_dirty, ny_dirty});
      checkShape(grid.shape(), {nu, nv});
      auto cfu = correction_factors(nu, ofactor, nx_dirty/2+1, supp, nthreads);
      auto cfv = correction_factors(nv, ofactor, ny_dirty/2+1, supp, nthreads);
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
    template<typename T> void dirty2grid_pre2(const mav<T,2> &dirty,
      mav<complex<T>,2> &grid, T w) const
      {
      checkShape(dirty.shape(), {nx_dirty, ny_dirty});
      checkShape(grid.shape(), {nu, nv});
      grid.fill(0);

      double x0 = -0.5*nx_dirty*psx,
             y0 = -0.5*ny_dirty*psy;
      execStatic(nx_dirty/2+1, nthreads, 0, [&](Scheduler &sched)
        {
        while (auto rng=sched.getNext()) for(auto i=rng.lo; i<rng.hi; ++i)
          {
          T fx = T(x0+i*psx);
          fx *= fx;
          for (size_t j=0; j<=ny_dirty/2; ++j)
            {
            T fy = T(y0+j*psy);
            auto ws = wscreen(fx, fy*fy, w, false);
            size_t ix = nu-nx_dirty/2+i;
            if (ix>=nu) ix-=nu;
            size_t jx = nv-ny_dirty/2+j;
            if (jx>=nv) jx-=nv;
            grid.v(ix,jx) = dirty(i,j)*ws; // lower left
            size_t i2 = nx_dirty-i, j2 = ny_dirty-j;
            size_t ix2 = nu-nx_dirty/2+i2;
            if (ix2>=nu) ix2-=nu;
            size_t jx2 = nv-ny_dirty/2+j2;
            if (jx2>=nv) jx2-=nv;
            if ((i>0)&&(i<i2))
              {
              grid.v(ix2,jx) = dirty(i2,j)*ws; // lower right
              if ((j>0)&&(j<j2))
                grid.v(ix2,jx2) = dirty(i2,j2)*ws; // upper right
              }
            if ((j>0)&&(j<j2))
              grid.v(ix,jx2) = dirty(i,j2)*ws; // upper left
            }
          }
        });
      }

    template<typename T> void dirty2grid(const mav<T,2> &dirty,
      mav<T,2> &grid) const
      {
      dirty2grid_pre(dirty, grid);
      hartley2_2D<T>(grid, grid, nthreads);
      }

    template<typename T> void dirty2grid_c_wscreen(const mav<T,2> &dirty,
      mav<complex<T>,2> &grid, T w) const
      {
      dirty2grid_pre2(dirty, grid, w);
      fmav<complex<T>> inout(grid);
      c2c(inout, inout, {0,1}, FORWARD, T(1), nthreads);
      }

    void getpix(double u_in, double v_in, double &u, double &v, int &iu0, int &iv0) const
      {
      u=fmod1(u_in*psx)*nu;
      iu0 = min(int(u+ushift)-int(nu), maxiu0);
      v=fmod1(v_in*psy)*nv;
      iv0 = min(int(v+vshift)-int(nv), maxiv0);
      }

    template<typename T> void apply_wscreen(const mav<complex<T>,2> &dirty,
      mav<complex<T>,2> &dirty2, double w, bool adjoint) const
      {
      checkShape(dirty.shape(), {nx_dirty, ny_dirty});
      checkShape(dirty2.shape(), {nx_dirty, ny_dirty});

      double x0 = -0.5*nx_dirty*psx,
             y0 = -0.5*ny_dirty*psy;
      execStatic(nx_dirty/2+1, nthreads, 0, [&](Scheduler &sched)
        {
        while (auto rng=sched.getNext()) for(auto i=rng.lo; i<rng.hi; ++i)
          {
          T fx = T(x0+i*psx);
          fx *= fx;
          for (size_t j=0; j<=ny_dirty/2; ++j)
            {
            T fy = T(y0+j*psy);
            auto ws = wscreen(fx, fy*fy, T(w), adjoint);
            dirty2(i,j) = dirty(i,j)*ws; // lower left
            size_t i2 = nx_dirty-i, j2 = ny_dirty-j;
            if ((i>0)&&(i<i2))
              {
              dirty2(i2,j) = dirty(i2,j)*ws; // lower right
              if ((j>0)&&(j<j2))
                dirty2(i2,j2) = dirty(i2,j2)*ws; // upper right
              }
            if ((j>0)&&(j<j2))
              dirty2(i,j2) = dirty(i,j2)*ws; // upper left
            }
          }
        });
      }
  };

constexpr int logsquare=4;

template<typename T, typename T2=complex<T>> class Helper
  {
  private:
    const GridderConfig &gconf;
    int nu, nv, nsafe, supp;
    T beta;
    const T2 *grid_r;
    T2 *grid_w;
    int su, sv;
    int iu0, iv0; // start index of the current visibility
    int bu0, bv0; // start index of the current buffer

    vector<T2> rbuf, wbuf;
    bool do_w_gridding;
    double w0, xdw;
    size_t nexp;
    size_t nvecs;
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
          grid_w[idxu*nv + idxv] += wbuf[iu*sv + iv];
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
          rbuf[iu*sv + iv] = grid_r[idxu*nv + idxv];
          if (++idxv>=nv) idxv=0;
          }
        if (++idxu>=nu) idxu=0;
        }
      }

  public:
    const T2 *p0r;
    T2 *p0w;
    T kernel[64] MRUTIL_ALIGNED(64);
    static constexpr size_t vlen=native_simd<T>::size();

    Helper(const GridderConfig &gconf_, const T2 *grid_r_, T2 *grid_w_,
      vector<std::mutex> &locks_, double w0_=-1, double dw_=-1)
      : gconf(gconf_), nu(gconf.Nu()), nv(gconf.Nv()), nsafe(gconf.Nsafe()),
        supp(gconf.Supp()), beta(T(gconf.Beta())), grid_r(grid_r_),
        grid_w(grid_w_), su(2*nsafe+(1<<logsquare)), sv(2*nsafe+(1<<logsquare)),
        bu0(-1000000), bv0(-1000000),
        rbuf(su*sv*(grid_r!=nullptr),T(0)),
        wbuf(su*sv*(grid_w!=nullptr),T(0)),
        do_w_gridding(dw_>0),
        w0(w0_),
        xdw(T(1)/dw_),
        nexp(2*supp + do_w_gridding),
        nvecs(vlen*((nexp+vlen-1)/vlen)),
        locks(locks_)
      {}
    ~Helper() { if (grid_w) dump(); }

    int lineJump() const { return sv; }
    T Wfac() const { return kernel[2*supp]; }
    void prep(const UVW &in)
      {
      double u, v;
      gconf.getpix(in.u, in.v, u, v, iu0, iv0);
      double xsupp=2./supp;
      double x0 = xsupp*(iu0-u);
      double y0 = xsupp*(iv0-v);
      for (int i=0; i<supp; ++i)
        {
        kernel[i  ] = T(x0+i*xsupp);
        kernel[i+supp] = T(y0+i*xsupp);
        }
      if (do_w_gridding)
        kernel[2*supp] = min(T(1), T(xdw*xsupp*abs(w0-in.w)));
      for (size_t i=nexp; i<nvecs; ++i)
        kernel[i]=0;
      for (size_t i=0; i<nvecs; ++i)
        {
        kernel[i] = T(1) - kernel[i]*kernel[i];
        kernel[i] = (kernel[i]<0) ? T(-200.) : beta*(sqrt(kernel[i])-T(1));
        }
      for (size_t i=0; i<nvecs; ++i)
        kernel[i] = exp(kernel[i]);
      if ((iu0<bu0) || (iv0<bv0) || (iu0+supp>bu0+su) || (iv0+supp>bv0+sv))
        {
        if (grid_w) { dump(); fill(wbuf.begin(), wbuf.end(), T(0)); }
        bu0=((((iu0+nsafe)>>logsquare)<<logsquare))-nsafe;
        bv0=((((iv0+nsafe)>>logsquare)<<logsquare))-nsafe;
        if (grid_r) load();
        }
      p0r = grid_r ? rbuf.data() + sv*(iu0-bu0) + iv0-bv0 : nullptr;
      p0w = grid_w ? wbuf.data() + sv*(iu0-bu0) + iv0-bv0 : nullptr;
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

template<typename T, typename Serv> void x2grid_c
  (const GridderConfig &gconf, Serv &srv, mav<complex<T>,2> &grid,
  double w0=-1, double dw=-1)
  {
  checkShape(grid.shape(), {gconf.Nu(), gconf.Nv()});
  MR_assert(grid.contiguous(), "grid is not contiguous");
  size_t supp = gconf.Supp();
  size_t nthreads = gconf.Nthreads();
  bool do_w_gridding = dw>0;
  vector<std::mutex> locks(gconf.Nu());

  size_t np = srv.Nvis();
  execGuided(np, nthreads, 100, 0.2, [&](Scheduler &sched)
    {
    Helper<T> hlp(gconf, nullptr, grid.vdata(), locks, w0, dw);
    int jump = hlp.lineJump();
    const T * MRUTIL_RESTRICT ku = hlp.kernel;
    const T * MRUTIL_RESTRICT kv = hlp.kernel+supp;

    while (auto rng=sched.getNext()) for(auto ipart=rng.lo; ipart<rng.hi; ++ipart)
      {
      UVW coord = srv.getCoord(ipart);
      auto flip = coord.FixW();
      hlp.prep(coord);
      auto * MRUTIL_RESTRICT ptr = hlp.p0w;
      auto v(srv.getVis(ipart));
      if (do_w_gridding) v*=hlp.Wfac();
      if (flip) v=conj(v);
      for (size_t cu=0; cu<supp; ++cu)
        {
        complex<T> tmp(v*ku[cu]);
        size_t cv=0;
        for (; cv+3<supp; cv+=4)
          {
          ptr[cv  ] += tmp*kv[cv  ];
          ptr[cv+1] += tmp*kv[cv+1];
          ptr[cv+2] += tmp*kv[cv+2];
          ptr[cv+3] += tmp*kv[cv+3];
          }
        for (; cv<supp; ++cv)
          ptr[cv] += tmp*kv[cv];
        ptr+=jump;
        }
      }
    });
  }

template<typename T, typename Serv> void grid2x_c
  (const GridderConfig &gconf, const mav<complex<T>,2> &grid,
  Serv &srv, double w0=-1, double dw=-1)
  {
  checkShape(grid.shape(), {gconf.Nu(), gconf.Nv()});
  MR_assert(grid.contiguous(), "grid is not contiguous");
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
    const T * MRUTIL_RESTRICT ku = hlp.kernel;
    const T * MRUTIL_RESTRICT kv = hlp.kernel+supp;

    while (auto rng=sched.getNext()) for(auto ipart=rng.lo; ipart<rng.hi; ++ipart)
      {
      UVW coord = srv.getCoord(ipart);
      auto flip = coord.FixW();
      hlp.prep(coord);
      complex<T> r = 0;
      const auto * MRUTIL_RESTRICT ptr = hlp.p0r;
      for (size_t cu=0; cu<supp; ++cu)
        {
        complex<T> tmp(0);
        size_t cv=0;
        for (; cv+3<supp; cv+=4)
          tmp += ptr[cv  ]*kv[cv  ]
               + ptr[cv+1]*kv[cv+1]
               + ptr[cv+2]*kv[cv+2]
               + ptr[cv+3]*kv[cv+3];
        for (; cv<supp; ++cv)
          tmp += ptr[cv] * kv[cv];
        r += tmp*ku[cu];
        ptr += jump;
        }
      if (flip) r=conj(r);
      if (do_w_gridding) r*=hlp.Wfac();
      srv.addVis(ipart, r);
      }
    });
  }

template<typename T> void apply_global_corrections(const GridderConfig &gconf,
  mav<T,2> &dirty, const ES_Kernel &kernel, double dw, bool divide_by_n)
  {
  auto nx_dirty=gconf.Nxdirty();
  auto ny_dirty=gconf.Nydirty();
  size_t nthreads = gconf.Nthreads();
  auto psx=gconf.Pixsize_x();
  auto psy=gconf.Pixsize_y();
  double x0 = -0.5*nx_dirty*psx,
         y0 = -0.5*ny_dirty*psy;
  auto cfu = correction_factors(gconf.Nu(), gconf.Ofactor(),
                                nx_dirty/2+1, gconf.Supp(), nthreads);
  auto cfv = correction_factors(gconf.Nv(), gconf.Ofactor(),
                                ny_dirty/2+1, gconf.Supp(), nthreads);
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
          fct = T(kernel.corfac(nm1*dw));
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
            fct = T(kernel.corfac(nm1*dw));
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

template<typename Serv> class WgridHelper
  {
  private:
    Serv &srv;
    double wmin, dw;
    size_t nplanes, supp;
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

    template<typename T> static void update_idx(vector<T> &v, const vector<T> &add,
      const vector<T> &del)
      {
      MR_assert(v.size()>=del.size(), "must not happen");
      vector<T> res;
      res.reserve((v.size()+add.size())-del.size());
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
      MR_assert(res.size()==(v.size()+add.size())-del.size(), "must not happen");
      v.swap(res);
      }

  public:
    WgridHelper(const GridderConfig &gconf, Serv &srv_, size_t verbosity_)
      : srv(srv_), verbosity(verbosity_), curplane(-1)
      {
      size_t nvis = srv.Nvis();
      size_t nthreads = gconf.Nthreads();
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
      nplanes = size_t((wmax-wmin)/dw+2);
      dw = (1.+1e-13)*(wmax-wmin)/(nplanes-1);

      supp = gconf.Supp();
      wmin -= (0.5*supp-1)*dw;
      wmax += (0.5*supp-1)*dw;
      nplanes += supp-2;
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
      vector<size_t> cnt(nplanes,0);
      for(size_t ipart=0; ipart<nvis; ++ipart)
        {
        int plane0 = max(0,int(1+(abs(srv.getCoord(ipart).w)-(0.5*supp*dw)-wmin)/dw));
        ++cnt[plane0];
        }

      // fill minplane
      for (size_t j=0; j<nplanes; ++j)
        minplane[j].resize(cnt[j]);
      vector<size_t> ofs(nplanes, 0);
      for (size_t ipart=0; ipart<nvis; ++ipart)
        {
        int plane0 = max(0,int(1+(abs(srv.getCoord(ipart).w)-(0.5*supp*dw)-wmin)/dw));
        minplane[plane0][ofs[plane0]++]=idx_t(ipart);
        }
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
      update_idx(subidx, minplane[curplane], curplane>=int(supp) ? minplane[curplane-supp] : vector<idx_t>());
      if (verbosity>1)
        cout << "Working on plane " << curplane << " containing " << subidx.size()
             << " visibilities" << endl;
      return true;
      }
  };

template<typename T, typename Serv> void x2dirty(
  const GridderConfig &gconf, Serv &srv, mav<T,2> &dirty,
  bool do_wstacking, size_t verbosity)
  {
  if (do_wstacking)
    {
    size_t nthreads = gconf.Nthreads();
    if (verbosity>0) cout << "Gridding using improved w-stacking" << endl;
    WgridHelper<Serv> hlp(gconf, srv, verbosity);
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
    apply_global_corrections(gconf, dirty, ES_Kernel(gconf.Supp(), gconf.Ofactor(), nthreads), dw, true);
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
  const GridderConfig &gconf,  const mav<T,2> &dirty,
  Serv &srv, bool do_wstacking, size_t verbosity)
  {
  if (do_wstacking)
    {
    size_t nx_dirty=gconf.Nxdirty(), ny_dirty=gconf.Nydirty();
    size_t nthreads = gconf.Nthreads();
    if (verbosity>0) cout << "Degridding using improved w-stacking" << endl;
    WgridHelper<Serv> hlp(gconf, srv, verbosity);
    double dw = hlp.DW();
    mav<T,2> tdirty({nx_dirty,ny_dirty});
    for (size_t i=0; i<nx_dirty; ++i)
      for (size_t j=0; j<ny_dirty; ++j)
        tdirty.v(i,j) = dirty(i,j);
    // correct for w gridding etc.
    apply_global_corrections(gconf, tdirty, ES_Kernel(gconf.Supp(), gconf.Ofactor(), nthreads), dw, true);
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
  const GridderConfig &gconf, const mav<T,2> &wgt,
  const mav<complex<T>,2> &ms)
  {
  size_t nrow=baselines.Nrows(),
         nchan=baselines.Nchannels(),
         nsafe=gconf.Nsafe();
  bool have_wgt=wgt.size()!=0;
  if (have_wgt) checkShape(wgt.shape(),{nrow,nchan});
  bool have_ms=ms.size()!=0;
  if (have_ms) checkShape(ms.shape(), {nrow,nchan});
  constexpr int side=1<<logsquare;
  size_t nbu = (gconf.Nu()+1+side-1) >> logsquare,
         nbv = (gconf.Nv()+1+side-1) >> logsquare;
  vector<idx_t> acc(nbu*nbv+1,0);
  vector<idx_t> tmp(nrow*nchan);

  for (idx_t irow=0, idx=0; irow<nrow; ++irow)
    for (idx_t ichan=0; ichan<nchan; ++ichan, ++idx)
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
        ++acc[nbv*iu0 + iv0 + 1];
        tmp[idx] = nbv*iu0 + iv0;
        }
      else
        tmp[idx] = ~idx_t(0);

  for (size_t i=1; i<acc.size(); ++i)
    acc[i] += acc[i-1];

  vector<idx_t> res(acc.back());
  for (size_t irow=0, idx=0; irow<nrow; ++irow)
    for (size_t ichan=0; ichan<nchan; ++ichan, ++idx)
      if (tmp[idx]!=(~idx_t(0)))
        res[acc[tmp[idx]]++] = baselines.getIdx(irow, ichan);
  return res;
  }

template<typename T> void ms2dirty(const mav<double,2> &uvw,
  const mav<double,1> &freq, const mav<complex<T>,2> &ms,
  const mav<T,2> &wgt, double pixsize_x, double pixsize_y, size_t nu, size_t nv, double epsilon,
  bool do_wstacking, size_t nthreads, mav<T,2> &dirty, size_t verbosity,
  bool negate_v=false)
  {
  Baselines baselines(uvw, freq, negate_v);
  GridderConfig gconf(dirty.shape(0), dirty.shape(1), nu, nv, epsilon, pixsize_x, pixsize_y, nthreads);
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
  GridderConfig gconf(dirty.shape(0), dirty.shape(1), nu, nv, epsilon, pixsize_x, pixsize_y, nthreads);
  mav<complex<T>,2> null_ms(nullptr, {0,0}, true);
  auto idx = getWgtIndices(baselines, gconf, wgt, null_ms);
  auto idx2 = mav<idx_t,1>(idx.data(),{idx.size()});
  ms.fill(0);
  auto serv = makeMsServ(baselines,idx2,ms,wgt);
  dirty2x(gconf, dirty, serv, do_wstacking, verbosity);
  }

} // namespace detail

// public names
using detail::ms2dirty;
using detail::dirty2ms;
} // namespace gridder

#endif
