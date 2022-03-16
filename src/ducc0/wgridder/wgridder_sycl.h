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

/* Copyright (C) 2022 Max-Planck-Society
   Authors: Martin Reinecke, Philipp Arras */

#ifndef DUCC0_WGRIDDER_SYCL_H
#define DUCC0_WGRIDDER_SYCL_H

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

#include "ducc0/infra/error_handling.h"
#include "ducc0/math/constants.h"
#include "ducc0/infra/threading.h"
#include "ducc0/infra/misc_utils.h"
#include "ducc0/infra/useful_macros.h"
#include "ducc0/infra/mav.h"
#include "ducc0/infra/timers.h"
#include "ducc0/math/gridding_kernel.h"
#include "ducc0/infra/sycl_utils.h"

namespace ducc0 {

namespace detail_wgridder_sycl {

#if defined(DUCC0_USE_SYCL)

using namespace std;
// the next line is necessary to address some sloppy name choices in hipSYCL
using std::min, std::max;

using namespace cl;

template<typename T> T sqr(T val) { return val*val; }

template<size_t ndim> void checkShape
  (const array<size_t, ndim> &shp1, const array<size_t, ndim> &shp2)
  { MR_assert(shp1==shp2, "shape mismatch"); }

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

    RowchanRange(uint32_t row_, uint16_t ch_begin_, uint16_t ch_end_)
      : row(row_), ch_begin(ch_begin_), ch_end(ch_end_) {}
    uint16_t size() const { return ch_end-ch_begin; }
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
    vector<double> f_over_c;
    size_t nrows, nchan;
    double umax, vmax;

  public:
    Baselines() = default;
    template<typename T> Baselines(const cmav<T,2> &coord_,
      const cmav<T,1> &freq, bool negate_v=false)
      {
      constexpr double speedOfLight = 299792458.;
      MR_assert(coord_.shape(1)==3, "dimension mismatch");
      nrows = coord_.shape(0);
      nchan = freq.shape(0);
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
      umax=vmax=0;
      for (size_t i=0; i<coord.size(); ++i)
        {
        coord[i] = UVW(coord_(i,0), vfac*coord_(i,1), coord_(i,2));
        umax = max(umax, abs(coord_(i,0)));
        vmax = max(vmax, abs(coord_(i,1)));
        }
      umax *= fcmax;
      vmax *= fcmax;
      }

    UVW effectiveCoord(size_t row, size_t chan) const
      { return coord[row]*f_over_c[chan]; }
    double absEffectiveW(size_t row, size_t chan) const
      { return abs(coord[row].w*f_over_c[chan]); }
    UVW baseCoord(size_t row) const
      { return coord[row]; }
    void prefetchRow(size_t row) const
      { DUCC0_PREFETCH_R(&coord[row]); }
    double ffact(size_t chan) const
      { return f_over_c[chan];}
    size_t Nrows() const { return nrows; }
    size_t Nchannels() const { return nchan; }
    double Umax() const { return umax; }
    double Vmax() const { return vmax; }

    const vector<UVW> &getUVW_raw() const { return coord; }
    const vector<double> &get_f_over_c() const { return f_over_c; }
  };

constexpr int logsquare=5;

template<typename Tcalc, typename Tacc, typename Tms, typename Timg> class Params
  {
  private:
    bool gridding;
    TimerHierarchy timers;
    const cmav<complex<Tms>,2> &ms_in;
    vmav<complex<Tms>,2> &ms_out;
    const cmav<Timg,2> &dirty_in;
    vmav<Timg,2> &dirty_out;
    const cmav<Tms,2> &wgt;
    const cmav<uint8_t,2> &mask;
    double pixsize_x, pixsize_y;
    size_t nxdirty, nydirty;
    double epsilon;
    bool do_wgridding;
    size_t nthreads;
    size_t verbosity;
    bool negate_v, divide_by_n;
    double sigma_min, sigma_max;

    Baselines bl;
    vector<RowchanRange> ranges;
    vector<uint32_t> vissum;
    vector<pair<Uvwidx, uint32_t>> blockstart;
 
    double wmin_d, wmax_d;
    size_t nvis;
    double wmin, dw;
    size_t nplanes;
    double nm1min, nm1max;

    double lshift, mshift, nshift;
    bool shifting, lmshift, no_nshift;

    size_t nu, nv;
    double ofactor;

    shared_ptr<HornerKernel> krn;

    size_t supp, nsafe;
    double ushift, vshift;
    int maxiu0, maxiv0;

    static_assert(sizeof(Tcalc)<=sizeof(Tacc), "bad type combination");
    static_assert(sizeof(Tms)<=sizeof(Tcalc), "bad type combination");
    static_assert(sizeof(Timg)<=sizeof(Tcalc), "bad type combination");

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

    void countRanges()
      {
      timers.push("building index");
      size_t nrow=bl.Nrows(),
             nchan=bl.Nchannels();

      if (do_wgridding)
        {
        dw = 0.5/ofactor/max(abs(nm1max+nshift), abs(nm1min+nshift));
        nplanes = size_t((wmax_d-wmin_d)/dw+supp);
        MR_assert(nplanes<(size_t(1)<<16), "too many w planes");
        wmin = (wmin_d+wmax_d)*0.5 - 0.5*(nplanes-1)*dw;
        }
      else
        dw = wmin = nplanes = 0;
      size_t nbunch = do_wgridding ? supp : 1;
      size_t max_allowed = 1024;

      struct tmp2
        {
        size_t sz=0;
        vector<vector<RowchanRange>> v;
        void add(const RowchanRange &rng, size_t max_allowed)
          {
          if (v.empty() || (sz>=max_allowed))
            { v.emplace_back(); sz=0; }
          v.back().push_back(rng);
          sz += rng.ch_end-rng.ch_begin;
          }
        };
      using Vmap = map<Uvwidx, tmp2>;
      struct bufmap
        {
        Vmap m;
        mutex mut;
        uint64_t dummy[8]; // separator to keep every entry on a different cache line
        };
      checkShape(wgt.shape(),{nrow,nchan});
      checkShape(ms_in.shape(), {nrow,nchan});
      checkShape(mask.shape(), {nrow,nchan});

      size_t ntiles_u = (nu>>logsquare) + 20;
      vector<bufmap> buf(ntiles_u);
      auto chunk = max<size_t>(1, nrow/(20*nthreads));
      auto xdw = 1./dw;
      auto shift = dw-(0.5*supp*dw)-wmin;
      execDynamic(nrow, nthreads, chunk, [&](Scheduler &sched)
        {
        vector<pair<uint16_t, uint16_t>> interbuf;
        while (auto rng=sched.getNext())
        for(auto irow=rng.lo; irow<rng.hi; ++irow)
          {
          bool on=false;
          Uvwidx uvwlast(0,0,0);
          size_t chan0=0;

          auto flush=[&]()
            {
            if (interbuf.empty()) return;
            auto tileidx = uvwlast.tile_u;
            lock_guard<mutex> lock(buf[tileidx].mut);
            auto &loc(buf[tileidx].m[uvwlast]);
            for (auto &x: interbuf)
              loc.add(RowchanRange(irow, x.first, x.second), max_allowed);
            interbuf.clear();
            };
          auto add=[&](uint16_t cb, uint16_t ce)
            { interbuf.emplace_back(cb, ce); };

          for (size_t ichan=0; ichan<nchan; ++ichan)
            {
            if (norm(ms_in(irow,ichan))*wgt(irow,ichan)*mask(irow,ichan)!=0)
              {
              auto uvw = bl.effectiveCoord(irow, ichan);
              uvw.FixW();
              double udum, vdum;
              int iu0, iv0, iw;
              getpix(uvw.u, uvw.v, udum, vdum, iu0, iv0);
              iu0 = (iu0+nsafe)>>logsquare;
              iv0 = (iv0+nsafe)>>logsquare;
              iw = do_wgridding ? max(0,int((uvw.w+shift)*xdw)) : 0;
              Uvwidx uvwcur(iu0, iv0, iw);
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

      size_t nranges=0, nblocks=0;
      for (const auto &x: buf)
        for (const auto &y: x.m)
          {
          nblocks += y.second.v.size();
          for (const auto &z: y.second.v)
            nranges += z.size();
          }
      blockstart.reserve(nblocks);
      ranges.reserve(nranges);
      size_t visacc=0;
      vissum.reserve(nranges+1);

// FIXME parallelize!
      for (auto &x: buf)
        for (auto &y: x.m)
          for (auto &z: y.second.v)
            {
            blockstart.push_back({y.first, uint32_t(ranges.size())});
            for (const auto &zz: z)
              {
              ranges.push_back(zz);
              vissum.push_back(visacc);
              visacc += zz.ch_end-zz.ch_begin; 
              }
            // z.resize(0); ??
            }
      vissum.push_back(visacc);

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
           << ", eps=" << (epsilon * (do_wgridding ? 3 : 2))
           << endl;
      cout << "  nrow=" << bl.Nrows() << ", nchan=" << bl.Nchannels()
           << ", nvis=" << nvis << "/" << (bl.Nrows()*bl.Nchannels()) << endl;
      if (do_wgridding)
        cout << "  w=[" << wmin_d << "; " << wmax_d << "], min(n-1)=" << nm1min
             << ", dw=" << dw << ", wmax/dw=" << wmax_d/dw << endl;
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
          double nval;
          if (tmp <= 1.) // northern hemisphere
            nval = sqrt(1.-tmp) - 1.;
          else
            nval = -sqrt(tmp-1.) -1.;
          nm1min = min(nm1min, nval);
          nm1max = max(nm1max, nval);
          }
      nshift = (no_nshift||(!do_wgridding)) ? 0. : -0.5*(nm1max+nm1min);
      shifting = lmshift || (nshift!=0);

      auto idx = getAvailableKernels<Tcalc>(epsilon, sigma_min, sigma_max);
      double mincost = 1e300;
      constexpr double nref_fft=2048;
      constexpr double costref_fft=0.0693;
      size_t minnu=0, minnv=0, minidx=KernelDB.size();
      for (size_t i=0; i<idx.size(); ++i)
        {
        const auto &krn(KernelDB[idx[i]]);
        auto supp = krn.W;
        auto ofactor = krn.ofactor;
        size_t nu=2*good_size_complex(size_t(nxdirty*ofactor*0.5)+1);
        size_t nv=2*good_size_complex(size_t(nydirty*ofactor*0.5)+1);
        double logterm = log(nu*nv)/log(nref_fft*nref_fft);
// FIXME: 0.3 is an estimated fudge factor
        double fftcost = 0.3*nu/nref_fft*nv/nref_fft*logterm*costref_fft;
        double gridcost = 2.2e-10*nvis*(supp*supp + ((2*supp+1)*(supp+3)));
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
      size_t nrow=bl.Nrows(),
             nchan=bl.Nchannels();
      checkShape(wgt.shape(),{nrow,nchan});
      checkShape(ms_in.shape(), {nrow,nchan});
      checkShape(mask.shape(), {nrow,nchan});

      nvis=0;
      wmin_d=1e300;
      wmax_d=-1e300;
      mutex mut;
      execParallel(nrow, nthreads, [&](size_t lo, size_t hi)
        {
        double lwmin_d=1e300, lwmax_d=-1e300;
        size_t lnvis=0;
        for(auto irow=lo; irow<hi; ++irow)
          for (size_t ichan=0, idx=irow*nchan; ichan<nchan; ++ichan, ++idx)
//            if (mask(irow,ichan) && (wgt(irow, ichan)!=0) && (norm(ms_in(irow,ichan)!=0)))
            if (norm(ms_in(irow,ichan))*wgt(irow,ichan)*mask(irow,ichan) != 0)
              {
              ++lnvis;
              double w = bl.absEffectiveW(irow, ichan);
              lwmin_d = min(lwmin_d, w);
              lwmax_d = max(lwmax_d, w);
              }
        {
        lock_guard<mutex> lock(mut);
        wmin_d = min(wmin_d, lwmin_d);
        wmax_d = max(wmax_d, lwmax_d);
        nvis += lnvis;
        }
        });
      timers.pop();
      }

class Baselines_GPU_prep
  {
  public:
    sycl::buffer<UVW,1> buf_uvw;
    sycl::buffer<double,1> buf_freq;

    Baselines_GPU_prep(const Baselines &bl)
      : buf_uvw(make_sycl_buffer(bl.getUVW_raw())),
        buf_freq(make_sycl_buffer(bl.get_f_over_c())) {}
  };

class Baselines_GPU
  {
  protected:
    sycl::accessor<UVW,1,sycl::access::mode::read> acc_uvw;
    sycl::accessor<double,1,sycl::access::mode::read> acc_f_over_c;

  public:
    Baselines_GPU(Baselines_GPU_prep &prep, sycl::handler &cgh)
      : acc_uvw(prep.buf_uvw, cgh, sycl::read_only),
        acc_f_over_c(prep.buf_freq, cgh, sycl::read_only)
      {}

    UVW effectiveCoord(size_t row, size_t chan) const
      {
      double f = acc_f_over_c[chan];
      return acc_uvw[row]*f;
      }
    double absEffectiveW(size_t row, size_t chan) const
      { return sycl::fabs(acc_uvw[row].w*acc_f_over_c[chan]); }
    UVW baseCoord(size_t row) const
      { return acc_uvw[row]; }
    double ffact(size_t chan) const
      { return acc_f_over_c[chan];}
    size_t Nrows() const { return acc_uvw.get_range().get(0); }
    size_t Nchannels() const { return acc_f_over_c.get_range().get(0); }
  };

class IndexComputer
  {
  public:
    sycl::buffer<RowchanRange, 1> buf_ranges;
    sycl::buffer<uint32_t, 1> buf_vissum;
    sycl::buffer<pair<Uvwidx, uint32_t>> buf_blockstart;

    IndexComputer(const vector<RowchanRange> &ranges,
      const vector<uint32_t> &vissum,
      const vector<pair<Uvwidx, uint32_t>> &blockstart)
      : buf_ranges(make_sycl_buffer(ranges)),
        buf_vissum(make_sycl_buffer(vissum)),
        buf_blockstart(make_sycl_buffer(blockstart))
      {}
  };

class GlobalCorrector
  {
  private:
    const Params &par;
    vector<double> cfu, cfv;
    sycl::buffer<double,1> bufcfu, bufcfv; 

    template<size_t maxsz> class Wcorrector
      {
      private:
        array<double, maxsz> x, wgtpsi;
        size_t n, supp;
    
      public:
        Wcorrector(const detail_gridding_kernel::KernelCorrection &corr)
          {
          const auto &x_ = corr.X();
          n = x_.size();
          MR_assert(n<=maxsz, "maxsz too small");
          const auto &wgtpsi_ = corr.Wgtpsi();
          supp = corr.Supp();
          for (size_t i=0; i<n; ++i)
            {
            x[i] = x_[i];
            wgtpsi[i] = wgtpsi_[i];
            }
          }
    
        double corfunc(double v) const
          {
          double tmp=0;
          for (size_t i=0; i<n; ++i)
            tmp += wgtpsi[i]*sycl::cos(pi*supp*v*x[i]);
          return 1./tmp;
          }
      };
   
    static double phase(double x, double y, double w, bool adjoint, double nshift)
      {
      double tmp = 1.-x-y;
      if (tmp<=0) return 0; // no phase factor beyond the horizon
      double nm1 = (-x-y)/(sycl::sqrt(tmp)+1); // more accurate form of sqrt(1-x-y)-1
      double phs = w*(nm1+nshift);
      if (adjoint) phs *= -1;
      if constexpr (is_same<Tcalc, double>::value)
        return twopi*phs;
      // we are reducing accuracy, so let's better do range reduction first
      return twopi*(phs-sycl::floor(phs));
      }
   
  public:
    GlobalCorrector(const Params &par_)
      : par(par_),
        cfu(par.krn->corfunc(par.nxdirty/2+1, 1./par.nu, par.nthreads)),
        cfv(par.krn->corfunc(par.nydirty/2+1, 1./par.nv, par.nthreads)),
        bufcfu(make_sycl_buffer(cfu)),
        bufcfv(make_sycl_buffer(cfv))
      {}

    void corr_degrid_narrow_field(sycl::queue &q,
      sycl::buffer<Tcalc, 2> &bufdirty, sycl::buffer<complex<Tcalc>, 2> &bufgrid)
      {
      // copy to grid and apply kernel correction
      q.submit([&](sycl::handler &cgh)
        {
        sycl::accessor accdirty{bufdirty, cgh, sycl::read_only};
        sycl::accessor acccfu{bufcfu, cgh, sycl::read_only};
        sycl::accessor acccfv{bufcfv, cgh, sycl::read_only};
        sycl::accessor accgrid{bufgrid, cgh, sycl::write_only};
        cgh.parallel_for(sycl::range<2>(par.nxdirty, par.nydirty), [accdirty,acccfu,acccfv,accgrid,nxdirty=par.nxdirty,nydirty=par.nydirty,nu=par.nu,nv=par.nv](sycl::item<2> item)
          {
          auto i = item.get_id(0);
          auto j = item.get_id(1);
          int icfu = sycl::abs(int(nxdirty/2)-int(i));
          int icfv = sycl::abs(int(nydirty/2)-int(j));
          size_t i2 = nu-nxdirty/2+i;
          if (i2>=nu) i2-=nu;
          size_t j2 = nv-nydirty/2+j;
          if (j2>=nv) j2-=nv;
          auto fctu = acccfu[icfu];
          auto fctv = acccfv[icfv];
          accgrid[i2][j2] = accdirty[i][j]*Tcalc(fctu*fctv);
          });
        });
      }

    void corr_grid_narrow_field(sycl::queue &q,
      sycl::buffer<complex<Tcalc>, 2> &bufgrid, sycl::buffer<Tcalc, 2> &bufdirty)
      {
      // copy to dirty image and apply kernel correction
      q.submit([&](sycl::handler &cgh)
        {
        sycl::accessor accdirty{bufdirty, cgh, sycl::write_only, sycl::no_init};
        sycl::accessor acccfu{bufcfu, cgh, sycl::read_only};
        sycl::accessor acccfv{bufcfv, cgh, sycl::read_only};
        sycl::accessor accgrid{bufgrid, cgh, sycl::read_only};
        cgh.parallel_for(sycl::range<2>(par.nxdirty, par.nydirty), [accdirty,acccfu,acccfv,accgrid,nxdirty=par.nxdirty,nydirty=par.nydirty,nu=par.nu,nv=par.nv](sycl::item<2> item)
          {
          auto i = item.get_id(0);
          auto j = item.get_id(1);
          int icfu = sycl::abs(int(nxdirty/2)-int(i));
          int icfv = sycl::abs(int(nydirty/2)-int(j));
          size_t i2 = nu-nxdirty/2+i;
          if (i2>=nu) i2-=nu;
          size_t j2 = nv-nydirty/2+j;
          if (j2>=nv) j2-=nv;
          auto fctu = acccfu[icfu];
          auto fctv = acccfv[icfv];
          accdirty[i][j] = (accgrid[i2][j2]*Tcalc(fctu*fctv)).real();
          });
        });
      }

    void apply_global_corrections(sycl::queue &q,
      sycl::buffer<Tcalc, 2> &bufdirty)
      {
      // apply global corrections to dirty image on GPU
      q.submit([&](sycl::handler &cgh)
        {
        Wcorrector<30> wcorr(par.krn->Corr());
        sycl::accessor accdirty{bufdirty, cgh, sycl::read_write};
        sycl::accessor acccfu{bufcfu, cgh, sycl::read_only};
        sycl::accessor acccfv{bufcfv, cgh, sycl::read_only};
        double x0 = par.lshift-0.5*par.nxdirty*par.pixsize_x,
               y0 = par.mshift-0.5*par.nydirty*par.pixsize_y;
        cgh.parallel_for(sycl::range<2>(par.nxdirty, par.nydirty), [nxdirty=par.nxdirty,nydirty=par.nydirty,accdirty,acccfu,acccfv,pixsize_x=par.pixsize_x,pixsize_y=par.pixsize_y,x0,y0,divide_by_n=par.divide_by_n,wcorr,nshift=par.nshift,dw=par.dw](sycl::item<2> item)
          {
          auto i = item.get_id(0);
          auto j = item.get_id(1);
          double fx = sqr(x0+i*pixsize_x);
          double fy = sqr(y0+j*pixsize_y);
          double fct;
          auto tmp = 1-fx-fy;
          if (tmp>=0)
            {
            auto nm1 = (-fx-fy)/(sycl::sqrt(tmp)+1); // accurate form of sqrt(1-x-y)-1
            fct = wcorr.corfunc((nm1+nshift)*dw);
            if (divide_by_n)
              fct /= nm1+1;
            }
          else // beyond the horizon, don't really know what to do here
            fct = divide_by_n ? 0 : wcorr.corfunc((sycl::sqrt(-tmp)-1)*dw);

          int icfu = sycl::abs(int(nxdirty/2)-int(i));
          int icfv = sycl::abs(int(nydirty/2)-int(j));
          accdirty[i][j]*=Tcalc(fct*acccfu[icfu]*acccfv[icfv]);
          });
        });
      }
    void degridding_wscreen(sycl::queue &q, double w,
      sycl::buffer<Tcalc, 2> &bufdirty, sycl::buffer<complex<Tcalc>, 2> &bufgrid)
      {
      // copy to grid and apply wscreen
      q.submit([&](sycl::handler &cgh)
        {
        sycl::accessor accdirty{bufdirty, cgh, sycl::read_only};
        sycl::accessor accgrid{bufgrid, cgh, sycl::write_only};
        double x0 = par.lshift-0.5*par.nxdirty*par.pixsize_x,
               y0 = par.mshift-0.5*par.nydirty*par.pixsize_y;
        cgh.parallel_for(sycl::range<2>(par.nxdirty, par.nydirty), [nxdirty=par.nxdirty, nydirty=par.nydirty, nu=par.nu, nv=par.nv, pixsize_x=par.pixsize_x, pixsize_y=par.pixsize_y,nshift=par.nshift,accgrid,accdirty,x0,y0,w](sycl::item<2> item)
          {
          auto i = item.get_id(0);
          auto j = item.get_id(1);
          size_t i2 = nu-nxdirty/2+i;
          if (i2>=nu) i2-=nu;
          size_t j2 = nv-nydirty/2+j;
          if (j2>=nv) j2-=nv;
          double fx = sqr(x0+i*pixsize_x);
          double fy = sqr(y0+j*pixsize_y);
          double myphase = phase(fx, fy, w, false, nshift);
          accgrid[i2][j2] = complex<Tcalc>(sycl::cos(myphase),sycl::sin(myphase))*accdirty[i][j];
          });
        });
      }
    void gridding_wscreen(sycl::queue &q, double w,
      sycl::buffer<complex<Tcalc>, 2> &bufgrid, sycl::buffer<Tcalc, 2> &bufdirty)
      {
      q.submit([&](sycl::handler &cgh)
        {
        sycl::accessor accdirty{bufdirty, cgh, sycl::read_write};
        sycl::accessor accgrid{bufgrid, cgh, sycl::read_only};
        double x0 = par.lshift-0.5*par.nxdirty*par.pixsize_x,
               y0 = par.mshift-0.5*par.nydirty*par.pixsize_y;
        cgh.parallel_for(sycl::range<2>(par.nxdirty, par.nydirty), [nxdirty=par.nxdirty, nydirty=par.nydirty, nu=par.nu, nv=par.nv, pixsize_x=par.pixsize_x, pixsize_y=par.pixsize_y,nshift=par.nshift,accgrid,accdirty,x0,y0,w](sycl::item<2> item)
          {
          auto i = item.get_id(0);
          auto j = item.get_id(1);
          size_t i2 = nu-nxdirty/2+i;
          if (i2>=nu) i2-=nu;
          size_t j2 = nv-nydirty/2+j;
          if (j2>=nv) j2-=nv;
          double fx = sqr(x0+i*pixsize_x);
          double fy = sqr(y0+j*pixsize_y);
          double myphase = phase(fx, fy, w, true, nshift);
          accdirty[i][j] += sycl::cos(myphase)*accgrid[i2][j2].real()
                           -sycl::sin(myphase)*accgrid[i2][j2].imag();
          });
        });
      }
  };
class RowchanComputer
  {
  public:
    sycl::accessor<RowchanRange,1,sycl::access::mode::read> acc_ranges;
    sycl::accessor<uint32_t,1,sycl::access::mode::read> acc_vissum;
    sycl::accessor<pair<Uvwidx,uint32_t>,1,sycl::access::mode::read> acc_blockstart;

  public:
    RowchanComputer(IndexComputer &idxcomp, sycl::handler &cgh)
      : acc_ranges(idxcomp.buf_ranges, cgh, sycl::read_only),
        acc_vissum(idxcomp.buf_vissum, cgh, sycl::read_only),
        acc_blockstart(idxcomp.buf_blockstart, cgh, sycl::read_only)
      {}

    void getRowChan(size_t iblock, size_t iwork, size_t &irow, size_t &ichan) const
      {
      auto xlo = acc_blockstart[iblock].second;
      auto xhi = (iblock+1==acc_blockstart.size()) ?
        acc_ranges.size() : acc_blockstart[iblock+1].second;
      auto wanted = acc_vissum[xlo]+iwork;
      if (wanted>=acc_vissum[xhi])
        { irow = ~size_t(0); return; }  // nothing to do for this item
      while (xlo+1<xhi)  // bisection search
        {
        auto xmid = (xlo+xhi)/2;
        (acc_vissum[xmid]<=wanted) ? xlo=xmid : xhi=xmid;
        }
      if (acc_vissum[xhi]<=wanted)
        xlo = xhi;
      irow = acc_ranges[xlo].row;
      ichan = acc_ranges[xlo].ch_begin + (wanted-acc_vissum[xlo]);
      }
  };

template<typename T> class KernelComputer
  {
  protected:
    sycl::accessor<T,1,sycl::access::mode::read> acc_coeff;
    size_t supp, D;

  public:
    KernelComputer(sycl::buffer<T,1> &buf_coeff, size_t supp_, sycl::handler &cgh)
      : acc_coeff(buf_coeff, cgh, sycl::read_only),
        supp(supp_), D(supp_+3) {}
    template<size_t Supp> inline void compute_uv(T ufrac, T vfrac, array<T,Supp> &ku, array<T,Supp> &kv) const
      {
//      if (Supp<supp) throw runtime_error("bad array size");
      auto x0 = T(ufrac)*T(-2)+T(supp-1);
      auto y0 = T(vfrac)*T(-2)+T(supp-1);
      for (size_t i=0; i<supp; ++i)
        {
        Tcalc resu=acc_coeff[i], resv=acc_coeff[i];
        for (size_t j=1; j<=D; ++j)
          {
          resu = resu*x0 + acc_coeff[j*supp+i];
          resv = resv*y0 + acc_coeff[j*supp+i];
          }
        ku[i] = resu;
        kv[i] = resv;
        }
      }
    template<size_t Supp> inline void compute_uvw(T ufrac, T vfrac, T wval, size_t nth, array<T,Supp> &ku, array<T,Supp> &kv) const
      {
//      if (Supp<supp) throw runtime_error("bad array size");
      auto x0 = T(ufrac)*T(-2)+T(supp-1);
      auto y0 = T(vfrac)*T(-2)+T(supp-1);
      auto z0 = T(wval-nth)*T(2)+T(supp-1);
      Tcalc resw=acc_coeff[nth];
      for (size_t j=1; j<=D; ++j)
        resw = resw*z0 + acc_coeff[j*supp+nth];
      for (size_t i=0; i<supp; ++i)
        {
        Tcalc resu=acc_coeff[i], resv=acc_coeff[i];
        for (size_t j=1; j<=D; ++j)
          {
          resu = resu*x0 + acc_coeff[j*supp+i];
          resv = resv*y0 + acc_coeff[j*supp+i];
          }
        ku[i] = resu*resw;
        kv[i] = resv;
        }
      }
  };

class CoordCalculator
  {
  private:
    size_t nu, nv;
    int maxiu0, maxiv0;
    double pixsize_x, pixsize_y, ushift, vshift;

  public:
    CoordCalculator (size_t nu_, size_t nv_, int maxiu0_, int maxiv0_, double pixsize_x_, double pixsize_y_, double ushift_, double vshift_)
      : nu(nu_), nv(nv_), maxiu0(maxiu0_), maxiv0(maxiv0_), pixsize_x(pixsize_x_), pixsize_y(pixsize_y_), ushift(ushift_), vshift(vshift_) {}

    [[gnu::always_inline]] void getpix(double u_in, double v_in, double &u, double &v, int &iu0, int &iv0) const
      {
      u = u_in*pixsize_x;
      u = (u-sycl::floor(u))*nu;
      iu0 = std::min(int(u+ushift)-int(nu), maxiu0);
      u -= iu0;
      v = v_in*pixsize_y;
      v = (v-sycl::floor(v))*nv;
      iv0 = std::min(int(v+vshift)-int(nv), maxiv0);
      v -= iv0;
      }
  };

template<typename T> static inline void do_shift(complex<T> &val, const UVW &coord,
  double lshift, double mshift, double nshift, double flip)
  {
  double fct = coord.u*lshift + coord.v*mshift + coord.w*nshift;
  if constexpr (is_same<double, T>::value)
    fct*=twopi;
  else
    // we are reducing accuracy,
    // so let's better do range reduction first
    fct = twopi*(fct-sycl::floor(fct));
  complex<T> phase(sycl::cos(T(fct)), T(flip)*sycl::sin(T(fct)));
  val *= phase;
  }

    void dirty2x()
      {
      timers.push("GPU degridding");
      if (do_wgridding)
        {
        { // Device buffer scope
timers.push("prep");
        sycl::queue q{sycl::default_selector()};
//print_device_info(q.get_device());

        auto bufdirty(make_sycl_buffer(dirty_in));
        // grid (only on GPU)
        sycl::buffer<complex<Tcalc>, 2> bufgrid{sycl::range<2>(nu,nv)};

        Baselines_GPU_prep bl_prep(bl);

        auto bufvis(make_sycl_buffer(ms_out));
        const auto &dcoef(krn->Coeff());
        vector<Tcalc> coef(dcoef.size());
        for (size_t i=0;i<coef.size(); ++i) coef[i] = Tcalc(dcoef[i]);
        auto bufcoef(make_sycl_buffer(coef));

q.wait(); timers.poppush("zeroing ms");
        sycl_zero_buffer(q, bufvis);

q.wait(); timers.poppush("indexcomp");
        // build index structure
        IndexComputer idxcomp(ranges, vissum, blockstart);

q.wait(); timers.poppush("copy HtoD");
  q.submit([&](sycl::handler &cgh)
    {
    Baselines_GPU blloc(bl_prep, cgh);
    KernelComputer<Tcalc> kcomp(bufcoef, supp, cgh);
    RowchanComputer rccomp(idxcomp, cgh);
    sycl::accessor accdirty{bufdirty, cgh, sycl::read_only};
    cgh.single_task([blloc,kcomp,rccomp,accdirty](){});
    });

        // apply global corrections to dirty image on GPU
q.wait(); timers.poppush("globcorr");
        GlobalCorrector globcorr(*this);
        globcorr.apply_global_corrections(q, bufdirty);

        CoordCalculator ccalc(nu, nv, maxiu0, maxiv0, pixsize_x, pixsize_y, ushift,vshift);
q.wait(); timers.poppush("FFT plan generation");
        sycl_fft_plan plan(bufgrid);
q.wait(); timers.pop();

        for (size_t pl=0; pl<nplanes; ++pl)
          {
q.wait(); timers.push("plane data structures");
          double w = wmin+pl*dw;
          vector<size_t> blidx;
          for (size_t i=0; i<blockstart.size(); ++i)
            {
            auto minpl = blockstart[i].first.minplane;
            if ((pl>=minpl) && (pl<minpl+supp))
              blidx.push_back(i);
            }
          auto bufblidx(make_sycl_buffer(blidx));
q.wait(); timers.poppush("copy HtoD idx");
  q.submit([&](sycl::handler &cgh)
    {
    sycl::accessor accblidx{bufblidx, cgh, sycl::read_only};
    cgh.single_task([accblidx](){});
    });
q.wait(); timers.poppush("zeroing grid");
          sycl_zero_buffer(q, bufgrid);

q.wait(); timers.poppush("wscreen");
          globcorr.degridding_wscreen(q, w, bufdirty, bufgrid);

q.wait(); timers.poppush("FFT");
          // FFT
          plan.exec(q, bufgrid, true);

q.wait(); timers.poppush("degridding proper");
          constexpr size_t blksz = 32768;
          for (size_t ofs=0; ofs<blidx.size(); ofs+=blksz)
            {
            q.submit([&](sycl::handler &cgh)
              {
              Baselines_GPU blloc(bl_prep, cgh);
              KernelComputer<Tcalc> kcomp(bufcoef, supp, cgh);
              RowchanComputer rccomp(idxcomp,cgh);

              sycl::accessor accblidx{bufblidx, cgh, sycl::read_only};
              sycl::accessor accgrid{bufgrid, cgh, sycl::read_only};
              sycl::accessor accvis{bufvis, cgh, sycl::write_only};

              constexpr size_t n_workitems = 32;
              sycl::range<2> global(min(blksz,blidx.size()-ofs), n_workitems);
              sycl::range<2> local(1, n_workitems);
              cgh.parallel_for(sycl::nd_range(global, local), [accgrid,accvis,nu=nu,nv=nv,supp=supp,shifting=shifting,lshift=lshift,mshift=mshift,nshift=nshift,rccomp,blloc,ccalc,kcomp,pl,w,dw=dw,ofs,accblidx](sycl::nd_item<2> item)
                {
                auto iblock = accblidx[item.get_global_id(0)+ofs];
                auto minplane = rccomp.acc_blockstart[iblock].first.minplane;

                for (auto iwork=item.get_global_id(1); ; iwork+=item.get_global_range(1))
                  {
                  size_t irow, ichan;
                  rccomp.getRowChan(iblock, iwork, irow, ichan);
                  if (irow==~size_t(0)) return;
      
                  auto coord = blloc.effectiveCoord(irow, ichan);
                  auto imflip = coord.FixW();
      
                  // compute fractional and integer indices in "grid"
                  double ufrac,vfrac;
                  int iu0, iv0;
                  ccalc.getpix(coord.u, coord.v, ufrac, vfrac, iu0, iv0);
      
                  // compute kernel values
                  array<Tcalc, 16> ukrn, vkrn;
                  size_t nth=pl-minplane;
                  auto wval=Tcalc((w-coord.w)/dw);
                  kcomp.compute_uvw(ufrac, vfrac, wval, nth, ukrn, vkrn);
      
                  // loop over supp*supp pixels from "grid"
                  complex<Tcalc> res=0;
                  auto iustart=size_t((iu0+nu)%nu);
                  auto ivstart=size_t((iv0+nv)%nv);
                  for (size_t i=0, realiu=iustart; i<supp;
                       ++i, realiu = (realiu+1<nu) ? realiu+1 : 0)
                    {
                    complex<Tcalc> tmp = 0;
                    for (size_t j=0, realiv=ivstart; j<supp;
                         ++j, realiv = (realiv+1<nv) ? realiv+1 : 0)
                      tmp += vkrn[j]*accgrid[realiu][realiv];
                    res += ukrn[i]*tmp;
                    }
                  res.imag(res.imag()*imflip);
      
                  if (shifting)
                    do_shift(res, coord, lshift, mshift, nshift, -imflip);
                  accvis[irow][ichan] += res;
                  }
                });
              });
            }
          q.wait();
timers.pop();
          } // end of loop over planes
q.wait(); timers.push("copying DtoH");
        }  // end of device buffer scope, buffers are written back
timers.poppush("weight application");
        if (wgt.stride(0)!=0)  // we need to apply weights!
          mav_apply([](auto &a, const auto &b){a*=b;}, nthreads, ms_out, wgt);
timers.pop();
        }
      else
        {
        { // Device buffer scope
        sycl::queue q{sycl::default_selector()};
q.wait(); timers.push("prep");

        auto bufdirty(make_sycl_buffer(dirty_in));
        // grid (only on GPU)
        sycl::buffer<complex<Tcalc>, 2> bufgrid{sycl::range<2>(nu,nv)};

        Baselines_GPU_prep bl_prep(bl);
        auto bufvis(make_sycl_buffer(ms_out));
        const auto &dcoef(krn->Coeff());
        vector<Tcalc> coef(dcoef.size());
        for (size_t i=0;i<coef.size(); ++i) coef[i] = Tcalc(dcoef[i]);
        auto bufcoef(make_sycl_buffer(coef));

q.wait(); timers.poppush("zeroing ms and grid");
        sycl_zero_buffer(q, bufvis);
        sycl_zero_buffer(q, bufgrid);

q.wait(); timers.poppush("indexcomp");
        // build index structure
        IndexComputer idxcomp(ranges, vissum, blockstart);
        CoordCalculator ccalc(nu, nv, maxiu0, maxiv0, pixsize_x, pixsize_y, ushift,vshift);

q.wait(); timers.poppush("copy HtoD");
  q.submit([&](sycl::handler &cgh)
    {
    Baselines_GPU blloc(bl_prep, cgh);
    KernelComputer<Tcalc> kcomp(bufcoef, supp, cgh);
    RowchanComputer rccomp(idxcomp, cgh);
    sycl::accessor accdirty{bufdirty, cgh, sycl::read_only};
    cgh.single_task([blloc,kcomp,rccomp,accdirty](){});
    });

q.wait(); timers.poppush("globcorr");
        {
        GlobalCorrector globcorr(*this);
        globcorr.corr_degrid_narrow_field(q, bufdirty, bufgrid);
        }
        // FFT
q.wait(); timers.poppush("FFT plan generation");
        sycl_fft_plan plan(bufgrid);
q.wait(); timers.poppush("FFT");
        plan.exec(q, bufgrid, true);

q.wait(); timers.poppush("degridding proper");
        constexpr size_t blksz = 32768;
        size_t nblock = blockstart.size();
        for (size_t ofs=0; ofs<nblock; ofs+= blksz)
          {
          q.submit([&](sycl::handler &cgh)
            {
            Baselines_GPU blloc(bl_prep, cgh);
            KernelComputer<Tcalc> kcomp(bufcoef, supp, cgh);
            RowchanComputer rccomp(idxcomp, cgh);
  
            sycl::accessor accgrid{bufgrid, cgh, sycl::read_only};
            sycl::accessor accvis{bufvis, cgh, sycl::write_only};
  
            constexpr size_t n_workitems = 64;
            sycl::range<2> global(min(blksz,nblock-ofs), n_workitems);
            sycl::range<2> local(1, n_workitems);
            cgh.parallel_for(sycl::nd_range<2>(global, local), [accgrid,accvis,nu=nu,nv=nv,supp=supp,shifting=shifting,lshift=lshift,mshift=mshift,rccomp,blloc,ccalc,kcomp,ofs](sycl::nd_item<2> item)
              {
              auto iblock = item.get_global_id(0)+ofs;
  
              for (auto iwork=item.get_global_id(1); ; iwork+=item.get_global_range(1))
                {
                size_t irow, ichan;
                rccomp.getRowChan(iblock, iwork, irow, ichan);
                if (irow==~size_t(0)) return;
    
                auto coord = blloc.effectiveCoord(irow, ichan);
                auto imflip = coord.FixW();
    
                // compute fractional and integer indices in "grid"
                double ufrac,vfrac;
                int iu0, iv0;
                ccalc.getpix(coord.u, coord.v, ufrac, vfrac, iu0, iv0);
    
                // compute kernel values
                array<Tcalc, 16> ukrn, vkrn;
                kcomp.compute_uv(ufrac, vfrac, ukrn, vkrn);
    
                // loop over supp*supp pixels from "grid"
                complex<Tcalc> res=0;
                auto iustart=size_t((iu0+nu)%nu);
                auto ivstart=size_t((iv0+nv)%nv);
                for (size_t i=0, realiu=iustart; i<supp;
                     ++i, realiu = (realiu+1<nu)?realiu+1 : 0)
                  {
                  complex<Tcalc> tmp = 0;
                  for (size_t j=0, realiv=ivstart; j<supp;
                       ++j, realiv = (realiv+1<nv)?realiv+1 : 0)
                    tmp += vkrn[j]*accgrid[realiu][realiv];
                  res += ukrn[i]*tmp;
                  }
                res.imag(res.imag()*imflip);
    
                if (shifting)
                  do_shift(res, coord, lshift, mshift, 0., -imflip);
                accvis[irow][ichan] = res;
                }
              });
            });
          }
q.wait(); timers.poppush("copying DtoH");
        }  // end of device buffer scope, buffers are written back
timers.poppush("weight application");
        if (wgt.stride(0)!=0)  // we need to apply weights!
          mav_apply([](auto &a, const auto &b){a*=b;}, nthreads, ms_out, wgt);
timers.pop();
        }
      timers.pop();
      }

    void x2dirty()
      {
      timers.push("GPU gridding");
      if (do_wgridding)
        {
timers.push("prep");
        bool do_weights = (wgt.stride(0)!=0);
        { // Device buffer scope
        sycl::queue q{sycl::default_selector()};

        auto bufdirty(make_sycl_buffer(dirty_out));
        sycl_zero_buffer(q, bufdirty);

        // grid (only on GPU)
        sycl::buffer<complex<Tcalc>, 2> bufgrid{sycl::range<2>(nu,nv)};
        sycl::buffer<Tcalc, 3> bufgridr{bufgrid.template reinterpret<Tcalc,3>(sycl::range<3>(nu,nv,2))};

        Baselines_GPU_prep bl_prep(bl);
        auto bufvis(make_sycl_buffer(ms_in));
        vmav<Tms,2> wgtx({1,1});
        auto bufwgt(make_sycl_buffer(do_weights ? wgt : wgtx));

        const auto &dcoef(krn->Coeff());
        vector<Tcalc> coef(dcoef.size());
        for (size_t i=0;i<coef.size(); ++i) coef[i] = Tcalc(dcoef[i]);
        auto bufcoef(make_sycl_buffer(coef));

q.wait(); timers.poppush("indexcomp");
        // build index structure
        IndexComputer idxcomp(ranges, vissum, blockstart);
        CoordCalculator ccalc(nu, nv, maxiu0, maxiv0, pixsize_x, pixsize_y, ushift,vshift);
        GlobalCorrector globcorr(*this);

q.wait(); timers.poppush("FFT plan generation");
        sycl_fft_plan plan(bufgrid);
q.wait(); timers.pop();
        for (size_t pl=0; pl<nplanes; ++pl)
          {
q.wait(); timers.push("plane data structures");
          double w = wmin+pl*dw;
          vector<size_t> blidx;
          for (size_t i=0; i<blockstart.size(); ++i)
            {
            auto minpl = blockstart[i].first.minplane;
            if ((pl>=minpl) && (pl<minpl+supp))
              blidx.push_back(i);
            }
          auto bufblidx(make_sycl_buffer(blidx));
q.wait(); timers.poppush("copy HtoD idx");
  q.submit([&](sycl::handler &cgh)
    {
    sycl::accessor accblidx{bufblidx, cgh, sycl::read_only};
    cgh.single_task([accblidx](){});
    });

q.wait(); timers.poppush("zeroing grid");
          sycl_zero_buffer(q, bufgrid);
          constexpr size_t blksz = 32768;
q.wait(); timers.poppush("copy HtoD");
  q.submit([&](sycl::handler &cgh)
    {
    sycl::accessor accvis{bufvis, cgh, sycl::read_only};
    sycl::accessor accwgt{bufwgt, cgh, sycl::read_only};
    cgh.single_task([accvis,accwgt](){});
    });
q.wait(); timers.poppush("gridding proper");
          for (size_t ofs=0; ofs<blidx.size(); ofs+= blksz)
            {
            q.submit([&](sycl::handler &cgh)
              {
              Baselines_GPU blloc(bl_prep, cgh);
              KernelComputer<Tcalc> kcomp(bufcoef, supp, cgh);
              RowchanComputer rccomp(idxcomp,cgh);
  
              sycl::accessor accblidx{bufblidx, cgh, sycl::read_only};
              sycl::accessor accgridr{bufgridr, cgh, sycl::read_write};
              sycl::accessor accvis{bufvis, cgh, sycl::read_only};
              sycl::accessor accwgt{bufwgt, cgh, sycl::read_only};
  
              constexpr size_t n_workitems = 96;
              sycl::range<2> global(min(blksz,blidx.size()-ofs), n_workitems);
              sycl::range<2> local(1, n_workitems);
              int nsafe = (supp+1)/2;
              size_t sidelen = 2*nsafe+(1<<logsquare);
              my_local_accessor<Tcalc,3> tile({sidelen,sidelen,2}, cgh);

              cgh.parallel_for(sycl::nd_range(global,local), [accgridr,accvis,nu=nu,nv=nv,supp=supp,shifting=shifting,lshift=lshift,mshift=mshift,nshift=nshift,rccomp,blloc,ccalc,kcomp,pl,sidelen,nsafe,tile,w,dw=dw,ofs,accblidx,accwgt,do_weights](sycl::nd_item<2> item)
                {
                auto iblock = accblidx[item.get_global_id(0)+ofs];
                auto minplane = rccomp.acc_blockstart[iblock].first.minplane;
  
                // preparation
                for (size_t i=item.get_global_id(1); i<sidelen*sidelen; i+=item.get_local_range(1))
                  {
                  size_t iu = i/sidelen, iv = i%sidelen;
                  tile[iu][iv][0]=Tcalc(0);
                  tile[iu][iv][1]=Tcalc(0);
                  }
                item.barrier(sycl::access::fence_space::local_space);
  
                for (auto iwork=item.get_global_id(1); ; iwork+=item.get_global_range(1))
                  {
                  size_t irow, ichan;
                  rccomp.getRowChan(iblock, iwork, irow, ichan);
                  if (irow==~size_t(0)) break;  // work done 
  
                  auto coord = blloc.effectiveCoord(irow, ichan);
                  auto imflip = coord.FixW();
    
                  // compute fractional and integer indices in "grid"
                  double ufrac,vfrac;
                  int iu0, iv0;
                  ccalc.getpix(coord.u, coord.v, ufrac, vfrac, iu0, iv0);
    
                  // compute kernel values
                  array<Tcalc, 16> ukrn, vkrn;
                  size_t nth=pl-minplane;
                  auto wval=Tcalc((w-coord.w)/dw);
                  kcomp.compute_uvw(ufrac, vfrac, wval, nth, ukrn, vkrn);
    
                  // loop over supp*supp pixels from "grid"
                  complex<Tcalc> val=accvis[irow][ichan];
                  if (do_weights) val *= accwgt[irow][ichan];
                  if (shifting)
                    do_shift(val, coord, lshift, mshift, nshift, imflip);
                  val.imag(val.imag()*imflip);
  
                  int bu0=((((iu0+nsafe)>>logsquare)<<logsquare))-nsafe;
                  int bv0=((((iv0+nsafe)>>logsquare)<<logsquare))-nsafe;
                  for (size_t i=0, ipos=iu0-bu0; i<supp; ++i, ++ipos)
                    {
                    auto tmp = ukrn[i]*val;
                    for (size_t j=0, jpos=iv0-bv0; j<supp; ++j, ++jpos)
                      {
                      auto tmp2 = vkrn[j]*tmp;
                      my_atomic_ref_l<Tcalc> rr(tile[ipos][jpos][0]);
                      rr.fetch_add(tmp2.real());
                      my_atomic_ref_l<Tcalc> ri(tile[ipos][jpos][1]);
                      ri.fetch_add(tmp2.imag());
                      }
                    }
                  }
  
                // add local buffer back to global buffer
                auto u_tile = rccomp.acc_blockstart[iblock].first.tile_u;
                auto v_tile = rccomp.acc_blockstart[iblock].first.tile_v;
                item.barrier(sycl::access::fence_space::local_space);
                for (size_t i=item.get_global_id(1); i<sidelen*sidelen; i+=item.get_global_range(1))
                  {
                  size_t iu = i/sidelen, iv = i%sidelen;
                  size_t igu = (iu+u_tile*(1<<logsquare)+nu-nsafe)%nu;
                  size_t igv = (iv+v_tile*(1<<logsquare)+nv-nsafe)%nv;
  
                  my_atomic_ref<Tcalc> rr(accgridr[igu][igv][0]);
                  rr.fetch_add(tile[iu][iv][0]);
                  my_atomic_ref<Tcalc> ri(accgridr[igu][igv][1]);
                  ri.fetch_add(tile[iu][iv][1]);
                  }
                });
              });
            }
q.wait(); timers.poppush("FFT");
          // FFT
          plan.exec(q, bufgrid, false);

q.wait(); timers.poppush("wscreen");
          globcorr.gridding_wscreen(q, w, bufgrid, bufdirty);
          q.wait();
timers.pop();
          } // end of loop over planes

q.wait(); timers.push("globcorr");
        // apply global corrections to dirty image on GPU
        globcorr.apply_global_corrections(q, bufdirty);
q.wait(); timers.poppush("copy DtoH");
        }  // end of device buffer scope, buffers are written back
timers.pop();
        }
      else
        {
timers.push("prep");
        bool do_weights = (wgt.stride(0)!=0);

        { // Device buffer scope
        sycl::queue q{sycl::default_selector()};
        // dirty image
        auto bufdirty(make_sycl_buffer(dirty_out));
        // grid (only on GPU)
        sycl::buffer<complex<Tcalc>, 2> bufgrid{sycl::range<2>(nu,nv)};
        sycl::buffer<Tcalc, 3> bufgridr{bufgrid.template reinterpret<Tcalc,3>(sycl::range<3>(nu,nv,2))};

        Baselines_GPU_prep bl_prep(bl);
        auto bufvis(make_sycl_buffer(ms_in));
        vmav<Tms,2> wgtx({1,1});
        auto bufwgt(make_sycl_buffer(do_weights ? wgt : wgtx));

        const auto &dcoef(krn->Coeff());
        vector<Tcalc> coef(dcoef.size());
        for (size_t i=0;i<coef.size(); ++i) coef[i] = Tcalc(dcoef[i]);
        auto bufcoef(make_sycl_buffer(coef));

q.wait(); timers.poppush("zeroing grid");
        sycl_zero_buffer(q, bufgrid);

        // build index structure
q.wait(); timers.poppush("indexcomp");
        IndexComputer idxcomp(ranges, vissum, blockstart);
        CoordCalculator ccalc(nu, nv, maxiu0, maxiv0, pixsize_x, pixsize_y, ushift,vshift);

q.wait(); timers.poppush("copy HtoD");
  q.submit([&](sycl::handler &cgh)
    {
    Baselines_GPU blloc(bl_prep, cgh);
    KernelComputer<Tcalc> kcomp(bufcoef, supp, cgh);
    RowchanComputer rccomp(idxcomp, cgh);
    sycl::accessor accvis{bufvis, cgh, sycl::read_only};
    sycl::accessor accwgt{bufwgt, cgh, sycl::read_only};
    sycl::accessor accgridr{bufgridr, cgh, sycl::read_only};
    cgh.single_task([accvis,accwgt,accgridr,blloc,kcomp,rccomp](){});
    });
q.wait(); timers.poppush("gridding proper");
        constexpr size_t blksz = 32768;
        size_t nblock = blockstart.size();
        for (size_t ofs=0; ofs<nblock; ofs+=blksz)
          {
          q.submit([&](sycl::handler &cgh)
            {
            Baselines_GPU blloc(bl_prep, cgh);
            KernelComputer<Tcalc> kcomp(bufcoef, supp, cgh);
            RowchanComputer rccomp(idxcomp, cgh);
  
            sycl::accessor accgridr{bufgridr, cgh, sycl::read_write};
            sycl::accessor accvis{bufvis, cgh, sycl::read_only};
            sycl::accessor accwgt{bufwgt, cgh, sycl::read_only};
  
            constexpr size_t n_workitems = 96;
            sycl::range<2> global(min(blksz,nblock-ofs), n_workitems);
            sycl::range<2> local(1, n_workitems);
            int nsafe = (supp+1)/2;
            size_t sidelen = 2*nsafe+(1<<logsquare);
            my_local_accessor<Tcalc,3> tile({sidelen,sidelen,2}, cgh);

            cgh.parallel_for(sycl::nd_range(global,local), [accgridr,accvis,tile,nu=nu,nv=nv,supp=supp,shifting=shifting,lshift=lshift,mshift=mshift,rccomp,blloc,ccalc,kcomp,nsafe,sidelen,ofs,accwgt,do_weights](sycl::nd_item<2> item)
              {
              auto iblock = item.get_global_id(0)+ofs;
  
              // preparation: zero local buffer
              for (size_t i=item.get_global_id(1); i<sidelen*sidelen; i+=item.get_global_range(1))
                {
                size_t iu = i/sidelen, iv = i%sidelen;
                tile[iu][iv][0] = Tcalc(0);
                tile[iu][iv][1] = Tcalc(0);
                }
              item.barrier();

              for (auto iwork=item.get_global_id(1); ; iwork+=item.get_global_range(1))
                {
                size_t irow, ichan;
                rccomp.getRowChan(iblock, iwork, irow, ichan);
//irow=ichan=0;
                if (irow==~size_t(0)) break;  // work done
                auto coord = blloc.effectiveCoord(irow, ichan);
                auto imflip = coord.FixW();
  
                // compute fractional and integer indices in "grid"
                double ufrac,vfrac;
                int iu0, iv0;
                ccalc.getpix(coord.u, coord.v, ufrac, vfrac, iu0, iv0);
  
                // compute kernel values
                array<Tcalc, 16> ukrn, vkrn;
                kcomp.compute_uv(ufrac, vfrac, ukrn, vkrn);
  
                // loop over supp*supp pixels from "grid"
                complex<Tcalc> val=accvis[irow][ichan];
                if(do_weights) val *= accwgt[irow][ichan];
                if (shifting)
                  do_shift(val, coord, lshift, mshift, 0., imflip);
                val.imag(val.imag()*imflip);
  
                int bu0=((((iu0+nsafe)>>logsquare)<<logsquare))-nsafe;
                int bv0=((((iv0+nsafe)>>logsquare)<<logsquare))-nsafe;
                for (size_t i=0, ipos=iu0-bu0; i<supp; ++i, ++ipos)
                  {
                  auto tmp = ukrn[i]*val;
                  for (size_t j=0, jpos=iv0-bv0; j<supp; ++j, ++jpos)
                    {
                    auto tmp2 = vkrn[j]*tmp;
                    my_atomic_ref_l<Tcalc> rr(tile[ipos][jpos][0]);
                    rr.fetch_add(tmp2.real());
                    my_atomic_ref_l<Tcalc> ri(tile[ipos][jpos][1]);
                    ri.fetch_add(tmp2.imag());
                    }
                  }
                }

              // add local buffer back to global buffer
              auto u_tile = rccomp.acc_blockstart[iblock].first.tile_u;
              auto v_tile = rccomp.acc_blockstart[iblock].first.tile_v;
              item.barrier();
              for (size_t i=item.get_global_id(1); i<sidelen*sidelen; i+=item.get_global_range(1))
                {
                size_t iu = i/sidelen, iv = i%sidelen;
                size_t igu = (iu+u_tile*(1<<logsquare)+nu-nsafe)%nu;
                size_t igv = (iv+v_tile*(1<<logsquare)+nv-nsafe)%nv;
                my_atomic_ref<Tcalc> rr(accgridr[igu][igv][0]);
                rr.fetch_add(tile[iu][iv][0]);
                my_atomic_ref<Tcalc> ri(accgridr[igu][igv][1]);
                ri.fetch_add(tile[iu][iv][1]);
                }
              });
            });
          }

q.wait(); timers.poppush("FFT plan generation");
        sycl_fft_plan plan(bufgrid);
q.wait(); timers.poppush("FFT");
        // FFT
        plan.exec(q, bufgrid, false);

q.wait(); timers.poppush("globcorr");
        {
        GlobalCorrector globcorr(*this);
        globcorr.corr_grid_narrow_field(q, bufgrid, bufdirty);
        }
q.wait(); timers.poppush("copy DtoH");
        }  // end of device buffer scope, buffers are written back
timers.pop();
        }
      timers.pop();
      }

  public:
    Params(const cmav<double,2> &uvw, const cmav<double,1> &freq,
           const cmav<complex<Tms>,2> &ms_in_, vmav<complex<Tms>,2> &ms_out_,
           const cmav<Timg,2> &dirty_in_, vmav<Timg,2> &dirty_out_,
           const cmav<Tms,2> &wgt_, const cmav<uint8_t,2> &mask_,
           double pixsize_x_, double pixsize_y_, double epsilon_,
           bool do_wgridding_, size_t nthreads_, size_t verbosity_,
           bool negate_v_, bool divide_by_n_, double sigma_min_,
           double sigma_max_, double center_x, double center_y, bool allow_nshift)
      : gridding(ms_out_.size()==0),
        timers(gridding ? "gridding" : "degridding"),
        ms_in(ms_in_), ms_out(ms_out_),
        dirty_in(dirty_in_), dirty_out(dirty_out_),
        wgt(wgt_), mask(mask_),
        pixsize_x(pixsize_x_), pixsize_y(pixsize_y_),
        nxdirty(gridding ? dirty_out.shape(0) : dirty_in.shape(0)),
        nydirty(gridding ? dirty_out.shape(1) : dirty_in.shape(1)),
        epsilon(epsilon_),
        do_wgridding(do_wgridding_),
        nthreads((nthreads_==0) ? get_default_nthreads() : nthreads_),
        verbosity(verbosity_),
        negate_v(negate_v_), divide_by_n(divide_by_n_),
        sigma_min(sigma_min_), sigma_max(sigma_max_),
        lshift(center_x), mshift(negate_v ? -center_y : center_y),
        lmshift((lshift!=0) || (mshift!=0)),
        no_nshift(!allow_nshift)
      {
      timers.push("Baseline construction");
      bl = Baselines(uvw, freq, negate_v);
      MR_assert(bl.Nrows()<(uint64_t(1)<<32), "too many rows in the MS");
      MR_assert(bl.Nchannels()<(uint64_t(1)<<16), "too many channels in the MS");
      timers.pop();
      // adjust for increased error when gridding in 2 or 3 dimensions
      epsilon /= do_wgridding ? 3 : 2;
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

template<typename Tcalc, typename Tacc, typename Tms, typename Timg> void ms2dirty_sycl(const cmav<double,2> &uvw,
  const cmav<double,1> &freq, const cmav<complex<Tms>,2> &ms,
  const cmav<Tms,2> &wgt_, const cmav<uint8_t,2> &mask_, double pixsize_x, double pixsize_y, double epsilon,
  bool do_wgridding, size_t nthreads, vmav<Timg,2> &dirty, size_t verbosity,
  bool negate_v=false, bool divide_by_n=true, double sigma_min=1.1,
  double sigma_max=2.6, double center_x=0, double center_y=0, bool allow_nshift=true)
  {
  auto ms_out(vmav<complex<Tms>,2>::build_empty());
  auto dirty_in(vmav<Timg,2>::build_empty());
  auto wgt(wgt_.size()!=0 ? wgt_ : wgt_.build_uniform(ms.shape(), 1.));
  auto mask(mask_.size()!=0 ? mask_ : mask_.build_uniform(ms.shape(), 1));
  Params<Tcalc, Tacc, Tms, Timg> par(uvw, freq, ms, ms_out, dirty_in, dirty, wgt, mask, pixsize_x, 
    pixsize_y, epsilon, do_wgridding, nthreads, verbosity, negate_v,
    divide_by_n, sigma_min, sigma_max, center_x, center_y, allow_nshift);
  }

template<typename Tcalc, typename Tacc, typename Tms, typename Timg> void dirty2ms_sycl(const cmav<double,2> &uvw,
  const cmav<double,1> &freq, const cmav<Timg,2> &dirty,
  const cmav<Tms,2> &wgt_, const cmav<uint8_t,2> &mask_, double pixsize_x, double pixsize_y,
  double epsilon, bool do_wgridding, size_t nthreads, vmav<complex<Tms>,2> &ms,
  size_t verbosity, bool negate_v=false, bool divide_by_n=true,
  double sigma_min=1.1, double sigma_max=2.6, double center_x=0, double center_y=0, bool allow_nshift=true)
  {
  if (ms.size()==0) return;  // nothing to do
  auto ms_in(ms.build_uniform(ms.shape(),1.));
  auto dirty_out(vmav<Timg,2>::build_empty());
  auto wgt(wgt_.size()!=0 ? wgt_ : wgt_.build_uniform(ms.shape(), 1.));
  auto mask(mask_.size()!=0 ? mask_ : mask_.build_uniform(ms.shape(), 1));
  Params<Tcalc, Tacc, Tms, Timg> par(uvw, freq, ms_in, ms, dirty, dirty_out, wgt, mask, pixsize_x,
    pixsize_y, epsilon, do_wgridding, nthreads, verbosity, negate_v,
    divide_by_n, sigma_min, sigma_max, center_x, center_y, allow_nshift);
  }

#else  // no SYCL support

template<typename Tcalc, typename Tacc, typename Tms, typename Timg> void ms2dirty_sycl(const cmav<double,2> &,
  const cmav<double,1> &, const cmav<complex<Tms>,2> &,
  const cmav<Tms,2> &, const cmav<uint8_t,2> &, double, double, double,
  bool, size_t, vmav<Timg,2> &, size_t,
  bool=false, bool=true, double=1.1,
  double=2.6, double=0, double=0, bool=true)
  { throw runtime_error("no SYCL support available"); }

template<typename Tcalc, typename Tacc, typename Tms, typename Timg> void dirty2ms_sycl(const cmav<double,2> &,
  const cmav<double,1> &, const cmav<Timg,2> &,
  const cmav<Tms,2> &, const cmav<uint8_t,2> &, double, double,
  double, bool, size_t, vmav<complex<Tms>,2> &,
  size_t, bool=false, bool=true,
  double=1.1, double=2.6, double=0, double=0, bool=true)
  { throw runtime_error("no SYCL support available"); }

#endif

}

using detail_wgridder_sycl::dirty2ms_sycl;
using detail_wgridder_sycl::ms2dirty_sycl;

}

#endif
