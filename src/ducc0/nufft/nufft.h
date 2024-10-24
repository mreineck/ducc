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

#ifndef DUCC0_NUFFT_H
#define DUCC0_NUFFT_H

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
#include <numeric>
#if ((!defined(DUCC0_NO_SIMD)) && (!defined(__AVX512F__)) && (defined(__AVX__)||defined(__SSE3__)))
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
#include "ducc0/nufft/nufft_common.h"
#include "ducc0/nufft/spread.h"

namespace ducc0 {

namespace detail_nufft {

template<typename Tcalc, typename Tacc, size_t ndim> class Nufft_ancestor
  {
  protected:
    TimerHierarchy timers;
    // requested epsilon value for this transform.
    double epsilon;
    // number of threads to use for this transform.
    size_t nthreads;

    // 1./<periodicity of coordinates>
    array<double, ndim> coordfct;

    // if true, start with zero mode
    // if false, start with most negative mode
    bool fft_order;

    // number of non-uniform points
    size_t npoints;

    // uniform grid dimensions
    array<size_t, ndim> nuni;

    // oversampled grid dimensions
    array<size_t, ndim> nover;

    // holds the indices of the nonuniform points in the order in which they
    // should be processed
    quick_array<uint32_t> coord_idx;

size_t krn_id;
    shared_ptr<PolynomialKernel> krn;

    size_t supp, nsafe;
    array<double, ndim> shift;

    array<int, ndim> maxi0;

    vector<vector<double>> corfac;

    // the base-2 logarithm of the linear dimension of a computational tile.
    constexpr static int log2tile = log2tile_<Tacc,ndim>;

    static_assert(sizeof(Tcalc)<=sizeof(Tacc),
      "Tacc must be at least as accurate as Tcalc");

    /*! Compute minimum index in the oversampled grid touched by the kernel
        around coordinate \a in. */
    template<typename Tcoord> [[gnu::always_inline]] void getpix(array<double,ndim> in,
      array<double,ndim> &out, array<int,ndim> &out0) const
      {
      // do range reduction in long double when Tcoord is double,
      // to avoid inaccuracies with very large grids
      using Tbig = typename conditional<is_same<Tcoord,double>::value, long double, double>::type;
      for (size_t i=0; i<ndim; ++i)
        {
        auto tmp = in[i]*coordfct[i];
        auto tmp2 = Tbig(tmp-floor(tmp))*nover[i];
        out0[i] = min(int(tmp2+shift[i])-int(nover[i]), maxi0[i]);
        out[i] = double(tmp2-out0[i]);
        }
      }

    /*! Compute index of the tile into which \a in falls. */
    template<typename Tcoord> [[gnu::always_inline]] array<uint32_t,ndim> get_tile(const array<double,ndim> &in) const
      {
      array<double,ndim> dum;
      array<int,ndim> i0;
      getpix<Tcoord>(in, dum, i0);
      array<uint32_t,ndim> res;
      for (size_t i=0; i<ndim; ++i)
        res[i] = uint32_t((i0[i]+nsafe)>>log2tile);
      return res;
      }
    template<typename Tcoord> [[gnu::always_inline]] array<uint32_t,ndim> get_tile(const array<double,ndim> &in, size_t lsq2) const
      {
      array<double,ndim> dum;
      array<int,ndim> i0;
      getpix<Tcoord>(in, dum, i0);
      array<uint32_t,ndim> res;
      for (size_t i=0; i<ndim; ++i)
        res[i] = uint32_t((i0[i]+nsafe)>>lsq2);
      return res;
      }

    template<typename Tcoord> void sort_coords(const cmav<Tcoord,2> &coords,
      const vmav<Tcoord,2> &coords_sorted)
      {
      timers.push("sorting coords");
      execParallel(npoints, nthreads, [&](size_t lo, size_t hi)
        {
        for (size_t i=lo; i<hi; ++i)
          for (size_t d=0; d<ndim; ++d)
            coords_sorted(i,d) = coords(coord_idx[i],d);
        });
      timers.pop();
      }

    template<typename Tpoints, typename Tgrid> bool prep_nu2u
      (const cmav<complex<Tpoints>,1> &points, const vmav<complex<Tgrid>,ndim> &uniform)
      {
      static_assert(sizeof(Tpoints)<=sizeof(Tcalc),
        "Tcalc must be at least as accurate as Tpoints");
      static_assert(sizeof(Tgrid)<=sizeof(Tcalc),
        "Tcalc must be at least as accurate as Tgrid");
      MR_assert(points.shape(0)==npoints, "number of points mismatch");
      MR_assert(uniform.shape()==nuni, "uniform grid dimensions mismatch");
      if (npoints==0)
        {
        mav_apply([](complex<Tgrid> &v){v=complex<Tgrid>(0);}, nthreads, uniform);
        return true;
        }
      return false;
      }
    template<typename Tpoints, typename Tgrid> bool prep_u2nu
      (const cmav<complex<Tpoints>,1> &points, const cmav<complex<Tgrid>,ndim> &uniform)
      {
      static_assert(sizeof(Tpoints)<=sizeof(Tcalc),
        "Tcalc must be at least as accurate as Tpoints");
      static_assert(sizeof(Tgrid)<=sizeof(Tcalc),
        "Tcalc must be at least as accurate as Tgrid");
      MR_assert(points.shape(0)==npoints, "number of points mismatch");
      MR_assert(uniform.shape()==nuni, "uniform grid dimensions mismatch");
      return npoints==0;
      }

   static string dim2string(const array<size_t, ndim> &arr)
      {
      ostringstream str;
      str << arr[0];
      for (size_t i=1; i<ndim; ++i) str << "x" << arr[i];
      return str.str();
      }

    void report(bool gridding)
      {
      cout << (gridding ? "Nu2u:" : "U2nu:") << endl
           << "  nthreads=" << nthreads << ", grid=(" << dim2string(nuni)
           << "), oversampled grid=(" << dim2string(nover) << "), supp="
           << supp << ", eps=" << epsilon << endl << "  npoints=" << npoints
           << endl << "  memory overhead: "
           << npoints*sizeof(uint32_t)/double(1<<30) << "GB (index) + "
           << accumulate(nover.begin(), nover.end(), 1, multiplies<>())*sizeof(complex<Tcalc>)/double(1<<30) << "GB (oversampled grid)" << endl;
      }

    static array<double, ndim> get_coordfct(const vector<double> &periodicity)
      {
      MR_assert(periodicity.size()==ndim, "periodicity size mismatch");
      array<double, ndim> res;
      for (size_t i=0; i<ndim; ++i)
        res[i] = 1./periodicity[i];
      return res;
      }

  public:
    Nufft_ancestor(bool gridding, size_t npoints_,
      const array<size_t,ndim> &uniform_shape, double epsilon_,
      size_t nthreads_, double sigma_min, double sigma_max,
      const vector<double> &periodicity, bool fft_order_)
      : timers(gridding ? "nu2u" : "u2nu"), epsilon(epsilon_),
        nthreads(adjust_nthreads(nthreads_)),
        coordfct(get_coordfct(periodicity)),
        fft_order(fft_order_), npoints(npoints_), nuni(uniform_shape)
      {
      MR_assert(npoints<=(~uint32_t(0)), "too many nonuniform points");

      timers.push("parameter calculation");
      vector<size_t> tdims{nuni.begin(), nuni.end()};
      auto [kidx, dims] = findNufftParameters<Tcalc,Tacc>
        (epsilon, sigma_min, sigma_max, tdims, npoints, gridding, nthreads);
      for (size_t i=0; i<ndim; ++i)
        {
        nover[i] = dims[i];
        MR_assert((nover[i]>>log2tile)<=max_ntile<ndim>, "oversampled grid too large");
        }
      timers.pop();

      krn = selectKernel(kidx);
      krn_id = kidx;
      supp = krn->support();
      nsafe = (supp+1)/2;

      for (size_t i=0; i<ndim; ++i)
        {
        shift[i] = supp*(-0.5)+1+nover[i];
        maxi0[i] = (nover[i]+nsafe)-supp;
        MR_assert(nover[i]>=2*nsafe, "oversampled length too small");
        MR_assert((nover[i]&1)==0, "oversampled dimensions must be even");
        }
      MR_assert(epsilon>0, "epsilon must be positive");

      timers.push("correction factors");
      for (size_t i=0; i<ndim; ++i)
        if ((i<1) || (nuni[i]!=nuni[i-1]) || (nover[i]!=nover[i-1]))
          corfac.push_back(krn->corfunc(nuni[i]/2+1, 1./nover[i], nthreads));
        else
          corfac.push_back(corfac.back());
      timers.pop();
      }
  };


template<typename Tcalc, typename Tacc, typename Tcoord, size_t ndim> class Nufft;

#define DUCC0_NUFFT_BOILERPLATE \
  private: \
    using parent=Nufft_ancestor<Tcalc, Tacc, ndim>; \
    using parent::coord_idx, parent::nthreads, parent::npoints, parent::supp, \
          parent::timers, parent::krn, parent::krn_id, parent::fft_order, parent::nuni, \
          parent::nover, parent::shift, parent::maxi0, parent::report, \
          parent::log2tile, parent::corfac, parent::sort_coords, \
          parent::prep_nu2u, parent::prep_u2nu; \
 \
    vmav<Tcoord,2> coords_sorted; \
    unique_ptr<Spreadinterp<Tcalc, Tacc, Tcoord, uint32_t, ndim>> spreadinterp; \
 \
  public: \
    using parent::parent; /* inherit constructor */ \
    Nufft(bool gridding, const cmav<Tcoord,2> &coords, \
          const array<size_t, ndim> &uniform_shape_, double epsilon_,  \
          size_t nthreads_, double sigma_min, double sigma_max, \
          const vector<double> &periodicity, bool fft_order_) \
      : parent(gridding, coords.shape(0), uniform_shape_, epsilon_, nthreads_, \
               sigma_min, sigma_max, periodicity, fft_order_), \
        coords_sorted({npoints,ndim},UNINITIALIZED) \
      { \
spreadinterp = make_unique<Spreadinterp<Tcalc, Tacc, Tcoord, uint32_t, ndim>>(coords, nover, krn_id, nthreads, periodicity); \
      build_index(coords); \
      sort_coords(coords, coords_sorted); \
      } \
    Nufft (bool gridding, size_t npoints_, \
      const array<size_t,ndim> &uniform_shape, double epsilon_, \
      size_t nthreads_, double sigma_min, double sigma_max, \
      const vector<double> &periodicity_, bool fft_order_) \
      : parent(gridding, npoints_, uniform_shape, epsilon_, nthreads_, \
               sigma_min, sigma_max, periodicity_, fft_order_) \
      { \
spreadinterp = make_unique<Spreadinterp<Tcalc, Tacc, Tcoord, uint32_t, ndim>>(npoints_, nover, krn_id, nthreads, periodicity_); \
      } \
 \
    template<typename Tpoints, typename Tgrid> void nu2u(bool forward, size_t verbosity, \
      const cmav<complex<Tpoints>,1> &points, const vmav<complex<Tgrid>,ndim> &uniform) \
      { \
      if (prep_nu2u(points, uniform)) return; \
      MR_assert(coords_sorted.size()!=0, "bad call"); \
      if (verbosity>0) report(true); \
      nonuni2uni(forward, coords_sorted, points, uniform); \
      if (verbosity>0) timers.report(cout); \
      } \
    template<typename Tpoints, typename Tgrid> void u2nu(bool forward, size_t verbosity, \
      const cmav<complex<Tgrid>,ndim> &uniform, const vmav<complex<Tpoints>,1> &points) \
      { \
      if (prep_u2nu(points, uniform)) return; \
      MR_assert(coords_sorted.size()!=0, "bad call"); \
      if (verbosity>0) report(false); \
      uni2nonuni(forward, uniform, coords_sorted, points); \
      if (verbosity>0) timers.report(cout); \
      } \
    template<typename Tpoints, typename Tgrid> void nu2u(bool forward, size_t verbosity, \
      const cmav<Tcoord,2> &coords, const cmav<complex<Tpoints>,1> &points, \
      const vmav<complex<Tgrid>,ndim> &uniform) \
      { \
      if (prep_nu2u(points, uniform)) return; \
      MR_assert(coords_sorted.size()==0, "bad call"); \
      if (verbosity>0) report(true); \
      build_index(coords); \
      nonuni2uni(forward, coords, points, uniform); \
      if (verbosity>0) timers.report(cout); \
      } \
    template<typename Tpoints, typename Tgrid> void u2nu(bool forward, size_t verbosity, \
      const cmav<complex<Tgrid>,ndim> &uniform, const cmav<Tcoord,2> &coords, \
      const vmav<complex<Tpoints>,1> &points) \
      { \
      if (prep_u2nu(points, uniform)) return; \
      MR_assert(coords_sorted.size()==0, "bad call"); \
      if (verbosity>0) report(false); \
      build_index(coords); \
      uni2nonuni(forward, uniform, coords, points); \
      if (verbosity>0) timers.report(cout); \
      }

/*! Helper class for carrying out 1D nonuniform FFTs of types 1 and 2.
    Tcalc: the floating-point type in which all kernel-related calculations
           are performed
    Tacc:  the floating-point type used for the grid on which data is
           accumulated in nu2u transforms. Can usually be the same as Tcalc,
           but may be chosen to be more accurate in specific situations.
    Tpoints: the floating-point type used for storing the values at the
           non-uniform points
    Tgrid: the floating-point type used for storing the values on the uniform
           grid.
    Tcoord: the floating-point type used for storing the coordinates of the
           non-uniform points.
 */
template<typename Tcalc, typename Tacc, typename Tcoord> class Nufft<Tcalc, Tacc, Tcoord, 1>: public Nufft_ancestor<Tcalc, Tacc, 1>
  {
  private:
    static constexpr size_t ndim=1;

  DUCC0_NUFFT_BOILERPLATE

  private:
    template<typename Tpoints, typename Tgrid> void nonuni2uni(bool forward,
      const cmav<Tcoord,2> &coords, const cmav<complex<Tpoints>,1> &points,
      const vmav<complex<Tgrid>,ndim> &uniform)
      {
      timers.push("nu2u proper");
      timers.push("allocating grid");
      auto grid = vmav<complex<Tcalc>,ndim>::build_noncritical(nover, UNINITIALIZED);
      timers.poppush("zeroing grid");
      mav_apply([](complex<Tcalc> &v){v=complex<Tcalc>(0);},nthreads,grid);
      timers.poppush("spreading");
//      constexpr size_t maxsupp = is_same<Tacc, float>::value ? 8 : 16;
//      spreading_helper<maxsupp>(supp, coords, points, grid);
(coords_sorted.size()>0) ? spreadinterp->spread(points, grid) : spreadinterp->spread(coords, points, grid);

      timers.poppush("FFT");
      auto fgrid(grid.to_fmav());
      c2c(fgrid, fgrid, {0}, forward, Tcalc(1), nthreads);
      timers.poppush("grid correction");
      execParallel(nuni[0], nthreads, [&](size_t lo, size_t hi)
        {
        for (auto i=lo; i<hi; ++i)
          {
          auto [icfu, iout, iin] = comp_indices(i, nuni[0], nover[0], fft_order);
          uniform(iout) = complex<Tgrid>(grid(iin)*Tcalc(corfac[0][icfu]));
          }
        });
      timers.pop();
      timers.pop();
      }

    template<typename Tpoints, typename Tgrid> void uni2nonuni(bool forward,
      const cmav<complex<Tgrid>,ndim> &uniform, const cmav<Tcoord,2> &coords,
      const vmav<complex<Tpoints>,1> &points)
      {
      timers.push("u2nu proper");
      timers.push("allocating grid");
      auto grid = vmav<complex<Tcalc>,ndim>::build_noncritical(nover, UNINITIALIZED);
      timers.poppush("zeroing grid");
      mav_apply([](complex<Tcalc> &v){v=complex<Tcalc>(0);},nthreads,grid);
      timers.poppush("grid correction");
      execParallel(nuni[0], nthreads, [&](size_t lo, size_t hi)
        {
        for (auto i=lo; i<hi; ++i)
          {
          auto [icfu, iin, iout] = comp_indices(i, nuni[0], nover[0], fft_order);
          grid(iout) = complex<Tcalc>(uniform(iin))*Tcalc(corfac[0][icfu]);
          }
        });
      timers.poppush("FFT");
      auto fgrid(grid.to_fmav());
      c2c(fgrid, fgrid, {0}, forward, Tcalc(1), nthreads);
      timers.poppush("interpolation");
//      constexpr size_t maxsupp = is_same<Tcalc, float>::value ? 8 : 16;
//      interpolation_helper<maxsupp>(supp, grid, coords, points);
(coords_sorted.size()>0) ?  spreadinterp->interp(grid, points) : spreadinterp->interp(grid, coords, points);

      timers.pop();
      timers.pop();
      }

    void build_index(const cmav<Tcoord,2> &coords)
      {
      timers.push("building index");
      MR_assert(coords.shape(0)==npoints, "number of coords mismatch");
      MR_assert(coords.shape(1)==ndim, "ndim mismatch");
      size_t ntiles_u = (nover[0]>>log2tile) + 3;
      coord_idx.resize(npoints);
      quick_array<uint32_t> key(npoints);
      execParallel(npoints, nthreads, [&](size_t lo, size_t hi)
        {
        for (size_t i=lo; i<hi; ++i)
          key[i] = parent::template get_tile<Tcoord>({coords(i,0)})[0];
        });
      bucket_sort2(key, coord_idx, ntiles_u, nthreads);
      timers.pop();
      }
  };

template<typename Tcalc, typename Tacc, typename Tcoord> class Nufft<Tcalc, Tacc, Tcoord, 2>: public Nufft_ancestor<Tcalc, Tacc, 2>
  {
  private:
    static constexpr size_t ndim=2;

  DUCC0_NUFFT_BOILERPLATE

    template<typename Tpoints, typename Tgrid> void nonuni2uni(bool forward,
      const cmav<Tcoord,2> &coords, const cmav<complex<Tpoints>,1> &points,
      const vmav<complex<Tgrid>,ndim> &uniform)
      {
      timers.push("nu2u proper");
      timers.push("allocating grid");
      auto grid = vmav<complex<Tcalc>,ndim>::build_noncritical(nover, UNINITIALIZED);
      timers.poppush("zeroing grid");
      mav_apply([](complex<Tcalc> &v){v=complex<Tcalc>(0);},nthreads,grid);
      timers.poppush("spreading");
//      constexpr size_t maxsupp = is_same<Tacc, float>::value ? 8 : 16;
//      spreading_helper<maxsupp>(supp, coords, points, grid);
(coords_sorted.size()>0) ? spreadinterp->spread(points, grid) : spreadinterp->spread(coords, points, grid);

      timers.poppush("FFT");
      {
      auto fgrid(grid.to_fmav());
      c2c(fgrid, fgrid, {1}, forward, Tcalc(1), nthreads);
      auto fgridl=fgrid.subarray({{},{0,(nuni[1]+1)/2}});
      c2c(fgridl, fgridl, {0}, forward, Tcalc(1), nthreads);
      if (nuni[1]>1)
        {
        auto fgridh=fgrid.subarray({{},{fgrid.shape(1)-nuni[1]/2,MAXIDX}});
        c2c(fgridh, fgridh, {0}, forward, Tcalc(1), nthreads);
        }
      }
      timers.poppush("grid correction");
      execParallel(nuni[0], nthreads, [&](size_t lo, size_t hi)
        {
        for (auto i=lo; i<hi; ++i)
          {
          auto [icfu, iout, iin] = comp_indices(i, nuni[0], nover[0], fft_order);
          for (size_t j=0; j<nuni[1]; ++j)
            {
            auto [icfv, jout, jin] = comp_indices(j, nuni[1], nover[1], fft_order);
            uniform(iout,jout) = complex<Tgrid>(grid(iin,jin)
              *Tcalc(corfac[0][icfu]*corfac[1][icfv]));
            }
          }
        });
      timers.pop();
      timers.pop();
      }

    template<typename Tpoints, typename Tgrid> void uni2nonuni(bool forward,
      const cmav<complex<Tgrid>,ndim> &uniform, const cmav<Tcoord,2> &coords,
      const vmav<complex<Tpoints>,1> &points)
      {
      timers.push("u2nu proper");
      timers.push("allocating grid");
      auto grid = vmav<complex<Tcalc>,ndim>::build_noncritical(nover, UNINITIALIZED);
      timers.poppush("zeroing grid");

      // only zero the parts of the grid that are not filled afterwards anyway
      { auto a0 = subarray<2>(grid, {{0,(nuni[0]+1)/2}, {nuni[1]/2,nover[1]-nuni[1]/2}}); quickzero(a0, nthreads); }
      { auto a0 = subarray<2>(grid, {{(nuni[0]+1)/2, nover[0]-nuni[0]/2}, {}}); quickzero(a0, nthreads); }
      if (nuni[0]>1)
        { auto a0 = subarray<2>(grid, {{nover[0]-nuni[0]/2,MAXIDX}, {nuni[1]/2, nover[1]-nuni[1]/2+1}}); quickzero(a0, nthreads); }
      timers.poppush("grid correction");
      execParallel(nuni[0], nthreads, [&](size_t lo, size_t hi)
        {
        for (auto i=lo; i<hi; ++i)
          {
          auto [icfu, iin, iout] = comp_indices(i, nuni[0], nover[0], fft_order);
          for (size_t j=0; j<nuni[1]; ++j)
            {
            auto [icfv, jin, jout] = comp_indices(j, nuni[1], nover[1], fft_order);
            grid(iout,jout) = complex<Tcalc>(uniform(iin,jin))
              *Tcalc(corfac[0][icfu]*corfac[1][icfv]);
            }
          }
        });
      timers.poppush("FFT");
      {
      auto fgrid(grid.to_fmav());
      auto fgridl=fgrid.subarray({{},{0,(nuni[1]+1)/2}});
      c2c(fgridl, fgridl, {0}, forward, Tcalc(1), nthreads);
      if (nuni[1]>1)
        {
        auto fgridh=fgrid.subarray({{},{fgrid.shape(1)-nuni[1]/2,MAXIDX}});
        c2c(fgridh, fgridh, {0}, forward, Tcalc(1), nthreads);
        }
      c2c(fgrid, fgrid, {1}, forward, Tcalc(1), nthreads);
      }
      timers.poppush("interpolation");
//      constexpr size_t maxsupp = is_same<Tcalc, float>::value ? 8 : 16;
//      interpolation_helper<maxsupp>(supp, grid, coords, points);
(coords_sorted.size()>0) ?  spreadinterp->interp(grid, points) : spreadinterp->interp(grid, coords, points);
      timers.pop();
      timers.pop();
      }

    void build_index(const cmav<Tcoord,2> &coords)
      {
      timers.push("building index");
      size_t ntiles_u = (nover[0]>>log2tile) + 3;
      size_t ntiles_v = (nover[1]>>log2tile) + 3;
      coord_idx.resize(npoints);
      quick_array<uint32_t> key(npoints);
      execParallel(npoints, nthreads, [&](size_t lo, size_t hi)
        {
        for (size_t i=lo; i<hi; ++i)
          {
          auto tile = parent::template get_tile<Tcoord>({coords(i,0), coords(i,1)});
          key[i] = tile[0]*ntiles_v + tile[1];
          }
        });
      bucket_sort2(key, coord_idx, ntiles_u*ntiles_v, nthreads);
      timers.pop();
      }
  };

template<typename Tcalc, typename Tacc, typename Tcoord> class Nufft<Tcalc, Tacc, Tcoord, 3>: public Nufft_ancestor<Tcalc, Tacc, 3>
  {
  private:
    static constexpr size_t ndim=3;

  DUCC0_NUFFT_BOILERPLATE

    template<typename Tpoints, typename Tgrid> void nonuni2uni(bool forward,
      const cmav<Tcoord,2> &coords, const cmav<complex<Tpoints>,1> &points,
      const vmav<complex<Tgrid>,ndim> &uniform)
      {
      timers.push("nu2u proper");
      timers.push("allocating grid");
      auto grid = vmav<complex<Tcalc>,ndim>::build_noncritical(nover, UNINITIALIZED);
      timers.poppush("zeroing grid");
      mav_apply([](complex<Tcalc> &v){v=complex<Tcalc>(0);},nthreads,grid);
      timers.poppush("spreading");
//      constexpr size_t maxsupp = is_same<Tacc, float>::value ? 8 : 16;
//      spreading_helper<maxsupp>(supp, coords, points, grid);
(coords_sorted.size()>0) ? spreadinterp->spread(points, grid) : spreadinterp->spread(coords, points, grid);
      timers.poppush("FFT");
      {
      auto fgrid(grid.to_fmav());
      slice slz{0,(nuni[2]+1)/2}, shz{fgrid.shape(2)-nuni[2]/2,MAXIDX};
      slice sly{0,(nuni[1]+1)/2}, shy{fgrid.shape(1)-nuni[1]/2,MAXIDX};
      c2c(fgrid, fgrid, {2}, forward, Tcalc(1), nthreads);
      auto fgridl=fgrid.subarray({{},{},slz});
      c2c(fgridl, fgridl, {1}, forward, Tcalc(1), nthreads);
      if (nuni[2]>1)
        {
        auto fgridh=fgrid.subarray({{},{},shz});
        c2c(fgridh, fgridh, {1}, forward, Tcalc(1), nthreads);
        }
      auto fgridll=fgrid.subarray({{},sly,slz});
      c2c(fgridll, fgridll, {0}, forward, Tcalc(1), nthreads);
      if (nuni[2]>1)
        {
        auto fgridlh=fgrid.subarray({{},sly,shz});
        c2c(fgridlh, fgridlh, {0}, forward, Tcalc(1), nthreads);
        }
      if (nuni[1]>1)
        {
        auto fgridhl=fgrid.subarray({{},shy,slz});
        c2c(fgridhl, fgridhl, {0}, forward, Tcalc(1), nthreads);
        if (nuni[2]>1)
          {
          auto fgridhh=fgrid.subarray({{},shy,shz});
          c2c(fgridhh, fgridhh, {0}, forward, Tcalc(1), nthreads);
          }
        }
      }
      timers.poppush("grid correction");
      execParallel(nuni[0], nthreads, [&](size_t lo, size_t hi)
        {
        for (auto i=lo; i<hi; ++i)
          {
          auto [icfu, iout, iin] = comp_indices(i, nuni[0], nover[0], fft_order);
          for (size_t j=0; j<nuni[1]; ++j)
            {
            auto [icfv, jout, jin] = comp_indices(j, nuni[1], nover[1], fft_order);
            for (size_t k=0; k<nuni[2]; ++k)
              {
              auto [icfw, kout, kin] = comp_indices(k, nuni[2], nover[2], fft_order);
              uniform(iout,jout,kout) = complex<Tgrid>(grid(iin,jin,kin)
                *Tcalc(corfac[0][icfu]*corfac[1][icfv]*corfac[2][icfw]));
              }
            }
          }
        });
      timers.pop();
      timers.pop();
      }

    template<typename Tpoints, typename Tgrid> void uni2nonuni(bool forward,
      const cmav<complex<Tgrid>,ndim> &uniform, const cmav<Tcoord,2> &coords,
      const vmav<complex<Tpoints>,1> &points)
      {
      timers.push("u2nu proper");
      timers.push("allocating grid");
      auto grid = vmav<complex<Tcalc>,ndim>::build_noncritical(nover, UNINITIALIZED);
      timers.poppush("zeroing grid");
      // TODO: not all entries need to be zeroed, perhaps some time can be saved here
      mav_apply([](complex<Tcalc> &v){v=complex<Tcalc>(0);},nthreads,grid);
      timers.poppush("grid correction");
      execParallel(nuni[0], nthreads, [&](size_t lo, size_t hi)
        {
        for (auto i=lo; i<hi; ++i)
          {
          auto [icfu, iin, iout] = comp_indices(i, nuni[0], nover[0], fft_order);
          for (size_t j=0; j<nuni[1]; ++j)
            {
            auto [icfv, jin, jout] = comp_indices(j, nuni[1], nover[1], fft_order);
            for (size_t k=0; k<nuni[2]; ++k)
              {
              auto [icfw, kin, kout] = comp_indices(k, nuni[2], nover[2], fft_order);
              grid(iout,jout,kout) = complex<Tcalc>(uniform(iin,jin,kin))
                *Tcalc(corfac[0][icfu]*corfac[1][icfv]*corfac[2][icfw]);
              }
            }
          }
        });
      timers.poppush("FFT");
      {
      auto fgrid(grid.to_fmav());
      slice slz{0,(nuni[2]+1)/2}, shz{fgrid.shape(2)-nuni[2]/2,MAXIDX};
      slice sly{0,(nuni[1]+1)/2}, shy{fgrid.shape(1)-nuni[1]/2,MAXIDX};
      auto fgridll=fgrid.subarray({{},sly,slz});
      c2c(fgridll, fgridll, {0}, forward, Tcalc(1), nthreads);
      if (nuni[2]>1)
        {
        auto fgridlh=fgrid.subarray({{},sly,shz});
        c2c(fgridlh, fgridlh, {0}, forward, Tcalc(1), nthreads);
        }
      if (nuni[1]>1)
        {
        auto fgridhl=fgrid.subarray({{},shy,slz});
        c2c(fgridhl, fgridhl, {0}, forward, Tcalc(1), nthreads);
        if (nuni[2]>1)
          {
          auto fgridhh=fgrid.subarray({{},shy,shz});
          c2c(fgridhh, fgridhh, {0}, forward, Tcalc(1), nthreads);
          }
        }
      auto fgridl=fgrid.subarray({{},{},slz});
      c2c(fgridl, fgridl, {1}, forward, Tcalc(1), nthreads);
      if (nuni[2]>1)
        {
        auto fgridh=fgrid.subarray({{},{},shz});
        c2c(fgridh, fgridh, {1}, forward, Tcalc(1), nthreads);
        }
      c2c(fgrid, fgrid, {2}, forward, Tcalc(1), nthreads);
      }
      timers.poppush("interpolation");
//      constexpr size_t maxsupp = is_same<Tcalc, float>::value ? 8 : 16;
//      interpolation_helper<maxsupp>(supp, grid, coords, points);
(coords_sorted.size()>0) ?  spreadinterp->interp(grid, points) : spreadinterp->interp(grid, coords, points);
      timers.pop();
      timers.pop();
      }

    void build_index(const cmav<Tcoord,2> &coords)
      {
      timers.push("building index");
      size_t ntiles_u = (nover[0]>>log2tile) + 3;
      size_t ntiles_v = (nover[1]>>log2tile) + 3;
      size_t ntiles_w = (nover[2]>>log2tile) + 3;
      size_t lsq2 = log2tile;
      while ((lsq2>=1) && (((ntiles_u*ntiles_v*ntiles_w)<<(3*(log2tile-lsq2)))<(size_t(1)<<28)))
        --lsq2;
      auto ssmall = log2tile-lsq2;
      auto msmall = (size_t(1)<<ssmall) - 1;

      coord_idx.resize(npoints);
      quick_array<uint32_t> key(npoints);
      execParallel(npoints, nthreads, [&](size_t lo, size_t hi)
        {
        for (size_t i=lo; i<hi; ++i)
          {
          auto tile = parent::template get_tile<Tcoord>({coords(i,0),coords(i,1),coords(i,2)},lsq2);
          auto lowkey = ((tile[0]&msmall)<<(2*ssmall))
                      | ((tile[1]&msmall)<<   ssmall)
                      |  (tile[2]&msmall);
          auto hikey = ((tile[0]>>ssmall)*ntiles_v*ntiles_w)
                     + ((tile[1]>>ssmall)*ntiles_w)
                     +  (tile[2]>>ssmall);
          key[i] = (hikey<<(3*ssmall)) | lowkey;
          }
        });
      bucket_sort2(key, coord_idx, (ntiles_u*ntiles_v*ntiles_w)<<(3*ssmall), nthreads);
      timers.pop();
      }
  };

#undef DUCC0_NUFFT_BOILERPLATE

template<typename Tcalc, typename Tacc, typename Tpoints, typename Tgrid, typename Tcoord>
  void nu2u(const cmav<Tcoord,2> &coord, const cmav<complex<Tpoints>,1> &points,
    bool forward, double epsilon, size_t nthreads,
    const vfmav<complex<Tgrid>> &uniform, size_t verbosity,
    double sigma_min, double sigma_max, const vector<double> &periodicity, bool fft_order)
  {
  auto ndim = uniform.ndim();
  MR_assert((ndim>=1) && (ndim<=3), "transform must be 1D/2D/3D");
  MR_assert(ndim==coord.shape(1), "dimensionality mismatch");
  if (ndim==1)
    {
    vmav<complex<Tgrid>,1> uniform2(uniform);
    Nufft<Tcalc, Tacc, Tcoord, 1> nufft(true, points.shape(0), uniform2.shape(),
      epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
    nufft.nu2u(forward, verbosity, coord, points, uniform2); 
    }
  else if (ndim==2)
    {
    vmav<complex<Tgrid>,2> uniform2(uniform);
    Nufft<Tcalc, Tacc, Tcoord, 2> nufft(true, points.shape(0), uniform2.shape(),
      epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
    nufft.nu2u(forward, verbosity, coord, points, uniform2); 
    }
  else if (ndim==3)
    {
    vmav<complex<Tgrid>,3> uniform2(uniform);
    Nufft<Tcalc, Tacc, Tcoord, 3> nufft(true, points.shape(0), uniform2.shape(),
      epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
    nufft.nu2u(forward, verbosity, coord, points, uniform2); 
    }
  }
template<typename Tcalc, typename Tacc, typename Tpoints, typename Tgrid, typename Tcoord>
  void u2nu(const cmav<Tcoord,2> &coord, const cfmav<complex<Tgrid>> &uniform,
    bool forward, double epsilon, size_t nthreads,
    const vmav<complex<Tpoints>,1> &points, size_t verbosity,
    double sigma_min, double sigma_max, const vector<double> &periodicity, bool fft_order)
  {
  auto ndim = uniform.ndim();
  MR_assert((ndim>=1) && (ndim<=3), "transform must be 1D/2D/3D");
  MR_assert(ndim==coord.shape(1), "dimensionality mismatch");
  if (ndim==1)
    {
    cmav<complex<Tgrid>,1> uniform2(uniform);
    Nufft<Tcalc, Tacc, Tcoord, 1> nufft(false, points.shape(0), uniform2.shape(),
      epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
    nufft.u2nu(forward, verbosity, uniform2, coord, points); 
    }
  else if (ndim==2)
    {
    cmav<complex<Tgrid>,2> uniform2(uniform);
    Nufft<Tcalc, Tacc, Tcoord, 2> nufft(false, points.shape(0), uniform2.shape(),
      epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
    nufft.u2nu(forward, verbosity, uniform2, coord, points); 
    }
  else if (ndim==3)
    {
    cmav<complex<Tgrid>,3> uniform2(uniform);
    Nufft<Tcalc, Tacc, Tcoord, 3> nufft(false, points.shape(0), uniform2.shape(),
      epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
    nufft.u2nu(forward, verbosity, uniform2, coord, points); 
    }
  }
} // namespace detail_nufft

// public names
using detail_nufft::findNufftKernel;
using detail_nufft::u2nu;
using detail_nufft::nu2u;
using detail_nufft::Nufft;

} // namespace ducc0

#endif
