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

/* Copyright (C) 2019-2025 Max-Planck-Society
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
#include "ducc0/nufft/spreadinterp.h"

namespace ducc0 {

namespace detail_nufft {

template<typename Tcalc> void nufft_FFT(bool gridding, bool forward,
  const vfmav<complex<Tcalc>> &grid, const vector<size_t> &nuni, size_t nthreads)
  {
  size_t ndim = grid.ndim();

  for (size_t iter=0; iter<ndim; ++iter)
    {
    auto idim = gridding ? ndim-1-iter : iter;
    if (idim+1==ndim)
      c2c(grid, grid, {idim}, forward, Tcalc(1), nthreads);
    else if (idim+2==ndim)
      {
      vector<slice> slices(ndim);
      vector<slice> sub {{0,(nuni[idim+1]+1)/2}, {grid.shape(idim+1)-nuni[idim+1]/2,MAXIDX}};
      for (size_t i=0; i<((nuni[idim+1]==1)?1:2); ++i)
        {
        slices[idim+1] = sub[i];
        auto subgrid=grid.subarray(slices);
        c2c(subgrid, subgrid, {idim}, forward, Tcalc(1), nthreads);
        }
      }
    else if (idim+3==ndim)
      {
      vector<slice> slices(ndim);
      vector<slice> sub1 {{0,(nuni[idim+1]+1)/2}, {grid.shape(idim+1)-nuni[idim+1]/2,MAXIDX}};
      vector<slice> sub2 {{0,(nuni[idim+2]+1)/2}, {grid.shape(idim+2)-nuni[idim+2]/2,MAXIDX}};
      for (size_t i=0; i<((nuni[idim+1]==1)?1:2); ++i)
        {
        slices[idim+1] = sub1[i];
        for (size_t j=0; j<((nuni[idim+2]==1)?1:2); ++j)
          {
          slices[idim+2] = sub2[j];
          auto subgrid=grid.subarray(slices);
          c2c(subgrid, subgrid, {idim}, forward, Tcalc(1), nthreads);
          }
        }
      }
    }
  }

template<typename Tcalc, typename Tgrid> void deconv_nu2u(
  const cfmav<complex<Tcalc>> &grid,
  const vfmav<complex<Tgrid>> &uniform,
  vector<vector<double>> &corfac,
  bool fft_order,
  size_t nthreads)
  {
  static_assert(sizeof(Tgrid)<=sizeof(Tcalc),
    "Tcalc must be at least as accurate as Tgrid");

  size_t ndim = grid.ndim();

  if (ndim==1)
    {
    cmav<complex<Tcalc>,1> grid2(grid);
    vmav<complex<Tgrid>,1> uni2(uniform);
    size_t nuni0=uni2.shape(0), nover0=grid2.shape(0);
    execParallel(nuni0, nthreads, [&](size_t lo, size_t hi)
      {
      for (auto i=lo; i<hi; ++i)
        {
        auto [icfu, iout, iin] = comp_indices(i, nuni0, nover0, fft_order);
        uni2(iout) = complex<Tgrid>(grid2(iin)*Tcalc(corfac[0][icfu]));
        }
      });
    }
  else if (ndim==2)
    {
    cmav<complex<Tcalc>,2> grid2(grid);
    vmav<complex<Tgrid>,2> uni2(uniform);
    size_t nuni0=uni2.shape(0), nover0=grid2.shape(0),
           nuni1=uni2.shape(1), nover1=grid2.shape(1);
    execParallel(nuni0, nthreads, [&](size_t lo, size_t hi)
      {
      for (auto i=lo; i<hi; ++i)
        {
        auto [icfu, iout, iin] = comp_indices(i, nuni0, nover0, fft_order);
        double cf0=corfac[0][icfu];
        for (size_t j=0; j<nuni1; ++j)
          {
          auto [icfv, jout, jin] = comp_indices(j, nuni1, nover1, fft_order);
          uni2(iout,jout) = complex<Tgrid>(grid2(iin,jin)
              *Tcalc(cf0*corfac[1][icfv]));
          }
        }
      });
    }
  else if (ndim==3)
    {
    cmav<complex<Tcalc>,3> grid2(grid);
    vmav<complex<Tgrid>,3> uni2(uniform);
    size_t nuni0=uni2.shape(0), nover0=grid2.shape(0),
           nuni1=uni2.shape(1), nover1=grid2.shape(1),
           nuni2=uni2.shape(2), nover2=grid2.shape(2);
    execParallel(nuni0, nthreads, [&](size_t lo, size_t hi)
      {
      for (auto i=lo; i<hi; ++i)
        {
        auto [icfu, iout, iin] = comp_indices(i, nuni0, nover0, fft_order);
        double cf0=corfac[0][icfu];
        for (size_t j=0; j<nuni1; ++j)
          {
          auto [icfv, jout, jin] = comp_indices(j, nuni1, nover1, fft_order);
          double cf01=cf0*corfac[1][icfv];
          for (size_t k=0; k<nuni2; ++k)
            {
            auto [icfw, kout, kin] = comp_indices(k, nuni2, nover2, fft_order);
            uni2(iout,jout,kout) = complex<Tgrid>(grid2(iin,jin,kin)
                *Tcalc(cf01*corfac[2][icfw]));
            }
          }
        }
      });
    }
  }
template<typename Tcalc, typename Tgrid> void deconv_u2nu(
  const cfmav<complex<Tgrid>> &uniform,
  const vfmav<complex<Tcalc>> &grid,
  vector<vector<double>> &corfac,
  bool fft_order,
  size_t nthreads)
  {
  size_t ndim = grid.ndim();

  if (ndim==1)
    {
    vmav<complex<Tcalc>,1> grid2(grid);
    cmav<complex<Tgrid>,1> uni2(uniform);
    size_t nuni0=uni2.shape(0), nover0=grid2.shape(0);
    execParallel(nuni0, nthreads, [&](size_t lo, size_t hi)
      {
      for (auto i=lo; i<hi; ++i)
        {
        auto [icfu, iin, iout] = comp_indices(i, nuni0, nover0, fft_order);
        grid2(iout) = complex<Tcalc>(uni2(iin))*Tcalc(corfac[0][icfu]);
        }
      });
    }
  else if (ndim==2)
    {
    vmav<complex<Tcalc>,2> grid2(grid);
    cmav<complex<Tgrid>,2> uni2(uniform);
    size_t nuni0=uni2.shape(0), nover0=grid2.shape(0),
           nuni1=uni2.shape(1), nover1=grid2.shape(1);
    execParallel(nuni0, nthreads, [&](size_t lo, size_t hi)
      {
      for (auto i=lo; i<hi; ++i)
        {
        auto [icfu, iin, iout] = comp_indices(i, nuni0, nover0, fft_order);
        double cf0=corfac[0][icfu];
        for (size_t j=0; j<nuni1; ++j)
          {
          auto [icfv, jin, jout] = comp_indices(j, nuni1, nover1, fft_order);
          grid2(iout,jout) = complex<Tcalc>(uni2(iin,jin))*Tcalc(cf0*corfac[1][icfv]);
          }
        }
      });
    }
  else if (ndim==3)
    {
    vmav<complex<Tcalc>,3> grid2(grid);
    cmav<complex<Tgrid>,3> uni2(uniform);
    size_t nuni0=uni2.shape(0), nover0=grid2.shape(0),
           nuni1=uni2.shape(1), nover1=grid2.shape(1),
           nuni2=uni2.shape(2), nover2=grid2.shape(2);
    execParallel(nuni0, nthreads, [&](size_t lo, size_t hi)
      {
      for (auto i=lo; i<hi; ++i)
        {
        auto [icfu, iin, iout] = comp_indices(i, nuni0, nover0, fft_order);
        double cf0=corfac[0][icfu];
        for (size_t j=0; j<nuni1; ++j)
          {
          auto [icfv, jin, jout] = comp_indices(j, nuni1, nover1, fft_order);
          double cf01=cf0*corfac[1][icfv];
          for (size_t k=0; k<nuni2; ++k)
            {
            auto [icfw, kin, kout] = comp_indices(k, nuni2, nover2, fft_order);
            grid2(iout,jout,kout) = complex<Tcalc>(uni2(iin,jin,kin))*Tcalc(cf01*corfac[2][icfw]);
            }
          }
        }
      });
    }
  }

template<typename Tcalc, typename Tacc> class Nufft_ancestor
  {
  protected:
    TimerHierarchy timers;
    // requested epsilon value for this transform.
    double epsilon;
    // number of threads to use for this transform.
    size_t nthreads;

    // if true, start with zero mode
    // if false, start with most negative mode
    bool fft_order;

    // number of non-uniform points
    size_t npoints;

    // uniform grid dimensions
    vector<size_t> nuni;

    // oversampled grid dimensions
    vector<size_t> nover;

size_t krn_id;
    shared_ptr<PolynomialKernel> krn;

    size_t supp;

    vector<vector<double>> corfac;

    static string dim2string(const vector<size_t> &arr)
      {
      ostringstream str;
      str << arr[0];
      for (size_t i=1; i<arr.size(); ++i) str << "x" << arr[i];
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

  public:
    Nufft_ancestor(bool gridding, size_t npoints_,
      const vector<size_t> &uniform_shape, double epsilon_,
      size_t nthreads_, double sigma_min, double sigma_max,
      bool fft_order_)
      : timers(gridding ? "nu2u" : "u2nu"), epsilon(epsilon_),
        nthreads(adjust_nthreads(nthreads_)),
        fft_order(fft_order_), npoints(npoints_), nuni(uniform_shape)
      {
      MR_assert(npoints<=(~uint32_t(0)), "too many nonuniform points");

      timers.push("parameter calculation");
      vector<size_t> tdims{nuni.begin(), nuni.end()};
      auto [kidx, dims] = findNufftParameters<Tcalc,Tacc>
        (epsilon, sigma_min, sigma_max, tdims, npoints, gridding, nthreads);
      nover = dims;
      timers.pop();

      krn = selectKernel(kidx);
      krn_id = kidx;
      supp = krn->support();

      MR_assert(epsilon>0, "epsilon must be positive");

      timers.push("correction factors");
      for (size_t i=0; i<nuni.size(); ++i)
        if ((i<1) || (nuni[i]!=nuni[i-1]) || (nover[i]!=nover[i-1]))
          corfac.push_back(krn->corfunc(nuni[i]/2+1, 1./nover[i], nthreads));
        else
          corfac.push_back(corfac.back());
      timers.pop();
      }

    const vector<size_t> &get_gridsize() const { return nover; }
  };


template<typename Tcalc, typename Tacc, typename Tcoord> class Nufft:
  public Nufft_ancestor<Tcalc, Tacc>
  {
  private:
    using parent=Nufft_ancestor<Tcalc, Tacc>;
    using parent::nthreads,
          parent::timers, parent::krn_id, parent::fft_order, parent::nuni,
          parent::nover, parent::report,
          parent::corfac;

    Spreadinterp2<Tcalc, Tacc, Tcoord, uint32_t> spreadinterp;

  public:
    using parent::parent; /* inherit constructor */
    using parent::get_gridsize;
    Nufft(bool gridding, const cmav<Tcoord,2> &coords,
          const vector<size_t> &uniform_shape_, double epsilon_, 
          size_t nthreads_, double sigma_min, double sigma_max,
          const vector<double> &periodicity, bool fft_order_,
          const vector<double> &corigin=vector<double>())
      : parent(gridding, coords.shape(0), uniform_shape_, epsilon_, nthreads_,
               sigma_min, sigma_max, fft_order_),
        spreadinterp(coords, nover, krn_id, nthreads, periodicity, corigin)
      {}
    Nufft (bool gridding, size_t npoints_,
      const vector<size_t> &uniform_shape, double epsilon_,
      size_t nthreads_, double sigma_min, double sigma_max,
      const vector<double> &periodicity, bool fft_order_,
          const vector<double> &corigin=vector<double>())
      : parent(gridding, npoints_, uniform_shape, epsilon_, nthreads_,
               sigma_min, sigma_max, fft_order_),
        spreadinterp(npoints_, nover, krn_id, nthreads, periodicity, corigin)
      {}

    template<typename Tpoints, typename Tgrid> void nu2u(bool forward, size_t verbosity,
      const cmav<complex<Tpoints>,1> &points, const vfmav<complex<Tgrid>> &uniform)
      {
      MR_assert(uniform.shape()==nuni, "uniform grid dimensions mismatch");
      if (points.shape(0)==0)
        {
        mav_apply([](complex<Tgrid> &v){v=complex<Tgrid>(0);}, nthreads, uniform);
        return;
        }
      if (verbosity>0) report(true);
      auto dummy = cmav<Tcoord,2>::build_empty();
      nonuni2uni(forward, dummy, points, uniform);
      if (verbosity>0) timers.report(cout);
      }
    template<typename Tpoints, typename Tgrid> void u2nu(bool forward, size_t verbosity,
      const cfmav<complex<Tgrid>> &uniform, const vmav<complex<Tpoints>,1> &points)
      {
      MR_assert(uniform.shape()==nuni, "uniform grid dimensions mismatch");
      if(points.shape(0)==0) return;
      if (verbosity>0) report(false);
      auto dummy = cmav<Tcoord,2>::build_empty();
      uni2nonuni(forward, uniform, dummy, points);
      if (verbosity>0) timers.report(cout);
      }
    template<typename Tpoints, typename Tgrid> void nu2u(bool forward, size_t verbosity,
      const cmav<Tcoord,2> &coords, const cmav<complex<Tpoints>,1> &points,
      const vfmav<complex<Tgrid>> &uniform)
      {
      MR_assert(uniform.shape()==nuni, "uniform grid dimensions mismatch");
      if ((points.shape(0)==0)&&(coords.shape(0)==0))
        {
        mav_apply([](complex<Tgrid> &v){v=complex<Tgrid>(0);}, nthreads, uniform);
        return;
        }
      if (verbosity>0) report(true);
      nonuni2uni(forward, coords, points, uniform);
      if (verbosity>0) timers.report(cout);
      }
    template<typename Tpoints, typename Tgrid> void u2nu(bool forward, size_t verbosity,
      const cfmav<complex<Tgrid>> &uniform, const cmav<Tcoord,2> &coords,
      const vmav<complex<Tpoints>,1> &points)
      {
      MR_assert(uniform.shape()==nuni, "uniform grid dimensions mismatch");
      if((points.shape(0)==0)&&(coords.shape(0)==0)) return;
      if (verbosity>0) report(false);
      uni2nonuni(forward, uniform, coords, points);
      if (verbosity>0) timers.report(cout);
      }
    template<typename Tpoints, typename Tgrid> void spread(
      const cmav<Tcoord,2> &coords, const cmav<complex<Tpoints>,1> &points,
      const vfmav<complex<Tgrid>> &grid)
      {
      MR_assert(grid.shape()==nover, "grid dimensions mismatch");
      spreadinterp.spread(coords, points, grid);
      }
    template<typename Tgrid> void spread_finish(bool forward,
      const vfmav<complex<Tgrid>> &grid, const vfmav<complex<Tgrid>> &uniform)
      {
      MR_assert(grid.shape()==nover, "grid dimensions mismatch");
      MR_assert(uniform.shape()==nuni, "grid dimensions mismatch");
      nufft_FFT(true, forward, grid, nuni, nthreads);
      deconv_nu2u(grid, uniform, corfac, fft_order, nthreads);
      }
    template<typename Tpoints, typename Tgrid> void interp(
      const cmav<Tcoord,2> &coords, const vmav<complex<Tpoints>,1> &points,
      const cfmav<complex<Tgrid>> &grid)
      {
      MR_assert(grid.shape()==nover, "grid dimensions mismatch");
      spreadinterp.interp(grid, coords, points);
      }
    template<typename Tgrid> void interp_prep(bool forward,
      const vfmav<complex<Tgrid>> &grid, const cfmav<complex<Tgrid>> &uniform)
      {
      MR_assert(grid.shape()==nover, "grid dimensions mismatch");
      MR_assert(uniform.shape()==nuni, "grid dimensions mismatch");
      deconv_u2nu(uniform, grid, corfac, fft_order, nthreads);
      nufft_FFT(false, forward, grid, nuni, nthreads);
      }

/*! Helper class for carrying out nonuniform FFTs of types 1 and 2.
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


  private:
    template<typename Tpoints, typename Tgrid> void nonuni2uni(bool forward,
      const cmav<Tcoord,2> &coords, const cmav<complex<Tpoints>,1> &points,
      const vfmav<complex<Tgrid>> &uniform)
      {
      timers.push("nu2u proper");
      timers.push("allocating grid");
      auto grid = vfmav<complex<Tcalc>>::build_noncritical(nover, UNINITIALIZED);
      timers.poppush("zeroing grid");
      mav_apply([](complex<Tcalc> &v){v=complex<Tcalc>(0);},nthreads,grid);
      timers.poppush("spreading");
      (coords.size()==0) ? spreadinterp.spread(points, grid)
                         : spreadinterp.spread(coords, points, grid);

      timers.poppush("FFT");
      nufft_FFT(true, forward, grid, nuni, nthreads);

      timers.poppush("grid correction");
      deconv_nu2u(grid, uniform, corfac, fft_order, nthreads);
      timers.pop();
      timers.pop();
      }

    template<typename Tpoints, typename Tgrid> void uni2nonuni(bool forward,
      const cfmav<complex<Tgrid>> &uniform, const cmav<Tcoord,2> &coords,
      const vmav<complex<Tpoints>,1> &points)
      {
      timers.push("u2nu proper");
      timers.push("allocating grid");
      auto grid = vfmav<complex<Tcalc>>::build_noncritical(nover, UNINITIALIZED);
      timers.poppush("zeroing grid");
      mav_apply([](complex<Tcalc> &v){v=complex<Tcalc>(0);},nthreads,grid);
      timers.poppush("grid correction");
      deconv_u2nu(uniform, grid, corfac, fft_order, nthreads);
      timers.poppush("FFT");
      nufft_FFT(false, forward, grid, nuni, nthreads);

      timers.poppush("interpolation");
      (coords.size()==0) ? spreadinterp.interp(grid, points)
                         : spreadinterp.interp(grid, coords, points);

      timers.pop();
      timers.pop();
      }
  };

template<typename Tcalc, typename Tacc, typename Tpoints, typename Tgrid, typename Tcoord>
  void nu2u(const cmav<Tcoord,2> &coord, const cmav<complex<Tpoints>,1> &points,
    bool forward, double epsilon, size_t nthreads,
    const vfmav<complex<Tgrid>> &uniform, size_t verbosity,
    double sigma_min, double sigma_max, const vector<double> &periodicity, bool fft_order)
  {
  auto ndim = uniform.ndim();
  MR_assert((ndim>=1) && (ndim<=3), "transform must be 1D/2D/3D");
  MR_assert(ndim==coord.shape(1), "dimensionality mismatch");
  Nufft<Tcalc, Tacc, Tcoord> nufft(true, points.shape(0), uniform.shape(),
    epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
  nufft.nu2u(forward, verbosity, coord, points, uniform); 
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
  Nufft<Tcalc, Tacc, Tcoord> nufft(false, points.shape(0), uniform.shape(),
    epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
  nufft.u2nu(forward, verbosity, uniform, coord, points); 
  }

template<typename T> auto get_mid_hdelta (const cmav<T,2> &v, size_t nthreads)
  {
  MR_assert(v.shape(0)>0, "at least one entry is required");
  size_t ndim = v.shape(1);
  vector<double> v1(ndim), v2(ndim);
  for (size_t d=0; d<ndim; ++d)
    v1[d] = v2[d] = v(0,d);

  Mutex mut;
  execStatic(v.shape(0), nthreads, 0, [&](auto &sched)
    {
    vector<double> lv1(v1), lv2(v2);
    while (auto rng=sched.getNext()) for (auto i=rng.lo; i<rng.hi; ++i)
      {
      for (size_t d=0; d<ndim; ++d)
        {
        lv1[d] = min(lv1[d], double(v(i,d)));
        lv2[d] = max(lv2[d], double(v(i,d)));
        }
      }
    {
    LockGuard lock(mut);
    for (size_t d=0; d<ndim; ++d)
      {
      v1[d] = min(v1[d], lv1[d]);
      v2[d] = max(v2[d], lv2[d]);
      }
    }
    });

  for (size_t d=0; d<ndim; ++d)
    {
    double mid = 0.5*(v1[d]+v2[d]);
    v2[d] = 0.5*(v2[d]-v1[d]);
    v1[d] = mid;
    }
  return make_tuple(v1,v2);
  }

template<typename Tcalc, typename Tacc, typename Tpoints, typename Tcoord> class Nufft3
  {
  private:
    vmav<complex<Tpoints>,1> fact_in, fact_out;
    vector<size_t> dims;

    size_t kidx;
    size_t nthreads;
    unique_ptr<Spreadinterp2<Tcalc, Tacc, Tcoord, uint32_t>> spreadinterp;
    unique_ptr<Nufft<Tcalc, Tacc, Tcoord>> nufft;

  public:
    Nufft3(const cmav<Tcoord,2> &coord_in, double epsilon, size_t nthreads_,
    const cmav<Tcoord,2> &coord_out, size_t /*verbosity*/,
    double sigma_min, double sigma_max)
      : nthreads(adjust_nthreads(nthreads_))
      {
      auto ndim = coord_in.shape(1);
      MR_assert((ndim>=1) && (ndim<=3), "transform must be 1D/2D/3D");
      MR_assert(ndim==coord_out.shape(1), "dimensionality mismatch");

      auto [mid_in, hdelta_in] = get_mid_hdelta(coord_in, nthreads);
      auto [mid_out, hdelta_out] = get_mid_hdelta(coord_out, nthreads);

      auto [kidx_, dims_, Ssafe] = findNufftParameters_type3<Tcalc,Tacc>
        (epsilon, sigma_min, sigma_max, hdelta_in, hdelta_out, coord_in.shape(0), nthreads);
      kidx = kidx_;
      dims = dims_;

      const auto &krn(getKernel(kidx));
      vector<double> gamma(ndim);
      for (size_t idim=0; idim<ndim; ++idim)
        gamma[idim] = dims[idim]/(2*krn.ofactor*Ssafe[idim]);

      fact_in.assign(vmav<complex<Tpoints>,1>({coord_in.shape(0)}));
      {
      execStatic(coord_in.shape(0), nthreads, 0, [&,mid_in=mid_in,mid_out=mid_out](auto &sched)
        {
        while (auto rng=sched.getNext()) for (auto i=rng.lo; i<rng.hi; ++i)
          {
          double phase=mid_out[0]*coord_in(i,0);
          if (ndim>1) phase += mid_out[1]*coord_in(i,1);
          if (ndim>2) phase += mid_out[2]*coord_in(i,2);
          fact_in(i) = complex<Tpoints>(polar(1., phase));
          }
        });

      vector<double> period_in;
      for (size_t d=0; d<ndim; ++d)
        period_in.push_back(2*pi*gamma[d]);

      spreadinterp = make_unique<Spreadinterp2<Tcalc, Tacc, Tcoord, uint32_t>>
        (coord_in, dims, kidx, nthreads, period_in, mid_in);
      }

      vector<double> period_out;
      for (size_t d=0; d<ndim; ++d)
        period_out.push_back(dims[d]/gamma[d]);

      nufft = make_unique<Nufft<Tcalc, Tacc, Tcoord>>(false, coord_out, dims,
        epsilon, nthreads, sigma_min, sigma_max, period_out, true, mid_out);

      auto krn2 = selectKernel(kidx);
      const auto &corr(krn2->Corr()); 
      fact_out.assign(vmav<complex<Tpoints>,1>({coord_out.shape(0)}));
      execStatic(coord_out.shape(0), nthreads, 0, [&,mid_in=mid_in,mid_out=mid_out](auto &sched)
        {
        while (auto rng=sched.getNext()) for (auto i=rng.lo; i<rng.hi; ++i)
          {
          double phihat=1, phase=0;
          for (size_t d=0; d<ndim; ++d)
            {
            phihat *= corr.template corfunc<Tpoints>(Tpoints((coord_out(i,d)-mid_out[d])*gamma[d]/dims[d]));
            phase += (coord_out(i,d)-mid_out[d])*mid_in[d];
            }
          fact_out(i) = complex<Tpoints>(polar(phihat, phase));
          }
        });
      }

    void exec(const cmav<complex<Tpoints>,1> &points_in,
              const vmav<complex<Tpoints>,1> &points_out,
              bool forward)
      {
      MR_assert(fact_in.shape()==points_in.shape(), "points_in shape mismatch");
      MR_assert(fact_out.shape()==points_out.shape(), "points_out shape mismatch");

      {
      // try to use points_out for temporary points_in_2 storage
      auto points_in_2 = make_unique<vmav<complex<Tpoints>,1>>(
        points_in.shape(0)<=points_out.shape(0) ?
        subarray<1>(points_out, {{0,points_in.shape(0)}}) :
        vmav<complex<Tpoints>,1>(points_in.shape()));

      execStatic(points_in.shape(0), nthreads, 0, [&](auto &sched)
        {
        while (auto rng=sched.getNext()) for (auto i=rng.lo; i<rng.hi; ++i)
          (*points_in_2)(i) = points_in(i) * (forward ? conj(fact_in(i)) : fact_in(i));
        });
      auto grid = vfmav<complex<Tcalc>>::build_noncritical(dims);
      spreadinterp->spread(*points_in_2, grid);
      points_in_2.reset();
      nufft->u2nu(forward, 0, grid, points_out);
      }

      execStatic(points_out.shape(0), nthreads, 0, [&](auto &sched)
        {
        while (auto rng=sched.getNext()) for (auto i=rng.lo; i<rng.hi; ++i)
          points_out(i) *= forward ? conj(fact_out(i)) : fact_out(i);
        });
      }
    void exec_adjoint(const cmav<complex<Tpoints>,1> &points_in,
                      const vmav<complex<Tpoints>,1> &points_out,
                      bool forward)
      {
      MR_assert(fact_out.shape()==points_in.shape(), "points_in shape mismatch");
      MR_assert(fact_in.shape()==points_out.shape(), "points_out shape mismatch");

      {
      // try to use points_out for temporary points_in_2 storage
      auto points_in_2 = make_unique<vmav<complex<Tpoints>,1>>(
        points_in.shape(0)<=points_out.shape(0) ?
        subarray<1>(points_out, {{0,points_in.shape(0)}}) :
        vmav<complex<Tpoints>,1>(points_in.shape()));

      execStatic(points_in.shape(0), nthreads, 0, [&](auto &sched)
        {
        while (auto rng=sched.getNext()) for (auto i=rng.lo; i<rng.hi; ++i)
          (*points_in_2)(i) = points_in(i) * (forward ? fact_out(i) : conj(fact_out(i)));
        });
      auto grid = vfmav<complex<Tcalc>>::build_noncritical(dims);
      nufft->nu2u(!forward, 0, *points_in_2, grid); 
      points_in_2.reset();
      spreadinterp->interp(grid, points_out);
      }

      execStatic(points_out.shape(0), nthreads, 0, [&](auto &sched)
        {
        while (auto rng=sched.getNext()) for (auto i=rng.lo; i<rng.hi; ++i)
          points_out(i) *= forward ? fact_in(i) : conj(fact_in(i));
        });
      }
  };

template<typename Tcalc, typename Tacc, typename Tpoints, typename Tcoord>
  void nu2nu(const cmav<Tcoord,2> &coord_in, const cmav<complex<Tpoints>,1> &points_in,
    bool forward, double epsilon, size_t nthreads,
    const cmav<Tcoord,2> &coord_out, const vmav<complex<Tpoints>,1> &points_out, size_t verbosity,
    double sigma_min, double sigma_max)
  {
  TimerHierarchy timers("nu2nu");
  nthreads = adjust_nthreads(nthreads);

  auto ndim = coord_in.shape(1);
  MR_assert((ndim>=1) && (ndim<=3), "transform must be 1D/2D/3D");
  MR_assert(ndim==coord_out.shape(1), "dimensionality mismatch");
  MR_assert(coord_in.shape(0)==points_in.shape(0), "points_in shape mismatch");
  MR_assert(coord_out.shape(0)==points_out.shape(0), "points_out shape mismatch");

  timers.push("coord min/max");
  auto [mid_in, hdelta_in] = get_mid_hdelta(coord_in, nthreads);
  auto [mid_out, hdelta_out] = get_mid_hdelta(coord_out, nthreads);

  timers.poppush("get spreading parameters");
  auto [kidx, dims, Ssafe] = findNufftParameters_type3<Tcalc,Tacc>
    (epsilon, sigma_min, sigma_max, hdelta_in, hdelta_out, points_in.shape(0), nthreads);

  //if (verbosity>0)
    //{
    //cout << "nu2nu: grid(";
    //for (size_t i=0; i+1<ndim; ++i) cout <<dims[i]<<"x";
    //cout << dims.back() << ")" << endl;
    //}

  const auto &krn(getKernel(kidx));
  auto krn2 = selectKernel(kidx);
  const auto &corr(krn2->Corr()); 

  Tpoints psign = forward ? -1 : 1;

  vector<double> gamma(ndim);
  for (size_t idim=0; idim<ndim; ++idim)
    gamma[idim] = dims[idim]/(2*krn.ofactor*Ssafe[idim]);

  timers.poppush("input rescaling & pre-pasing");
  // try to use points_out for temporary points_in_2 storage
  auto points_in_2 = make_unique<vmav<complex<Tpoints>,1>>(
    points_in.shape(0)<=points_out.shape(0) ?
    subarray<1>(points_out, {{0,points_in.shape(0)}}) :
    vmav<complex<Tpoints>,1>(points_in.shape()));
  // prephase input values
  execStatic(points_in.shape(0), nthreads, 0, [&,mid_in=mid_in,mid_out=mid_out](auto &sched)
    {
    while (auto rng=sched.getNext()) for (auto i=rng.lo; i<rng.hi; ++i)
      {
      double phase=0;
      for (size_t d=0; d<ndim; ++d)
        phase += mid_out[d]*coord_in(i,d);
      (*points_in_2)(i) = points_in(i)*complex<Tpoints>(polar(1., psign*phase));
      }
    });

  {
  timers.poppush("spreading");
  vector<double> period_in;
  for (size_t d=0; d<ndim; ++d)
    period_in.push_back(2*pi*gamma[d]);
  Spreadinterp2<Tcalc, Tacc, Tcoord, uint32_t> spreadinterp
    (coord_in.shape(0), dims, kidx, nthreads, period_in, mid_in);
  auto grid = vfmav<complex<Tcalc>>::build_noncritical(dims);
  spreadinterp.spread(coord_in, *points_in_2, grid);
  points_in_2.reset();

  timers.poppush("u2nu");
  vector<double> period_out;
  for (size_t d=0; d<ndim; ++d)
    period_out.push_back(dims[d]/gamma[d]);
  Nufft<Tcalc, Tacc, Tcoord> nufft(false, points_out.shape(0), dims,
    epsilon, nthreads, sigma_min, sigma_max, period_out, true, mid_out);
  nufft.u2nu(forward, 0, grid, coord_out, points_out); 
  }

  timers.poppush("output post-phasing and deconvolution");
  execStatic(points_out.shape(0), nthreads, 0, [&,mid_in=mid_in,mid_out=mid_out,dims=dims](auto &sched)
    {
    while (auto rng=sched.getNext()) for (auto i=rng.lo; i<rng.hi; ++i)
      {
      double phihat=1, phase=0;
      for (size_t d=0; d<ndim; ++d)
        {
        phihat *= corr.template corfunc<Tpoints>(Tpoints((coord_out(i,d)-mid_out[d])*gamma[d]/dims[d]));
        phase += (coord_out(i,d)-mid_out[d])*mid_in[d];
        }
      points_out(i) *= complex<Tpoints>(polar(phihat, psign*phase));
      }
    });

  timers.pop();
  if (verbosity>0) timers.report(cout);
  }

} // namespace detail_nufft

// public names
using detail_nufft::findNufftKernel;
using detail_nufft::u2nu;
using detail_nufft::nu2u;
using detail_nufft::nu2nu;
using detail_nufft::Nufft;
using detail_nufft::Nufft3;

} // namespace ducc0

#endif
