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

template<typename Tcalc, typename Tacc, size_t ndim> class Nufft_ancestor
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
    array<size_t, ndim> nuni;

    // oversampled grid dimensions
    array<size_t, ndim> nover;

size_t krn_id;
    shared_ptr<PolynomialKernel> krn;

    size_t supp;

    vector<vector<double>> corfac;

    static_assert(sizeof(Tcalc)<=sizeof(Tacc),
      "Tacc must be at least as accurate as Tcalc");

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

  public:
    Nufft_ancestor(bool gridding, size_t npoints_,
      const array<size_t,ndim> &uniform_shape, double epsilon_,
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
      for (size_t i=0; i<ndim; ++i)
        nover[i] = dims[i];
      timers.pop();

      krn = selectKernel(kidx);
      krn_id = kidx;
      supp = krn->support();

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


template<typename Tcalc, typename Tacc, typename Tcoord, size_t ndim> class Nufft:
  public Nufft_ancestor<Tcalc, Tacc, ndim>
  {
  private:
    using parent=Nufft_ancestor<Tcalc, Tacc, ndim>;
    using parent::nthreads,
          parent::timers, parent::krn_id, parent::fft_order, parent::nuni,
          parent::nover, parent::report,
          parent::corfac,
          parent::prep_nu2u, parent::prep_u2nu;

    Spreadinterp<Tcalc, Tacc, Tcoord, uint32_t, ndim> spreadinterp;

  public:
    using parent::parent; /* inherit constructor */
    Nufft(bool gridding, const cmav<Tcoord,2> &coords,
          const array<size_t, ndim> &uniform_shape_, double epsilon_, 
          size_t nthreads_, double sigma_min, double sigma_max,
          const vector<double> &periodicity, bool fft_order_)
      : parent(gridding, coords.shape(0), uniform_shape_, epsilon_, nthreads_,
               sigma_min, sigma_max, fft_order_),
        spreadinterp(coords, nover, krn_id, nthreads, periodicity)
      {}
    Nufft (bool gridding, size_t npoints_,
      const array<size_t,ndim> &uniform_shape, double epsilon_,
      size_t nthreads_, double sigma_min, double sigma_max,
      const vector<double> &periodicity, bool fft_order_)
      : parent(gridding, npoints_, uniform_shape, epsilon_, nthreads_,
               sigma_min, sigma_max, fft_order_),
        spreadinterp(npoints_, nover, krn_id, nthreads, periodicity)
      {}

    template<typename Tpoints, typename Tgrid> void nu2u(bool forward, size_t verbosity,
      const cmav<complex<Tpoints>,1> &points, const vmav<complex<Tgrid>,ndim> &uniform)
      {
      if (prep_nu2u(points, uniform)) return;
      if (verbosity>0) report(true);
      auto dummy = cmav<Tcoord,2>::build_empty();
      nonuni2uni(forward, dummy, points, uniform);
      if (verbosity>0) timers.report(cout);
      }
    template<typename Tpoints, typename Tgrid> void u2nu(bool forward, size_t verbosity,
      const cmav<complex<Tgrid>,ndim> &uniform, const vmav<complex<Tpoints>,1> &points)
      {
      if (prep_u2nu(points, uniform)) return;
      if (verbosity>0) report(false);
      auto dummy = cmav<Tcoord,2>::build_empty();
      uni2nonuni(forward, uniform, dummy, points);
      if (verbosity>0) timers.report(cout);
      }
    template<typename Tpoints, typename Tgrid> void nu2u(bool forward, size_t verbosity,
      const cmav<Tcoord,2> &coords, const cmav<complex<Tpoints>,1> &points,
      const vmav<complex<Tgrid>,ndim> &uniform)
      {
      if (prep_nu2u(points, uniform)) return;
      if (verbosity>0) report(true);
      nonuni2uni(forward, coords, points, uniform);
      if (verbosity>0) timers.report(cout);
      }
    template<typename Tpoints, typename Tgrid> void u2nu(bool forward, size_t verbosity,
      const cmav<complex<Tgrid>,ndim> &uniform, const cmav<Tcoord,2> &coords,
      const vmav<complex<Tpoints>,1> &points)
      {
      if (prep_u2nu(points, uniform)) return;
      if (verbosity>0) report(false);
      uni2nonuni(forward, uniform, coords, points);
      if (verbosity>0) timers.report(cout);
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
      const vmav<complex<Tgrid>,ndim> &uniform)
      {
      timers.push("nu2u proper");
      timers.push("allocating grid");
      auto grid = vmav<complex<Tcalc>,ndim>::build_noncritical(nover, UNINITIALIZED);
      timers.poppush("zeroing grid");
      mav_apply([](complex<Tcalc> &v){v=complex<Tcalc>(0);},nthreads,grid);
      timers.poppush("spreading");
      (coords.size()==0) ? spreadinterp.spread(points, grid)
                         : spreadinterp.spread(coords, points, grid);

      timers.poppush("FFT");
      nufft_FFT(true, forward, grid.to_fmav(), vector<size_t>(&nuni[0], &nuni[ndim]), nthreads);

      timers.poppush("grid correction");
      deconv_nu2u(grid.to_fmav(), uniform.to_fmav(), corfac, fft_order, nthreads);
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
      deconv_u2nu(uniform.to_fmav(), grid.to_fmav(), corfac, fft_order, nthreads);
      timers.poppush("FFT");
      nufft_FFT(false, forward, grid.to_fmav(), vector<size_t>(&nuni[0], &nuni[ndim]), nthreads);

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
