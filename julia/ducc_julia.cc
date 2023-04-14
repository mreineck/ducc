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

/* Copyright (C) 2022-2023 Max-Planck-Society, Leo A. Bianchi
   Authors: Martin Reinecke, Leo A. Bianchi */

/*
Compilation: (the -I path must point to the src/ directory in the ducc0 checkout)

g++ -O3 -march=native -ffast-math -I ../src/ ducc_julia.cc -Wfatal-errors -pthread -std=c++17 -fPIC -c -fvisibility=hidden

Creating the shared library:

g++ -O3 -march=native -o ducc_julia.so ducc_julia.o -Wfatal-errors -pthread -std=c++17 -shared -fPIC

CONVENTIONS USED FOR THIS WRAPPER:
 - passed ArrayDescriptors are in Julia order, i.e. all axis-swapping etc. will
   take place on the C++ side, if necessary
 - if axis indices or array indices are passed, they are assumed to be one-based
*/

#include "ducc0/infra/threading.cc"
#include "ducc0/infra/mav.cc"
#include "ducc0/math/gl_integrator.cc"
#include "ducc0/math/gridding_kernel.cc"
#include "ducc0/nufft/nufft.h"
#include "ducc0/bindings/typecode.h"
#include "ducc0/bindings/array_descriptor.h"
#include "ducc0/sht/sht.cc"

using namespace ducc0;
using namespace std;

template<typename T> cmav<T,2> get_coord(const ArrayDescriptor &desc)
  {
  auto res(to_cmav<true,T,2>(desc));
  // flip coord axis!
  return cmav<T,2>(res.data()+(res.shape(1)-1)*res.stride(1),
    res.shape(), {res.stride(0), -res.stride(1)});
  }

#define DUCC0_JULIA_TRY_BEGIN try{
#define DUCC0_JULIA_TRY_END } catch(const exception &e) { cout << e.what() << endl; return 1; } return 0;

#if defined _WIN32 || defined __CYGWIN__
#define DUCC0_INTERFACE_FUNCTION extern "C" [[gnu::dllexport]]
#else
#define DUCC0_INTERFACE_FUNCTION extern "C" [[gnu::visibility("default")]]
#endif

// FFT

DUCC0_INTERFACE_FUNCTION
int fft_c2c(const ArrayDescriptor *in_, ArrayDescriptor *out_,
  const ArrayDescriptor *axes_, int forward, double fct, size_t nthreads)
  {
  DUCC0_JULIA_TRY_BEGIN
  const auto &in(*in_);
  auto &out(*out_);
  const auto &axes(*axes_);
  auto myaxes(to_vector_subtract_1<false, uint64_t, size_t>(axes));
  for (auto &a: myaxes) a = in.ndim-1-a;
  if (in.dtype==Typecode<complex<double>>::value)
    {
    auto myin(to_cfmav<true,complex<double>>(in));
    auto myout(to_vfmav<true,complex<double>>(out));
    c2c(myin, myout, myaxes, forward, fct, nthreads);
    }
  else if (in.dtype==Typecode<complex<float>>::value)
    {
    auto myin(to_cfmav<true,complex<float>>(in));
    auto myout(to_vfmav<true,complex<float>>(out));
    c2c(myin, myout, myaxes, forward, float(fct), nthreads);
    }
  else
    MR_fail("bad datatype");
  DUCC0_JULIA_TRY_END
  }
DUCC0_INTERFACE_FUNCTION
int fft_r2c(const ArrayDescriptor *in_, ArrayDescriptor *out_,
  const ArrayDescriptor *axes_, int forward, double fct, size_t nthreads)
  {
  DUCC0_JULIA_TRY_BEGIN
  const auto &in(*in_);
  auto &out(*out_);
  const auto &axes(*axes_);
  auto myaxes(to_vector_subtract_1<false, uint64_t, size_t>(axes));
  for (auto &a: myaxes) a = in.ndim-1-a;
  if (in.dtype==Typecode<double>::value)
    {
    auto myin(to_cfmav<true,double>(in));
    auto myout(to_vfmav<true,complex<double>>(out));
    r2c(myin, myout, myaxes, forward, fct, nthreads);
    }
  else if (in.dtype==Typecode<float>::value)
    {
    auto myin(to_cfmav<true,float>(in));
    auto myout(to_vfmav<true,complex<float>>(out));
    r2c(myin, myout, myaxes, forward, float(fct), nthreads);
    }
  else
    MR_fail("bad datatype");
  DUCC0_JULIA_TRY_END
  }
DUCC0_INTERFACE_FUNCTION
int fft_c2r(const ArrayDescriptor *in_, ArrayDescriptor *out_,
  const ArrayDescriptor *axes_, int forward, double fct, size_t nthreads)
  {
  DUCC0_JULIA_TRY_BEGIN
  const auto &in(*in_);
  auto &out(*out_);
  const auto &axes(*axes_);
  auto myaxes(to_vector_subtract_1<false, uint64_t, size_t>(axes));
  for (auto &a: myaxes) a = in.ndim-1-a;
  if (in.dtype==Typecode<complex<double>>::value)
    {
    auto myin(to_cfmav<true,complex<double>>(in));
    auto myout(to_vfmav<true,double>(out));
    c2r(myin, myout, myaxes, forward, fct, nthreads);
    }
  else if (in.dtype==Typecode<complex<float>>::value)
    {
    auto myin(to_cfmav<true,complex<float>>(in));
    auto myout(to_vfmav<true,float>(out));
    c2r(myin, myout, myaxes, forward, float(fct), nthreads);
    }
  else
    MR_fail("bad datatype");
  DUCC0_JULIA_TRY_END
  }
DUCC0_INTERFACE_FUNCTION
int fft_r2r_genuine_fht(const ArrayDescriptor *in_, ArrayDescriptor *out_,
  const ArrayDescriptor *axes_, double fct, size_t nthreads)
  {
  DUCC0_JULIA_TRY_BEGIN
  const auto &in(*in_);
  auto &out(*out_);
  const auto &axes(*axes_);
  auto myaxes(to_vector_subtract_1<false, uint64_t, size_t>(axes));
  for (auto &a: myaxes) a = in.ndim-1-a;
  if (in.dtype==Typecode<double>::value)
    {
    auto myin(to_cfmav<true,double>(in));
    auto myout(to_vfmav<true,double>(out));
    r2r_genuine_fht(myin, myout, myaxes, fct, nthreads);
    }
  else if (in.dtype==Typecode<float>::value)
    {
    auto myin(to_cfmav<true,float>(in));
    auto myout(to_vfmav<true,float>(out));
    r2r_genuine_fht(myin, myout, myaxes, float(fct), nthreads);
    }
  else
    MR_fail("bad datatype");
  DUCC0_JULIA_TRY_END
  }

// NUFFT

DUCC0_INTERFACE_FUNCTION
double nufft_best_epsilon(size_t ndim, int singleprec,
  double ofactor_min, double ofactor_max)
  {
  try
    {
    return bestEpsilon(ndim, bool(singleprec), ofactor_min, ofactor_max);
    }
  catch(const exception &e)
    { cout << e.what() << endl; return -1.; }
  }

DUCC0_INTERFACE_FUNCTION
int nufft_u2nu(const ArrayDescriptor *grid_,
                     const ArrayDescriptor *coord_,
                     int forward,
                     double epsilon,
                     size_t nthreads,
                     ArrayDescriptor *out_,
                     size_t verbosity,
                     double sigma_min,
                     double sigma_max,
                     double periodicity,
                     int fft_order)
  {
  DUCC0_JULIA_TRY_BEGIN
  const auto &grid(*grid_);
  const auto &coord(*coord_);
  auto &out(*out_);
  if (coord.dtype==Typecode<double>::value)
    {
    auto mycoord = get_coord<double>(coord);
    if (grid.dtype==Typecode<complex<double>>::value)
      {
      auto mygrid(to_cfmav<true,complex<double>>(grid));
      auto myout(to_vmav<true,complex<double>,1>(out));
      MR_assert(mycoord.shape(0)==myout.shape(0), "npoints mismatch");
      MR_assert(mycoord.shape(1)==mygrid.ndim(), "dimensionality mismatch");
      u2nu<double,double>(mycoord,mygrid,forward,epsilon,nthreads,myout,
        verbosity,sigma_min,sigma_max,periodicity,fft_order);
      }
    else if (grid.dtype==Typecode<complex<float>>::value)
      {
      auto mygrid(to_cfmav<true,complex<float>>(grid));
      auto myout(to_vmav<true,complex<float>,1>(out));
      MR_assert(mycoord.shape(0)==myout.shape(0), "npoints mismatch");
      MR_assert(mycoord.shape(1)==mygrid.ndim(), "dimensionality mismatch");
      u2nu<float,float>(mycoord,mygrid,forward,epsilon,nthreads,myout,
        verbosity,sigma_min,sigma_max,periodicity,fft_order);
      }
    else
      MR_fail("bad datatype");
    }
  else if (coord.dtype==Typecode<float>::value)
    {
    auto mycoord = get_coord<float>(coord);
    if (grid.dtype==Typecode<complex<float>>::value)
      {
      auto mygrid(to_cfmav<true,complex<float>>(grid));
      auto myout(to_vmav<true,complex<float>,1>(out));
      MR_assert(mycoord.shape(0)==myout.shape(0), "npoints mismatch");
      MR_assert(mycoord.shape(1)==mygrid.ndim(), "dimensionality mismatch");
      u2nu<float,float>(mycoord,mygrid,forward,epsilon,nthreads,myout,
        verbosity,sigma_min,sigma_max,periodicity,fft_order);
      }
    else
      MR_fail("bad datatype");
    }
  DUCC0_JULIA_TRY_END
  }

DUCC0_INTERFACE_FUNCTION
int nufft_nu2u(const ArrayDescriptor *points_,
                       const ArrayDescriptor *coord_,
                       int forward,
                       double epsilon,
                       size_t nthreads,
                       ArrayDescriptor *out_,
                       size_t verbosity,
                       double sigma_min,
                       double sigma_max,
                       double periodicity,
                       int fft_order)
  {
  DUCC0_JULIA_TRY_BEGIN
  const auto &points(*points_);
  const auto &coord(*coord_);
  auto &out(*out_);
  if (coord.dtype==Typecode<double>::value)
    {
    auto mycoord = get_coord<double>(coord);
    if (points.dtype==Typecode<complex<double>>::value)
      {
      auto mypoints(to_cmav<true,complex<double>,1>(points));
      auto myout(to_vfmav<true,complex<double>>(out));
      MR_assert(mycoord.shape(0)==mypoints.shape(0), "npoints mismatch");
      MR_assert(mycoord.shape(1)==myout.ndim(), "dimensionality mismatch");
      nu2u<double,double>(mycoord,mypoints,forward,epsilon,nthreads,myout,
        verbosity,sigma_min,sigma_max,periodicity,fft_order);
      }
    else if (points.dtype==Typecode<complex<float>>::value)
      {
      auto mypoints(to_cmav<true,complex<float>,1>(points));
      auto myout(to_vfmav<true,complex<float>>(out));
      MR_assert(mycoord.shape(0)==mypoints.shape(0), "npoints mismatch");
      MR_assert(mycoord.shape(1)==myout.ndim(), "dimensionality mismatch");
      nu2u<float,float>(mycoord,mypoints,forward,epsilon,nthreads,myout,
        verbosity,sigma_min,sigma_max,periodicity,fft_order);
      }
    else
      MR_fail("bad datatype");
    }
  else if (coord.dtype==Typecode<float>::value)
    {
    auto mycoord = get_coord<float>(coord);
    if (points.dtype==Typecode<complex<float>>::value)
      {
      auto mypoints(to_cmav<true,complex<float>,1>(points));
      auto myout(to_vfmav<true,complex<float>>(out));
      MR_assert(mycoord.shape(0)==mypoints.shape(0), "npoints mismatch");
      MR_assert(mycoord.shape(1)==myout.ndim(), "dimensionality mismatch");
      nu2u<float,float>(mycoord,mypoints,forward,epsilon,nthreads,myout,
        verbosity,sigma_min,sigma_max,periodicity,fft_order);
      }
    else
      MR_fail("bad datatype");
    }
  DUCC0_JULIA_TRY_END
  }

struct Tplan
  {
  size_t npoints;
  vector<size_t> shp;
  size_t coord_type;
  void *plan;
  };

DUCC0_INTERFACE_FUNCTION
Tplan *nufft_make_plan(int nu2u,
                             const ArrayDescriptor *shape_,
                             const ArrayDescriptor *coord_,
                             double epsilon,
                             size_t nthreads,
                             double sigma_min,
                             double sigma_max,
                             double periodicity,
                             int fft_order)
  {
  try
    {
    const auto &shape(*shape_);
    const auto &coord(*coord_);
    auto myshape = to_vector<true, uint64_t, size_t>(shape);
    auto ndim = myshape.size();
    MR_assert(coord.ndim==2, "bad coordinate dimensionality");
    MR_assert(coord.shape[0]==ndim, "dimensionality mismatch");
    auto res = new Tplan{coord.shape[1],myshape,coord.dtype,nullptr};
    if (coord.dtype==Typecode<double>::value)
      {
      auto mycoord = get_coord<double>(coord);
      if (ndim==1)
        res->plan = new Nufft<double, double, double, 1>(nu2u, mycoord, array<size_t,1>{myshape[0]},
          epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
      else if (ndim==2)
        res->plan = new Nufft<double, double, double, 2>(nu2u, mycoord, array<size_t,2>{myshape[0],myshape[1]},
          epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
      else if (ndim==3)
        res->plan = new Nufft<double, double, double, 3>(nu2u, mycoord, array<size_t,3>{myshape[0],myshape[1],myshape[2]},
          epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
      else
        MR_fail("bad number of dimensions");
      return res;
      }
    else if (coord.dtype==Typecode<float>::value)
      {
      auto mycoord = get_coord<float>(coord);
      if (ndim==1)
        res->plan = new Nufft<float, float, float, 1>(nu2u, mycoord, array<size_t,1>{myshape[0]},
          epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
      else if (ndim==2)
        res->plan = new Nufft<float, float, float, 2>(nu2u, mycoord, array<size_t,2>{myshape[0],myshape[1]},
          epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
      else if (ndim==3)
        res->plan = new Nufft<float, float, float, 3>(nu2u, mycoord, array<size_t,3>{myshape[0],myshape[1],myshape[2]},
          epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
      else
        MR_fail("bad number of dimensions");
      return res;
      }
    MR_fail("bad coordinate data type");
    }
  catch(const exception &e)
    { cout << e.what() << endl; return nullptr; }
  }

DUCC0_INTERFACE_FUNCTION
int nufft_delete_plan(Tplan *plan)
  {
  DUCC0_JULIA_TRY_BEGIN
  if (plan->shp.size()==1)
    (plan->coord_type==Typecode<double>::value) ?
      delete reinterpret_cast<Nufft<double, double, double, 1> *>(plan->plan)
    : delete reinterpret_cast<Nufft<float,float,float, 1> *>(plan->plan);
  else if (plan->shp.size()==2)
    (plan->coord_type==Typecode<double>::value) ?
      delete reinterpret_cast<Nufft<double, double, double, 2> *>(plan->plan)
    : delete reinterpret_cast<Nufft<float,float,float, 2> *>(plan->plan);
  else if (plan->shp.size()==3)
    (plan->coord_type==Typecode<double>::value) ?
      delete reinterpret_cast<Nufft<double, double, double, 3> *>(plan->plan)
    : delete reinterpret_cast<Nufft<float,float,float, 3> *>(plan->plan);
  else
    MR_fail("bad number of dimensions");
  delete plan;
  DUCC0_JULIA_TRY_END
  }

DUCC0_INTERFACE_FUNCTION
int nufft_nu2u_planned(Tplan *plan, int forward, size_t verbosity,
  const ArrayDescriptor *points_, ArrayDescriptor *uniform_)
  {
  DUCC0_JULIA_TRY_BEGIN
  const auto &points(*points_);
  auto &uniform(*uniform_);
  MR_assert(uniform.ndim==plan->shp.size(), "dimensionality mismatch");
  for (size_t i=0; i<uniform.ndim; ++i)
    MR_assert(uniform.shape[i]==plan->shp[uniform.ndim-1-i], "array dimension mismatch");
  if (points.dtype==Typecode<complex<double>>::value)
    {
    MR_assert(plan->coord_type==Typecode<double>::value, "data type mismatch");
    auto mypoints(to_cmav<true,complex<double>,1>(points));
    if (plan->shp.size()==1)
      {
      auto myout(to_vmav<true,complex<double>,1>(uniform));
      auto rplan = reinterpret_cast<Nufft<double, double, double, 1> *>(plan->plan);
      rplan->nu2u(forward, verbosity, mypoints, myout);
      }
    else if (plan->shp.size()==2)
      {
      auto myout(to_vmav<true,complex<double>,2>(uniform));
      auto rplan = reinterpret_cast<Nufft<double, double, double, 2> *>(plan->plan);
      rplan->nu2u(forward, verbosity, mypoints, myout);
      }
    else if (plan->shp.size()==3)
      {
      auto myout(to_vmav<true,complex<double>,3>(uniform));
      auto rplan = reinterpret_cast<Nufft<double, double, double, 3> *>(plan->plan);
      rplan->nu2u(forward, verbosity, mypoints, myout);
      }
    else
      MR_fail("bad number of dimensions");
    }
  else if (points.dtype==Typecode<complex<float>>::value)
    {
    MR_assert(plan->coord_type==Typecode<float>::value, "data type mismatch");
    auto mypoints(to_cmav<true,complex<float>,1>(points));
    if (plan->shp.size()==1)
      {
      auto myout(to_vmav<true,complex<float>,1>(uniform));
      auto rplan = reinterpret_cast<Nufft<float, float, float, 1> *>(plan->plan);
      rplan->nu2u(forward, verbosity, mypoints, myout);
      }
    else if (plan->shp.size()==2)
      {
      auto myout(to_vmav<true,complex<float>,2>(uniform));
      auto rplan = reinterpret_cast<Nufft<float, float, float, 2> *>(plan->plan);
      rplan->nu2u(forward, verbosity, mypoints, myout);
      }
    else if (plan->shp.size()==3)
      {
      auto myout(to_vmav<true,complex<float>,3>(uniform));
      auto rplan = reinterpret_cast<Nufft<float, float, float, 3> *>(plan->plan);
      rplan->nu2u(forward, verbosity, mypoints, myout);
      }
    else
      MR_fail("bad number of dimensions");
    }
  else
    MR_fail("unsupported data type");
  DUCC0_JULIA_TRY_END
  }

DUCC0_INTERFACE_FUNCTION
int nufft_u2nu_planned(Tplan *plan, int forward, size_t verbosity,
  const ArrayDescriptor *uniform_, ArrayDescriptor *points_)
  {
  DUCC0_JULIA_TRY_BEGIN
  const auto &uniform(*uniform_);
  auto &points(*points_);
  MR_assert(uniform.ndim==plan->shp.size(), "dimensionality mismatch");
  for (size_t i=0; i<uniform.ndim; ++i)
    MR_assert(uniform.shape[i]==plan->shp[uniform.ndim-1-i], "array dimension mismatch");
  if (points.dtype==Typecode<complex<double>>::value)
    {
    MR_assert(plan->coord_type==Typecode<double>::value, "data type mismatch");
    auto mypoints(to_vmav<true,complex<double>,1>(points));
    if (plan->shp.size()==1)
      {
      auto myuniform(to_cmav<true,complex<double>,1>(uniform));
      auto rplan = reinterpret_cast<Nufft<double, double, double, 1> *>(plan->plan);
      rplan->u2nu(forward, verbosity, myuniform, mypoints);
      }
    else if (plan->shp.size()==2)
      {
      auto myuniform(to_cmav<true,complex<double>,2>(uniform));
      auto rplan = reinterpret_cast<Nufft<double, double, double, 2> *>(plan->plan);
      rplan->u2nu(forward, verbosity, myuniform, mypoints);
      }
    else if (plan->shp.size()==3)
      {
      auto myuniform(to_cmav<true,complex<double>,3>(uniform));
      auto rplan = reinterpret_cast<Nufft<double, double, double, 3> *>(plan->plan);
      rplan->u2nu(forward, verbosity, myuniform, mypoints);
      }
    else
      MR_fail("bad number of dimensions");
    }
  else if (points.dtype==Typecode<complex<float>>::value)
    {
    MR_assert(plan->coord_type==Typecode<float>::value, "data type mismatch");
    auto mypoints(to_vmav<true,complex<float>,1>(points));
    if (plan->shp.size()==1)
      {
      auto myuniform(to_cmav<true,complex<float>,1>(uniform));
      auto rplan = reinterpret_cast<Nufft<float, float, float, 1> *>(plan->plan);
      rplan->u2nu(forward, verbosity, myuniform, mypoints);
      }
    else if (plan->shp.size()==2)
      {
      auto myuniform(to_cmav<true,complex<float>,2>(uniform));
      auto rplan = reinterpret_cast<Nufft<float, float, float, 2> *>(plan->plan);
      rplan->u2nu(forward, verbosity, myuniform, mypoints);
      }
    else if (plan->shp.size()==3)
      {
      auto myuniform(to_cmav<true,complex<float>,3>(uniform));
      auto rplan = reinterpret_cast<Nufft<float, float, float, 3> *>(plan->plan);
      rplan->u2nu(forward, verbosity, myuniform, mypoints);
      }
    else
      MR_fail("bad number of dimensions");
    }
  else
    MR_fail("unsupported data type");
  DUCC0_JULIA_TRY_END
  }

DUCC0_INTERFACE_FUNCTION
int sht_alm2leg(const ArrayDescriptor *alm_, size_t spin,
  size_t lmax, const ArrayDescriptor *mval_, const ArrayDescriptor *mstart_,
  ptrdiff_t lstride, const ArrayDescriptor *theta_, size_t nthreads,
  ArrayDescriptor *leg_)
  {
  DUCC0_JULIA_TRY_BEGIN
  auto mval(to_cmav<true,size_t,1>(*mval_));
  auto mstart(subtract_1(to_cmav_with_typecast<true,ptrdiff_t,size_t,1>(*mstart_)));
  auto theta(to_cmav<true,double,1>(*theta_));
  if (alm_->dtype==Typecode<complex<double>>::value)
    {
    auto alm(to_cmav<true,complex<double>,2>(*alm_));
    auto leg(to_vmav<true,complex<double>,3>(*leg_));
    alm2leg(alm, leg, spin, lmax, mval, mstart, lstride, theta, nthreads, ALM2MAP);
    }
  else if (alm_->dtype==Typecode<complex<float>>::value)
    {
    auto alm(to_cmav<true,complex<float>,2>(*alm_));
    auto leg(to_vmav<true,complex<float>,3>(*leg_));
    alm2leg(alm, leg, spin, lmax, mval, mstart, lstride, theta, nthreads, ALM2MAP);
    }
  else
    MR_fail("unsupported data type");
  DUCC0_JULIA_TRY_END
  }

DUCC0_INTERFACE_FUNCTION
int sht_leg2alm(const ArrayDescriptor *leg_, size_t spin,
  size_t lmax, const ArrayDescriptor *mval_, const ArrayDescriptor *mstart_,
  ptrdiff_t lstride, const ArrayDescriptor *theta_, size_t nthreads,
  ArrayDescriptor *alm_)
  {
  DUCC0_JULIA_TRY_BEGIN
  auto mval(to_cmav<true,size_t,1>(*mval_));
  auto mstart(subtract_1(to_cmav_with_typecast<true,ptrdiff_t,size_t,1>(*mstart_)));
  auto theta(to_cmav<true,double,1>(*theta_));
  if (leg_->dtype==Typecode<complex<double>>::value)
    {
    auto leg(to_cmav<true,complex<double>,3>(*leg_));
    auto alm(to_vmav<true,complex<double>,2>(*alm_));
    leg2alm(alm, leg, spin, lmax, mval, mstart, lstride, theta, nthreads);
    }
  else if (leg_->dtype==Typecode<complex<float>>::value)
    {
    auto leg(to_cmav<true,complex<float>,3>(*leg_));
    auto alm(to_vmav<true,complex<float>,2>(*alm_));
    leg2alm(alm, leg, spin, lmax, mval, mstart, lstride, theta, nthreads);
    }
  else
    MR_fail("unsupported data type");
  DUCC0_JULIA_TRY_END
  }

DUCC0_INTERFACE_FUNCTION
int sht_leg2map(const ArrayDescriptor *leg_,
  const ArrayDescriptor *nphi_, const ArrayDescriptor *phi0_,
  const ArrayDescriptor *ringstart_,
  ptrdiff_t pixstride, size_t nthreads, ArrayDescriptor *map_)
  {
  DUCC0_JULIA_TRY_BEGIN
  auto nphi(to_cmav<true,size_t,1>(*nphi_));
  auto phi0(to_cmav<true,double,1>(*phi0_));
  auto ringstart(subtract_1(to_cmav<true,size_t,1>(*ringstart_)));
  if (leg_->dtype==Typecode<complex<double>>::value)
    {
    auto leg(to_cmav<true,complex<double>,3>(*leg_));
    auto map(to_vmav<true,double,2>(*map_));
    leg2map(map, leg, nphi, phi0, ringstart, pixstride, nthreads);
    }
  else if (leg_->dtype==Typecode<complex<float>>::value)
    {
    auto leg(to_cmav<true,complex<float>,3>(*leg_));
    auto map(to_vmav<true,float,2>(*map_));
    leg2map(map, leg, nphi, phi0, ringstart, pixstride, nthreads);
    }
  else
    MR_fail("unsupported data type");
  DUCC0_JULIA_TRY_END
  }

DUCC0_INTERFACE_FUNCTION
  int sht_map2leg(const ArrayDescriptor *map_,
    const ArrayDescriptor *nphi_, const ArrayDescriptor *phi0_,
    const ArrayDescriptor *ringstart_,
    ptrdiff_t pixstride, size_t nthreads, ArrayDescriptor *leg_)
    {
    DUCC0_JULIA_TRY_BEGIN
    auto nphi(to_cmav<true,size_t,1>(*nphi_));
    auto phi0(to_cmav<true,double,1>(*phi0_));
    auto ringstart(subtract_1(to_cmav<true,size_t,1>(*ringstart_)));
    if (map_->dtype==Typecode<double>::value)
      {
      auto map(to_cmav<true,double,2>(*map_));
      auto leg(to_vmav<true,complex<double>,3>(*leg_));
      map2leg(map, leg, nphi, phi0, ringstart, pixstride, nthreads);
      }
    else if (map_->dtype==Typecode<float>::value)
      {
      auto map(to_cmav<true,float,2>(*map_));
      auto leg(to_vmav<true,complex<float>,3>(*leg_));
      map2leg(map, leg, nphi, phi0, ringstart, pixstride, nthreads);
      }
    else
      MR_fail("unsupported data type");
    DUCC0_JULIA_TRY_END
    }

#undef DUCC0_JULIA_TRY_BEGIN
#undef DUCC0_JULIA_TRY_END
