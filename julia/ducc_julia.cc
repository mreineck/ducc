// file ducc_julia.cc

/*
Compilation: (the -I path must point to the src/ directory in the ducc0 checkout)

g++ -O3 -march=native -ffast-math -I ../src/ ducc_julia.cc -Wfatal-errors -pthread -std=c++17 -fPIC -c

Creating the shared library:

g++ -O3 -march=native -o ducc_julia.so ducc_julia.o -Wfatal-errors -pthread -std=c++17 -shared -fPIC
*/

#include "ducc0/infra/threading.cc"
#include "ducc0/infra/mav.cc"
#include "ducc0/math/gl_integrator.cc"
#include "ducc0/math/gridding_kernel.cc"
#include "ducc0/nufft/nufft.h"
#include "ducc0/bindings/typecode.h"
#include "ducc0/bindings/array_descriptor.h"

using namespace ducc0;
using namespace std;

template<typename T> cmav<T,2> get_coord(const ArrayDescriptor &desc)
  {
  auto res(to_cmav<T,2>(desc));
  // flip coord axis!
  return cmav<T,2>(res.data()+(res.shape(1)-1)*res.stride(1),
    res.shape(), {res.stride(0), -res.stride(1)});
  }

#define DUCC0_JULIA_TRY_BEGIN try{
#define DUCC0_JULIA_TRY_END } catch(const exception &e) { cout << e.what() << endl; return 1; } return 0;

extern "C" {

int nufft_u2nu(ArrayDescriptor grid,
                     ArrayDescriptor coord,
                     int forward,
                     double epsilon,
                     size_t nthreads,
                     ArrayDescriptor out,
                     size_t verbosity,
                     double sigma_min,
                     double sigma_max,
                     double periodicity,
                     int fft_order)
  {
  DUCC0_JULIA_TRY_BEGIN
  if (coord.dtype==Typecode<double>::value)
    {
    auto mycoord = get_coord<double>(coord);
    if (grid.dtype==Typecode<complex<double>>::value)
      {
      auto mygrid(to_cfmav<complex<double>>(grid));
      auto myout(to_vmav<complex<double>,1>(out));
      MR_assert(mycoord.shape(0)==myout.shape(0), "npoints mismatch");
      MR_assert(mycoord.shape(1)==mygrid.ndim(), "dimensionality mismatch");
      u2nu<double,double>(mycoord,mygrid,forward,epsilon,nthreads,myout,
        verbosity,sigma_min,sigma_max,periodicity,fft_order);
      }
    else if (grid.dtype==Typecode<complex<float>>::value)
      {
      auto mygrid(to_cfmav<complex<float>>(grid));
      auto myout(to_vmav<complex<float>,1>(out));
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
      auto mygrid(to_cfmav<complex<float>>(grid));
      auto myout(to_vmav<complex<float>,1>(out));
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

int nufft_nu2u(ArrayDescriptor points,
                       ArrayDescriptor coord,
                       int forward,
                       double epsilon,
                       size_t nthreads,
                       ArrayDescriptor out,
                       size_t verbosity,
                       double sigma_min,
                       double sigma_max,
                       double periodicity,
                       int fft_order)
  {
  DUCC0_JULIA_TRY_BEGIN
  if (coord.dtype==Typecode<double>::value)
    {
    auto mycoord = get_coord<double>(coord);
    if (points.dtype==Typecode<complex<double>>::value)
      {
      auto mypoints(to_cmav<complex<double>,1>(points));
      auto myout(to_vfmav<complex<double>>(out));
      MR_assert(mycoord.shape(0)==mypoints.shape(0), "npoints mismatch");
      MR_assert(mycoord.shape(1)==myout.ndim(), "dimensionality mismatch");
      nu2u<double,double>(mycoord,mypoints,forward,epsilon,nthreads,myout,
        verbosity,sigma_min,sigma_max,periodicity,fft_order);
      }
    else if (points.dtype==Typecode<complex<float>>::value)
      {
      auto mypoints(to_cmav<complex<float>,1>(points));
      auto myout(to_vfmav<complex<float>>(out));
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
      auto mypoints(to_cmav<complex<float>,1>(points));
      auto myout(to_vfmav<complex<float>>(out));
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

Tplan *nufft_make_plan(int nu2u,
                             ArrayDescriptor shape,
                             ArrayDescriptor coord,
                             double epsilon,
                             size_t nthreads,
                             double sigma_min,
                             double sigma_max,
                             double periodicity,
                             int fft_order)
  {
  try
    {
    auto myshape(to_cmav<uint64_t, 1>(shape));
    auto ndim = myshape.shape(0);
    MR_assert(coord.ndim==2, "bad coordinate dimensionality");
    MR_assert(coord.shape[0]==ndim, "dimensionality mismatch");
    auto res = new Tplan{coord.shape[1],vector<size_t>(ndim),coord.dtype,nullptr};
    for (size_t i=0; i<ndim; ++i)
      res->shp[i] = myshape(ndim-1-i);
    if (coord.dtype==Typecode<double>::value)
      {
      auto mycoord = get_coord<double>(coord);
      if (ndim==1)
        res->plan = new Nufft<double, double, double, 1>(nu2u, mycoord, array<size_t,1>{myshape(0)},
          epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
      else if (ndim==2)
        res->plan = new Nufft<double, double, double, 2>(nu2u, mycoord, array<size_t,2>{myshape(1),myshape(0)},
          epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
      else if (ndim==3)
        res->plan = new Nufft<double, double, double, 3>(nu2u, mycoord, array<size_t,3>{myshape(2),myshape(1),myshape(0)},
          epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
      else
        MR_fail("bad number of dimensions");
      return res;
      }
    else if (coord.dtype==Typecode<float>::value)
      {
      auto mycoord = get_coord<float>(coord);
      if (ndim==1)
        res->plan = new Nufft<float, float, float, 1>(nu2u, mycoord, array<size_t,1>{myshape(0)},
          epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
      else if (ndim==2)
        res->plan = new Nufft<float, float, float, 2>(nu2u, mycoord, array<size_t,2>{myshape(1),myshape(0)},
          epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
      else if (ndim==3)
        res->plan = new Nufft<float, float, float, 3>(nu2u, mycoord, array<size_t,3>{myshape(2),myshape(1),myshape(0)},
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

int nufft_nu2u_planned(Tplan *plan, int forward, size_t verbosity,
  ArrayDescriptor points, ArrayDescriptor uniform)
  {
  DUCC0_JULIA_TRY_BEGIN
  MR_assert(uniform.ndim==plan->shp.size(), "dimensionality mismatch");
  for (size_t i=0; i<uniform.ndim; ++i)
    MR_assert(uniform.shape[i]==plan->shp[i], "array dimension mismatch");
  if (points.dtype==Typecode<complex<double>>::value)
    {
    MR_assert(plan->coord_type==Typecode<double>::value, "data type mismatch");
    auto mypoints(to_cmav<complex<double>,1>(points));
    if (plan->shp.size()==1)
      {
      auto myout(to_vmav<complex<double>,1>(uniform));
      auto rplan = reinterpret_cast<Nufft<double, double, double, 1> *>(plan->plan);
      rplan->nu2u(forward, verbosity, mypoints, myout);
      }
    else if (plan->shp.size()==2)
      {
      auto myout(to_vmav<complex<double>,2>(uniform));
      auto rplan = reinterpret_cast<Nufft<double, double, double, 2> *>(plan->plan);
      rplan->nu2u(forward, verbosity, mypoints, myout);
      }
    else if (plan->shp.size()==3)
      {
      auto myout(to_vmav<complex<double>,3>(uniform));
      auto rplan = reinterpret_cast<Nufft<double, double, double, 3> *>(plan->plan);
      rplan->nu2u(forward, verbosity, mypoints, myout);
      }
    else
      MR_fail("bad number of dimensions");
    }
  else if (points.dtype==Typecode<complex<float>>::value)
    {
    MR_assert(plan->coord_type==Typecode<float>::value, "data type mismatch");
    auto mypoints(to_cmav<complex<float>,1>(points));
    if (plan->shp.size()==1)
      {
      auto myout(to_vmav<complex<float>,1>(uniform));
      auto rplan = reinterpret_cast<Nufft<float, float, float, 1> *>(plan->plan);
      rplan->nu2u(forward, verbosity, mypoints, myout);
      }
    else if (plan->shp.size()==2)
      {
      auto myout(to_vmav<complex<float>,2>(uniform));
      auto rplan = reinterpret_cast<Nufft<float, float, float, 2> *>(plan->plan);
      rplan->nu2u(forward, verbosity, mypoints, myout);
      }
    else if (plan->shp.size()==3)
      {
      auto myout(to_vmav<complex<float>,3>(uniform));
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

int nufft_u2nu_planned(Tplan *plan, int forward, size_t verbosity,
  ArrayDescriptor uniform, ArrayDescriptor points)
  {
  DUCC0_JULIA_TRY_BEGIN
  MR_assert(uniform.ndim==plan->shp.size(), "dimensionality mismatch");
  for (size_t i=0; i<uniform.ndim; ++i)
    MR_assert(uniform.shape[i]==plan->shp[i], "array dimension mismatch");
  if (points.dtype==Typecode<complex<double>>::value)
    {
    MR_assert(plan->coord_type==Typecode<double>::value, "data type mismatch");
    auto mypoints(to_vmav<complex<double>,1>(points));
    if (plan->shp.size()==1)
      {
      auto myuniform(to_cmav<complex<double>,1>(uniform));
      auto rplan = reinterpret_cast<Nufft<double, double, double, 1> *>(plan->plan);
      rplan->u2nu(forward, verbosity, myuniform, mypoints);
      }
    else if (plan->shp.size()==2)
      {
      auto myuniform(to_cmav<complex<double>,2>(uniform));
      auto rplan = reinterpret_cast<Nufft<double, double, double, 2> *>(plan->plan);
      rplan->u2nu(forward, verbosity, myuniform, mypoints);
      }
    else if (plan->shp.size()==3)
      {
      auto myuniform(to_cmav<complex<double>,3>(uniform));
      auto rplan = reinterpret_cast<Nufft<double, double, double, 3> *>(plan->plan);
      rplan->u2nu(forward, verbosity, myuniform, mypoints);
      }
    else
      MR_fail("bad number of dimensions");
    }
  else if (points.dtype==Typecode<complex<float>>::value)
    {
    MR_assert(plan->coord_type==Typecode<float>::value, "data type mismatch");
    auto mypoints(to_vmav<complex<float>,1>(points));
    if (plan->shp.size()==1)
      {
      auto myuniform(to_cmav<complex<float>,1>(uniform));
      auto rplan = reinterpret_cast<Nufft<float, float, float, 1> *>(plan->plan);
      rplan->u2nu(forward, verbosity, myuniform, mypoints);
      }
    else if (plan->shp.size()==2)
      {
      auto myuniform(to_cmav<complex<float>,2>(uniform));
      auto rplan = reinterpret_cast<Nufft<float, float, float, 2> *>(plan->plan);
      rplan->u2nu(forward, verbosity, myuniform, mypoints);
      }
    else if (plan->shp.size()==3)
      {
      auto myuniform(to_cmav<complex<float>,3>(uniform));
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

}

#undef DUCC0_JULIA_TRY_BEGIN
#undef DUCC0_JULIA_TRY_END
