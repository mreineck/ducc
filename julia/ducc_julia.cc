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

using namespace ducc0;
using namespace std;

extern "C" {

/*
ndim:         number of dimensions (1/2/3)
npoints:      number of non-uniform points
shape:        points to a dense Julia array of shape(ndim,) containing the grid
              dimensions in Julia order
grid:         points to a dense Julia array of shape (2,shape)
              the leading dimension is for real and imaginary parts
coord:        points to a dense Julia array of shape(ndim,npoints)
forward:      0 ==> FFT exponent is  1
              1 ==> FFT exponent is -1
epsilon:      desired accuracy
nthreads:     number of threads to use. 0 means using all available threads. 
out:          points to a mutable dense Julia array of shape (2,npoints)
              the leading dimension is for real and imaginary parts
sigma_min:    minimum oversampling factor. If unsure, use 1.1
sigma_max:    maximum oversampling factor. If unsure, use 2.6
periodicity:  assumed periodicity of the coordinates of the nonuniform points
fft_order:    0: Fourier grids start with most negative k mode
              1: Fourier grids start with zero k mode (usual FFT convention)
*/
void nufft_u2nu_julia_double (size_t ndim,
                              size_t npoints,
                              const size_t *shape,
                              const double *grid,
                              const double *coord,
                              int forward,
                              double epsilon,
                              size_t nthreads,
                              double *out,
                              size_t verbosity,
                              double sigma_min,
                              double sigma_max,
                              double periodicity,
                              int fft_order)
  {
  cmav<double,2> mycoord(coord,{npoints,ndim});
  vector<size_t> myshape(ndim);
  for (size_t i=0; i<ndim; ++i)
    myshape[i] = shape[ndim-1-i];
  cfmav<complex<double>> mygrid(reinterpret_cast<const complex<double> *>(grid), myshape);
  vmav<complex<double>,1> myout(reinterpret_cast<complex<double> *>(out), {npoints});
  u2nu<double,double>(mycoord,mygrid,forward,epsilon,nthreads,myout,verbosity,sigma_min,sigma_max,periodicity,fft_order);
  }

/*
ndim:         number of dimensions (1/2/3)
npoints:      number of non-uniform points
shape:        points to a dense Julia array of shape(ndim,) containing the grid
              dimensions in Julia order
points:       points to a dense Julia array of shape (2,npoints)
              the leading dimension is for real and imaginary parts
coord:        points to a dense Julia array of shape(ndim,npoints)
forward:      0 ==> FFT exponent is  1
              1 ==> FFT exponent is -1
epsilon:      desired accuracy
nthreads:     number of threads to use. 0 means using all available threads. 
out:          points to a dense mutable Julia array of shape (2,shape)
              the leading dimension is for real and imaginary parts
sigma_min:    minimum oversampling factor. If unsure, use 1.1
sigma_max:    maximum oversampling factor. If unsure, use 2.6
periodicity:  assumed periodicity of the coordinates of the nonuniform points
fft_order:    0: Fourier grids start with most negative k mode
              1: Fourier grids start with zero k mode (usual FFT convention)
*/
void nufft_nu2u_julia_double (size_t ndim,
                              size_t npoints,
                              const size_t *shape,
                              const double *points,
                              const double *coord,
                              int forward,
                              double epsilon,
                              size_t nthreads,
                              double *out,
                              size_t verbosity,
                              double sigma_min,
                              double sigma_max,
                              double periodicity,
                              int fft_order)
  {
  cmav<double,2> mycoord(coord,{npoints,ndim});
  vector<size_t> myshape(ndim);
  for (size_t i=0; i<ndim; ++i)
    myshape[i] = shape[ndim-1-i];
  vfmav<complex<double>> myout(reinterpret_cast<complex<double> *>(out), myshape);
  cmav<complex<double>,1> mypoints(reinterpret_cast<const complex<double> *>(points), {npoints});
  nu2u<double,double>(mycoord,mypoints,forward,epsilon,nthreads,myout,verbosity,sigma_min,sigma_max,periodicity,fft_order);
  }

struct Tplan
  {
  size_t npoints;
  vector<size_t> shp;
  void *plan;
  };

Tplan *make_nufft_plan_double(int nu2u,
                             size_t ndim,
                             size_t npoints,
                             const size_t *shape,
                             const double *coord,
                             double epsilon,
                             size_t nthreads,
                             double sigma_min,
                             double sigma_max,
                             double periodicity,
                             int fft_order)
  {
  auto res = new Tplan{npoints,vector<size_t>(ndim),nullptr};
  cmav<double,2> mycoord(coord,{npoints,ndim});
  for (size_t i=0; i<ndim; ++i)
    res->shp[i] = shape[ndim-1-i];
  if (ndim==1)
    {
    array<size_t,1> myshape({shape[0]});
    res->plan = new Nufft<double, double, double, 1>(nu2u, mycoord, myshape,
      epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
    }
  else if (ndim==2)
    {
    array<size_t,2> myshape({shape[1],shape[0]});
    res->plan = new Nufft<double, double, double, 2>(nu2u, mycoord, myshape,
      epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
    }
  else if (ndim==3)
    {
    array<size_t,3> myshape({shape[2],shape[1],shape[0]});
    res->plan = new Nufft<double, double, double, 3>(nu2u, mycoord, myshape,
      epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
    }
  else
    MR_fail("bad number of dimensions");
  return res;
  }

void delete_nufft_plan_double(Tplan *plan)
  {
  if (plan->shp.size()==1)
    delete reinterpret_cast<Nufft<double, double, double, 1> *>(plan->plan);
  else if (plan->shp.size()==2)
    delete reinterpret_cast<Nufft<double, double, double, 2> *>(plan->plan);
  else if (plan->shp.size()==3)
    delete reinterpret_cast<Nufft<double, double, double, 3> *>(plan->plan);
  else
    MR_fail("bad number of dimensions");
  delete plan;
  }

void planned_nu2u(Tplan *plan, int forward, size_t verbosity,
  const double *points, double *uniform)
  {
  cmav<complex<double>,1> mypoints(reinterpret_cast<const complex<double> *>(points), {plan->npoints});
  if (plan->shp.size()==1)
    {
    array<size_t,1> myshape({plan->shp[0]});
    vmav<complex<double>,1> myout(reinterpret_cast<complex<double> *>(uniform), myshape);
    auto rplan = reinterpret_cast<Nufft<double, double, double, 1> *>(plan->plan);
    rplan->nu2u(forward, verbosity, mypoints, myout);
    }
  else if (plan->shp.size()==2)
    {
    array<size_t,2> myshape({plan->shp[0],plan->shp[1]});
    vmav<complex<double>,2> myout(reinterpret_cast<complex<double> *>(uniform), myshape);
    auto rplan = reinterpret_cast<Nufft<double, double, double, 2> *>(plan->plan);
    rplan->nu2u(forward, verbosity, mypoints, myout);
    }
  else if (plan->shp.size()==3)
    {
    array<size_t,3> myshape({plan->shp[0],plan->shp[1],plan->shp[2]});
    vmav<complex<double>,3> myout(reinterpret_cast<complex<double> *>(uniform), myshape);
    auto rplan = reinterpret_cast<Nufft<double, double, double, 3> *>(plan->plan);
    rplan->nu2u(forward, verbosity, mypoints, myout);
    }
  else
    MR_fail("bad number of dimensions");
  }

void planned_u2nu(Tplan *plan, int forward, size_t verbosity,
  const double *uniform, double *points)
  {
  vmav<complex<double>,1> mypoints(reinterpret_cast<complex<double> *>(points), {plan->npoints});
  if (plan->shp.size()==1)
    {
    array<size_t,1> myshape({plan->shp[0]});
    cmav<complex<double>,1> myuniform(reinterpret_cast<const complex<double> *>(uniform), myshape);
    auto rplan = reinterpret_cast<Nufft<double, double, double, 1> *>(plan->plan);
    rplan->u2nu(forward, verbosity, myuniform, mypoints);
    }
  else if (plan->shp.size()==2)
    {
    array<size_t,2> myshape({plan->shp[0],plan->shp[1]});
    cmav<complex<double>,2> myuniform(reinterpret_cast<const complex<double> *>(uniform), myshape);
    auto rplan = reinterpret_cast<Nufft<double, double, double, 2> *>(plan->plan);
    rplan->u2nu(forward, verbosity, myuniform, mypoints);
    }
  else if (plan->shp.size()==3)
    {
    array<size_t,3> myshape({plan->shp[0],plan->shp[1],plan->shp[2]});
    cmav<complex<double>,3> myuniform(reinterpret_cast<const complex<double> *>(uniform), myshape);
    auto rplan = reinterpret_cast<Nufft<double, double, double, 3> *>(plan->plan);
    rplan->u2nu(forward, verbosity, myuniform, mypoints);
    }
  else
    MR_fail("bad number of dimensions");
  }
}
