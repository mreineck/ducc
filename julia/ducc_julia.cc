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

}
