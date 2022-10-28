#include <stdlib.h>

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
                              int fft_order);

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
                              int fft_order);
int main()
  {
  const double pi=3.141592653589793238462643383279502884197;
  int ndim=3;
  int npoints=10000000;
  size_t shape[]={100,200,300};
  double *grid = calloc(shape[0]*shape[1]*shape[2], 2*sizeof(double));
  double *coord = calloc(ndim*npoints, sizeof(double));
  double *points = calloc(npoints, 2*sizeof(double));
  nufft_u2nu_julia_double (ndim, npoints, shape, grid, coord, 1, 1e-5, 4, points, 1, 1.1, 2.5, 2*pi, 0);
  nufft_nu2u_julia_double (ndim, npoints, shape, points, coord, 1, 1e-5, 4, grid, 1, 1.1, 2.5, 2*pi, 0);
  }
