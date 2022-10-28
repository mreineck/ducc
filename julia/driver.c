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

void *make_nufft_plan_double(int nu2u,
                             size_t ndim,
                             size_t npoints,
                             const size_t *shape,
                             const double *coord,
                             double epsilon,
                             size_t nthreads,
                             double sigma_min,
                             double sigma_max,
                             double periodicity,
                             int fft_order);
void delete_nufft_plan_double(void *plan);
void planned_nu2u(void *plan, int forward, size_t verbosity,
  const double *points, double *uniform);
void planned_u2nu(void *plan, int forward, size_t verbosity,
  const double *uniform, double *points);

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

  void *plan = make_nufft_plan_double(0, ndim, npoints, shape, coord, 1e-5, 4, 1.1, 2.6, 2*pi, 0);
  planned_u2nu(plan, 1, 1, grid, points);
  delete_nufft_plan_double(plan);
  plan = make_nufft_plan_double(1, ndim, npoints, shape, coord, 1e-5, 4, 1.1, 2.6, 2*pi, 0);
  planned_nu2u(plan, 1, 1, points, grid);
  delete_nufft_plan_double(plan);
  }
