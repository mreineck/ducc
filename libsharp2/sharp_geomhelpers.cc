/*
 *  This file is part of libsharp2.
 *
 *  libsharp2 is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  libsharp2 is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with libsharp2; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/* libsharp2 is being developed at the Max-Planck-Institut fuer Astrophysik */

/*! \file sharp_geomhelpers.c
 *  Spherical transform library
 *
 *  Copyright (C) 2006-2019 Max-Planck-Society
 *  \author Martin Reinecke
 */

#include <cmath>
#include <vector>
#include "libsharp2/sharp_geomhelpers.h"
#include "mr_util/gl_integrator.h"
#include "mr_util/fft.h"
#include "mr_util/error_handling.h"

using namespace std;

unique_ptr<sharp_geom_info> sharp_make_subset_healpix_geom_info (int nside, int stride, int nrings,
  const int *rings, const double *weight)
  {
  const double pi=3.141592653589793238462643383279502884197;
  ptrdiff_t npix=(ptrdiff_t)nside*nside*12;
  ptrdiff_t ncap=2*(ptrdiff_t)nside*(nside-1);

  vector<double> theta(nrings), weight_(nrings), phi0(nrings);
  vector<int> nph(nrings), stride_(nrings);
  vector<ptrdiff_t> ofs(nrings);
  ptrdiff_t curofs=0, checkofs; /* checkofs used for assertion introduced when adding rings arg */
  for (int m=0; m<nrings; ++m)
    {
    int ring = (rings==NULL)? (m+1) : rings[m];
    ptrdiff_t northring = (ring>2*nside) ? 4*nside-ring : ring;
    stride_[m] = stride;
    if (northring < nside)
      {
      theta[m] = 2*asin(northring/(sqrt(6.)*nside));
      nph[m] = 4*northring;
      phi0[m] = pi/nph[m];
      checkofs = 2*northring*(northring-1)*stride;
      }
    else
      {
      double fact1 = (8.*nside)/npix;
      double costheta = (2*nside-northring)*fact1;
      theta[m] = acos(costheta);
      nph[m] = 4*nside;
      if ((northring-nside) & 1)
        phi0[m] = 0;
      else
        phi0[m] = pi/nph[m];
      checkofs = (ncap + (northring-nside)*nph[m])*stride;
      ofs[m] = curofs;
      }
    if (northring != ring) /* southern hemisphere */
      {
      theta[m] = pi-theta[m];
      checkofs = (npix - nph[m])*stride - checkofs;
      ofs[m] = curofs;
      }
    weight_[m]=4.*pi/npix*((weight==NULL) ? 1. : weight[northring-1]);
    if (rings==NULL) {
      MR_assert(curofs==checkofs, "Bug in computing ofs[m]");
    }
    ofs[m] = curofs;
    curofs+=nph[m];
    }

  return sharp_make_geom_info(nrings, nph.data(), ofs.data(), stride_.data(), phi0.data(), theta.data(), weight_.data());
  }

unique_ptr<sharp_geom_info> sharp_make_weighted_healpix_geom_info (int nside, int stride,
  const double *weight)
  {
  return sharp_make_subset_healpix_geom_info(nside, stride, 4 * nside - 1, NULL, weight);
  }

unique_ptr<sharp_geom_info> sharp_make_gauss_geom_info (int nrings, int nphi, double phi0,
  int stride_lon, int stride_lat)
  {
  const double pi=3.141592653589793238462643383279502884197;

  vector<int> nph(nrings), stride_(nrings);
  vector<double> phi0_(nrings);
  vector<ptrdiff_t> ofs(nrings);

  mr::GL_Integrator integ(nrings);
  auto theta = integ.coords();
  auto weight = integ.weights();
  for (int m=0; m<nrings; ++m)
    {
    theta[m] = acos(-theta[m]);
    nph[m]=nphi;
    phi0_[m]=phi0;
    ofs[m]=(ptrdiff_t)m*stride_lat;
    stride_[m]=stride_lon;
    weight[m]*=2*pi/nphi;
    }

  return sharp_make_geom_info (nrings, nph.data(), ofs.data(), stride_.data(), phi0_.data(), theta.data(), weight.data());
  }

/* Weights from Waldvogel 2006: BIT Numerical Mathematics 46, p. 195 */
unique_ptr<sharp_geom_info> sharp_make_fejer1_geom_info (int nrings, int ppring, double phi0,
  int stride_lon, int stride_lat)
  {
  const double pi=3.141592653589793238462643383279502884197;

  vector<double> theta(nrings), weight(nrings), phi0_(nrings);
  vector<int> nph(nrings), stride_(nrings);
  vector<ptrdiff_t> ofs(nrings);

  weight[0]=2.;
  for (int k=1; k<=(nrings-1)/2; ++k)
    {
    weight[2*k-1]=2./(1.-4.*k*k)*cos((k*pi)/nrings);
    weight[2*k  ]=2./(1.-4.*k*k)*sin((k*pi)/nrings);
    }
  if ((nrings&1)==0) weight[nrings-1]=0.;
  mr::r2r_fftpack({size_t(nrings)}, {sizeof(double)}, {sizeof(double)}, {0}, false, false, weight.data(), weight.data(), 1.);

  for (int m=0; m<(nrings+1)/2; ++m)
    {
    theta[m]=pi*(m+0.5)/nrings;
    theta[nrings-1-m]=pi-theta[m];
    nph[m]=nph[nrings-1-m]=ppring;
    phi0_[m]=phi0_[nrings-1-m]=phi0;
    ofs[m]=(ptrdiff_t)m*stride_lat;
    ofs[nrings-1-m]=(ptrdiff_t)((nrings-1-m)*stride_lat);
    stride_[m]=stride_[nrings-1-m]=stride_lon;
    weight[m]=weight[nrings-1-m]=weight[m]*2*pi/(nrings*nph[m]);
    }

  return sharp_make_geom_info (nrings, nph.data(), ofs.data(), stride_.data(), phi0_.data(), theta.data(), weight.data());
  }

/* Weights from Waldvogel 2006: BIT Numerical Mathematics 46, p. 195 */
unique_ptr<sharp_geom_info> sharp_make_cc_geom_info (int nrings, int ppring, double phi0,
  int stride_lon, int stride_lat)
  {
  const double pi=3.141592653589793238462643383279502884197;

  vector<double> theta(nrings), weight(nrings,0.), phi0_(nrings);
  vector<int> nph(nrings), stride_(nrings);
  vector<ptrdiff_t> ofs(nrings);

  int n=nrings-1;
  double dw=-1./(n*n-1.+(n&1));
  weight[0]=2.+dw;
  for (int k=1; k<=(n/2-1); ++k)
    weight[2*k-1]=2./(1.-4.*k*k) + dw;
  weight[2*(n/2)-1]=(n-3.)/(2*(n/2)-1) -1. -dw*((2-(n&1))*n-1);
  mr::r2r_fftpack({size_t(n)}, {sizeof(double)}, {sizeof(double)}, {0}, false, false, weight.data(), weight.data(), 1.);
  weight[n]=weight[0];

  for (int m=0; m<(nrings+1)/2; ++m)
    {
    theta[m]=pi*m/(nrings-1.);
    if (theta[m]<1e-15) theta[m]=1e-15;
    theta[nrings-1-m]=pi-theta[m];
    nph[m]=nph[nrings-1-m]=ppring;
    phi0_[m]=phi0_[nrings-1-m]=phi0;
    ofs[m]=(ptrdiff_t)m*stride_lat;
    ofs[nrings-1-m]=(ptrdiff_t)((nrings-1-m)*stride_lat);
    stride_[m]=stride_[nrings-1-m]=stride_lon;
    weight[m]=weight[nrings-1-m]=weight[m]*2*pi/(n*nph[m]);
    }

  return sharp_make_geom_info (nrings, nph.data(), ofs.data(), stride_.data(), phi0_.data(), theta.data(), weight.data());
  }

/* Weights from Waldvogel 2006: BIT Numerical Mathematics 46, p. 195 */
unique_ptr<sharp_geom_info> sharp_make_fejer2_geom_info (int nrings, int ppring, double phi0,
  int stride_lon, int stride_lat)
  {
  const double pi=3.141592653589793238462643383279502884197;

  vector<double> theta(nrings), weight(nrings+1, 0.), phi0_(nrings);
  vector<int> nph(nrings), stride_(nrings);
  vector<ptrdiff_t> ofs(nrings);

  int n=nrings+1;
  weight[0]=2.;
  for (int k=1; k<=(n/2-1); ++k)
    weight[2*k-1]=2./(1.-4.*k*k);
  weight[2*(n/2)-1]=(n-3.)/(2*(n/2)-1) -1.;
  mr::r2r_fftpack({size_t(n)}, {sizeof(double)}, {sizeof(double)}, {0}, false, false, weight.data(), weight.data(), 1.);
  for (int m=0; m<nrings; ++m)
    weight[m]=weight[m+1];

  for (int m=0; m<(nrings+1)/2; ++m)
    {
    theta[m]=pi*(m+1)/(nrings+1.);
    theta[nrings-1-m]=pi-theta[m];
    nph[m]=nph[nrings-1-m]=ppring;
    phi0_[m]=phi0_[nrings-1-m]=phi0;
    ofs[m]=(ptrdiff_t)m*stride_lat;
    ofs[nrings-1-m]=(ptrdiff_t)((nrings-1-m)*stride_lat);
    stride_[m]=stride_[nrings-1-m]=stride_lon;
    weight[m]=weight[nrings-1-m]=weight[m]*2*pi/(n*nph[m]);
    }

  return sharp_make_geom_info (nrings, nph.data(), ofs.data(), stride_.data(), phi0_.data(), theta.data(), weight.data());
  }

unique_ptr<sharp_geom_info> sharp_make_mw_geom_info (int nrings, int ppring, double phi0,
  int stride_lon, int stride_lat)
  {
  const double pi=3.141592653589793238462643383279502884197;

  vector<double> theta(nrings), phi0_(nrings);
  vector<int> nph(nrings), stride_(nrings);
  vector<ptrdiff_t> ofs(nrings);

  for (int m=0; m<nrings; ++m)
    {
    theta[m]=pi*(2.*m+1.)/(2.*nrings-1.);
    if (theta[m]>pi-1e-15) theta[m]=pi-1e-15;
    nph[m]=ppring;
    phi0_[m]=phi0;
    ofs[m]=(ptrdiff_t)m*stride_lat;
    stride_[m]=stride_lon;
    }

  return sharp_make_geom_info (nrings, nph.data(), ofs.data(), stride_.data(), phi0_.data(), theta.data(), NULL);
  }
