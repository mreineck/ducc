/*
 *  This file is part of DUCC.
 *
 *  DUCC is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  DUCC is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with DUCC; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/*
 *  DUCC is being developed at the Max-Planck-Institut fuer Astrophysik
 */

/*
 *  Copyright (C) 2017-2021 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <complex>

#include "ducc0/sht/sht.h"
#include "ducc0/sht/alm.h"
#include "ducc0/infra/string_utils.h"
#include "ducc0/infra/error_handling.h"
#include "ducc0/math/constants.h"
#include "ducc0/bindings/pybind_utils.h"

namespace ducc0 {

namespace detail_pymodule_sht {

using namespace std;

namespace py = pybind11;

auto None = py::none();

py::array Py_GL_weights(size_t nlat, size_t nlon)
  {
  auto res = make_Pyarr<double>({nlat});
  auto res2 = to_vmav<double,1>(res);
  GL_Integrator integ(nlat);
  auto wgt = integ.weights();
  for (size_t i=0; i<res2.shape(0); ++i)
    res2(i) = wgt[i]*twopi/nlon;
  return move(res);
  }

py::array Py_GL_thetas(size_t nlat)
  {
  auto res = make_Pyarr<double>({nlat});
  auto res2 = to_vmav<double,1>(res);
  GL_Integrator integ(nlat);
  auto x = integ.coords();
  for (size_t i=0; i<res2.shape(0); ++i)
    res2(i) = acos(-x[i]);
  return move(res);
  }

template<typename T> py::array Py2_rotate_alm(const py::array &alm_, int64_t lmax,
  double psi, double theta, double phi, size_t nthreads)
  {
  auto a1 = to_cmav<complex<T>,1>(alm_);
  auto alm = make_Pyarr<complex<T>>({a1.shape(0)});
  auto a2 = to_vmav<complex<T>,1>(alm);
  {
  py::gil_scoped_release release;
  for (size_t i=0; i<a1.shape(0); ++i) a2(i)=a1(i);
  Alm_Base base(lmax,lmax);
  rotate_alm(base, a2, psi, theta, phi, nthreads);
  }
  return move(alm);
  }
py::array Py_rotate_alm(const py::array &alm, int64_t lmax,
  double psi, double theta, double phi, size_t nthreads)
  {
  if (isPyarr<complex<float>>(alm))
    return Py2_rotate_alm<float>(alm, lmax, psi, theta, phi, nthreads);
  if (isPyarr<complex<double>>(alm))
    return Py2_rotate_alm<double>(alm, lmax, psi, theta, phi, nthreads);
  MR_fail("type matching failed: 'alm' has neither type 'c8' nor 'c16'");
  }

void getmstuff(size_t lmax, const py::object &mval_, const py::object &mstart_,
  vmav<size_t,1> &mval, vmav<size_t,1> &mstart)
  {
  MR_assert(mval_.is_none()==mstart_.is_none(), "mval and mstart must be supplied together");
  if (mval_.is_none())
    {
    vmav<size_t,1> tmv({lmax+1});
    mval.assign(tmv);
    vmav<size_t,1> tms({lmax+1});
    mstart.assign(tms);
    for (size_t m=0, idx=0; m<=lmax; ++m, idx+=lmax+1-m)
      {
      mval(m) = m;
      mstart(m) = idx;
      }
    }
  else
    {
    auto tmval = to_cmav<int64_t,1>(mval_);
    auto tmstart = to_cmav<int64_t,1>(mstart_);
    size_t nm = tmval.shape(0);
    MR_assert(nm==tmstart.shape(0), "size mismatch between mval and mstart");
    vmav<size_t,1> tmv({nm});
    mval.assign(tmv);
    vmav<size_t,1> tms({nm});
    mstart.assign(tms);
    for (size_t i=0; i<nm; ++i)
      {
      auto m = tmval(i);
      MR_assert((m>=0) && (m<=int64_t(lmax)), "bad m value");
      mval(i) = size_t(m);
      mstart(i) = size_t(tmstart(i));
      }
    }
  }
cmav<size_t,1> get_mstart(size_t lmax, const py::object &mstart_)
  {
  if (mstart_.is_none())
    {
    vmav<size_t,1> mstart({lmax+1});
    for (size_t m=0, idx=0; m<=lmax; ++m, idx+=lmax+1-m)
      mstart(m) = idx;
    return mstart;
    }
  auto mstart = to_cmav<size_t,1>(mstart_);
  MR_assert(mstart.shape(0)==lmax+1, "bad mstart size");
  return mstart;
  }

py::array Py_get_gridweights(const string &type, size_t ntheta)
  {
  auto wgt_ = make_Pyarr<double>({ntheta});
  auto wgt = to_vmav<double,1>(wgt_);
  get_gridweights(type, wgt);
  return wgt_;
  }

size_t min_almdim(size_t lmax, const cmav<size_t,1> &mval, const cmav<size_t,1> &mstart, ptrdiff_t lstride)
  {
  size_t res=0;
  for (size_t i=0; i<mval.shape(0); ++i)
    {
    auto ifirst = ptrdiff_t(mstart(i)) + ptrdiff_t(mval(i))*lstride;
    MR_assert(ifirst>=0, "impossible a_lm memory layout");
    auto ilast = ptrdiff_t(mstart(i)) + ptrdiff_t(lmax)*lstride;
    MR_assert(ilast>=0, "impossible a_lm memory layout");
    res = max(res, size_t(max(ifirst, ilast)));
    }
  return res+1;
  }
size_t min_mapdim(const cmav<size_t,1> &nphi, const cmav<size_t,1> &ringstart, ptrdiff_t pixstride)
  {
  size_t res=0;
  for (size_t i=0; i<nphi.shape(0); ++i)
    {
    auto ilast = ptrdiff_t(ringstart(i)) + ptrdiff_t(nphi(i)-1)*pixstride;
    MR_assert(ilast>=0, "impossible map memory layout");
    res = max(res, max(ringstart(i), size_t(ilast)));
    }
  return res+1;
  }

template<typename T> py::array Py2_alm2leg(const py::array &alm_, size_t spin, size_t lmax, const py::object &mval_, const py::object &mstart_, ptrdiff_t lstride, const py::array &theta_, size_t nthreads, py::object &leg__)
  {
  auto alm = to_cmav<complex<T>,2>(alm_);
  auto theta = to_cmav<double,1>(theta_);
  vmav<size_t,1> mval, mstart;
  getmstuff(lmax, mval_, mstart_, mval, mstart);
  MR_assert(alm.shape(1)>=min_almdim(lmax, mval, mstart, lstride), "bad a_lm array size");
  auto leg_ = get_optional_Pyarr<complex<T>>(leg__, {alm.shape(0),theta.shape(0),mval.shape(0)});
  auto leg = to_vmav<complex<T>,3>(leg_);
  {
  py::gil_scoped_release release;
  alm2leg(alm, leg, spin, lmax, mval, mstart, lstride, theta, nthreads, ALM2MAP);
  }
  return leg_;
  }
py::array Py_alm2leg(const py::array &alm, size_t lmax, const py::array &theta, size_t spin, const py::object &mval, const py::object &mstart, ptrdiff_t lstride, size_t nthreads, py::object &leg)
  {
  if (isPyarr<complex<float>>(alm))
    return Py2_alm2leg<float>(alm, spin, lmax, mval, mstart, lstride, theta, nthreads, leg);
  if (isPyarr<complex<double>>(alm))
    return Py2_alm2leg<double>(alm, spin, lmax, mval, mstart, lstride, theta, nthreads, leg);
  MR_fail("type matching failed: 'alm' has neither type 'c8' nor 'c16'");
  }
template<typename T> py::array Py2_alm2leg_deriv1(const py::array &alm_, size_t lmax, const py::object &mval_, const py::object &mstart_, ptrdiff_t lstride, const py::array &theta_, size_t nthreads, py::object &leg__)
  {
  auto alm = to_cmav<complex<T>,2>(alm_);
  auto theta = to_cmav<double,1>(theta_);
  vmav<size_t,1> mval, mstart;
  getmstuff(lmax, mval_, mstart_, mval, mstart);
  MR_assert(alm.shape(0)==1, "need exactly 1 a_lm component");
  MR_assert(alm.shape(1)>=min_almdim(lmax, mval, mstart, lstride), "bad a_lm array size");
  auto leg_ = get_optional_Pyarr<complex<T>>(leg__, {2,theta.shape(0),mval.shape(0)});
  auto leg = to_vmav<complex<T>,3>(leg_);
  {
  py::gil_scoped_release release;
  alm2leg(alm, leg, 0, lmax, mval, mstart, lstride, theta, nthreads, ALM2MAP_DERIV1);
  }
  return leg_;
  }
py::array Py_alm2leg_deriv1(const py::array &alm, size_t lmax, const py::array &theta, const py::object &mval, const py::object &mstart, ptrdiff_t lstride, size_t nthreads, py::object &leg)
  {
  if (isPyarr<complex<float>>(alm))
    return Py2_alm2leg_deriv1<float>(alm, lmax, mval, mstart, lstride, theta, nthreads, leg);
  if (isPyarr<complex<double>>(alm))
    return Py2_alm2leg_deriv1<double>(alm, lmax, mval, mstart, lstride, theta, nthreads, leg);
  MR_fail("type matching failed: 'alm' has neither type 'c8' nor 'c16'");
  }
template<typename T> py::array Py2_leg2alm(const py::array &leg_, const py::array &theta_, size_t spin, size_t lmax, const py::object &mval_, const py::object &mstart_, ptrdiff_t lstride, size_t nthreads, py::object &alm__)
  {
  auto leg = to_cmav<complex<T>,3>(leg_);
  auto theta = to_cmav<double,1>(theta_);
  MR_assert(leg.shape(1)==theta.shape(0), "bad leg array size");
  vmav<size_t,1> mval, mstart;
  getmstuff(lmax, mval_, mstart_, mval, mstart);
  auto alm_ = get_optional_Pyarr_minshape<complex<T>>(alm__, {leg.shape(0),min_almdim(lmax, mval, mstart, lstride)});
  auto alm = to_vmav<complex<T>,2>(alm_);
  MR_assert(alm.shape(0)==leg.shape(0), "bad number of components in a_lm array");
  {
  py::gil_scoped_release release;
  leg2alm(alm, leg, spin, lmax, mval, mstart, lstride, theta, nthreads);
  }
  return alm_;
  }
py::array Py_leg2alm(const py::array &leg, size_t lmax, const py::array &theta, size_t spin, const py::object &mval, const py::object &mstart, ptrdiff_t lstride, size_t nthreads, py::object &alm)
  {
  if (isPyarr<complex<float>>(leg))
    return Py2_leg2alm<float>(leg, theta, spin, lmax, mval, mstart, lstride, nthreads, alm);
  if (isPyarr<complex<double>>(leg))
    return Py2_leg2alm<double>(leg, theta, spin, lmax, mval, mstart, lstride, nthreads, alm);
  MR_fail("type matching failed: 'leg' has neither type 'c8' nor 'c16'");
  }
template<typename T> py::array Py2_map2leg(const py::array &map_, const py::array &nphi_, const py::array &phi0_, const py::array &ringstart_, size_t mmax, ptrdiff_t pixstride, size_t nthreads, py::object &leg__)
  {
  auto map = to_cmav<T,2>(map_);
  auto nphi = to_cmav<size_t,1>(nphi_);
  auto phi0 = to_cmav<double,1>(phi0_);
  auto ringstart = to_cmav<size_t,1>(ringstart_);
  MR_assert(map.shape(1)>=min_mapdim(nphi, ringstart, pixstride), "bad map array size");
  auto leg_ = get_optional_Pyarr<complex<T>>(leg__, {map.shape(0),nphi.shape(0),mmax+1});
  auto leg = to_vmav<complex<T>,3>(leg_);
  {
  py::gil_scoped_release release;
  map2leg(map, leg, nphi, phi0, ringstart, pixstride, nthreads);
  }
  return leg_;
  }
py::array Py_map2leg(const py::array &map, const py::array &nphi, const py::array &phi0, const py::array &ringstart, size_t mmax, ptrdiff_t pixstride, size_t nthreads, py::object &leg)
  {
  if (isPyarr<float>(map))
    return Py2_map2leg<float>(map, nphi, phi0, ringstart, mmax, pixstride, nthreads, leg);
  if (isPyarr<double>(map))
    return Py2_map2leg<double>(map, nphi, phi0, ringstart, mmax, pixstride, nthreads, leg);
  MR_fail("type matching failed: 'map' has neither type 'f4' nor 'f8'");
  }
template<typename T> py::array Py2_leg2map(const py::array &leg_, const py::array &nphi_, const py::array &phi0_, const py::array &ringstart_, ptrdiff_t pixstride, size_t nthreads, py::object &map__)
  {
  auto leg = to_cmav<complex<T>,3>(leg_);
  auto nphi = to_cmav<size_t,1>(nphi_);
  auto phi0 = to_cmav<double,1>(phi0_);
  auto ringstart = to_cmav<size_t,1>(ringstart_);
  auto map_ = get_optional_Pyarr_minshape<T>(map__, {leg.shape(0),min_mapdim(nphi, ringstart, pixstride)});
  auto map = to_vmav<T,2>(map_);
  MR_assert(map.shape(0)==leg.shape(0), "bad number of components in map array");
  {
  py::gil_scoped_release release;
  leg2map(map, leg, nphi, phi0, ringstart, pixstride, nthreads);
  }
  return map_;
  }
py::array Py_leg2map(const py::array &leg, const py::array &nphi, const py::array &phi0, const py::array &ringstart, ptrdiff_t pixstride, size_t nthreads, py::object &map)
  {
  if (isPyarr<complex<float>>(leg))
    return Py2_leg2map<float>(leg, nphi, phi0, ringstart, pixstride, nthreads, map);
  if (isPyarr<complex<double>>(leg))
    return Py2_leg2map<double>(leg, nphi, phi0, ringstart, pixstride, nthreads, map);
  MR_fail("type matching failed: 'leg' has neither type 'c8' nor 'c16'");
  }

// FIXME: open questions
// - do we build mstart automatically and just take mmax?
// - phi0, ringstart = None => build assuming phi0=0, rings sequential?
// - accept scalar nphi, phi0?
template<typename T> py::array Py2_synthesis(const py::array &alm_,
  py::object &map__, size_t spin, size_t lmax,
  const py::object &mstart_, ptrdiff_t lstride, 
  const py::array &theta_, 
  const py::array &nphi_,
  const py::array &phi0_, const py::array &ringstart_,
  ptrdiff_t pixstride, size_t nthreads)
  {
  auto alm = to_cmav<complex<T>,2>(alm_);
  auto mstart = get_mstart(lmax, mstart_);
  auto theta = to_cmav<double,1>(theta_);
  auto phi0 = to_cmav<double,1>(phi0_);
  auto nphi = to_cmav<size_t,1>(nphi_);
  auto ringstart = to_cmav<size_t,1>(ringstart_);
  auto map_ = get_optional_Pyarr_minshape<T>(map__, {alm.shape(0), min_mapdim(nphi, ringstart, pixstride)});
  auto map = to_vmav<T,2>(map_);
  MR_assert(map.shape(0)==alm.shape(0), "bad number of components in map array");
  {
  py::gil_scoped_release release;
  synthesis(alm, map, spin, lmax, mstart, lstride, theta, nphi, phi0, ringstart, pixstride, nthreads, ALM2MAP);
  }
  return map_;
  }
py::array Py_synthesis(const py::array &alm, const py::array &theta,
  size_t lmax, const py::object &mstart, 
  const py::array &nphi,
  const py::array &phi0, const py::array &ringstart, size_t spin, ptrdiff_t lstride, ptrdiff_t pixstride,
  size_t nthreads, py::object &map)
  {
  if (isPyarr<complex<float>>(alm))
    return Py2_synthesis<float>(alm, map, spin, lmax, mstart, lstride, theta, 
      nphi, phi0, ringstart, pixstride, nthreads);
  else if (isPyarr<complex<double>>(alm))
    return Py2_synthesis<double>(alm, map, spin, lmax, mstart, lstride, theta, 
      nphi, phi0, ringstart, pixstride, nthreads);
  MR_fail("type matching failed: 'alm' has neither type 'c8' nor 'c16'");
  }
template<typename T> py::array Py2_synthesis_deriv1(const py::array &alm_,
  py::object &map__, size_t lmax,
  const py::object &mstart_, ptrdiff_t lstride, 
  const py::array &theta_, 
  const py::array &nphi_,
  const py::array &phi0_, const py::array &ringstart_,
  ptrdiff_t pixstride, size_t nthreads)
  {
  auto alm = to_cmav<complex<T>,2>(alm_);
  auto mstart = get_mstart(lmax, mstart_);
  auto theta = to_cmav<double,1>(theta_);
  auto phi0 = to_cmav<double,1>(phi0_);
  auto nphi = to_cmav<size_t,1>(nphi_);
  auto ringstart = to_cmav<size_t,1>(ringstart_);
  auto map_ = get_optional_Pyarr_minshape<T>(map__, {alm.shape(0), min_mapdim(nphi, ringstart, pixstride)});
  auto map = to_vmav<T,2>(map_);
  MR_assert(map.shape(0)==2, "bad number of components in map array");
  {
  py::gil_scoped_release release;
  synthesis(alm, map, 1, lmax, mstart, lstride, theta, nphi, phi0, ringstart, pixstride, nthreads, ALM2MAP_DERIV1);
  }
  return map_;
  }
py::array Py_synthesis_deriv1(const py::array &alm, const py::array &theta,
  size_t lmax, const py::object &mstart, 
  const py::array &nphi,
  const py::array &phi0, const py::array &ringstart, ptrdiff_t lstride, ptrdiff_t pixstride,
  size_t nthreads, py::object &map)
  {
  if (isPyarr<complex<float>>(alm))
    return Py2_synthesis_deriv1<float>(alm, map, lmax, mstart, lstride, theta, 
      nphi, phi0, ringstart, pixstride, nthreads);
  else if (isPyarr<complex<double>>(alm))
    return Py2_synthesis_deriv1<double>(alm, map, lmax, mstart, lstride, theta, 
      nphi, phi0, ringstart, pixstride, nthreads);
  MR_fail("type matching failed: 'alm' has neither type 'c8' nor 'c16'");
  }


template<typename T> py::array_t<T> check_build_map(const py::object &map, size_t ncomp, const py::object &ntheta, const py::object &nphi)
  {
  if (map.is_none())
    {
    MR_assert((!ntheta.is_none()) && (!nphi.is_none()),
      "you need to specify either 'map' or 'ntheta' and 'nphi'");
    return make_Pyarr<T>({ncomp, ntheta.cast<size_t>(), nphi.cast<size_t>()});
    }
  else
    {
    py::array_t<T> tmap = map;
    MR_assert((size_t(tmap.ndim())==3) && (size_t(tmap.shape(0))==ncomp), "map size mismatch");
    if (!ntheta.is_none())
      MR_assert(size_t(tmap.shape(1))==ntheta.cast<size_t>(), "ntheta mismatch");
    if (!nphi.is_none())
      MR_assert(size_t(tmap.shape(2))==nphi.cast<size_t>(), "nphi mismatch");
    return tmap;
    }
  }
template<typename T> py::array_t<complex<T>> check_build_alm(const py::object &alm, size_t ncomp, size_t lmax, size_t mmax)
  {
  size_t nalm = ((mmax+1)*(mmax+2))/2 + (mmax+1)*(lmax-mmax);
  if (alm.is_none())
    {
    MR_assert(lmax>=mmax, "mmax must not be larger than lmax");
    return make_Pyarr<complex<T>>({ncomp, nalm});
    }
  else
    {
    py::array_t<complex<T>> talm = alm;
    MR_assert((size_t(talm.ndim())==2) && (size_t(talm.shape(0))==ncomp)
      && (size_t(talm.shape(1))==nalm), "alm size mismatch");
    return talm;
    }
  }

template<typename T> py::array Py2_synthesis_2d(const py::array &alm_,
  size_t spin, size_t lmax, const string &geometry, const py::object & ntheta,
  const py::object &nphi, size_t mmax, size_t nthreads, py::object &map__)
  {
  auto alm = to_cmav<complex<T>,2>(alm_);
  auto map_ = check_build_map<T>(map__, alm.shape(0), ntheta, nphi);
  auto map = to_vmav<T,3>(map_);
  MR_assert(map.shape(0)==alm.shape(0), "bad number of components in map array");
  {
  py::gil_scoped_release release;
  synthesis_2d(alm, map, spin, lmax, mmax, geometry, nthreads);
  }
  return map_;
  }
py::array Py_synthesis_2d(const py::array &alm, size_t spin, size_t lmax, const string &geometry, const py::object &ntheta, const py::object &nphi, const py::object &mmax_, size_t nthreads, py::object &map)
  {
  size_t mmax = mmax_.is_none() ? lmax : mmax_.cast<size_t>();
  if (isPyarr<complex<float>>(alm))
    return Py2_synthesis_2d<float>(alm, spin, lmax, geometry, ntheta, nphi, mmax, nthreads, map);
  else if (isPyarr<complex<double>>(alm))
    return Py2_synthesis_2d<double>(alm, spin, lmax, geometry, ntheta, nphi, mmax, nthreads, map);
  MR_fail("type matching failed: 'alm' has neither type 'c8' nor 'c16'");
  }
template<typename T> py::array Py2_adjoint_synthesis_2d(
  const py::array &map_, size_t spin, size_t lmax, const string &geometry, size_t mmax, size_t nthreads, py::object &alm__)
  {
  auto map = to_cmav<T,3>(map_);
  auto alm_ = check_build_alm<T>(alm__, map.shape(0), lmax, mmax);
  auto alm = to_vmav<complex<T>,2>(alm_);
  MR_assert(map.shape(0)==alm.shape(0), "bad number of components in map array");
  {
  py::gil_scoped_release release;
  adjoint_synthesis_2d(alm, map, spin, lmax, mmax, geometry, nthreads);
  }
  return alm_;
  }
py::array Py_adjoint_synthesis_2d(
  const py::array &map, size_t spin, size_t lmax, const string &geometry, const py::object &mmax_, size_t nthreads, py::object &alm)
  {
  size_t mmax = mmax_.is_none() ? lmax : mmax_.cast<size_t>();
  if (isPyarr<float>(map))
    return Py2_adjoint_synthesis_2d<float>(map, spin, lmax, geometry, mmax, nthreads, alm);
  else if (isPyarr<double>(map))
    return Py2_adjoint_synthesis_2d<double>(map, spin, lmax, geometry, mmax, nthreads, alm);
  MR_fail("type matching failed: 'alm' has neither type 'c8' nor 'c16'");
  }
template<typename T> py::array Py2_synthesis_2d_deriv1(const py::array &alm_,
  size_t lmax, const string &geometry, const py::object & ntheta,
  const py::object &nphi, size_t mmax, size_t nthreads, py::object &map__)
  {
  auto alm = to_cmav<complex<T>,2>(alm_);
  auto map_ = check_build_map<T>(map__, 2, ntheta, nphi);
  auto map = to_vmav<T,3>(map_);
  MR_assert((map.shape(0)==2) && (alm.shape(0)==1), "incorrect number of components");
  {
  py::gil_scoped_release release;
  synthesis_2d(alm, map, 1, lmax, mmax, geometry, nthreads, ALM2MAP_DERIV1);
  }
  return map_;
  }
py::array Py_synthesis_2d_deriv1(const py::array &alm, size_t lmax, const string &geometry, const py::object &ntheta, const py::object &nphi, const py::object &mmax_, size_t nthreads, py::object &map)
  {
  size_t mmax = mmax_.is_none() ? lmax : mmax_.cast<size_t>();
  if (isPyarr<complex<float>>(alm))
    return Py2_synthesis_2d_deriv1<float>(alm, lmax, geometry, ntheta, nphi, mmax, nthreads, map);
  else if (isPyarr<complex<double>>(alm))
    return Py2_synthesis_2d_deriv1<double>(alm, lmax, geometry, ntheta, nphi, mmax, nthreads, map);
  MR_fail("type matching failed: 'alm' has neither type 'c8' nor 'c16'");
  }

template<typename T> py::array Py2_adjoint_synthesis(py::object &alm__,
  size_t lmax, const py::object &mstart_, ptrdiff_t lstride, 
  const py::array &map_, const py::array &theta_, const py::array &phi0_,
  const py::array &nphi_, const py::array &ringstart_, size_t spin, ptrdiff_t pixstride,
  size_t nthreads)
  {
  auto mstart = get_mstart(lmax, mstart_);
  auto map = to_cmav<T,2>(map_);
  auto theta = to_cmav<double,1>(theta_);
  auto phi0 = to_cmav<double,1>(phi0_);
  auto nphi = to_cmav<size_t,1>(nphi_);
  auto ringstart = to_cmav<size_t,1>(ringstart_);
  vmav<size_t,1> mval(mstart.shape());
  for (size_t i=0; i<mval.shape(0); ++i)
    mval(i) = i;
  auto alm_ = get_optional_Pyarr_minshape<complex<T>>(alm__, {map.shape(0),min_almdim(lmax, mval, mstart, lstride)});
  auto alm = to_vmav<complex<T>,2>(alm_);
  MR_assert(alm.shape(0)==map.shape(0), "bad number of components in a_lm array");
  {
  py::gil_scoped_release release;
  adjoint_synthesis(alm, map, spin, lmax, mstart, lstride, theta, nphi, phi0, ringstart, pixstride, nthreads);
  }
  return alm_;
  }
py::array Py_adjoint_synthesis(const py::array &map, const py::array &theta,
 size_t lmax,
  const py::object &mstart,
  const py::array &nphi,
  const py::array &phi0, const py::array &ringstart, size_t spin, ptrdiff_t lstride, ptrdiff_t pixstride,
  size_t nthreads,
  py::object &alm)
  {
  if (isPyarr<float>(map))
    return Py2_adjoint_synthesis<float>(alm, lmax, mstart, lstride, map, theta,
      phi0, nphi, ringstart, spin, pixstride, nthreads);
  else if (isPyarr<double>(map))
    return Py2_adjoint_synthesis<double>(alm, lmax, mstart, lstride, map, theta,
      phi0, nphi, ringstart, spin, pixstride, nthreads);
  MR_fail("type matching failed: 'alm' has neither type 'c8' nor 'c16'");
  }

template<typename T> py::array Py2_analysis_2d(
  const py::array &map_, size_t spin, size_t lmax, const string &geometry, size_t mmax, size_t nthreads, py::object &alm__)
  {
  auto map = to_cmav<T,3>(map_);
  auto alm_ = check_build_alm<T>(alm__, map.shape(0), lmax, mmax);
  auto alm = to_vmav<complex<T>,2>(alm_);
  MR_assert(map.shape(0)==alm.shape(0), "bad number of components in map array");
  {
  py::gil_scoped_release release;
  analysis_2d(alm, map, spin, lmax, mmax, geometry, nthreads);
  }
  return alm_;
  }
py::array Py_analysis_2d(
  const py::array &map, size_t spin, size_t lmax, const string &geometry, py::object &mmax_, size_t nthreads, py::object &alm)
  {
  size_t mmax = mmax_.is_none() ? lmax : mmax_.cast<size_t>();
  if (isPyarr<float>(map))
    return Py2_analysis_2d<float>(map, spin, lmax, geometry, mmax, nthreads, alm);
  else if (isPyarr<double>(map))
    return Py2_analysis_2d<double>(map, spin, lmax, geometry, mmax, nthreads, alm);
  MR_fail("type matching failed: 'alm' has neither type 'c8' nor 'c16'");
  }

template<typename T> py::array Py2_adjoint_analysis_2d(const py::array &alm_,
  size_t spin, size_t lmax, const string &geometry, const py::object & ntheta,
  const py::object &nphi, size_t mmax, size_t nthreads, py::object &map__)
  {
  auto alm = to_cmav<complex<T>,2>(alm_);
  auto map_ = check_build_map<T>(map__, alm.shape(0), ntheta, nphi);
  auto map = to_vmav<T,3>(map_);
  MR_assert(map.shape(0)==alm.shape(0), "bad number of components in map array");
  {
  py::gil_scoped_release release;
  adjoint_analysis_2d(alm, map, spin, lmax, mmax, geometry, nthreads);
  }
  return map_;
  }
py::array Py_adjoint_analysis_2d(const py::array &alm, size_t spin, size_t lmax, const string &geometry, const py::object &ntheta, const py::object &nphi, const py::object &mmax_, size_t nthreads, py::object &map)
  {
  size_t mmax = mmax_.is_none() ? lmax : mmax_.cast<size_t>();
  if (isPyarr<complex<float>>(alm))
    return Py2_adjoint_analysis_2d<float>(alm, spin, lmax, geometry, ntheta, nphi, mmax, nthreads, map);
  else if (isPyarr<complex<double>>(alm))
    return Py2_adjoint_analysis_2d<double>(alm, spin, lmax, geometry, ntheta, nphi, mmax, nthreads, map);
  MR_fail("type matching failed: 'alm' has neither type 'c8' nor 'c16'");
  }


template<typename T> class Py_sharpjob
  {
  private:
    int64_t lmax_, mmax_, ntheta_, nphi_, nside_, npix_;
    string geom;
    int nthreads;

  public:
    Py_sharpjob () : lmax_(0), mmax_(0), ntheta_(0), nphi_(0), nside_(0),
      npix_(0), nthreads(1) {}

    string repr() const
      {
      return "<sharpjob_d: lmax=" + dataToString(lmax_) +
        ", mmax=" + dataToString(mmax_) + ", npix=", dataToString(npix_) +".>";
      }

    void set_nthreads(int64_t nthreads_)
      { nthreads = int(nthreads_); }
    void set_gauss_geometry(int64_t ntheta, int64_t nphi)
      {
      MR_assert((ntheta>0)&&(nphi>0),"bad grid dimensions");
      geom = "GL";
      ntheta_ = ntheta;
      nphi_ = nphi;
      npix_=ntheta*nphi;
      }
    void set_healpix_geometry(int64_t nside)
      {
      MR_assert(nside>0,"bad Nside value");
      geom = "HP";
      nside_ = nside;
      npix_=12*nside*nside;
      }
    void set_fejer1_geometry(int64_t ntheta, int64_t nphi)
      {
      MR_assert((ntheta>0)&&(nphi>0),"bad grid dimensions");
      geom = "F1";
      ntheta_ = ntheta;
      nphi_ = nphi;
      npix_=ntheta*nphi;
      }
    void set_fejer2_geometry(int64_t ntheta, int64_t nphi)
      {
      MR_assert((ntheta>0)&&(nphi>0),"bad grid dimensions");
      geom = "F2";
      ntheta_ = ntheta;
      nphi_ = nphi;
      npix_=ntheta*nphi;
      }
    void set_cc_geometry(int64_t ntheta, int64_t nphi)
      {
      MR_assert((ntheta>0)&&(nphi>0),"bad grid dimensions");
      geom = "CC";
      ntheta_ = ntheta;
      nphi_ = nphi;
      npix_=ntheta*nphi;
      }
    void set_dh_geometry(int64_t ntheta, int64_t nphi)
      {
      MR_assert((ntheta>0)&&(nphi>0),"bad grid dimensions");
      geom = "DH";
      ntheta_ = ntheta;
      nphi_ = nphi;
      npix_=ntheta*nphi;
      }
    void set_mw_geometry(int64_t ntheta, int64_t nphi)
      {
      MR_assert((ntheta>0)&&(nphi>0),"bad grid dimensions");
      geom = "MW";
      ntheta_ = ntheta;
      nphi_ = nphi;
      npix_=ntheta*nphi;
      }
    void set_triangular_alm_info (int64_t lmax, int64_t mmax)
      {
      MR_assert(mmax>=0,"negative mmax");
      MR_assert(mmax<=lmax,"mmax must not be larger than lmax");
      lmax_=lmax; mmax_=mmax;
      }

    int64_t n_alm() const
      { return ((mmax_+1)*(mmax_+2))/2 + (mmax_+1)*(lmax_-mmax_); }

    py::array alm2map (const py::array_t<complex<double>> &alm_) const
      {
      MR_assert(npix_>0,"no map geometry specified");
      MR_assert (alm_.size()==n_alm(),
        "incorrect size of a_lm array");
      auto map_=make_Pyarr<double>({size_t(npix_)});
      auto map=to_vmav<double,1>(map_);
      auto alm=to_cmav<complex<double>,1>(alm_);
      cmav<complex<double>,2> ar(alm.data(), {1, size_t(n_alm())}, {0, alm.stride(0)});
      if (geom=="HP")
        {
        auto mstart = get_mstart(lmax_, None);
        Healpix_Base2 base(nside_, RING, SET_NSIDE);
        auto nrings = size_t(4*nside_-1);
        auto theta_= make_Pyarr<double>({nrings});
        vmav<double,1> theta({nrings}), phi0({nrings});
        vmav<size_t,1> nphi({nrings}), ringstart({nrings});
        for (size_t r=0, rs=nrings-1; r<=rs; ++r, --rs)
          {
          int64_t startpix, ringpix;
          double ringtheta;
          bool shifted;
          base.get_ring_info2 (r+1, startpix, ringpix, ringtheta, shifted);
          theta(r) = ringtheta;
          theta(rs) = pi-ringtheta;
          nphi(r) = nphi(rs) = size_t(ringpix);
          phi0(r) = phi0(rs) = shifted ? (pi/ringpix) : 0.;
          ringstart(r) = size_t(startpix);
          ringstart(rs) = size_t(base.Npix() - startpix - ringpix);
          }
        vmav<double,2> mr(map.data(), {1, size_t(npix_)}, {0, map.stride(0)});
        synthesis(ar, mr, 0, lmax_, mstart, 1, theta, nphi, phi0, ringstart, 1, nthreads);
        }
      else
        {
        vmav<double,3> mr(map.data(), {1, size_t(ntheta_), size_t(nphi_)}, {0, map.stride(0)*nphi_, map.stride(0)});
        synthesis_2d(ar, mr, 0, lmax_, mmax_, geom, nthreads);
        }
      return map_;
      }
    py::array alm2map_adjoint (const py::array_t<double> &map_) const
      {
      MR_assert(npix_>0,"no map geometry specified");
      MR_assert (map_.size()==npix_,"incorrect size of map array");
      auto alm_=make_Pyarr<complex<double>>({size_t(n_alm())});
      auto alm=to_vmav<complex<double>,1>(alm_);
      vmav<complex<double>,2> ar(alm.data(), {1, size_t(n_alm())}, {0, alm.stride(0)});
      auto map=to_cmav<double,1>(map_);
      if (geom=="HP")
        {
        auto mstart = get_mstart(lmax_, None);
        Healpix_Base2 base(nside_, RING, SET_NSIDE);
        auto nrings = size_t(4*nside_-1);
        auto theta_= make_Pyarr<double>({nrings});
        vmav<double,1> theta({nrings}), phi0({nrings});
        vmav<size_t,1> nphi({nrings}), ringstart({nrings});
        for (size_t r=0, rs=nrings-1; r<=rs; ++r, --rs)
          {
          int64_t startpix, ringpix;
          double ringtheta;
          bool shifted;
          base.get_ring_info2 (r+1, startpix, ringpix, ringtheta, shifted);
          theta(r) = ringtheta;
          theta(rs) = pi-ringtheta;
          nphi(r) = nphi(rs) = size_t(ringpix);
          phi0(r) = phi0(rs) = shifted ? (pi/ringpix) : 0.;
          ringstart(r) = size_t(startpix);
          ringstart(rs) = size_t(base.Npix() - startpix - ringpix);
          }
        cmav<double,2> mr(map.data(), {1, size_t(npix_)}, {0, map.stride(0)});
        adjoint_synthesis(ar, mr, 0, lmax_, mstart, 1, theta, nphi, phi0, ringstart, 1, nthreads);
        }
      else
        {
        cmav<double,3> mr(map.data(), {1, size_t(ntheta_), size_t(nphi_)}, {0, map.stride(0)*nphi_, map.stride(0)});
        adjoint_synthesis_2d(ar, mr, 0, lmax_, mmax_, geom, nthreads);
        }
      return alm_;
      }
    py::array map2alm (const py::array_t<double> &map_) const
      {
      MR_assert(npix_>0,"no map geometry specified");
      MR_assert (map_.size()==npix_,"incorrect size of map array");
      auto alm_=make_Pyarr<complex<double>>({size_t(n_alm())});
      auto alm=to_vmav<complex<double>,1>(alm_);
      vmav<complex<double>,2> ar(alm.data(), {1, size_t(n_alm())}, {0, alm.stride(0)});
      auto map=to_cmav<double,1>(map_);
      cmav<double,3> mr(map.data(), {1, size_t(ntheta_), size_t(nphi_)}, {0, map.stride(0)*nphi_, map.stride(0)});
      analysis_2d(ar, mr, 0, lmax_, mmax_, geom, nthreads);
      return alm_;
      }
    py::array alm2map_spin (const py::array_t<complex<double>> &alm_, int64_t spin) const
      {
      MR_assert(npix_>0,"no map geometry specified");
      auto map_=make_Pyarr<double>({2, size_t(npix_)});
      auto map=to_vmav<double,2>(map_);
      auto alm=to_cmav<complex<double>,2>(alm_);
      MR_assert((alm.shape(0)==2)&&(alm.shape(1)==size_t(n_alm())),
        "incorrect size of a_lm array");
      if (geom=="HP")
        {
        auto mstart = get_mstart(lmax_, None);
        Healpix_Base2 base(nside_, RING, SET_NSIDE);
        auto nrings = size_t(4*nside_-1);
        auto theta_= make_Pyarr<double>({nrings});
        vmav<double,1> theta({nrings}), phi0({nrings});
        vmav<size_t,1> nphi({nrings}), ringstart({nrings});
        for (size_t r=0, rs=nrings-1; r<=rs; ++r, --rs)
          {
          int64_t startpix, ringpix;
          double ringtheta;
          bool shifted;
          base.get_ring_info2 (r+1, startpix, ringpix, ringtheta, shifted);
          theta(r) = ringtheta;
          theta(rs) = pi-ringtheta;
          nphi(r) = nphi(rs) = size_t(ringpix);
          phi0(r) = phi0(rs) = shifted ? (pi/ringpix) : 0.;
          ringstart(r) = size_t(startpix);
          ringstart(rs) = size_t(base.Npix() - startpix - ringpix);
          }
        synthesis(alm, map, spin, lmax_, mstart, 1, theta, nphi, phi0, ringstart, 1, nthreads);
        }
      else
        {
        vmav<double,3> mr(map.data(), {2, size_t(ntheta_), size_t(nphi_)}, {map.stride(0), map.stride(1)*nphi_, map.stride(1)});
        synthesis_2d(alm, mr, spin, lmax_, mmax_, geom, nthreads);
        }
      return map_;
      }
    py::array map2alm_spin (const py::array_t<double> &map_, int64_t spin) const
      {
      MR_assert(npix_>0,"no map geometry specified");
      MR_assert (map_.shape(1)==npix_,"incorrect size of map array");
      auto alm_=make_Pyarr<complex<double>>({2, size_t(n_alm())});
      auto alm=to_vmav<complex<double>,2>(alm_);
      auto map=to_cmav<double,2>(map_);
      cmav<double,3> mr(map.data(), {2, size_t(ntheta_), size_t(nphi_)}, {map.stride(0), map.stride(1)*nphi_, map.stride(1)});
      analysis_2d(alm, mr, spin, lmax_, mmax_, geom, nthreads);
      return alm_;
      }
  };

constexpr const char *sht_DS = R"""(
Python interface for spherical harmonic transforms and manipulation of
spherical harmonic coefficients.

Error conditions are reported by raising exceptions.
)""";
constexpr const char *sht_experimental_DS = R"""(
Experimental interface to the SHT functionality.

Notes
-----

The functionality in this module is not considered to have a stable interface
and also may be moved to other modules in the future. If you use it, be prepared
to adjust your code at some point ion the future!
)""";

constexpr const char *rotate_alm_DS = R"""(
Rotates a set of spherical harmonic coefficients according to the given Euler angles.

Parameters
----------
alm: numpy.ndarray(((lmax+1)*(lmax=2)/2,), dtype=numpy complex64 or numpy.complex128)
    the spherical harmonic coefficients, in the order
    (0,0), (1,0), (2,0), ... (lmax,0), (1,1), (2,1), ..., (lmax, lmax)
lmax : int >= 0
    Maximum multipole order l of the data set.
psi : float
    First rotation angle about the z-axis. All angles are in radians,
    the rotations are active and the referential system is assumed to be
    right handed.
theta : float
    Second rotation angl about the original (unrotated) y-axis
phi : float
    Third rotation angle about the original (unrotated) z-axis.
nthreads: int >= 0
    the number of threads to use for the computation
    if 0, use as many threads as there are hardware threads available on the system

Returns
-------
numpy.ndarray(same shape and dtype as alm)
)""";

constexpr const char *alm2leg_DS = R"""(
Transforms a set of spherical harmonic coefficients to Legendre coefficients
dependent on theta and m.

Parameters
----------
alm: numpy.ndarray((ncomp, x), dtype=numpy.complex64 or numpy.complex128)
    the set of spherical harmonic coefficients.
    ncomp must be 1 if spin is 0, else 2.
    The second dimension must be large enough to accommodate all entries, which
    are stored according to the parameters `lmax`, 'mval`, `mstart`, and `lstride`.
leg: None or numpy.ndarray((ncomp, ntheta, nm), same dtype as `alm`)
    output array containing the Legendre coefficients
    if `None`, a new suitable array is allocated
spin: int >= 0
    the spin to use for the transform
    if spin==0, ncomp must be 1, otherwise 2
lmax: int >= 0
    the maximum l moment of the transform (inclusive)
mval: numpy.ndarray((nm,), dtype = numpy.uint64)
    the m moments for which the transform should be carried out
    entries must be unique and <= lmax
mstart: numpy.ndarray((nm,), dtype = numpy.uint64)
    the (hypothetical) index in the second dimension of `alm` on which the
    entry with l=0, m=mval[mi] would be stored, for mi in mval
lstride: int
    the index stride in the second dimension of `alm` between the entries for
    `l` and `l+1`, but the same `m`.
theta: numpy.ndarray((ntheta,), dtype=numpy.float64)
    the colatitudes of the map rings
nthreads: int >= 0
    the number of threads to use for the computation
    if 0, use as many threads as there are hardware threads available on the system

Returns
-------
numpy.ndarray((ncomp, ntheta, nm), same dtype as `alm`)
    the Legendre coefficients. If `leg` was supplied, this will be the same object.
)""";

constexpr const char *alm2leg_deriv1_DS = R"""(
Transforms a set of spin-0 spherical harmonic coefficients to Legendre
coefficients of the first derivatives with respect to colatiude and longitude,
dependent on theta and m.

Parameters
----------
alm: numpy.ndarray((1, x), dtype=numpy.complex64 or numpy.complex128)
    the set of spherical harmonic coefficients.
    The second dimension must be large enough to accommodate all entries, which
    are stored according to the parameters `lmax`, 'mval`, `mstart`, and `lstride`.
leg: None or numpy.ndarray((2, ntheta, nm), same dtype as `alm`)
    output array containing the Legendre coefficients
    if `None`, a new suitable array is allocated
lmax: int >= 0
    the maximum l moment of the transform (inclusive)
mval: numpy.ndarray((nm,), dtype = numpy.uint64)
    the m moments for which the transform should be carried out
    entries must be unique and <= lmax
mstart: numpy.ndarray((nm,), dtype = numpy.uint64)
    the (hypothetical) index in the second dimension of `alm` on which the
    entry with l=0, m=mval[mi] would be stored, for mi in mval
lstride: int
    the index stride in the second dimension of `alm` between the entries for
    `l` and `l+1`, but the same `m`.
theta: numpy.ndarray((ntheta,), dtype=numpy.float64)
    the colatitudes of the map rings
nthreads: int >= 0
    the number of threads to use for the computation
    if 0, use as many threads as there are hardware threads available on the system

Returns
-------
numpy.ndarray((2, ntheta, nm), same dtype as `alm`)
    the Legendre coefficients. If `leg` was supplied, this will be the same object.
    The first component contains coefficients representing a map of df/dtheta;
    the second component those of 1./sin(theta) df/dphi.
)""";

constexpr const char *leg2alm_DS = R"""(
Transforms a set of Legendre coefficients to spherical harmonic coefficients

Parameters
----------
leg: numpy.ndarray((ncomp, ntheta, nm), dtype=numpy.complex64 or numpy.complex128)
    ncomp must be 1 if spin is 0, else 2
alm: None or numpy.ndarray((ncomp, x), same dtype as `leg`)
    the set of spherical harmonic coefficients.
    The second dimension must be large enough to accommodate all entries, which
    are stored according to the parameters `lmax`, 'mval`, `mstart`, and `lstride`.
    if `None`, a new suitable array is allocated
spin: int >= 0
    the spin to use for the transform
    if spin==0, ncomp must be 1, otherwise 2
lmax: int >= 0
    the maximum l moment of the transform (inclusive)
mval: numpy.ndarray((nm,), dtype = numpy.uint64)
    the m moments for which the transform should be carried out
    entries must be unique and <= lmax
mstart: numpy.ndarray((nm,), dtype = numpy.uint64)
    the (hypothetical) index in the second dimension of `alm` on which the
    entry with l=0, m=mval[mi] would be stored, for mi in mval
lstride: int
    the index stride in the second dimension of `alm` between the entries for
    `l` and `l+1`, but the same `m`.
theta: numpy.ndarray((ntheta,), dtype=numpy.float64)
    the colatitudes of the map rings
nthreads: int >= 0
    the number of threads to use for the computation
    if 0, use as many threads as there are hardware threads available on the system

Returns
-------
numpy.ndarray((ncomp, *), same dtype as `leg`)
    the Legendre coefficients.
    if `alm` was supplied, this will be the same object
    If newly allocated, the smallest possible second dimensions will be chosen.
)""";

constexpr const char *map2leg_DS = R"""(
Transforms a map or several maps to Legendre coefficients
dependent on theta and m.

Parameters
----------
map: numpy.ndarray((ncomp, x), dtype=numpy.float32 or numpy.float64)
    the map pixel data.
    The second dimension must be large enough to accommodate all pixels, which
    are stored according to the parameters `nphi`, 'ringstart`, and `pixstride`.
leg: None or numpy.ndarray((ncomp, ntheta, mmax+1), dtype=numpy.complex of same accuracy as `map`)
    output array containing the Legendre coefficients
    if `None`, a new suitable array is allocated
nphi: numpy.ndarray((ntheta,), dtype=numpy.uint64)
    number of pixels in every ring
phi0: numpy.ndarray((ntheta,), dtype=numpy.float64)
    azimuth (in radians) of the first pixel in every ring
ringstart: numpy.ndarray((ntheta,), dtype=numpy.uint64)
    the index in the second dimension of `map` at which the first pixel of every
    ring is stored
pixstride: int
    the index stride in the second dimension of `map` between two subsequent
    pixels in a ring
mmax: int
    the maximum m moment to compute in this transform. If `leg`
    is provided, `mmax` must be equal to `leg.shape[2]=1`.
nthreads: int >= 0
    the number of threads to use for the computation
    if 0, use as many threads as there are hardware threads available on the system

Returns
-------
numpy.ndarray((ncomp, ntheta, nm), dtype=numpy.complex of same accuracy as `map`)
    the Legendre coefficients
    if `leg` was supplied, this will be the same object

Notes
-----
In contrast to `leg2alm` and `alm2leg` the `m` values are assumed to form a
range from 0 to mmax, inclusively. 
)""";

constexpr const char *leg2map_DS = R"""(
Transforms one or more sets of Legendre coefficients to maps.

Parameters
----------
leg: numpy.ndarray((ncomp, ntheta, mmax+1), numppy.complex64 or numpy.complex128)
    input array containing the Legendre coefficients
map: None or numpy.ndarray((ncomp, x), dtype=numpy.float of same accuracy as `leg`
    the map pixel data.
    The second dimension must be large enough to accommodate all pixels, which
    are stored according to the parameters `nphi`, 'ringstart`, and `pixstride`.
    if `None`, a new suitable array is allocated
nphi: numpy.ndarray((ntheta,), dtype=numpy.uint64)
    number of pixels in every ring
phi0: numpy.ndarray((ntheta,), dtype=numpy.float64)
    azimuth (in radians) of the first pixel in every ring
ringstart: numpy.ndarray((ntheta,), dtype=numpy.uint64)
    the index in the second dimension of `map` at which the first pixel of every
    ring is stored
pixstride: int
    the index stride in the second dimension of `map` between two subsequent
    pixels in a ring
nthreads: int >= 0
    the number of threads to use for the computation
    if 0, use as many threads as there are hardware threads available on the system

Returns
-------
numpy.ndarray((ncomp, x), dtype=numpy.float of same accuracy as `leg`)
    the map pixel data.
    If `map` was supplied, this will be the same object
    If newly allocated, the smallest possible second dimensions will be chosen.

Notes
-----
In contrast to `leg2alm` and `alm2leg` the `m` values are assumed to form a
range from 0 to mmax, inclusively. 
)""";

constexpr const char *synthesis_2d_DS = R"""(
Transforms one or two sets of spherical harmonic coefficients to 2D maps.

Parameters
----------
alm: numpy.ndarray((ncomp, x), dtype=numpy.complex64 or numpy.complex128)
    the set of spherical harmonic coefficients.
    The second dimension must be large enough to accommodate all entries, which
    are stored according to the healpy convention.
map: numpy.ndarray((ncomp, ntheta, nphi), dtype=numpy.float of same accuracy as alm)
    storage for the output map.
    If not supplied, a new array is allocated.
ntheta, nphi: int > 0
    dimensions of the output map.
    If not supplied, `map` must be supplied.
    If supplied, and `map` is also supplied, must match with the array dimensions
spin: int >= 0
    the spin to use for the transform.
    If spin==0, ncomp must be 1, otherwise 2
lmax: int >= 0
    the maximum l moment of the transform (inclusive).
mmax: int >= 0 and <= lmax
    the maximum m moment of the transform (inclusive).
    If not supplied, mmax is assumed to be equal to lmax
geometry: one of "CC", "F1", "MW", "MWflip", "GL", "DH", "F2"
    the distribution of rings over the theta range
        - CC: Clenshaw-Curtis, equidistant, first and last ring on poles
        - F1: Fejer's first rule, equidistant, first and last ring half a ring
          width from the poles
        - MW: McEwen & Wiaux scheme, equidistant, first ring half a ring width from
          the north pole, last ring on the south pole
        - MWflip: flipped McEwen & Wiaux scheme, equidistant, first ring on the
          north pole, last ring half a ring width from the south pole
        - GL: Gauss-Legendre, non-equidistant
        - DH: Driscoll-Healy, equidistant, first ring on north pole, last ring one
          ring width from the south pole
        - F2: Fejer's second rule, equidistant, first and last ring one ring width
          from the poles.
nthreads: int >= 0
    the number of threads to use for the computation.
    If 0, use as many threads as there are hardware threads available on the system

Returns
-------
numpy.ndarray((ncomp, ntheta, nphi), dtype=numpy.float of same accuracy as alm)
    the computed map. If the map parameter was specified, this is identical with
    map.
)""";

constexpr const char *synthesis_2d_deriv1_DS = R"""(
Transforms a set of spherical harmonic coefficients to two 2D maps containing
the derivatives with respect to theta and phi.

Parameters
----------
alm: numpy.ndarray((1, x), dtype=numpy.complex64 or numpy.complex128)
    the set of spherical harmonic coefficients.
    The second dimension must be large enough to accommodate all entries, which
    are stored according to the healpy convention.
map: numpy.ndarray((2, ntheta, nphi), dtype=numpy.float of same accuracy as alm)
    storage for the output map.
    If not supplied, a new array is allocated.
ntheta, nphi: int > 0
    dimensions of the output map.
    If not supplied, `map` must be supplied.
    If supplied, and `map` is also supplied, must match with the array dimensions
lmax: int >= 0
    the maximum l (and m) moment of the transform (inclusive)
mmax: int >= 0 and <= lmax
    the maximum m moment of the transform (inclusive).
    If not supplied, mmax is assumed to be equal to lmax
geometry: one of "CC", "F1", "MW", "MWflip", "GL", "DH", "F2"
    the distribution of rings over the theta range
        - CC: Clenshaw-Curtis, equidistant, first and last ring on poles
        - F1: Fejer's first rule, equidistant, first and last ring half a ring
          width from the poles
        - MW: McEwen & Wiaux scheme, equidistant, first ring half a ring width from
          the north pole, last ring on the south pole
        - MWflip: flipped McEwen & Wiaux scheme, equidistant, first ring on the
          north pole, last ring half a ring width from the south pole
        - GL: Gauss-Legendre, non-equidistant
        - DH: Driscoll-Healy, equidistant, first ring on north pole, last ring one
          ring width from the south pole
        - F2: Fejer's second rule, equidistant, first and last ring one ring width
          from the poles.
nthreads: int >= 0
    the number of threads to use for the computation.
    If 0, use as many threads as there are hardware threads available on the system

Returns
-------
numpy.ndarray((ncomp, ntheta, nphi), dtype=numpy.float of same accuracy as alm)
    the maps containing the derivatives with respect to theta and phi.
    If the map parameter was specified, this is identical with map.
)""";

constexpr const char *adjoint_synthesis_2d_DS = R"""(
Transforms one or two 2D maps to spherical harmonic coefficients.
This is the adjoint operation of `synthesis_2D`.

Parameters
----------
alm: numpy.ndarray((ncomp, x), dtype=numpy.complex64 or numpy.complex128)
    storage for the spherical harmonic coefficients.
    The second dimension must be large enough to accommodate all entries, which
    are stored according to the healpy convention.
    If not supplied, a new array is allocated.
map: numpy.ndarray((ncomp, ntheta, nphi), dtype=numpy.float of same accuracy as alm)
    The input map.
spin: int >= 0
    the spin to use for the transform.
    If spin==0, ncomp must be 1, otherwise 2
lmax: int >= 0
    the maximum l (and m) moment of the transform (inclusive)
mmax: int >= 0 and <= lmax
    the maximum m moment of the transform (inclusive).
    If not supplied, mmax is assumed to be equal to lmax
geometry: one of "CC", "F1", "MW", "MWflip", "GL", "DH", "F2"
    the distribution of rings over the theta range
        - CC: Clenshaw-Curtis, equidistant, first and last ring on poles
        - F1: Fejer's first rule, equidistant, first and last ring half a ring
          width from the poles
        - MW: McEwen & Wiaux scheme, equidistant, first ring half a ring width from
          the north pole, last ring on the south pole
        - MWflip: flipped McEwen & Wiaux scheme, equidistant, first ring on the
          north pole, last ring half a ring width from the south pole
        - GL: Gauss-Legendre, non-equidistant
        - DH: Driscoll-Healy, equidistant, first ring on north pole, last ring one
          ring width from the south pole
        - F2: Fejer's second rule, equidistant, first and last ring one ring width
          from the poles.
nthreads: int >= 0
    the number of threads to use for the computation.
    If 0, use as many threads as there are hardware threads available on the system

Returns
-------
numpy.ndarray((ncomp, x), dtype=numpy.complex64 or numpy.complex128)
    the computed spherical harmonic coefficients
    If the `alm` parameter was specified, this is identical to `alm`.
)""";

constexpr const char *analysis_2d_DS = R"""(
Transforms one or two 2D maps to spherical harmonic coefficients.
This is the inverse operation of `synthesis_2D`.

Parameters
----------
alm: numpy.ndarray((ncomp, x), dtype=numpy.complex64 or numpy.complex128)
    storage for the spherical harmonic coefficients.
    The second dimension must be large enough to accommodate all entries, which
    are stored according to the healpy convention
    If not supplied, a new array is allocated.
map: numpy.ndarray((ncomp, ntheta, nphi), dtype=numpy.float of same accuracy as alm)
    The input map.
spin: int >= 0
    the spin to use for the transform.
    If spin==0, ncomp must be 1, otherwise 2
lmax: int >= 0
    the maximum l (and m) moment of the transform (inclusive)
mmax: int >= 0 and <= lmax
    the maximum m moment of the transform (inclusive).
    If not supplied, mmax is assumed to be equal to lmax
geometry: one of "CC", "F1", "MW", "MWflip", "GL", "DH", "F2"
    the distribution of rings over the theta range
        - CC: Clenshaw-Curtis, equidistant, first and last ring on poles
        - F1: Fejer's first rule, equidistant, first and last ring half a ring
          width from the poles
        - MW: McEwen & Wiaux scheme, equidistant, first ring half a ring width from
          the north pole, last ring on the south pole
        - MWflip: flipped McEwen & Wiaux scheme, equidistant, first ring on the
          north pole, last ring half a ring width from the south pole
        - GL: Gauss-Legendre, non-equidistant
        - DH: Driscoll-Healy, equidistant, first ring on north pole, last ring one
          ring width from the south pole
        - F2: Fejer's second rule, equidistant, first and last ring one ring width
          from the poles.
nthreads: int >= 0
    the number of threads to use for the computation.
    If 0, use as many threads as there are hardware threads available on the system

Returns
-------
numpy.ndarray((ncomp, x), dtype=numpy.complex64 or numpy.complex128)
    the computed spherical harmonic coefficients
    If the `alm` parameter was specified, this is identical to `alm`.

Notes
-----
The maximum ``m`` moment to which this function can analyze its input map is
``min(lmax, (nphi-1)//2)``.

The maximum ``l``  moment to which this function can analyze its input map
depends on the geometry, and is

    - ``ntheta-2`` for CC
    - ``ntheta-1`` for F1, MW, MWflip, and GL
    - ``(ntheta-2)//2`` for DH
    - ``(ntheta-1)//2`` for F2

For the CC and F1 geometries this limit is considerably higher than the one
obtainable by simply applying quadrature weights. This improvement is achieved
by temporary upsampling along meridians to apply the weights at a higher
resolution.
)""";

constexpr const char *synthesis_DS = R"""(
Transforms one or two sets of spherical harmonic coefficients to maps on the sphere.

Parameters
----------
alm: numpy.ndarray((ncomp, x), dtype=numpy.complex64 or numpy.complex128)
    the set of spherical harmonic coefficients.
    The second dimension must be large enough to accommodate all entries, which
    are stored according to the healpy convention.
map: None or numpy.ndarray((ncomp, x), dtype=numpy.float of same accuracy as `alm`
    the map pixel data.
    The second dimension must be large enough to accommodate all pixels, which
    are stored according to the parameters `nphi`, 'ringstart`, and `pixstride`.
    if `None`, a new suitable array is allocated
theta: numpy.ndarray((ntheta,), dtype=numpy.float64)
    the colatitudes of the map rings
nphi: numpy.ndarray((ntheta,), dtype=numpy.uint64)
    number of pixels in every ring
phi0: numpy.ndarray((ntheta,), dtype=numpy.float64)
    azimuth (in radians) of the first pixel in every ring
mstart: numpy.ndarray((mmax+1,), dtype = numpy.uint64)
    the (hypothetical) index in the second dimension of `alm` on which the
    entry with (l=0, m) would be stored. If not supplied, a contiguous storage
    scheme in the order m=0,1,2,... is assumed.
ringstart: numpy.ndarray((ntheta,), dtype=numpy.uint64)
    the index in the second dimension of `map` at which the first pixel of every
    ring is stored
lstride: int
    the index stride in the second dimension of `alm` between the entries for
    `l` and `l+1`, but the same `m`.
pixstride: int
    the index stride in the second dimension of `map` between two subsequent
    pixels in a ring
nthreads: int >= 0
    the number of threads to use for the computation
    if 0, use as many threads as there are hardware threads available on the system
spin: int >= 0
    the spin to use for the transform.
    If spin==0, ncomp must be 1, otherwise 2
lmax: int >= 0
    the maximum l moment of the transform (inclusive).

Returns
-------
numpy.ndarray((ncomp, x), dtype=numpy.float of same accuracy as `alm`)
    the map pixel data.
    If `map` was supplied, this will be the same object
    If newly allocated, the smallest possible second dimensions will be chosen.
)""";

constexpr const char *adjoint_synthesis_DS = R"""(
Transforms one or two maps to spherical harmonic coefficients.
This is the adjoint operation of `synthesis`.

Parameters
----------
alm: None or numpy.ndarray((ncomp, x), dtype=numpy.complex of same precision as `map`)
    the set of spherical harmonic coefficients.
    The second dimension must be large enough to accommodate all entries, which
    are stored according to the healpy convention.
    if `None`, a new suitable array is allocated
map: numpy.ndarray((ncomp, x), dtype=numpy.float32 or numpy.float64
    The second dimension must be large enough to accommodate all pixels, which
    are stored according to the parameters `nphi`, 'ringstart`, and `pixstride`.
theta: numpy.ndarray((ntheta,), dtype=numpy.float64)
    the colatitudes of the map rings
nphi: numpy.ndarray((ntheta,), dtype=numpy.uint64)
    number of pixels in every ring
phi0: numpy.ndarray((ntheta,), dtype=numpy.float64)
    azimuth (in radians) of the first pixel in every ring
mstart: numpy.ndarray((mmax+1,), dtype = numpy.uint64)
    the (hypothetical) index in the second dimension of `alm` on which the
    entry with (l=0, m) would be stored. If not supplied, a contiguous storage
    scheme in the order m=0,1,2,... is assumed.
ringstart: numpy.ndarray((ntheta,), dtype=numpy.uint64)
    the index in the second dimension of `map` at which the first pixel of every
    ring is stored
lstride: int
    the index stride in the second dimension of `alm` between the entries for
    `l` and `l+1`, but the same `m`.
pixstride: int
    the index stride in the second dimension of `map` between two subsequent
    pixels in a ring
nthreads: int >= 0
    the number of threads to use for the computation
    if 0, use as many threads as there are hardware threads available on the system
spin: int >= 0
    the spin to use for the transform.
    If spin==0, ncomp must be 1, otherwise 2
lmax: int >= 0
    the maximum l moment of the transform (inclusive).

Returns
-------
numpy.ndarray((ncomp, x), dtype=numpy.complex of same accuracy as `map`)
    the set of spherical harmonic coefficients.
    The second dimension must be large enough to accommodate all entries, which
    are stored according to the healpy convention.
    If newly allocated, the smallest possible second dimensions will be chosen.
)""";

constexpr const char *synthesis_deriv1_DS = R"""(
Transforms a set of spherical harmonic coefficients to two maps containing
the derivatives with respect to theta and phi.

Parameters
----------
alm: numpy.ndarray((1, x), dtype=numpy.complex64 or numpy.complex128)
    the set of spherical harmonic coefficients.
    The second dimension must be large enough to accommodate all entries, which
    are stored according to the healpy convention.
map: None or numpy.ndarray((2, x), dtype=numpy.float of same accuracy as `alm`
    the map pixel data.
    The second dimension must be large enough to accommodate all pixels, which
    are stored according to the parameters `nphi`, 'ringstart`, and `pixstride`.
    if `None`, a new suitable array is allocated
theta: numpy.ndarray((ntheta,), dtype=numpy.float64)
    the colatitudes of the map rings
nphi: numpy.ndarray((ntheta,), dtype=numpy.uint64)
    number of pixels in every ring
phi0: numpy.ndarray((ntheta,), dtype=numpy.float64)
    azimuth (in radians) of the first pixel in every ring
mstart: numpy.ndarray((mmax+1,), dtype = numpy.uint64)
    the (hypothetical) index in the second dimension of `alm` on which the
    entry with (l=0, m) would be stored. If not supplied, a contiguous storage
    scheme in the order m=0,1,2,... is assumed.
ringstart: numpy.ndarray((ntheta,), dtype=numpy.uint64)
    the index in the second dimension of `map` at which the first pixel of every
    ring is stored
lstride: int
    the index stride in the second dimension of `alm` between the entries for
    `l` and `l+1`, but the same `m`.
pixstride: int
    the index stride in the second dimension of `map` between two subsequent
    pixels in a ring
nthreads: int >= 0
    the number of threads to use for the computation
    if 0, use as many threads as there are hardware threads available on the system
lmax: int >= 0
    the maximum l moment of the transform (inclusive).

Returns
-------
numpy.ndarray((2, x), dtype=numpy.float of same accuracy as `alm`)
    the map pixel data.
    If `map` was supplied, this will be the same object
    If newly allocated, the smallest possible second dimensions will be chosen.
)""";

constexpr const char *sharpjob_d_DS = R"""(
Interface class to some of libsharp2's functionality.

Notes
-----
This class is considered obsolescent and will be removed in the future.
)""";

void add_sht(py::module_ &msup)
  {
  using namespace pybind11::literals;
  auto m = msup.def_submodule("sht");
  m.doc() = sht_DS;
  auto m2 = m.def_submodule("experimental");
  m2.doc() = sht_experimental_DS;

  m2.def("synthesis", &Py_synthesis, synthesis_DS, py::kw_only(), "alm"_a, "theta"_a,
    "lmax"_a, "mstart"_a=None, "nphi"_a, "phi0"_a, "ringstart"_a, "spin"_a,
    "lstride"_a=1, "pixstride"_a=1, "nthreads"_a=1, "map"_a=None);
  m2.def("adjoint_synthesis", &Py_adjoint_synthesis, adjoint_synthesis_DS, py::kw_only(), "map"_a, "theta"_a,
    "lmax"_a, "mstart"_a=None, "nphi"_a, "phi0"_a, "ringstart"_a, "spin"_a,
    "lstride"_a=1, "pixstride"_a=1, "nthreads"_a=1, "alm"_a=None);
  m2.def("synthesis_deriv1", &Py_synthesis_deriv1, synthesis_deriv1_DS, py::kw_only(), "alm"_a, "theta"_a,
    "lmax"_a, "mstart"_a=None, "nphi"_a, "phi0"_a, "ringstart"_a,
    "lstride"_a=1, "pixstride"_a=1, "nthreads"_a=1, "map"_a=None);

  m2.def("synthesis_2d", &Py_synthesis_2d, synthesis_2d_DS, py::kw_only(), "alm"_a, "spin"_a, "lmax"_a, "geometry"_a, "ntheta"_a=None, "nphi"_a=None, "mmax"_a=None, "nthreads"_a=1, "map"_a=None);
  m2.def("adjoint_synthesis_2d", &Py_adjoint_synthesis_2d, adjoint_synthesis_2d_DS, py::kw_only(), "map"_a, "spin"_a, "lmax"_a, "geometry"_a, "mmax"_a=None, "nthreads"_a=1, "alm"_a=None);
  m2.def("synthesis_2d_deriv1", &Py_synthesis_2d_deriv1, synthesis_2d_deriv1_DS, py::kw_only(), "alm"_a, "lmax"_a, "geometry"_a, "ntheta"_a=None, "nphi"_a=None, "mmax"_a=None, "nthreads"_a=1, "map"_a=None);
  m2.def("analysis_2d", &Py_analysis_2d, analysis_2d_DS, py::kw_only(), "map"_a, "spin"_a, "lmax"_a, "geometry"_a, "mmax"_a=None, "nthreads"_a=1, "alm"_a=None);
  m2.def("adjoint_analysis_2d", &Py_adjoint_analysis_2d, py::kw_only(), "alm"_a, "spin"_a, "lmax"_a, "geometry"_a, "ntheta"_a=None, "nphi"_a=None, "mmax"_a=None, "nthreads"_a=1, "map"_a=None);

  m2.def("GL_weights",&Py_GL_weights, "nlat"_a, "nlon"_a);
  m2.def("GL_thetas",&Py_GL_thetas, "nlat"_a);
  m2.def("get_gridweights", &Py_get_gridweights, "type"_a, "ntheta"_a);
  m2.def("alm2leg", &Py_alm2leg, alm2leg_DS, py::kw_only(), "alm"_a, "lmax"_a, "theta"_a, "spin"_a=0, "mval"_a=None, "mstart"_a=None, "lstride"_a=1, "nthreads"_a=1, "leg"_a=None);
  m2.def("alm2leg_deriv1", &Py_alm2leg_deriv1, alm2leg_deriv1_DS, py::kw_only(), "alm"_a, "lmax"_a, "theta"_a, "mval"_a=None, "mstart"_a=None, "lstride"_a=1, "nthreads"_a=1, "leg"_a=None);
  m2.def("leg2alm", &Py_leg2alm, leg2alm_DS, py::kw_only(), "leg"_a, "lmax"_a, "theta"_a, "spin"_a=0, "mval"_a=None, "mstart"_a=None, "lstride"_a=1, "nthreads"_a=1, "alm"_a=None);
  m2.def("map2leg", &Py_map2leg, map2leg_DS, py::kw_only(), "map"_a, "nphi"_a, "phi0"_a, "ringstart"_a, "mmax"_a, "pixstride"_a=1, "nthreads"_a=1, "leg"_a=None);
  m2.def("leg2map", &Py_leg2map, leg2map_DS, py::kw_only(), "leg"_a, "nphi"_a, "phi0"_a, "ringstart"_a, "pixstride"_a=1, "nthreads"_a=1, "map"_a=None);
  m.def("rotate_alm", &Py_rotate_alm, rotate_alm_DS, "alm"_a, "lmax"_a, "psi"_a, "theta"_a,
    "phi"_a, "nthreads"_a=1);

  py::class_<Py_sharpjob<double>> (m, "sharpjob_d", py::module_local(),sharpjob_d_DS)
    .def(py::init<>())
    .def("set_nthreads", &Py_sharpjob<double>::set_nthreads, "nthreads"_a)
    .def("set_gauss_geometry", &Py_sharpjob<double>::set_gauss_geometry,
      "ntheta"_a,"nphi"_a)
    .def("set_healpix_geometry", &Py_sharpjob<double>::set_healpix_geometry,
      "nside"_a)
    .def("set_fejer1_geometry", &Py_sharpjob<double>::set_fejer1_geometry,
      "ntheta"_a, "nphi"_a)
    .def("set_fejer2_geometry", &Py_sharpjob<double>::set_fejer2_geometry,
      "ntheta"_a, "nphi"_a)
    .def("set_cc_geometry", &Py_sharpjob<double>::set_cc_geometry,
      "ntheta"_a, "nphi"_a)
    .def("set_dh_geometry", &Py_sharpjob<double>::set_dh_geometry,
      "ntheta"_a, "nphi"_a)
    .def("set_mw_geometry", &Py_sharpjob<double>::set_mw_geometry,
      "ntheta"_a, "nphi"_a)
    .def("set_triangular_alm_info",
      &Py_sharpjob<double>::set_triangular_alm_info, "lmax"_a, "mmax"_a)
    .def("n_alm", &Py_sharpjob<double>::n_alm)
    .def("alm2map", &Py_sharpjob<double>::alm2map,"alm"_a)
    .def("alm2map_adjoint", &Py_sharpjob<double>::alm2map_adjoint,"map"_a)
    .def("map2alm", &Py_sharpjob<double>::map2alm,"map"_a)
    .def("alm2map_spin", &Py_sharpjob<double>::alm2map_spin,"alm"_a,"spin"_a)
    .def("map2alm_spin", &Py_sharpjob<double>::map2alm_spin,"map"_a,"spin"_a)
    .def("__repr__", &Py_sharpjob<double>::repr);
  }

}

using detail_pymodule_sht::add_sht;

}

