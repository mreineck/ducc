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

/*! \file sharp_almhelpers.cc
 *  Spherical transform library
 *
 *  Copyright (C) 2008-2020 Max-Planck-Society
 *  \author Martin Reinecke
 */

#include "ducc0/infra/error_handling.h"
#include "ducc0/sharp/sharp_almhelpers.h"

namespace ducc0 {

namespace detail_sharp {

using namespace std;

using dcmplx = complex<double>;
using fcmplx = complex<float>;

sharp_standard_alm_info::sharp_standard_alm_info (size_t lmax__, size_t nm_, ptrdiff_t stride_,
  const size_t *mval__, const ptrdiff_t *mstart)
  : lmax_(lmax__), mval_(nm_), mvstart(nm_), stride(stride_)
  {
  for (size_t mi=0; mi<nm_; ++mi)
    {
    mval_[mi] = mval__[mi];
    mvstart[mi] = mstart[mi];
    }
  }

sharp_standard_alm_info::sharp_standard_alm_info (size_t lmax__, size_t mmax_, ptrdiff_t stride_,
  const ptrdiff_t *mstart)
  : lmax_(lmax__), mval_(mmax_+1), mvstart(mmax_+1), stride(stride_)
  {
  for (size_t i=0; i<=mmax_; ++i)
    {
    mval_[i]=i;
    mvstart[i] = mstart[i];
    }
  }

template<typename T> void sharp_standard_alm_info::tclear (T *alm) const
  {
  for (size_t mi=0;mi<mval_.size();++mi)
    for (size_t l=mval_[mi];l<=lmax_;++l)
      reinterpret_cast<T *>(alm)[mvstart[mi]+ptrdiff_t(l)*stride]=0.;
  }
void sharp_standard_alm_info::clear_alm(const any &alm) const
  {
  if (alm.type()==typeid(dcmplx *)) tclear(any_cast<dcmplx *>(alm));
  else if (alm.type()==typeid(fcmplx *)) tclear(any_cast<fcmplx *>(alm));
  else MR_fail("bad a_lm data type");
  }
template<typename T> void sharp_standard_alm_info::tget(size_t mi, const T *alm, mav<dcmplx,1> &almtmp) const
  {
  for (auto l=mval_[mi]; l<=lmax_; ++l)
    almtmp.v(l) = alm[mvstart[mi]+ptrdiff_t(l)*stride];
  }
void sharp_standard_alm_info::get_alm(size_t mi, const any &alm, mav<dcmplx,1> &almtmp) const
  {
  if (alm.type()==typeid(dcmplx *)) tget(mi, any_cast<dcmplx *>(alm), almtmp);
  else if (alm.type()==typeid(const dcmplx *)) tget(mi, any_cast<const dcmplx *>(alm), almtmp);
  else if (alm.type()==typeid(fcmplx *)) tget(mi, any_cast<fcmplx *>(alm), almtmp);
  else if (alm.type()==typeid(const fcmplx *)) tget(mi, any_cast<const fcmplx *>(alm), almtmp);
  else MR_fail("bad a_lm data type");
  }
template<typename T> void sharp_standard_alm_info::tadd(size_t mi, const mav<dcmplx,1> &almtmp, T *alm) const
  {
  for (auto l=mval_[mi]; l<=lmax_; ++l)
    alm[mvstart[mi]+ptrdiff_t(l)*stride] += T(almtmp(l));
  }
void sharp_standard_alm_info::add_alm(size_t mi, const mav<dcmplx,1> &almtmp, const any &alm) const
  {
  if (alm.type()==typeid(dcmplx *)) tadd(mi, almtmp, any_cast<dcmplx *>(alm));
  else if (alm.type()==typeid(fcmplx *)) tadd(mi, almtmp, any_cast<fcmplx *>(alm));
  else MR_fail("bad a_lm data type");
  }

ptrdiff_t sharp_standard_alm_info::index (size_t l, size_t mi)
  { return mvstart[mi]+stride*ptrdiff_t(l); }
/* This currently requires all m values from 0 to nm-1 to be present.
   It might be worthwhile to relax this criterion such that holes in the m
   distribution are permissible. */
size_t sharp_standard_alm_info::mmax() const
  {
  //FIXME: if gaps are allowed, we have to search the maximum m in the array
  auto nm_=mval_.size();
  vector<bool> mcheck(nm_,false);
  for (auto m_cur : mval_)
    {
    MR_assert(m_cur<nm_, "not all m values are present");
    MR_assert(mcheck[m_cur]==false, "duplicate m value");
    mcheck[m_cur]=true;
    }
  return nm_-1;
  }

unique_ptr<sharp_standard_alm_info> sharp_make_triangular_alm_info (size_t lmax, size_t mmax, ptrdiff_t stride)
  {
  vector<ptrdiff_t> mvstart(mmax+1);
  size_t tval = 2*lmax+1;
  for (size_t m=0; m<=mmax; ++m)
    mvstart[m] = stride*ptrdiff_t((m*(tval-m))>>1);
  return make_unique<sharp_standard_alm_info>(lmax, mmax, stride, mvstart.data());
  }

unique_ptr<sharp_standard_alm_info> sharp_make_rectangular_alm_info (size_t lmax, size_t mmax, ptrdiff_t stride)
  {
  vector<ptrdiff_t> mvstart(mmax+1);
  for (size_t m=0; m<=mmax; ++m)
    mvstart[m] = stride*ptrdiff_t(m*(lmax+1));
  return make_unique<sharp_standard_alm_info>(lmax, mmax, stride, mvstart.data());
  }

}}
