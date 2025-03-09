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
 *  Copyright (C) 2023-2025 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#ifndef DUCC0_MCM_H
#define DUCC0_MCM_H

#include "ducc0/infra/simd.h"
#include "ducc0/infra/mav.h"
#include "ducc0/math/constants.h"
#include "ducc0/math/wigner3j.h"

namespace ducc0 {

using namespace std;

template<typename Tout> void coupling_matrix_spin0_tri(const cmav<double,2> &spec,
  size_t lmax, const vmav<Tout,2> &mat, size_t nthreads)
  {
  size_t nspec=spec.shape(0);
  MR_assert(spec.shape(1)>=1, "spec.shape[1] is too small.");
  auto lmax_spec = spec.shape(1)-1;
  MR_assert(mat.shape(0)==nspec, "number of spectra and matrices mismatch");
  MR_assert(mat.shape(1)==((lmax+1)*(lmax+2))/2, "bad number of matrix entries");
  using Tsimd = native_simd<double>;
  constexpr size_t vlen = Tsimd::size();
  auto lmax_spec_used = min(2*lmax, lmax_spec);
  auto spec2(vmav<double,2>::build_noncritical({nspec, lmax_spec_used+1+vlen-1}, UNINITIALIZED));
  for (size_t l=0; l<=lmax_spec_used; ++l)
    for (size_t i=0; i<nspec; ++i)
      spec2(i,l) = spec(i,l)/ducc0::fourpi*(2.*l+1.);
  for (size_t l=lmax_spec_used+1; l<spec2.shape(1); ++l)
    for (size_t i=0; i<nspec; ++i)
      spec2(i,l) = 0.;
  execDynamic(lmax+1, nthreads, 1, [&](ducc0::Scheduler &sched)
    {
    vmav<Tsimd,1> resfullv({lmax+1});
    vmav<Tsimd,1> val_({nspec});
    Tsimd * DUCC0_RESTRICT val = val_.data();
    Tsimd lofs;
    for (size_t k=0; k<vlen; ++k)
      lofs[k]=double(k);
    while (auto rng=sched.getNext()) for(int el1=int(rng.lo); el1<int(rng.hi); ++el1)
      {
      for (int el2=el1; el2<=int(lmax); el2+=vlen)
        {
        int el3min = el2-el1;
        size_t idx_out = el1*(lmax+1) - (el1*(el1+1))/2 + el2;
        if (el3min<=int(lmax_spec))
          {
          wigner3j_00_vec_squared_compact(Tsimd(el1), Tsimd(el2)+lofs,
            subarray<1>(resfullv, {{size_t(0), size_t(el1+1)}}));
          const Tsimd * DUCC0_RESTRICT res = resfullv.data();

          // FIXME: use generic lambdas in C++20
          if (nspec==1)
            {
            Tsimd val=0;
            int max_i = min(el1+el2, int(lmax_spec)) - el3min;
            for (int i=0, i2=0; i<=max_i; i+=2, ++i2)
              {
              int el3 = el3min+i;
              val += res[i2]*Tsimd(&spec2(0,el3), element_aligned_tag());
              }
            for (size_t k=0; k<vlen; ++k)
              if (el2+k<=lmax)
                mat(0, idx_out+k) = Tout(val[k]);
            }
          else if (nspec==2)
            {
            constexpr size_t nspec=2;
            array<Tsimd,nspec> val;
            for (size_t ispec=0; ispec<nspec; ++ispec)
              val[ispec]=0;
            int max_i = min(el1+el2, int(lmax_spec)) - el3min;
            for (int i=0, i2=0; i<=max_i; i+=2, ++i2)
              {
              int el3 = el3min+i;
              for (size_t ispec=0; ispec<nspec; ++ispec)
                val[ispec] += res[i2]*Tsimd(&spec2(ispec,el3), element_aligned_tag());
              }
            for (size_t ispec=0; ispec<nspec; ++ispec)
              for (size_t k=0; k<vlen; ++k)
                if (el2+k<=lmax)
                  mat(ispec, idx_out+k) = Tout(val[ispec][k]);
            }
          else if (nspec==3)
            {
            constexpr size_t nspec=3;
            array<Tsimd,nspec> val;
            for (size_t ispec=0; ispec<nspec; ++ispec)
              val[ispec]=0;
            int max_i = min(el1+el2, int(lmax_spec)) - el3min;
            for (int i=0, i2=0; i<=max_i; i+=2, ++i2)
              {
              int el3 = el3min+i;
              for (size_t ispec=0; ispec<nspec; ++ispec)
                val[ispec] += res[i2]*Tsimd(&spec2(ispec,el3), element_aligned_tag());
              }
            for (size_t ispec=0; ispec<nspec; ++ispec)
              for (size_t k=0; k<vlen; ++k)
                if (el2+k<=lmax)
                  mat(ispec, idx_out+k) = Tout(val[ispec][k]);
            }
          else if (nspec==4)
            {
            constexpr size_t nspec=4;
            array<Tsimd,nspec> val;
            for (size_t ispec=0; ispec<nspec; ++ispec)
              val[ispec]=0;
            int max_i = min(el1+el2, int(lmax_spec)) - el3min;
            for (int i=0, i2=0; i<=max_i; i+=2, ++i2)
              {
              int el3 = el3min+i;
              for (size_t ispec=0; ispec<nspec; ++ispec)
                val[ispec] += res[i2]*Tsimd(&spec2(ispec,el3), element_aligned_tag());
              }
            for (size_t ispec=0; ispec<nspec; ++ispec)
              for (size_t k=0; k<vlen; ++k)
                if (el2+k<=lmax)
                  mat(ispec, idx_out+k) = Tout(val[ispec][k]);
            }
          else if (nspec<=50)
            {
            array<Tsimd,50> val;
            for (size_t ispec=0; ispec<nspec; ++ispec)
              val[ispec]=0;
            int max_i = min(el1+el2, int(lmax_spec)) - el3min;
            for (int i=0, i2=0; i<=max_i; i+=2, ++i2)
              {
              int el3 = el3min+i;
              for (size_t ispec=0; ispec<nspec; ++ispec)
                val[ispec] += res[i2]*Tsimd(&spec2(ispec,el3), element_aligned_tag());
              }
            for (size_t ispec=0; ispec<nspec; ++ispec)
              for (size_t k=0; k<vlen; ++k)
                if (el2+k<=lmax)
                  mat(ispec, idx_out+k) = Tout(val[ispec][k]);
            }
          else
            {
            for (size_t ispec=0; ispec<nspec; ++ispec)
              val[ispec]=0;
            int max_i = min(el1+el2, int(lmax_spec)) - el3min;
            for (int i=0, i2=0; i<=max_i; i+=2, ++i2)
              {
              int el3 = el3min+i;
              for (size_t ispec=0; ispec<nspec; ++ispec)
                val[ispec] += res[i2]*Tsimd(&spec2(ispec,el3), element_aligned_tag());
              }
            for (size_t ispec=0; ispec<nspec; ++ispec)
              for (size_t k=0; k<vlen; ++k)
                if (el2+k<=lmax)
                  mat(ispec, idx_out+k) = Tout(val[ispec][k]);
            }
          }
        else
          for (size_t ispec=0; ispec<nspec; ++ispec)
            for (size_t k=0; k<vlen; ++k)
              if (el2+k<=lmax)
                mat(ispec, idx_out+k) = Tout(0);
        }
      }
    });
  }

template<int is00, int is02, int is20, int is22, int im00, int im02, int im20, int impp, int immm, typename Tout> void coupling_matrix_spin0and2_tri(
  const cmav<double,3> &spec, size_t lmax, const vmav<Tout,3> &mat, size_t nthreads)
  {
  constexpr size_t ncomp_spec=size_t(max(is00, max(is02, max(is20, is22)))) + 1;
  static_assert(ncomp_spec>0, "need at least one spectral component");
  static_assert(ncomp_spec <= (is00>=0)+(is02>=0)+(is20>=0)+(is22>=0),
    "gaps in spectral component indices");
  static_assert((is00==0)||(is02==0)||(is20==0)||(is22==0),
    "gaps in spectral component indices");
  static_assert((ncomp_spec<2) || (is00==1)||(is02==1)||(is20==1)||(is22==1),
    "gaps in spectral component indices");
  static_assert((ncomp_spec<3) || (is00==2)||(is02==2)||(is20==2)||(is22==2),
    "gaps in spectral component indices");
  static_assert((ncomp_spec<4) || (is00==3)||(is02==3)||(is20==3)||(is22==3),
    "gaps in spectral component indices");

  constexpr size_t ncomp_mat = size_t(max(im00, max(im02, max(im20, max(impp, immm))))) + 1;
  static_assert(ncomp_mat>0, "need at least one matrix component");
  static_assert(ncomp_mat == (im00>=0)+(im02>=0)+(im20>=0)+(impp>=0)+(immm>=0),
    "gaps in matrix component indices");
  static_assert((im00==0)+(im02==0)+(im20==0)+(impp==0)+(immm==0)==1,
    "gaps in matrix component indices");
  static_assert((ncomp_mat<2) || ((im00==1)+(im02==1)+(im20==1)+(impp==1)+(immm==1)==1),
    "gaps in matrix component indices");
  static_assert((ncomp_mat<3) || ((im00==2)+(im02==2)+(im20==2)+(impp==2)+(immm==2)==1),
    "gaps in matrix component indices");
  static_assert((ncomp_mat<4) || ((im00==3)+(im02==3)+(im20==3)+(impp==3)+(immm==3)==1),
    "gaps in matrix component indices");
  static_assert((ncomp_mat<5) || ((im00==4)+(im02==4)+(im20==4)+(impp==4)+(immm==4)==1),
    "gaps in matrix component indices");

  if constexpr ((im02<0) && (im20<0) && (impp<0) && (immm<0))
    return coupling_matrix_spin0_tri(subarray<2>(spec, {{},{is00},{}}),
      lmax, subarray<2>(mat, {{},{im00},{}}), nthreads);

  size_t nspec=spec.shape(0);
  MR_assert(spec.shape(1)==ncomp_spec, "spec.shape[1] must be .", ncomp_spec);
  MR_assert(spec.shape(2)>=1, "lmax_spec is too small.");
  MR_assert(mat.shape(0)==nspec, "number of spectra and matrices mismatch");
  MR_assert(mat.shape(1)==ncomp_mat, "bad number of matrix components");
  MR_assert(mat.shape(2)==((lmax+1)*(lmax+2))/2, "bad number of matrix entries");
  auto lmax_spec = spec.shape(2)-1;
  using Tsimd = native_simd<double>;
  constexpr size_t vlen = Tsimd::size();
  auto lmax_spec_used = min(2*lmax, lmax_spec);
  auto spec2(vmav<double,3>::build_noncritical
    ({nspec, ncomp_spec, lmax_spec_used+1+vlen-1+1}, UNINITIALIZED));
  for (size_t l=0; l<=lmax_spec_used; ++l)
    for (size_t j=0; j<ncomp_spec; ++j)
      for (size_t i=0; i<nspec; ++i)
        spec2(i,j,l) = spec(i,j,l)/ducc0::fourpi*(2.*l+1.);
  for (size_t l=lmax_spec_used+1; l<spec2.shape(2); ++l)
    for (size_t j=0; j<ncomp_spec; ++j)
      for (size_t i=0; i<nspec; ++i)
        spec2(i,j,l) = 0.;
  execDynamic(lmax+1, nthreads, 1, [&](ducc0::Scheduler &sched)
    {
// FIXME: these two lines are necessary for Visual C++, no idea why
    constexpr size_t ncomp_mat = size_t(max(im00, max(im02, max(im20, max(impp, immm))))) + 1;
    constexpr size_t ncomp_spec=size_t(max(is00, max(is02, max(is20, is22)))) + 1;
// res arrays are one larger to make loops simpler below
    vmav<Tsimd,2> wig({2, 2*lmax+1+1});
    vmav<array<Tsimd,ncomp_mat>,1> val_({nspec});
    array<Tsimd,ncomp_mat> * DUCC0_RESTRICT val = val_.data();
    Tsimd lofs;
    for (size_t k=0; k<vlen; ++k)
      lofs[k]=double(k);
    while (auto rng=sched.getNext()) for(int el1=int(rng.lo); el1<int(rng.hi); ++el1)
      {
      for (int el2=el1; el2<=int(lmax); el2+=vlen)
        {
        int el3min = el2-el1;
        int el3max = el2+el1;
        size_t idx_out = el1*(lmax+1) - (el1*(el1+1))/2 + el2;
        if (el3min<=int(lmax_spec))
          {
          auto tmp=subarray<2>(wig,{{},{size_t(el3min), size_t(el3max+2)}});
          // only compute 00 wigners if necessary
          if constexpr ((im00>=0) || (im02>=0) || (im20>=0))
            flexible_wigner3j_vec(Tsimd(el1), Tsimd(el2)+lofs, 0, 0,
              Tsimd(el3min)+lofs, subarray<1>(tmp, {{0}, {}}));
          // we always need those if we arrive here
          flexible_wigner3j_vec(Tsimd(el1), Tsimd(el2)+lofs, -2, 2,
            Tsimd(el3min)+lofs, subarray<1>(tmp, {{1}, {}}));
          const Tsimd * DUCC0_RESTRICT wp0 = &wig(0,0);
          const Tsimd * DUCC0_RESTRICT wp1 = &wig(1,0);
          int maxidx = min(el3max, int(lmax_spec));

          // FIXME: use generic lambdas in C++20
          if (nspec==1)
            {
            array<Tsimd,ncomp_mat> val;
            for (size_t j=0; j<ncomp_mat; ++j)
              val[j]=0;
            for (int el3=el3min; el3<=maxidx; el3+=2)
              {
              const Tsimd w0=wp0[el3], w1=wp1[el3];
              const Tsimd w00=w0*w0, w01=w0*w1, w11=w1*w1;
              const Tsimd w11p1=wp1[el3+1]*wp1[el3+1];
              array<Tsimd, ncomp_spec> sp;
              for (size_t i=0; i<ncomp_spec; ++i)
                sp[i] = Tsimd(&spec2(0,i,el3), element_aligned_tag());
              if constexpr (im00>=0)
                val[im00] += w00*sp[is00];
              if constexpr (im02>=0)
                val[im02] += w01*sp[is02];
              if constexpr (im20>=0)
                val[im20] += w01*sp[is20];
              if constexpr (impp>=0)
                val[impp] += w11*sp[is22];
              if constexpr (immm>=0)
                val[immm] += w11p1*Tsimd(&spec2(0,is22,el3+1), element_aligned_tag());
              }
            for (size_t j=0; j<ncomp_mat; ++j)
              for (size_t k=0; k<vlen; ++k)
                if (el2+k<=lmax)
                  mat(0, j, idx_out+k) = Tout(val[j][k]);
            }
          else if (nspec==2)
            {
            constexpr size_t nspec=2;
            array<array<Tsimd,ncomp_mat>,nspec> val;
            for (size_t ispec=0; ispec<nspec; ++ispec)
              for (size_t j=0; j<ncomp_mat; ++j)
                val[ispec][j]=0;
            for (int el3=el3min; el3<=maxidx; el3+=2)
              {
              const Tsimd w0=wp0[el3], w1=wp1[el3];
              const Tsimd w00=w0*w0, w01=w0*w1, w11=w1*w1;
              const Tsimd w11p1=wp1[el3+1]*wp1[el3+1];
              for (size_t ispec=0; ispec<nspec; ++ispec)
                {
                array<Tsimd, ncomp_spec> sp;
                for (size_t i=0; i<ncomp_spec; ++i)
                  sp[i] = Tsimd(&spec2(ispec,i,el3), element_aligned_tag());
                if constexpr (im00>=0)
                  val[ispec][im00] += w00*sp[is00];
                if constexpr (im02>=0)
                  val[ispec][im02] += w01*sp[is02];
                if constexpr (im20>=0)
                  val[ispec][im20] += w01*sp[is20];
                if constexpr (impp>=0)
                  val[ispec][impp] += w11*sp[is22];
                if constexpr (immm>=0)
                  val[ispec][immm] += w11p1*Tsimd(&spec2(ispec,is22,el3+1), element_aligned_tag());
                }
              }
            for (size_t ispec=0; ispec<nspec; ++ispec)
              for (size_t j=0; j<ncomp_mat; ++j)
                for (size_t k=0; k<vlen; ++k)
                  if (el2+k<=lmax)
                    mat(ispec, j, idx_out+k) = Tout(val[ispec][j][k]);
            }
          else if (nspec<=50)
            {
            array<array<Tsimd,ncomp_mat>,50> val;
            for (size_t ispec=0; ispec<nspec; ++ispec)
              for (size_t j=0; j<ncomp_mat; ++j)
                val[ispec][j]=0;
            for (int el3=el3min; el3<=maxidx; el3+=2)
              {
              const Tsimd w0=wp0[el3], w1=wp1[el3];
              const Tsimd w00=w0*w0, w01=w0*w1, w11=w1*w1;
              const Tsimd w11p1=wp1[el3+1]*wp1[el3+1];
              for (size_t ispec=0; ispec<nspec; ++ispec)
                {
                array<Tsimd, ncomp_spec> sp;
                for (size_t i=0; i<ncomp_spec; ++i)
                  sp[i] = Tsimd(&spec2(ispec,i,el3), element_aligned_tag());
                if constexpr (im00>=0)
                  val[ispec][im00] += w00*sp[is00];
                if constexpr (im02>=0)
                  val[ispec][im02] += w01*sp[is02];
                if constexpr (im20>=0)
                  val[ispec][im20] += w01*sp[is20];
                if constexpr (impp>=0)
                  val[ispec][impp] += w11*sp[is22];
                if constexpr (immm>=0)
                  val[ispec][immm] += w11p1*Tsimd(&spec2(ispec,is22,el3+1), element_aligned_tag());
                }
              }
            for (size_t ispec=0; ispec<nspec; ++ispec)
              for (size_t j=0; j<ncomp_mat; ++j)
                for (size_t k=0; k<vlen; ++k)
                  if (el2+k<=lmax)
                    mat(ispec, j, idx_out+k) = Tout(val[ispec][j][k]);
            }
          else
            {
            for (size_t ispec=0; ispec<nspec; ++ispec)
              for (size_t j=0; j<ncomp_mat; ++j)
                val[ispec][j]=0;
            for (int el3=el3min; el3<=maxidx; el3+=2)
              {
              const Tsimd w0=wp0[el3], w1=wp1[el3];
              const Tsimd w00=w0*w0, w01=w0*w1, w11=w1*w1;
              const Tsimd w11p1=wp1[el3+1]*wp1[el3+1];
              for (size_t ispec=0; ispec<nspec; ++ispec)
                {
                array<Tsimd, ncomp_spec> sp;
                for (size_t i=0; i<ncomp_spec; ++i)
                  sp[i] = Tsimd(&spec2(ispec,i,el3), element_aligned_tag());
                if constexpr (im00>=0)
                  val[ispec][im00] += w00*sp[is00];
                if constexpr (im02>=0)
                  val[ispec][im02] += w01*sp[is02];
                if constexpr (im20>=0)
                  val[ispec][im20] += w01*sp[is20];
                if constexpr (impp>=0)
                  val[ispec][impp] += w11*sp[is22];
                if constexpr (immm>=0)
                  val[ispec][immm] += w11p1*Tsimd(&spec2(ispec,is22,el3+1), element_aligned_tag());
                }
              }
            for (size_t ispec=0; ispec<nspec; ++ispec)
              for (size_t j=0; j<ncomp_mat; ++j)
                for (size_t k=0; k<vlen; ++k)
                  if (el2+k<=lmax)
                    mat(ispec, j, idx_out+k) = Tout(val[ispec][j][k]);
            }
          }
        else
          for (size_t ispec=0; ispec<nspec; ++ispec)
            for (size_t j=0; j<ncomp_mat; ++j)
              for (size_t k=0; k<vlen; ++k)
                if (el2+k<=lmax)
                  mat(ispec, j, idx_out+k) = Tout(0);
        }
      }
    });
  }

template<typename Tout> void coupling_matrix_spin0and2_pure(const cmav<double,3> &spec,
  size_t lmax, const vmav<Tout,4> &mat, size_t nthreads)
  {
  using Tsimd = native_simd<double>;
  constexpr size_t vlen=Tsimd::size();
  constexpr size_t ncomp_spec=4;
  constexpr size_t ncomp_mat=4;
  size_t nspec=spec.shape(0);
  MR_assert(spec.shape(1)==ncomp_spec, "spec.shape[1] must be 4.");
  MR_assert(spec.shape(2)>=1, "lmax_spec is too small.");
  MR_assert(mat.shape(0)==nspec, "number of spectra and matrices mismatch");
  MR_assert(mat.shape(1)==ncomp_mat, "bad number of matrix components");
  MR_assert(mat.shape(2)==lmax+1, "bad number of matrix entries");
  MR_assert(mat.shape(3)==lmax+1, "bad number of matrix entries");
  auto lmax_spec = spec.shape(2)-1;
  auto lmax_spec_used = min(2*lmax, lmax_spec);
  auto spec2(vmav<double,3>::build_noncritical
    ({nspec, ncomp_spec, lmax_spec_used+1+vlen-1+1}, UNINITIALIZED));
  for (size_t l=0; l<=lmax_spec_used; ++l)
    for (size_t j=0; j<ncomp_spec; ++j)
      for (size_t i=0; i<nspec; ++i)
        spec2(i,j,l) = spec(i,j,l)/ducc0::fourpi*(2.*l+1.);
  for (size_t l=lmax_spec_used+1; l<spec2.shape(2); ++l)
    for (size_t j=0; j<ncomp_spec; ++j)
      for (size_t i=0; i<nspec; ++i)
        spec2(i,j,l) = 0.;
  vector<double> nom1(2*lmax+1+vlen-1+1), nom2(2*lmax+1+vlen-1+1);
  for (size_t el3=0; el3<nom1.size(); ++el3)
    {
    nom1[el3] = 2.*sqrt((el3+1.)*el3);
    nom2[el3] = sqrt((el3+2.)*(el3+1.)*el3*(el3-1.));
    }
  execDynamic(lmax+1, nthreads, 1, [&](ducc0::Scheduler &sched)
    {
    // res arrays are one larger to make loops simpler below
    vmav<Tsimd,2> wig({6, 2*lmax+1+1});
    constexpr size_t nvcomp = 7;
    vmav<array<Tsimd,nvcomp>,1> val_({nspec});
    array<Tsimd,nvcomp> * DUCC0_RESTRICT val = val_.data();
    Tsimd lofs;
    for (size_t k=0; k<vlen; ++k)
      lofs[k]=double(k);
    while (auto rng=sched.getNext()) for(int el1=int(rng.lo); el1<int(rng.hi); ++el1)
      {
      for (int xel2=el1; xel2<=int(lmax); xel2+=vlen)
        {
        Tsimd el2=Tsimd(xel2)+lofs;
        int el3min = abs(xel2-el1);
        int el3max = el1+xel2;
        Tsimd xdenom1 = blend(el2>Tsimd(1.), sqrt(Tsimd(1.) / ((el2-1.)*(el2+2.))), Tsimd(0.)),
              xdenom2 = blend(el2>Tsimd(1.), sqrt(Tsimd(1.) / ((el2+2.)*(el2+1.)*el2*(el2-1.))), Tsimd(0.));
        double xxdenom1 = (el1>1) ? sqrt(1. / ((el1-1.)*(el1+2.))) : 0,
               xxdenom2 = (el1>1) ? sqrt(1. / ((el1+2.)*(el1+1.)*el1*(el1-1.))): 0;
        if (el3min<=int(lmax_spec))
          {
          {
          auto tmp = subarray<2>(wig, {{}, {size_t(el3min), size_t(el3max+2)}});
          constexpr array<int,6> m1 {{0, -2, -2, -2, -2, -2}};
          constexpr array<int,6> m2 {{0,  2,  1,  0,  1,  0}};
          array<Tsimd,6> xl1 {{el1, el1, el1, el1, el2, el2}};
          array<Tsimd,6> xl2 {{el2, el2, el2, el2, el1, el1}};
          for (size_t ii=0; ii<6; ++ii)
            flexible_wigner3j_vec(xl1[ii], xl2[ii], m1[ii], m2[ii],
              Tsimd(el3min)+lofs, subarray<1>(tmp, {{ii}, {}}));
          }

          for (size_t ispec=0; ispec<nspec; ++ispec)
            for (size_t j=0; j<nvcomp; ++j)
              val[ispec][j]=0;
          int maxidx = min(el3max, int(lmax_spec));
          for (int el3=el3min; el3<=maxidx; el3+=2)
            {
            Tsimd fac_b = Tsimd(&nom1[el3],element_aligned_tag())*xdenom1,
                  fac_c = Tsimd(&nom2[el3],element_aligned_tag())*xdenom2,
                  xfac_b = Tsimd(&nom1[el3],element_aligned_tag())*xxdenom1,
                  xfac_c = Tsimd(&nom2[el3],element_aligned_tag())*xxdenom2;
//                  fac_b2 = Tsimd(&nom1[el3+1],element_aligned_tag())*xdenom1,
//                  fac_c2 = Tsimd(&nom2[el3+1],element_aligned_tag())*xdenom2,
//                  xfac_b2 = Tsimd(&nom1[el3+1],element_aligned_tag())*xxdenom1,
//                  xfac_c2 = Tsimd(&nom2[el3+1],element_aligned_tag())*xxdenom2;
            for (size_t ispec=0; ispec<nspec; ++ispec)
              {
              const Tsimd s0(&spec2(ispec,0,el3), element_aligned_tag()),
                          s1(&spec2(ispec,1,el3), element_aligned_tag()),
                          s2(&spec2(ispec,2,el3), element_aligned_tag()),
                          s3(&spec2(ispec,3,el3), element_aligned_tag());
              val[ispec][0] += wig(0,el3)*wig(0,el3)*s0;
              auto combin = wig(1,el3) + fac_b*wig(2,el3) + fac_c*wig(3,el3);
              val[ispec][1] += wig(0,el3)*combin*s1;
              val[ispec][2] += wig(0,el3)*combin*s2;
              val[ispec][3] += combin*combin*Tsimd(&spec2(ispec,3,el3), element_aligned_tag());
              auto xcombin = wig(1,el3) + xfac_b*wig(4,el3) + xfac_c*wig(5,el3);
              val[ispec][4] += wig(0,el3)*xcombin*s1;
              val[ispec][5] += wig(0,el3)*xcombin*s2;
              val[ispec][6] += xcombin*xcombin*s3;
//              auto combin2 = wig(1,el3+1) + fac_b2*wig(2,el3+1) + fac_c2*wig(3,el3+1);
//              val[ispec][7] += combin2*combin2*Tsimd(&spec2(ispec,3,el3+1), element_aligned_tag());
//              auto  xcombin2 = wig(1,el3+1) + xfac_b2*wig(4,el3+1) + xfac_c2*wig(5,el3+1);
//              val[ispec][8] += xcombin2*xcombin2*Tsimd(&spec2(ispec,3,el3+1), element_aligned_tag());
              }
            }
          for (size_t ispec=0; ispec<nspec; ++ispec)
            for (size_t k=0; k<vlen; ++k)
              if (el2[k]<=lmax)
                {
                mat(ispec, 0, xel2+k, el1) = Tout((2*el1+1.)*val[ispec][0][k]);
                mat(ispec, 1, xel2+k, el1) = Tout((2*el1+1.)*val[ispec][1][k]);
                mat(ispec, 2, xel2+k, el1) = Tout((2*el1+1.)*val[ispec][2][k]);
                mat(ispec, 3, xel2+k, el1) = Tout((2*el1+1.)*val[ispec][3][k]);
                mat(ispec, 0, el1, xel2+k) = Tout((2*el2[k]+1.)*val[ispec][0][k]);
                mat(ispec, 1, el1, xel2+k) = Tout((2*el2[k]+1.)*val[ispec][4][k]);
                mat(ispec, 2, el1, xel2+k) = Tout((2*el2[k]+1.)*val[ispec][5][k]);
                mat(ispec, 3, el1, xel2+k) = Tout((2*el2[k]+1.)*val[ispec][6][k]);
//                mat(ispec, 4, xel2+k, el1) = (2*el1+1.)*val[ispec][4][k];
//                mat(ispec, 4, el1, xel2+k) = (2*el2[k]+1.)*val[ispec][8][k];
                }
          }
        else
          for (size_t ispec=0; ispec<nspec; ++ispec)
            for (size_t j=0; j<ncomp_mat; ++j)
              for (size_t k=0; k<vlen; ++k)
                if (el2[k]<=lmax)
                  mat(ispec, j, xel2+k, el1) = mat(ispec, j, el1, xel2+k) = 0.;
        }
      }
    });
  }

}

#endif
