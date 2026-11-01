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
 *  Copyright (C) 2023-2026 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#ifndef DUCC0_MCM_H
#define DUCC0_MCM_H

#include "ducc0/infra/simd.h"
#include "ducc0/infra/mav.h"
#include "ducc0/math/constants.h"
#include "ducc0/math/wigner3j.h"

namespace DUCC0_NAMESPACE {

using namespace std;

namespace detail_mcm {

static bool neededForToeplitz(int l1, int l2, int l_exact, int l_toeplitz, int dl_band)
  {
  if (l_exact<0) return true;  // we want everything
  if (l1<l2) swap(l1,l2);
  if (l2<=l_exact) return true;
  if (l2==l_toeplitz) return true;
  if ((l2<l_toeplitz) && (l1-l2<=dl_band)) return true;
  if (l1==l2) return true;
  return false;
  }

template<typename T> static void toeplitz_fill(const vmav<T,2> &mat,
  const cmav<T,1> &diag0, int l_exact, int l_toeplitz, int dl_band, size_t nthreads)
  {
  int s0=mat.shape(0), s1=mat.shape(1);
  int lmax=max(s0-1,s1-1);
  int lsmall=min(s0-1,s1-1);
  if ((l_exact<0) || (l_exact>=lmax))  // nothing to do
    return;
  MR_assert(l_toeplitz<=lsmall, "l_toeplitz too large");
  MR_assert(l_exact<=lsmall, "l_exact too large");
  vector<double> diag(lmax+1);
  for (int l=0; l<=lmax; ++l)
    diag[l] = sqrt(double(diag0(l)));
  vector<double> row_toep(lmax+1-l_toeplitz);
  vector<double> row_exact(lmax+1-l_exact);
  if (s1>=s0)
    {
    for (int l=l_toeplitz; l<=lmax; ++l)
      row_toep[l-l_toeplitz] = mat(l_toeplitz,l)/(diag[l_toeplitz]*diag[l]);
    for (int l=l_exact; l<=lmax; ++l)
      row_exact[l-l_exact] = mat(l_exact,l)/(diag[l_exact]*diag[l]);
    }
  else
    {
    for (int l=l_toeplitz; l<=lmax; ++l)
      row_toep[l-l_toeplitz] = mat(l,l_toeplitz)/(diag[l_toeplitz]*diag[l]);
    for (int l=l_exact; l<=lmax; ++l)
      row_exact[l-l_exact] = mat(l,l_exact)/(diag[l_exact]*diag[l]);
    }

  execDynamic(lsmall-l_exact, nthreads, 8, [&](Scheduler &sched)
    {
    while (auto rng=sched.getNext())
      for (size_t xl1=rng.lo; xl1<rng.hi; ++xl1)
        {
        int l1 = int(xl1)+l_exact+1;
        for (int l2=l1+1; l2<=lmax; ++l2)
          {
          if ((l1<l_toeplitz)&&((l2-l1)<=dl_band))  // already computed exactly
            continue;
          if (l1==l_toeplitz)  // already computed exactly
            continue;
          T val=0;
          if ((l2-l1)<=(lmax-l_toeplitz))  // use column at l_toeplitz
            val = T(row_toep[l2-l1]*diag[l1]*diag[l2]);
          else  // use column at l_exact
            val = T(row_exact[l2-l1]*diag[l1]*diag[l2]);
          if (l2<s1) mat(l1,l2) = val;
          if (l2<s0) mat(l2,l1) = val;
          }
        }
    });
  }

template<typename Tsimd, typename Tout> inline void zero_mat
  (int el1, int el2, size_t s1_, size_t s2_, const vmav<Tout,3> &mat, const vmav<Tout,2> &diag)
  {
  int s1=int(s1_), s2=int(s2_);
  size_t nmat=mat.shape(0);
  constexpr auto vlen=int(Tsimd::size());
  for (size_t imat=0; imat<nmat; ++imat)
    {
    for (int k=0; k<vlen; ++k)
      {
      if ((el1<s1) && (el2+k<s2)) mat(imat, el1, el2+k) = Tout(0);
      if ((el2+k<s1) && (el1<s2)) mat(imat, el2+k, el1) = Tout(0);
      }
    if (el1==el2) diag(imat, el1) = Tout(0);
    }
  }

template<typename Tsimd, typename Tout, typename Tval> inline void store_mat02
  (int el1, int el2, size_t s1_, size_t s2_, const vector<int> &optype,
   const vmav<Tout,3> &mat, const vmav<Tout,2> &diag, const Tval &val)
  {
  int s1=int(s1_), s2=int(s2_);
  constexpr auto vlen=int(Tsimd::size());
  for (size_t ispec=0, imat=0; ispec<optype.size(); ++ispec)
    {
    auto op = optype[ispec];
    for (int k=0; k<vlen; ++k)
      {
      if ((el2+k<s1) && (el1<s2))
        {
        if (op==4)
          {
          mat(imat, el2+k, el1) = Tout(val[ispec][2][k]);
          mat(imat+1, el2+k, el1) = Tout(val[ispec][3][k]);
          }
        else
          mat(imat, el2+k, el1) = Tout(val[ispec][op][k]);
        }
      if ((el1<s1) && (el2+k<s2))
        {
        if (op==4)
          {
          mat(imat, el1, el2+k) = Tout(val[ispec][2][k]);
          mat(imat+1, el1, el2+k) = Tout(val[ispec][3][k]);
          }
        else
          mat(imat, el1, el2+k) = Tout(val[ispec][op][k]);
        }
      }
    if (el1==el2)
      {
      if (op==4)
        {
        diag(imat, el1) = Tout(val[ispec][2][0]);
        diag(imat+1, el1) = Tout(val[ispec][3][0]);
        }
      else
        diag(imat, el1) = Tout(val[ispec][op][0]);
      }
    imat += (op==4) ? 2 : 1;
    }
  }

template<size_t opmask, typename Tsimd, size_t nspec> inline array<array<Tsimd,4>,nspec> sum_wig_general
  (int el1, int el2, size_t lmax_spec, const Wigner3j_direct<Tsimd> &w3j,
   const vmav<double,2> &spec2)
  {
  int el3min = el2-el1;
  int el3max = el2+el1;
  int maxidx = min(el3max, int(lmax_spec));
  array<array<Tsimd,4>,nspec> val;
  for (size_t ispec=0; ispec<nspec; ++ispec)
    for (size_t j=0; j<4; ++j)
      val[ispec][j]=0;

  for (int el3=el3min, ofs=0; el3<=maxidx; el3+=2, ++ofs)
    {
    array<Tsimd,4> wigval=w3j.template calc<opmask>(ofs);
    for (size_t ispec=0; ispec<nspec; ++ispec)
      {
      if constexpr(opmask&7)
        {
        Tsimd sp = loadu<Tsimd>(&spec2(ispec,el3));
        if constexpr (opmask&1)
          val[ispec][0] += wigval[0]*sp;
        if constexpr (opmask&2)
          val[ispec][1] += wigval[1]*sp;
        if constexpr (opmask&4)
          val[ispec][2] += wigval[2]*sp;
        }
      if constexpr (opmask&8)
        val[ispec][3] += wigval[3]*loadu<Tsimd>(&spec2(ispec,el3+1));
      }
    }
  return val;
  }
template<size_t opmask, typename Tsimd, typename Tval> inline void sum_wig_general
  (int el1, int el2, size_t lmax_spec, size_t nspec, const Wigner3j_direct<Tsimd> &w3j,
   const vmav<double,2> &spec2, Tval &val)
  {
  int el3min = el2-el1;
  int el3max = el2+el1;
  int maxidx = min(el3max, int(lmax_spec));
  for (size_t ispec=0; ispec<nspec; ++ispec)
    for (size_t j=0; j<4; ++j)
      val[ispec][j]=0;

  for (int el3=el3min, ofs=0; el3<=maxidx; el3+=2, ++ofs)
    {
    array<Tsimd,4> wigval=w3j.template calc<opmask>(ofs);
    for (size_t ispec=0; ispec<nspec; ++ispec)
      {
      Tsimd sp = loadu<Tsimd>(&spec2(ispec,el3));
      if constexpr (opmask&1)
        val[ispec][0] += wigval[0]*sp;
      if constexpr (opmask&2)
        val[ispec][1] += wigval[1]*sp;
      if constexpr (opmask&4)
        val[ispec][2] += wigval[2]*sp;
      if constexpr (opmask&8)
        val[ispec][3] += wigval[3]*loadu<Tsimd>(&spec2(ispec,el3+1));
      }
    }
  }

template<size_t opmask, typename Tout> void coupling_matrix_rect(
  const cmav<double,2> &spec, const vmav<Tout,3> &mat,
  const vector<int> &optype, int l_exact, int l_toeplitz, int dl_band,
  size_t nthreads)
  {
  size_t nspec=spec.shape(0);
  MR_assert(optype.size()==nspec, "incorrect optype size");
  MR_assert(spec.shape(1)>=1, "lmax_spec is too small.");
  size_t nmat=0;
  for (auto op: optype)
    {
    MR_assert((op>=0) && (op<5), "bad optype entry");
    nmat += (op<4) ? 1 : 2;
    }
  MR_assert(mat.shape(0)==nmat, "incorrect number of output matrices");
  MR_assert(mat.size()>0, "matrix must not be zero-sized");
  size_t s1=mat.shape(1), s2=mat.shape(2);
  size_t lmax = max(s1-1, s2-1);
  size_t lsmall = min(s1-1, s2-1);
  auto lmax_spec = spec.shape(1)-1;
  using Tsimd = native_simd<double>;
  constexpr size_t vlen = Tsimd::size();
  auto lmax_spec_used = min(2*lmax, lmax_spec);
  auto spec2(vmav<double,2>::build_noncritical
    ({nspec, lmax_spec_used+1+vlen-1+1}, PAGE_IN(nthreads)));
  for (size_t l=0; l<=lmax_spec_used; ++l)
    for (size_t i=0; i<nspec; ++i)
      spec2(i,l) = spec(i,l)/fourpi*(2.*l+1.);
  for (size_t l=lmax_spec_used+1; l<spec2.shape(1); ++l)
    for (size_t i=0; i<nspec; ++i)
      spec2(i,l) = 0.;
  auto diag(vmav<Tout,2>::build_noncritical({nmat, lmax+1}));
  Wigner3j_direct_tables<Tsimd> wtables(lmax);
  execDynamic(lmax+1, nthreads, 1, [&](Scheduler &sched)
    {
    Wigner3j_direct<Tsimd> w3j(wtables);
    vmav<array<Tsimd,4>,1> val_({nspec});
    array<Tsimd,4> * DUCC0_RESTRICT val = val_.data();
    while (auto rng=sched.getNext()) for(int el1=int(rng.lo); el1<int(rng.hi); ++el1)
      {
      for (int el2=el1; el2<=((el1<=int(lsmall))?int(lmax):el1); el2+=vlen)
        {
        bool necessary=false;
        for (size_t i=0; i<vlen; ++i)
          if (neededForToeplitz(el1, el2+i, l_exact, l_toeplitz, dl_band))
            necessary=true;
        if (!necessary) continue;

        w3j.prep<opmask>(el1,el2);
        int el3min = el2-el1;
        if (el3min<=int(lmax_spec))
          {
          if (nspec==1)
            {
            auto val = sum_wig_general<opmask, Tsimd, 1> (el1, el2, lmax_spec, w3j, spec2);
            store_mat02<Tsimd, Tout> (el1, el2, s1, s2, optype, mat, diag, val);
            }
          else if (nspec==2)
            {
            auto val = sum_wig_general<opmask, Tsimd, 2> (el1, el2, lmax_spec, w3j, spec2);
            store_mat02<Tsimd, Tout> (el1, el2, s1, s2, optype, mat, diag, val);
            }
          else if (nspec==3)
            {
            auto val = sum_wig_general<opmask, Tsimd, 3> (el1, el2, lmax_spec, w3j, spec2);
            store_mat02<Tsimd, Tout> (el1, el2, s1, s2, optype, mat, diag, val);
            }
          else if (nspec==4)
            {
            auto val = sum_wig_general<opmask, Tsimd, 4> (el1, el2, lmax_spec, w3j, spec2);
            store_mat02<Tsimd, Tout> (el1, el2, s1, s2, optype, mat, diag, val);
            }
          else if (nspec<=50)
            {
            array<array<Tsimd,4>,50> val;
            sum_wig_general<opmask, Tsimd> (el1, el2, lmax_spec, nspec, w3j, spec2, val);
            store_mat02<Tsimd, Tout> (el1, el2, s1, s2, optype, mat, diag, val);
            }
          else
            {
            sum_wig_general<opmask, Tsimd> (el1, el2, lmax_spec, nspec, w3j, spec2, val);
            store_mat02<Tsimd, Tout> (el1, el2, s1, s2, optype, mat, diag, val);
            }
          }
        else
          zero_mat<Tsimd, Tout> (el1, el2, s1, s2, mat, diag);
        }
      }
    });
  if (l_exact>=0)
    for (size_t imat=0; imat<mat.shape(0); ++imat)
      toeplitz_fill(subarray<2>(mat,{{imat},{},{}}), subarray<1>(diag,{{imat},{}}), l_exact, l_toeplitz, dl_band, nthreads);
  }

}

using detail_mcm::coupling_matrix_rect;

}

#endif
