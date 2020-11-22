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
 *  Copyright (C) 2020 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#ifndef DUCC_TRANSPOSE_H
#define DUCC_TRANSPOSE_H

#include <algorithm>
#include "ducc0/infra/error_handling.h"
#include "ducc0/infra/mav.h"

namespace ducc0 {

namespace detail_transpose {

using namespace std;
using shape_t=fmav_info::shape_t;
using stride_t=fmav_info::stride_t;

template<typename T1, typename T2> inline void rearrange(T1 &v, const T2 &idx)
  {
  T1 tmp(v);
  for (size_t i=0; i<idx.size(); ++i)
    v[i] = tmp[idx[i]];
  }

inline auto prep(const fmav_info &in, const fmav_info &out)
  {
  MR_assert(in.shape()==out.shape(), "shape mismatch");
  shape_t shp;
  stride_t si, so;
  for (size_t i=0; i<in.ndim(); ++i)
    if (in.shape(i)!=1) // remove axes of length 1
      {
      shp.push_back(in.shape(i));
      si.push_back(in.stride(i));
      so.push_back(out.stride(i));
      }
  // sort dimensions in order of descending output stride
  vector<size_t> idx(shp.size());
  iota(idx.begin(), idx.end(), 0);
  sort(idx.begin(), idx.end(),
    [&so](size_t i1, size_t i2) {return so[i1] > so[i2];});
  rearrange(shp, idx);
  rearrange(si, idx);
  rearrange(so, idx);
  // try merging dimensions
//  [...]
  if (shp.size()>1)
    {
    // move axis with smallest remaining input stride to second-to-last place
    auto iminstr = min_element(si.begin(), si.end()-1) - si.begin();
    size_t s2l = si.size()-2;
    swap (shp[iminstr], shp[s2l]);
    swap (si[iminstr], si[s2l]);
    swap (so[iminstr], so[s2l]);
    }

  return make_tuple(shp, si, so);
  }

template<typename T, typename Func> void sthelper1(const T *in, T *out,
  size_t s0, ptrdiff_t sti0, ptrdiff_t sto0, Func func)
  {
  for (size_t i=0; i<s0; ++i, in+=sti0, out+=sto0)
    func(*in, *out);
  }

inline bool critical(ptrdiff_t s)
  {
  s = (s>=0) ? s : -s;
  return (s>4096) && ((s&(s-1))==0);
  }

template<typename T, typename Func> void sthelper2(const T * DUCC0_RESTRICT in,
  T * DUCC0_RESTRICT out, size_t s0, size_t s1, ptrdiff_t sti0, ptrdiff_t sti1,
  ptrdiff_t sto0, ptrdiff_t sto1, Func func)
  {
  if ((sti0<=sti1) && (sto0<=sto1)) // no need to block
    {
    for (size_t i1=0; i1<s1; ++i1, in+=sti1, out+=sto1)
      {
      auto pi0=in;
      auto po0=out;
      for (size_t i0=0; i0<s0; ++i0, pi0+=sti0, po0+=sto0)
        func(*pi0, *po0);
      }
    return;
    }
  if ((sti0>=sti1) && (sto0>=sto1)) // no need to block
    {
    for (size_t i0=0; i0<s0; ++i0, in+=sti0, out+=sto0)
      {
      auto pi1=in;
      auto po1=out;
      for (size_t i1=0; i1<s1; ++i1, pi1+=sti1, po1+=sto1)
        func(*pi1, *po1);
      }
    return;
    }
  // OK, we have to do a real transpose
  // select blockig sizes depending on critical strides
  bool crit0 = critical(sizeof(T)*sti0) || critical(sizeof(T)*sto0);
  bool crit1 = critical(sizeof(T)*sti1) || critical(sizeof(T)*sto1);
  size_t bs0 = crit0 ? 8 : 8;
  size_t bs1 = crit1 ? 8 : 8;
  // make sure that the smallest absolute stride goes in the innermost loop
  if (min(abs(sti0),abs(sto0))<min(abs(sti1),abs(sto1)))
    {
    swap(s0,s1);
    swap(sti0,sti1);
    swap(sto0,sto1);
    swap(bs0, bs1);
    }
  for (size_t ii0=0; ii0<s0; ii0+=bs0)
    {
    size_t ii0e = min(s0, ii0+bs0);
    for (size_t ii1=0; ii1<s1; ii1+=bs1)
      {
      size_t ii1e = min(s1, ii1+bs1);
      for (size_t i0=ii0; i0<ii0e; ++i0)
        for (size_t i1=ii1; i1<ii1e; ++i1)
          func(in[i0*sti0+i1*sti1], out[i0*sto0+i1*sto1]);
      }
    }
  }

template<typename T, typename Func> void iter(const fmav<T> &in,
  fmav<T> &out, size_t dim, ptrdiff_t idxin, ptrdiff_t idxout, Func func)
  {
  size_t ndim = in.ndim();
  if (dim+2==ndim)
    sthelper2(in.cdata()+idxin, out.vdata()+idxout, in.shape(ndim-2), in.shape(ndim-1),
      in.stride(ndim-2), in.stride(ndim-1), out.stride(ndim-2),
      out.stride(ndim-1), func);
  else
    for (size_t i=0; i<in.shape(dim); ++i)
      iter(in, out, dim+1, idxin+i*in.stride(dim), idxout+i*out.stride(dim), func);
  }

template<typename T, typename Func> void transpose(const fmav<T> &in,
  fmav<T> &out, Func func)
  {
  auto [shp, si, so] = prep(in, out);
  fmav<T> in2(in, shp, si), out2(out, shp, so);
  if (in2.ndim()==1)  // 1D, just iterate
    {
    sthelper1(in2.cdata(), out2.vdata(), in2.shape(0), in2.stride(0), out2.stride(0), func);
    return;
    }
  iter(in2, out2, 0, 0, 0, func);
  }
template<typename T> void transpose(const fmav<T> &in, fmav<T> &out)
  { transpose(in, out, [](const T &in, T&out) { out=in; }); }

}

using detail_transpose::transpose;

}

#endif
