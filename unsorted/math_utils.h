/*
 *  This file is part of libcxxsupport.
 *
 *  libcxxsupport is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  libcxxsupport is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with libcxxsupport; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/*
 *  libcxxsupport is being developed at the Max-Planck-Institut fuer Astrophysik
 *  and financially supported by the Deutsches Zentrum fuer Luft- und Raumfahrt
 *  (DLR).
 */

/*! \file math_utils.h
 *  Various convenience mathematical functions.
 *
 *  Copyright (C) 2002-2015 Max-Planck-Society
 *  \author Martin Reinecke
 */

#ifndef PLANCK_MATH_UTILS_H
#define PLANCK_MATH_UTILS_H

#include <cmath>
#include <vector>
#include <algorithm>
#include "datatypes.h"

/*! Returns \a atan2(y,x) if \a x!=0 or \a y!=0; else returns 0. */
inline double safe_atan2 (double y, double x)
  {
  using namespace std;
  return ((x==0.) && (y==0.)) ? 0.0 : atan2(y,x);
  }

/*! Helper function for linear interpolation (or extrapolation).
    The array must be ordered in ascending order; no two values may be equal. */
template<typename T, typename Iter, typename Comp> inline void interpol_helper
  (const Iter &begin, const Iter &end, const T &val, Comp comp, tsize &idx,
  T &frac)
  {
  using namespace std;
  planck_assert((end-begin)>1,"sequence too small for interpolation");
  idx = lower_bound(begin,end,val,comp)-begin;
  if (idx>0) --idx;
  idx = min(tsize(end-begin-2),idx);
  frac = (val-begin[idx])/(begin[idx+1]-begin[idx]);
  }

/*! Helper function for linear interpolation (or extrapolation).
    The array must be ordered in ascending order; no two values may be equal. */
template<typename T, typename Iter> inline void interpol_helper
  (const Iter &begin, const Iter &end, const T &val, tsize &idx, T &frac)
  { interpol_helper (begin,end,val,std::less<T>(),idx,frac); }

/*! \} */

template<typename T> class kahan_adder
  {
  private:
    T sum, c;
  public:
    kahan_adder(): sum(0), c(0) {}

    void add (const T &val)
      {
      volatile T tc=c; // volatile to disable over-eager optimizers
      volatile T y=val-tc;
      volatile T t=sum+y;
      tc=(t-sum)-y;
      sum=t;
      c=tc;
      }
    T result() const { return sum; }
  };

template<typename T> class tree_adder
  {
  private:
    std::vector<T> state;
    tsize n;

  public:
    tree_adder(): state(1,T(0)), n(0) {}

    void add (const T &val)
      {
      state[0]+=val; ++n;
      if (n>(tsize(1)<<(state.size()-1)))
        state.push_back(T(0));
      int shift=0;
      while (((n>>shift)&1)==0)
        {
        state[shift+1]+=state[shift];
        state[shift]=T(0);
        ++shift;
        }
      }
    T result() const
      {
      T sum(0);
      for (tsize i=0; i<state.size(); ++i)
        sum+=state[i];
      return sum;
      }
  };

template<typename Iter> bool checkNan (Iter begin, Iter end)
  {
  while (begin!=end)
    {
    if (*begin != *begin) return true;
    ++begin;
    }
  return false;
  }

#endif
