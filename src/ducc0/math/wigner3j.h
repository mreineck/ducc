/*
 *  This file is part of ducc0.
 *
 *  This code is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This code is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this code; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/** \file ducc0/math/wigner3j.h
 *  Computation of Wigner-3j symbols
 *  Algorithm implemented according to Schulten & Gordon:
 *  J. Math. Phys. 16, p. 10 (1975)
 *
 *  Copyright (C) 2009-2023 Max-Planck-Society
 *  \author Martin Reinecke
 */

#ifndef DUCC0_WIGNER3J_H
#define DUCC0_WIGNER3J_H

#include <vector>
#include "ducc0/infra/mav.h"

namespace ducc0 {

namespace detail_wigner3j {

using namespace std;

/**
  Compute the Wigner 3j symbols for parameters \a l2, \a l3, \a m2, \a m3
  following the algorithm in the SLATEC algorithm DRC3JJ.
  The results are returned in \a res, which is resized appropriately.
 */
void wigner3j (double l2, double l3, double m2, double m3, vector<double> &res);
void wigner3j (double l2, double l3, double m2, double m3, const vmav<double,1> &res);

void wigner3j_00_squared_compact (double l2, double l3, const vmav<double,1> &res);

template<typename Tsimd> void wigner3j_00_vec_squared_compact (Tsimd l2, Tsimd l3, const vmav<Tsimd,1> &res);

void flexible_wigner3j (double l2, double l3, double m2, double m3, double l1min, const vmav<double,1> &res);
template<typename Tsimd> void flexible_wigner3j_vec
  (Tsimd l2, Tsimd l3, double m2, double m3, Tsimd l1min, const vmav<Tsimd,1> &res);

/**
  Compute the Wigner 3j symbols for parameters \a l2, \a l3, \a m2, \a m3
  following the algorithm in the SLATEC algorithm DRC3JJ.
  The results are returned in \a res, which is resized appropriately.
  The l1 value of the first entry in \a res is returned in \a l1min.
 */
void wigner3j_int (int l2, int l3, int m2, int m3, int &l1min, vector<double> &res);
void wigner3j_int (int l2, int l3, int m2, int m3, int &l1min, const vmav<double,1> &res);
int wigner3j_ncoef_int(int l2, int l3, int m2, int m3);

template<typename Tsimd> class Wigner3j_direct
  {
  private:
    vector<double> g, fct;
    Tsimd iota;
    static constexpr size_t safety = 16; // safety margin beyond lmax
    inline static Tsimd sqr(Tsimd arg) { return arg*arg; }

  public:
    Wigner3j_direct(size_t lmax)
      : g(2*lmax+1+safety), fct(2*lmax+1+safety)
      {
      for (size_t i=0; i<g.size(); i++)
        {
        // FIXME: it may be more accurate to do the g recurrence in logarithms.
        g[i] = (i==0) ? 1. : g[i-1]*((i-0.5)/i);
        fct[i] = 1./(g[i]*(2*i+1));
        }
      for (size_t i=0; i<Tsimd::size(); ++i)
        iota[i] = double(i);
      }

    // ofs = (el3-el3min)/2
    Tsimd get_00_sq(int el1, int el2, int ofs) const
      {
      return Tsimd(&fct[el2+ofs], element_aligned_tag()) * Tsimd(&g[el2-el1+ofs], element_aligned_tag()) * g[ofs] * g[el1-ofs];
      }
    // ofs = (el3-el3min)/2
    Tsimd get_p2m2_sq(int el1, int el2, int ofs) const
      {
      auto el2v = double(el2) + iota;
      auto lmbda_sq = el1*(el1+1.)*(el2v+1.)*(el2v+2.);
  
      auto lmbda2 = (el2v+ofs+1.) * (2.*(el1-ofs)+1.);
  
      auto A_sq = lmbda_sq * sqr(1. + 2./el1 * (1. - lmbda2/((el1+1.)*(el2v+1.))));
  
      auto pref_5_num_sq = 4.*lmbda2 * (2.*(el2v-el1+ofs+1) - 1.) * (el2v-el1+ofs+1) * ofs * (2*(el2v+ofs)+3.) * (el1-ofs+1.) * (2*ofs-1.);
      auto B_sq = pref_5_num_sq / lmbda_sq;
  
      auto threej_000_sq = get_00_sq(el1, el2, ofs);
      auto threej_000_2_sq = get_00_sq(el1, el2+2, ofs-1);
  
      auto inner_sq = A_sq * threej_000_sq - 2. * sqrt(A_sq*B_sq*threej_000_sq * threej_000_2_sq) + B_sq * threej_000_2_sq;
      auto eta_sq = ((el1-1.)*(el1+2.)*(el2v-1.)*el2v);
      return inner_sq / eta_sq;
      }
  };

}

using detail_wigner3j::wigner3j;
using detail_wigner3j::wigner3j_int;
using detail_wigner3j::wigner3j_ncoef_int;

using detail_wigner3j::wigner3j_00_vec_squared_compact;

using detail_wigner3j::flexible_wigner3j;
using detail_wigner3j::flexible_wigner3j_vec;

using detail_wigner3j::Wigner3j_direct;
}

#endif
