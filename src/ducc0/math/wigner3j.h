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

// Direct evaluation of Wigner 3j symbols, following https://arxiv.org/abs/2602.15605
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

    double simple_000(int el1, int el2, int el3) const
      {
if (el1>el2) return simple_000(el2,el3,el1);
auto J = el1+el2+el3;
if (J&1) return 0;
MR_assert((J&1)==0, "oops");
double sign = (J&2) ? -1 : 1;
      auto el3min = el2-el1;
if ((el3<abs(el3min)) || (el3>el2+el1)) return 0;
      auto ofs = (el3-el3min)/2;
      return sign*sqrt(fct[el2+ofs] * g[el2-el1+ofs] * g[ofs] * g[el1-ofs]);
      }
    double simple_000_sq(int el1, int el2, int el3) const
      {
if (el1>el2) return simple_000_sq(el2,el3,el1);
auto J = el1+el2+el3;
if (J&1) return 0;
      auto el3min = el2-el1;
if ((el3<abs(el3min)) || (el3>el2+el1)) return 0;
      auto ofs = (el3-el3min)/2;
      return fct[el2+ofs] * g[el2-el1+ofs] * g[ofs] * g[el1-ofs];
      }
    double simple_0m1p1(int el1, int el2, int el3) const
      {
      auto J = el1+el2+el3;
      auto Jmpp = J-2*el1;
      auto Jpmp = J-2*el2;
      auto Jppm = J-2*el3;
      if (J&1)  // odd
        {
// eq 25
        return -0.5*simple_000(el1,el2,el3+1)*sqrt((J+2.)*(Jmpp+1.)*(Jpmp+1.)*Jppm/(el2*(el2+1.)*el3*(el3+1.)));
        }
      else
        {
// eq 34
        return sqrt((el2+1.)*(el3+1.)/(el2*el3))*simple_000(el1,el2,el3)
             + 0.5*simple_000(el1, el2+1, el3+1)*sqrt((J+2.)*(J+3.)*(Jmpp+1.)*(Jmpp+2.)/(el2*(el2+1.)*el3*(el3+1.)));
        }
      }
    double simple_0m1p1_sq(int el1, int el2, int el3) const
      {
      auto J = el1+el2+el3;
      auto Jmpp = J-2*el1;
      auto Jpmp = J-2*el2;
      auto Jppm = J-2*el3;
      if (J&1)  // odd
        {
// eq 25
        return 0.25*simple_000_sq(el1,el2,el3+1)*(J+2.)*(Jmpp+1.)*(Jpmp+1.)*Jppm/(el2*(el2+1.)*el3*(el3+1.));
        }
      else
        {
// eq 34
auto t1sq = (el2+1.)*(el3+1.)/(el2*el3)*simple_000_sq(el1,el2,el3);
auto t2sq = 0.25*simple_000_sq(el1, el2+1, el3+1)*(J+2.)*(J+3.)*(Jmpp+1.)*(Jmpp+2.)/(el2*(el2+1.)*el3*(el3+1.));
return t1sq+t2sq-2.*sqrt(t1sq*t2sq);
        auto tmp= sqrt((el2+1.)*(el3+1.)/(el2*el3))*simple_000(el1,el2,el3)
             + 0.5*simple_000(el1, el2+1, el3+1)*sqrt((J+2.)*(J+3.)*(Jmpp+1.)*(Jmpp+2.)/(el2*(el2+1.)*el3*(el3+1.)));
        return tmp*tmp;
        }
      }
    double simple_0m2p2(int el1, int el2, int el3) const
      {  // eq 56
      auto J = el1+el2+el3;
      auto Jmpp = J-2*el1;
      auto Jpmp = J-2*el2;
      auto Jppm = J-2*el3;
      auto lambda = sqrt(el2*(el2+1.)*(el3+1.)*(el3+2.));
      auto eta = sqrt((el2-1.)*(el2+2.)*(el3-1.)*el3);
      auto Lambda = sqrt((J+2.)*(Jmpp+1)*(Jpmp+1)*Jppm);
      return (lambda * simple_000(el1,el2,el3)
             + 2*sqrt(el3*(el3+2.))*simple_0m1p1(el1,el2,el3)
             -Lambda*simple_0m1p1(el1,el2,el3+1))/eta;
      }

    double simple_0m2p2_sq(int el1, int el2, int el3) const
      {  // eq 56
      auto J = el1+el2+el3;
      auto Jmpp = J-2*el1;
      auto Jpmp = J-2*el2;
      auto Jppm = J-2*el3;
MR_assert(J&1,"oops");
      auto eta_sq = (el2-1.)*(el2+2.)*(el3-1.)*el3;
      auto Lambda_sq = (J+2.)*(Jmpp+1)*(Jpmp+1)*Jppm;

//auto term1 = -sqrt(1./(el2+1.));
//auto term2 = -sqrt(el2+1.);
auto termsumsq = (el2+1.) + 2. + 1./(el2+1.);
//auto term25 = simple_000(el1,el2,el3+1)*sqrt((J+2.)*(Jmpp+1)*(Jpmp+1)*Jppm*(el3+2.)/(el2*(el3+1.))) * (term1+term2);
auto term25_sq = simple_000_sq(el1,el2,el3+1)*(J+2.)*(Jmpp+1)*(Jpmp+1)*Jppm*(el3+2.)/(el2*(el3+1.)) * termsumsq;
//auto term3 = -Lambda*0.5*simple_000(el1, el2+1, el3+2)*sqrt((J+3.)*(J+4.)*(Jmpp+2.)*(Jmpp+3.)/(el2*(el2+1.)*(el3+1.)*(el3+2.)));
auto term3_sq = Lambda_sq*0.25*simple_000_sq(el1, el2+1, el3+2)*(J+3.)*(J+4.)*(Jmpp+2.)*(Jmpp+3.)/(el2*(el2+1.)*(el3+1.)*(el3+2.));
// term25 and term 3 have opposite signs?
//      auto res = term25 + term3;
auto res = term25_sq+term3_sq-2*sqrt(term25_sq*term3_sq);
return res/eta_sq;
      }
    Tsimd get_EB_el3(int el1, int el2, int el3) const
      {  // eq 56
      auto el2v = double(el2) + iota;
      auto el3v = double(el3) + iota;
      auto J = el1+el2v+el3v;
      auto Jmpp = J-2*el1;
      auto Jpmp = J-2*el2v;
      auto Jppm = J-2*el3v;
//MR_assert(J&1,"oops");
      auto eta_sq = (el2v-1.)*(el2v+2.)*(el3v-1.)*el3v;
      auto Lambda_sq = (J+2.)*(Jmpp+1)*(Jpmp+1)*Jppm;

      auto termsumsq = (el2v+1.) + 2. + 1./(el2v+1.);
      auto term25_sq = get_TT_el3(el1,el2,el3+1)*(J+2.)*(Jmpp+1)*(Jpmp+1)*Jppm*(el3v+2.)/(el2v*(el3v+1.)) * termsumsq;
      auto term3_sq = Lambda_sq*0.25*get_TT_el3(el1, el2+1, el3+2)*(J+3.)*(J+4.)*(Jmpp+2.)*(Jmpp+3.)/(el2v*(el2v+1.)*(el3v+1.)*(el3v+2.));
      auto res = term25_sq+term3_sq-2*sqrt(term25_sq*term3_sq);
      return res/eta_sq;
      }

    // ofs = (el3-el3min)/2
    Tsimd get_TT(int el1, int el2, int ofs) const
      {
      return Tsimd(&fct[el2+ofs], element_aligned_tag()) * Tsimd(&g[el2-el1+ofs], element_aligned_tag()) * g[ofs] * g[el1-ofs];
      }
    Tsimd get_TT_el3(int el1, int el2, int el3) const
      {
int ofs = (el3-(el2-el1))/2;
      return Tsimd(&fct[el2+ofs], element_aligned_tag()) * Tsimd(&g[el2-el1+ofs], element_aligned_tag()) * g[ofs] * g[el1-ofs];
      }
    // ofs = (el3-el3min)/2
    Tsimd get_EE(int el1, int el2, int ofs) const
      {
      auto el2v = double(el2) + iota;
      auto lmbda_sq = el1*(el1+1.)*(el2v+1.)*(el2v+2.);
  
      auto lmbda2 = (el2v+ofs+1.) * (2.*(el1-ofs)+1.);
  
      auto A_sq = lmbda_sq * sqr(1. + 2./el1 * (1. - lmbda2/((el1+1.)*(el2v+1.))));
  
      auto pref_5_num_sq = 4.*lmbda2 * (2.*(el2v-el1+ofs+1) - 1.) * (el2v-el1+ofs+1) * ofs * (2*(el2v+ofs)+3.) * (el1-ofs+1.) * (2*ofs-1.);
      auto B_sq = pref_5_num_sq / lmbda_sq;
  
      auto threej_000_sq = get_TT(el1, el2, ofs);
      auto threej_000_2_sq = get_TT(el1, el2+2, ofs-1);
  
      auto inner_sq = A_sq * threej_000_sq - 2. * sqrt(A_sq*B_sq*threej_000_sq * threej_000_2_sq) + B_sq * threej_000_2_sq;
      auto eta_sq = (el1-1.)*(el1+2.)*(el2v-1.)*el2v;
      return inner_sq / eta_sq;
      }
    // ofs = (el3-el3min)/2
    Tsimd get_TE(int el1, int el2, int ofs) const
      {
      auto el2v = double(el2) + iota;
      auto lmbda_sq = el1*(el1+1.)*(el2v+1.)*(el2v+2.);
  
      auto lmbda2 = (el2v+ofs+1.) * (2.*(el1-ofs)+1.);
  
      auto A_sq = lmbda_sq * sqr(1. + 2./el1 * (1. - lmbda2/((el1+1.)*(el2v+1.))));

      auto pref_5_num_sq = 4.*lmbda2 * (2.*(el2v-el1+ofs+1) - 1.) * (el2v-el1+ofs+1) * ofs * (2*(el2v+ofs)+3.) * (el1-ofs+1.) * (2*ofs-1.);
      auto B_sq = pref_5_num_sq / lmbda_sq;

//el3 = 2*ofs + el3min = 2*ofs + el2-el1
//J = el1+el2+el2-el1+2*ofs == even
// Note: the signs on the two roots don't matter, as long as we just get them both right or wrong together
      auto x_eta_sq = Tsimd(1.)/((el1-1.)*(el1+2.)*(el2v-1.)*el2v);
      auto threej_000_sq = get_TT(el1, el2, ofs);
      auto tmp1 = sqrt(A_sq*threej_000_sq*x_eta_sq);
      auto threej_000_2_sq = get_TT(el1, el2+2, ofs-1);
      auto tmp2 = -sqrt(B_sq*threej_000_2_sq*x_eta_sq);
  
      auto threej_0p2m2 = tmp1+tmp2;
      return sqrt(threej_000_sq)*threej_0p2m2;
      }
    Tsimd get_EB_el3(int el1, int el2, int el3) const
      {  // eq 56
      auto el1v = double(el1) + iota;
      auto el3v = double(el3) + iota;
      auto J = el11+el2+el3v;
      auto Jmpp = J-2*el1v;
      auto Jpmp = J-2*el2;
      auto Jppm = J-2*el3v;
//MR_assert(J&1,"oops");
      auto eta_sq = (el2-1.)*(el2+2.)*(el3v-1.)*el3v;
      auto Lambda_sq = (J+2.)*(Jmpp+1)*(Jpmp+1)*Jppm;

      auto termsumsq = (el2+1.) + 2. + 1./(el2+1.);
      auto term25_sq = get_TT_el3(el1,el2,el3+1)*(J+2.)*(Jmpp+1)*(Jpmp+1)*Jppm*(el3v+2.)/(el2v*(el3v+1.)) * termsumsq;
      auto term3_sq = Lambda_sq*0.25*get_TT_el3(el1, el2+1, el3+2)*(J+3.)*(J+4.)*(Jmpp+2.)*(Jmpp+3.)/(el2v*(el2v+1.)*(el3v+1.)*(el3v+2.));
#else
      auto Jmpp = J-2*el1;
      auto Jpmp = J-2*el2v;
      auto Jppm = J-2*el3v;
//MR_assert(J&1,"oops");
      auto eta_sq = (el2v-1.)*(el2v+2.)*(el3v-1.)*el3v;
      auto Lambda_sq = (J+2.)*(Jmpp+1)*(Jpmp+1)*Jppm;

      auto termsumsq = (el2v+1.) + 2. + 1./(el2v+1.);
      auto term25_sq = get_TT_el3(el1,el2,el3+1)*(J+2.)*(Jmpp+1)*(Jpmp+1)*Jppm*(el3v+2.)/(el2v*(el3v+1.)) * termsumsq;
      auto term3_sq = Lambda_sq*0.25*get_TT_el3(el1, el2+1, el3+2)*(J+3.)*(J+4.)*(Jmpp+2.)*(Jmpp+3.)/(el2v*(el2v+1.)*(el3v+1.)*(el3v+2.));
      auto res = term25_sq+term3_sq-2*sqrt(term25_sq*term3_sq);
      return res/eta_sq;
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
