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
 *  Copyright (C) 2009-2026 Max-Planck-Society
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

// Support class for direct evaluation of Wigner 3j symbols,
// following https://arxiv.org/abs/2602.15605
// NOTE: compared to the paper, this tabulates the square root of the
// values, to save a sqrt() call in a few places.
template<typename Tsimd> class Wigner3j_direct_tables
  {
  private:
    vector<double> g, fct;
    Tsimd iota;
    static constexpr size_t safety = 4*Tsimd::size(); // safety margin beyond lmax
    inline static Tsimd sqr(Tsimd arg) { return arg*arg; }

  public:
    Wigner3j_direct_tables(size_t lmax)
      : g(2*lmax+1+safety), fct(2*lmax+1+safety)
      {
      double gcur = 1.;
      for (size_t i=0; i<g.size(); ++i, gcur*=(i-0.5)/i)
        {
        g[i] = sqrt(gcur);
        fct[i] = sqrt(1./(gcur*(2*i+1)));
        }
      for (size_t i=0; i<Tsimd::size(); ++i)
        iota[i] = double(i);
      }

    const vector<double> &G() const { return g; }
    const vector<double> &Fct() const { return fct; }
  };

template<typename Tsimd> class Wigner3j_direct
  {
  private:
    const vector<double> &g;
    const vector<double> &fct;
    Tsimd iota;

    int el1, el2;
    Tsimd el2v;

    // EE/TE
    Tsimd lmbda, x_lmbda_sq, lmbda_t1, lmbda_t2, x_eta_sq, x_eta;

    // EB
    Tsimd termsumsq, ebtmp1, ebtmp2;

    inline static Tsimd sqr(Tsimd arg) { return arg*arg; }

  public:
    Wigner3j_direct (const Wigner3j_direct_tables<Tsimd> &inp) : g(inp.G()), fct(inp.Fct())
      {
      for (size_t i=0; i<Tsimd::size(); ++i)
        iota[i] = double(i);
      }

    template<size_t opmask> void prep (int el1_, int el2_)
      {
      MR_assert(el1_>=0, "el1 must not be negative");
      MR_assert(el2_>=el1_, "el2 must not be smaller than el1");
      el1 = el1_;
      el2 = el2_;
      el2v = double(el2) + iota;

      // If el1<2, all the EE/TE/EB symbols are zero
      // Initialize everything to 0, avoiding NaNs
      if (el1<2)
        {
        x_eta_sq = lmbda = x_lmbda_sq = lmbda_t1 = lmbda_t2 = x_eta = termsumsq = ebtmp1 = ebtmp2 = 0;
        }
      else
        {
        // EE/TE/EB
        if constexpr (opmask&14)
          x_eta_sq = Tsimd(1.)/((el1-1.)*(el1+2.)*(el2v-1.)*el2v);
  
        // EE/TE
        if constexpr(opmask&6)
          {
          lmbda = sqrt(el1*(el1+1.)*(el2v+1.)*(el2v+2.));
          x_lmbda_sq = Tsimd(1.)/(lmbda*lmbda);
          lmbda_t1 = lmbda *(1.+2./el1);
          lmbda_t2 = lmbda/(((el1+1.)*(el2v+1.)*el1));
          x_eta = sqrt(x_eta_sq);
          }
  
        // EB
        if constexpr(opmask&8)
          {
          termsumsq = (el1+1.) + 2. + 1./(el1+1.);
          ebtmp1 = Tsimd(1.)/((el1+1.)*(el2v+2.));
          ebtmp2 = Tsimd(1.)/(el1*(el2v+1.));
          }
        }
      }

    template<size_t opmask> std::array<Tsimd,4> calc(int ofs) const
      {
      std::array<Tsimd,4> res;
      // we use this for TT/EE/TE
      Tsimd threej_000;
      if constexpr (opmask&7)
        threej_000 = loadu<Tsimd>(&fct[el2+ofs]) * loadu<Tsimd>(&g[el2-el1+ofs]) * g[ofs] * g[el1-ofs];
      // TT
      if constexpr (opmask&1)
        res[0] = threej_000*threej_000;
      // EE/TE
      if constexpr (opmask&6)
        {
        Tsimd Jpmp = 2.*ofs;  // actually scalar
        Tsimd J = Jpmp+2*el2v;
        Tsimd Jmpp = J-2*el1;
        Tsimd el3v = el2v-el1 + Jpmp;
        Tsimd Jppm = J-2*el3v;  // actually scalar
        
        auto lmbda2 = (J+2.) * (Jppm+1.);
            
        auto A = lmbda_t1 - lmbda2*lmbda_t2;
        auto B_sq = 0.25 * x_lmbda_sq * lmbda2 * (Jmpp+1.) * (Jmpp+2.) * (J+3.) * (Jppm+2.)  * Jpmp * (Jpmp-1.);
  
        auto threej_000_2 = loadu<Tsimd>(&fct[el2+1+ofs]) * loadu<Tsimd>(&g[el2+1-el1+ofs]) * g[ofs-1] * g[el1+1-ofs];

        auto tmp1 = A*threej_000;
        auto tmp2 = -sqrt(B_sq)*threej_000_2;

        auto threej_0p2m2 = (tmp1+tmp2)*x_eta;
        if constexpr(opmask&2)  // TE
          res[1] = threej_000*threej_0p2m2;
        if constexpr(opmask&4)  // EE
          res[2] = threej_0p2m2*threej_0p2m2;
        }
      // EB
      if constexpr(opmask&8)
        {
        // Note the "+1" here, since we are shifted, and J will be odd
        auto el3v = 2.*ofs+el2v+1-el1;
        auto J = el3v+el1+el2v;
        auto Jmpp = J-2*el1;
        auto Jpmp = J-2*el2v;
        auto Jppm = J-2*el3v;
        auto Lambda_sq = (J+2.)*(Jppm+1.)*(Jmpp+1.)*Jpmp;

        auto t1 = loadu<Tsimd>(&g[el2+1-el1+ofs])*g[ofs];
        auto t2 = sqr(loadu<Tsimd>(&fct[el2+1+ofs]) * g[el1-ofs]*t1);
        auto t3 = sqr(loadu<Tsimd>(&fct[el2+2+ofs]) * g[el1+1-ofs]*t1);
        auto term25_sq = t2*(el2v+2.)* termsumsq;
        auto term3_sq = t3*0.25*(J+3.)*(J+4.)*(Jppm+2.)*(Jppm+3.)*ebtmp1;
        auto tmp = term25_sq+term3_sq-2.*sqrt(term25_sq*term3_sq);
        res[3] = tmp*Lambda_sq*x_eta_sq*ebtmp2;
        }
      return res;
      }
  };

}

using detail_wigner3j::wigner3j;
using detail_wigner3j::wigner3j_int;
using detail_wigner3j::wigner3j_ncoef_int;

using detail_wigner3j::wigner3j_00_vec_squared_compact;

using detail_wigner3j::flexible_wigner3j;
using detail_wigner3j::flexible_wigner3j_vec;

using detail_wigner3j::Wigner3j_direct_tables;
using detail_wigner3j::Wigner3j_direct;
}

#endif
