/*
 *  This file is part of the MR utility library.
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

/** \file ducc0/math/solvers.h
 *  Various solvers for linear equation systems
 *
 *  \copyright Copyright (C) 2022 Max-Planck-Society, Ignace Bogaert
 *  \author Martin Reinecke
 */

#ifndef DUCC0_SOLVERS_H
#define DUCC0_SOLVERS_H

#include <cmath>
#include "ducc0/infra/mav.h"

namespace ducc0 {

namespace detail_solvers {

using namespace std;

template<typename T> auto sym_ortho(T a, T b, T &c, T &s, T &r)
  {
  r = hypot(a,b);
  c = a/r;
  s = b/r;
  }

/* Code based on scipy's LSMR implementation
   Original copyright:
   Copyright (C) 2010 David Fong and Michael Saunders

   David Chin-lung Fong            clfong@stanford.edu
   Institute for Computational and Mathematical Engineering
   Stanford University

   Michael Saunders                saunders@stanford.edu
   Systems Optimization Laboratory
   Dept of MS&E, Stanford University. */
template <typename Tx, typename Tb, size_t xdim, size_t bdim,
  typename Top, typename Top_adj, typename Tnormx, typename Tnormb>
  auto lsmr(Top op, Top_adj op_adj, Tnormx fnormx, Tnormb fnormb,
            const cmav<Tb,bdim> &b, vmav<Tx,xdim> &x, const cmav<Tx,xdim> &x0,
            double damp, double atol, double btol, double conlim,
            size_t maxiter, bool verbose, size_t nthreads)
  {
  static_assert(is_same<Tx,float >::value || is_same<Tx,complex<float >>::value
              ||is_same<Tx,double>::value || is_same<Tx,complex<double>>::value,
                "bad type for x");
  static_assert(is_same<Tb,float >::value || is_same<Tb,complex<float >>::value
              ||is_same<Tb,double>::value || is_same<Tb,complex<double>>::value,
                "bad type for b");
  using Tfx = typename conditional<is_same<Tx,float>::value
                                 ||is_same<Tx,complex<float>>::value, float, double>::type;
  using Tfb = typename conditional<is_same<Tb,float>::value
                                 ||is_same<Tb,complex<float>>::value, float, double>::type;
  static_assert(is_same<Tfx,Tfb>::value, "mixed single/double precision detected");
  vmav<Tb, bdim> u(b.shape());
  mav_apply([](auto &v1, const auto &v2) { v1=v2; }, nthreads, u, b);
  auto normb = fnormb(b);
  MR_assert(x.shape()==x0.shape(), "shape mismatch");
  mav_apply([](auto &v1, const auto &v2) { v1=v2; }, nthreads, x, x0);

  vmav<Tx, xdim> xtmp(x0.shape());
  vmav<Tb, bdim> btmp(b.shape());
  {
  op(x,btmp);
  mav_apply([](auto &v1, const auto &v2) { v1-=v2; }, nthreads, u, btmp);
  }
  auto beta = fnormb(u);

  vmav<Tx, xdim> v(x0.shape());
  double alpha = 0;
  if (beta>0)
    {
    mav_apply([xbeta=Tfb(1./beta)](auto &v1) { v1*=xbeta; }, nthreads, u);
    op_adj(u,v);
    alpha = fnormx(v);
    }
  else
    mav_apply([](auto &v1) { v1=0; }, nthreads, v);

  if (alpha>0)
    mav_apply([xalpha=Tfx(1./alpha)](auto &v1) { v1*=xalpha; }, nthreads, v);

  // Initialize variables for 1st iteration.
  size_t itn = 0;
  double zetabar = alpha*beta,
         alphabar = alpha,
         rho = 1,
         rhobar = 1,
         cbar = 1,
         sbar = 0;

  vmav<Tx, xdim> h(v.shape());
  mav_apply([](auto &v1, const auto &v2) { v1=v2; }, nthreads, h, v);
  vmav<Tx, xdim> hbar(h.shape());
  mav_apply([](auto &v1) { v1=0; }, nthreads, hbar);

  // Initialize variables for estimation of ||r||.
  double betadd = beta,
         betad = 0,
         rhodold = 1,
         tautildeold = 0,
         thetatilde = 0,
         zeta = 0,
         d = 0;

  // Initialize variables for estimation of ||A|| and cond(A)
  double normA2 = alpha * alpha,
         maxrbar = 0,
         minrbar = 1e+100,
         normA = sqrt(normA2),
         condA = 1,
         normx = 0;

  // Items for use in stopping rules, normb set earlier
  size_t istop = 0;
  double ctol = (conlim>0) ? (1/conlim) : 0;
  auto normr = beta;

  auto normar = alpha*beta;

  if (verbose)
    cout << "0" << " " << normr << " " << normar << " " << normA << " " << condA << endl;

  if (normar==0)  // x0 is a solution
    return make_tuple(x, istop, itn, normr, normar, normA, condA, normx, normb);

  if (normb==0)  // zero vector is a solution
    {
    mav_apply([](auto &v1) { v1=0; }, nthreads, x);
    return make_tuple(x, istop, itn, normr, normar, normA, condA, normx, normb);
    }

  for (itn=1; itn<=maxiter; ++itn)
    {
    // Perform the next step of the bidiagonalization to obtain the
    // next  beta, u, alpha, v.  These satisfy the relations
    //     beta*u  =  A@v   -  alpha*u,
    //     alpha*v  =  A'@u  -  beta*v.

    op(v, btmp);
    mav_apply([alpha](auto &v1, const auto &v2) {v1 = v2-Tfb(alpha)*v1;}, nthreads, u, btmp);
    beta = fnormb(u);

    if (beta>0)
      {
      mav_apply([xbeta=Tb(1./beta)](auto &v1) {v1 *= xbeta;}, nthreads, u);
      op_adj(u, xtmp);
      mav_apply([beta](auto &v1, const auto &v2) {v1 = v2-Tfx(beta)*v1;}, nthreads, v, xtmp);
      alpha = fnormx(v);
      if (alpha>0)
        mav_apply([xalpha=Tfx(1./alpha)](auto &v1) {v1 *= xalpha;}, nthreads, v);
      }

    // At this point, beta = beta_{k+1}, alpha = alpha_{k+1}.

    // Construct rotation Qhat_{k,2k+1}.
    double chat, shat, alphahat;
    sym_ortho(alphabar, damp, chat, shat, alphahat);

    // Use a plane rotation (Q_i) to turn B_i to R_i
    auto rhoold = rho;
    double c, s;
    sym_ortho(alphahat, beta, c, s, rho);
    auto thetanew = s*alpha;
    alphabar = c*alpha;

    // Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar
    auto rhobarold = rhobar;
    auto zetaold = zeta;
    auto thetabar = sbar * rho;
    auto rhotemp = cbar * rho;
    sym_ortho(cbar*rho, thetanew, cbar, sbar, rhobar);
    zeta = cbar * zetabar;
    zetabar = - sbar * zetabar;

    // Update h, h_hat, x.
    {
    auto fct = Tfx(- thetabar * rho / (rhoold * rhobarold));
    auto fct2 = Tfx(zeta / (rho * rhobar));
    auto fct3 = Tfx(-thetanew/rho);
    mav_apply([fct, fct2, fct3](auto &vhbar, auto &vx, auto &vh, const auto &vv)
      {
      vhbar = vhbar*fct + vh;
      vx += vhbar*fct2;
      vh = vh*fct3 + vv;
      }, nthreads, hbar, x, h, v);
    }

    // Estimate of ||r||.

    // Apply rotation Qhat_{k,2k+1}.
    auto betaacute = chat * betadd;
    auto betacheck = -shat * betadd;

    // Apply rotation Q_{k,k+1}.
    auto betahat = c * betaacute;
    betadd = -s * betaacute;

    // Apply rotation Qtilde_{k-1}.
    // betad = betad_{k-1} here.

    auto thetatildeold = thetatilde;
    double ctildeold, stildeold, rhotildeold;
    sym_ortho(rhodold, thetabar, ctildeold, stildeold, rhotildeold);
    thetatilde = stildeold * rhobar;
    rhodold = ctildeold * rhobar;
    betad = - stildeold * betad + ctildeold * betahat;

    // betad   = betad_k here.
    // rhodold = rhod_k  here.

    tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold;
    auto taud = (zeta - thetatilde * tautildeold) / rhodold;
    d += betacheck * betacheck;
    normr = sqrt(d + (betad - taud)*(betad-taud) + betadd * betadd);

    // Estimate ||A||.
    normA2 += beta * beta;
    normA = sqrt(normA2);
    normA2 += alpha * alpha;

    // Estimate cond(A).
    maxrbar = max(maxrbar, rhobarold);
    if (itn>1)
      minrbar = min(minrbar, rhobarold);
    condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp);

    // Test for convergence.

    // Compute norms for convergence testing.
    normar = abs(zetabar);
    normx = fnormx(x);

    // Now use these norms to estimate certain other quantities,
    // some of which will be small near a solution.
    auto test1 = normr / normb;
    auto test2 = (normA*normr==0) ? 1./0. : normar / (normA*normr);
    auto test3 = 1 / condA;
    auto t1 = test1 / (1 + normA * normx / normb);
    auto rtol = btol + atol * normA * normx / normb;

    // The following tests guard against extremely small values of
    // atol, btol or ctol.  (The user may have set any or all of
    // the parameters atol, btol, conlim to 0.)
    // The effect is equivalent to the normAl tests using
    // atol = eps,  btol = eps,  conlim = 1/eps.

    if (itn >= maxiter) istop=7;  // did not find a solution in time
    if (1+test3 <= 1) istop=6;
    if (1+test2 <= 1) istop=5;
    if (1+t1 <= 1) istop=4;

    // Allow for tolerances set by the user.
    if (test3 <= ctol) istop=3;  // cond(A) > conlim
    if (test2 <= atol) istop=2;  // x solves least-squares problem
    if (test1 <= rtol) istop=1;  // x is an approximate solution

    if (verbose)
      cout << itn << " " << normr << " " << normar << " " << normA << " " << condA << endl;

    if (istop>0)
      break;
    }
  return make_tuple(x, istop, itn, normr, normar, normA, condA, normx, normb);
  }

}

using detail_solvers::lsmr;

}

#endif
