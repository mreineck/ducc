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

/** \file ducc0/math/solvers.h
 *  Various solvers for linear equation systems
 *
 *  \copyright Copyright (C) 2022-2025 Max-Planck-Society
 *  \author Martin Reinecke
 */

#if 0

/* Code based on scipy's LSQR implementation
   Original reference:
   C. C. Paige and M. A. Saunders, Algorithm 583; LSQR: Sparse linear
   equations and least-squares problems, TOMS 8(2), 195--209 (1982).
*/
/* NOTE: "x" must contain an initial guess (if in doubt, use a zero vector) */
template <fp_or_complex Tx, fp_or_complex Tb, size_t xdim, size_t bdim,
  typename Top, typename Top_adj, typename Tnormx, typename Tnormb>
  requires same_as<real_type_t<Tx>, real_type_t<Tb>>
    && invocable<Top&,     const vmav<Tx,xdim>&, const vmav<Tb,bdim>&>
    && invocable<Top_adj&, const vmav<Tb,bdim>&, const vmav<Tx,xdim>&>
  auto lsqr(Top op, Top_adj op_adj, Tnormx fnormx, Tnormb fnormb,
            const cmav<Tb,bdim> &b, const vmav<Tx,xdim> &x,
            double damp, double atol, double btol, double conlim,
            size_t maxiter, bool verbose, size_t nthreads)
  {
  using Tf = real_type_t<Tx>;
  vmav<Tb, bdim> u(b.shape(), UNINITIALIZED);
  mav_apply([](auto &v1, const auto &v2) { v1=v2; }, nthreads, u, b);
  auto bnorm = fnormb(b);

  vmav<Tx, xdim> xtmp(x.shape(), UNINITIALIZED);
  vmav<Tb, bdim> btmp(b.shape(), UNINITIALIZED);
  {
  op(x,btmp);
  mav_apply([](auto &v1, const auto &v2) { v1-=v2; }, nthreads, u, btmp);
  }
  auto beta = fnormb(u);

  vmav<Tx, xdim> v(x.shape(), UNINITIALIZED);
  double alpha = 0;
  if (beta>0)
    {
    mav_apply([xbeta=Tf(1./beta)](auto &v1) { v1*=xbeta; }, nthreads, u);
    op_adj(u,v);
    alpha = fnormx(v);
    }
  else
    mav_apply([](auto &v1, const auto &v2) { v1=v2; }, nthreads, v, x);

  if (alpha>0)
    mav_apply([xalpha=Tf(1./alpha)](auto &v1) { v1*=xalpha; }, nthreads, v);

  vmav<Tx, xdim> w(x.shape(), UNINITIALIZED);
  mav_apply([](auto &v1, const auto &v2) { v1=v2; }, nthreads, w, v);

  size_t istop = 0,
         itn = 0;
  double rhobar = alpha,
         phibar = beta,
         rnorm = beta,
         r1norm = rnorm,
         anorm = 0,
         acond = 0,
         xnorm = 0,
         xxnorm = 0,
         dampsq = damp*damp,
         arnorm = alpha*beta,
         ddnorm = 0;
  double cs2=-1,
         sn2=0,
         z=0,
         ctol=0,
         res2=0;
  if (conlim>0)
    ctol = 1/conlim;

  if (arnorm==0)  // initial guess is a solution
    return make_tuple(x, istop, itn, rnorm, arnorm, anorm, acond, xnorm, bnorm);

  for (itn=1; itn<=maxiter; ++itn)
    {
    // Perform the next step of the bidiagonalization to obtain the
    // next  beta, u, alpha, v.  These satisfy the relations
    //     beta*u  =  A@v   -  alpha*u,
    //     alpha*v  =  A'@u  -  beta*v.

    op(v, btmp);
    mav_apply([alpha](auto &v1, const auto &v2) {v1 = v2-Tf(alpha)*v1;}, nthreads, u, btmp);
    beta = fnormb(u);

    if (beta>0)
      {
      mav_apply([xbeta=Tb(1./beta)](auto &v1) {v1 *= xbeta;}, nthreads, u);
      anorm = sqrt(anorm*anorm + alpha*alpha + beta*beta + dampsq);
      op_adj(u, xtmp);
      mav_apply([beta](auto &v1, const auto &v2) {v1 = v2-Tf(beta)*v1;}, nthreads, v, xtmp);
      alpha = fnormx(v);
      if (alpha>0)
        mav_apply([xalpha=Tf(1./alpha)](auto &v1) {v1 *= xalpha;}, nthreads, v);
      }

    // Use a plane rotation to eliminate the damping parameter.
    // This alters the diagonal (rhobar) of the lower-bidiagonal matrix.
    double rhobar1, cs1, sn1, psi;
    if (damp>0)
      {
      rhobar1 = sqrt(rhobar*rhobar + dampsq);
      cs1 = rhobar / rhobar1;
      sn1 = damp / rhobar1;
      psi = sn1 * phibar;
      phibar = cs1 * phibar;
      }
    else
      {
      cs1 = 1;
      sn1 = 0;
      rhobar1 = rhobar;
      psi = 0.;
      }

    // Use a plane rotation to eliminate the subdiagonal element (beta)
    // of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.
    double cs, sn, rho;
    sym_ortho(rhobar1, beta, cs, sn, rho);

    double theta = sn * alpha;
    rhobar = -cs * alpha;
    double phi = cs * phibar;
    phibar = sn * phibar;
    double tau = sn * phi;

    // Update x and w.
    double t1 = phi / rho;
    double t2 = -theta / rho;
    {
    double tmp = fnormx(w)/rho;
    ddnorm += tmp*tmp;
    }

    mav_apply([t1=Tf(t1),t2=Tf(t2)](auto &v1, auto &v2, const auto &v3)
      {
      v1 += t1*v2;
      v2 = v3+t2*v2;
      }, nthreads, x, w, v);

    // Use a plane rotation on the right to eliminate the
    // super-diagonal element (theta) of the upper-bidiagonal matrix.
    // Then use the result to estimate norm(x).
    auto delta = sn2 * rho;
    auto gambar = -cs2 * rho;
    auto rhs = phi - delta * z;
    auto zbar = rhs / gambar;
    xnorm = sqrt(xxnorm + zbar*zbar);
    auto gamma = sqrt(gambar*gambar + theta*theta);
    cs2 = gambar / gamma;
    sn2 = theta / gamma;
    z = rhs / gamma;
    xxnorm += z*z;

    // Test for convergence.
    // First, estimate the condition of the matrix  Abar,
    // and the norms of  rbar  and  Abar'rbar.
    acond = anorm * sqrt(ddnorm);
    auto res1 = phibar*phibar;
    res2 += psi*psi;
    rnorm = sqrt(res1 + res2);
    arnorm = alpha * abs(tau);

    // Distinguish between
    //     r1norm = ||b - Ax|| and
    //     r2norm = rnorm in current code
    //            = sqrt(r1norm^2 + damp^2*||x - x0||^2).
    // Estimate r1norm from
    //     r1norm = sqrt(r2norm^2 - damp^2*||x - x0||^2).
    // Although there is cancellation, it might be accurate enough.
    if (damp>0)
      {
      auto r1sq = rnorm*rnorm - dampsq*xxnorm;
      r1norm = sqrt(abs(r1sq));
      if (r1sq<0)
        r1norm = -r1norm;
      }
    else
      r1norm = rnorm;

    // Now use these norms to estimate certain other quantities,
    // some of which will be small near a solution.
    auto test1 = rnorm / bnorm;
    auto test2 = (anorm*rnorm==0) ?
      numeric_limits<double>::max() : arnorm / (anorm*rnorm);
    auto test3 = 1 / acond;
    t1 = test1 / (1 + anorm * xnorm / bnorm);
    auto rtol = btol + atol * anorm * xnorm / bnorm;

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
      cout << itn << " " << rnorm << " " << arnorm << " " << anorm << " " << acond << endl;

    if (istop>0)
      break;
    }
  return make_tuple(x, istop, itn, rnorm, arnorm, anorm, acond, xnorm, bnorm);
  }

#endif
