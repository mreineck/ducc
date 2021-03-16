/*
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

/*! \file alm.h
 *  Class for storing spherical harmonic coefficients.
 *
 *  Copyright (C) 2003-2021 Max-Planck-Society
 *  \author Martin Reinecke
 *
 *  For the a_lm rotation code:
 *  Copyright (c) 2018 Richard Mikael Slevinsky and Everett Dawes
 */

#ifndef DUCC0_ALM_H
#define DUCC0_ALM_H

#if 1
#include <complex>
#include <cmath>
#include "ducc0/infra/threading.h"
#endif

#include "ducc0/infra/mav.h"
#include "ducc0/infra/error_handling.h"

namespace ducc0 {

namespace detail_alm {

using namespace std;

/*! Base class for calculating the storage layout of spherical harmonic
    coefficients. */
class Alm_Base
  {
  protected:
    size_t lmax, arrsize;
    vector<size_t> mval;
    vector<ptrdiff_t> mstart;

  public:
    /*! Returns the total number of coefficients for maximum quantum numbers
        \a l and \a m. */
    static size_t Num_Alms (size_t l, size_t m)
      {
      MR_assert(m<=l,"mmax must not be larger than lmax");
      return ((m+1)*(m+2))/2 + (m+1)*(l-m);
      }
    size_t Num_Alms() const
      { return arrsize; }

    Alm_Base (size_t lmax_, const vector<size_t> &mval_,
              const vector<ptrdiff_t> &mstart_)
      : lmax(lmax_), mval(mval_)
      {
      MR_assert(mval.size()>0, "no m indices supplied");
      MR_assert(mstart_.size()==mval.size(), "mval and mstart have different sizes");
      for (size_t i=0; i<mval.size(); ++i)
        {
        MR_assert(mval[i]<=lmax, "m >= lmax");
        if (i>0)
          MR_assert(mval[i]>mval[i-1], "m not strictly ascending");
        }
      mstart.resize(mval.back()+1, -2*lmax);
      arrsize=0;
      for (size_t i=0; i<mval.size(); ++i)
        {
        mstart[mval[i]] = mstart_[i];
        arrsize = size_t(max(ptrdiff_t(arrsize), mstart_[i]+ptrdiff_t(lmax+1)));
        }
      }

    Alm_Base (size_t lmax_, const vector<size_t> &mval_)
      : lmax(lmax_), mval(mval_)
      {
      MR_assert(mval.size()>0, "no m indices supplied");
      for (size_t i=0; i<mval.size(); ++i)
        {
        MR_assert(mval[i]<=lmax, "m >= lmax");
        if (i>0)
          MR_assert(mval[i]>mval[i-1], "m not strictly ascending");
        }
      mstart.resize(mval.back()+1, -2*lmax);
      for (size_t i=0, cnt=0; i<mval.size(); ++i, cnt+=lmax-mval[i]+1)
        mstart[mval[i]] = ptrdiff_t(cnt)-ptrdiff_t(mval[i]);
      arrsize = size_t(mstart.back()+ptrdiff_t(lmax+1));
      }

    /*! Constructs an Alm_Base object with given \a lmax and \a mmax. */
    Alm_Base (size_t lmax_, size_t mmax_)
      : lmax(lmax_), mval(mmax_+1), mstart(mmax_+1)
      {
      ptrdiff_t idx = 0;
      for (size_t m=0; m<=mmax_; ++m)
        {
        mval[m] = m;
        mstart[m] = idx-m;
        idx += lmax-m+1;
        }
      arrsize = Num_Alms(lmax_, mmax_);
      }

    /*! Returns the maximum \a l */
    size_t Lmax() const { return lmax; }
    /*! Returns the maximum \a m */
    size_t Mmax() const { return mval.back(); }

    size_t n_entries() const { return arrsize; }

    /*! Returns an array index for a given m, from which the index of a_lm
        can be obtained by adding l. */
    size_t index_l0 (size_t m) const
      { return mstart[m]; }

    /*! Returns the array index of the specified coefficient. */
    size_t index (size_t l, size_t m) const
      { return index_l0(m) + l; }

    bool conformable(const Alm_Base &other) const
      {
      return (lmax==other.lmax) && (mval==other.mval) && (mstart==other.mstart);
      }
    bool complete() const
      { return mval.size() == lmax+1; }
  };

/*! Class for storing spherical harmonic coefficients. */
template<typename T> class Alm: public Alm_Base
  {
  private:
    mav<T,1> alm;

    template<typename Func> void applyLM(Func func)
      {
      for (auto m: mval)
        for (size_t l=m; l<=lmax; ++l)
          func(l,m,alm.v(index(l,m)));
      }

  public:
    /*! Constructs an Alm object with given \a lmax and \a mmax. */
    Alm (mav<T,1> &data, size_t lmax_, size_t mmax_)
      : Alm_Base(lmax_, mmax_), alm(data)
      { MR_assert(alm.size()==Num_Alms(lmax, mmax_), "bad array size"); }
    Alm (const mav<T,1> &data, size_t lmax_, size_t mmax_)
      : Alm_Base(lmax_, mmax_), alm(data)
      { MR_assert(alm.size()==Num_Alms(lmax, mmax_), "bad array size"); }
    Alm (size_t lmax_=0, size_t mmax_=0)
      : Alm_Base(lmax_,mmax_), alm ({Num_Alms(lmax,mmax_)}) {}

    /*! Sets all coefficients to zero. */
    void SetToZero ()
      { alm.fill(0); }

    /*! Multiplies all coefficients by \a factor. */
    template<typename T2> void Scale (const T2 &factor)
      { for (size_t m=0; m<alm.size(); ++m) alm.v(m)*=factor; }
    /*! \a a(l,m) *= \a factor[l] for all \a l,m. */
    template<typename T2> void ScaleL (const mav<T2,1> &factor)
      {
      MR_assert(factor.size()>size_t(lmax),
        "alm.ScaleL: factor array too short");
      applyLM([&factor](size_t l, size_t /*m*/, T &v){v*=factor(l);}); 
      }
    /*! \a a(l,m) *= \a factor[m] for all \a l,m. */
    template<typename T2> void ScaleM (const mav<T2,1> &factor)
      {
      MR_assert(factor.size()>size_t(Mmax()),
        "alm.ScaleM: factor array too short");
      applyLM([&factor](size_t /*l*/, size_t m, T &v){v*=factor(m);}); 
      }
    /*! Adds \a num to a_00. */
    template<typename T2> void Add (const T2 &num)
      {
      MR_assert(mval[0]==0, "cannot add a constant: no m=0 mode present");
      alm.v(index_l0(0))+=num;
      }

    /*! Returns a reference to the specified coefficient. */
    T &operator() (size_t l, size_t m)
      { return alm.v(index(l,m)); }
    /*! Returns a constant reference to the specified coefficient. */
    const T &operator() (size_t l, size_t m) const
      { return alm(index(l,m)); }
    /*! Returns a constant reference to the specified coefficient. */
    const T &c(size_t l, size_t m) const
      { return alm(index(l,m)); }

    /*! Returns a pointer for a given m, from which the address of a_lm
        can be obtained by adding l. */
    T *mstart (size_t m)
      { return &alm.v(index_l0(m)); }
    /*! Returns a pointer for a given m, from which the address of a_lm
        can be obtained by adding l. */
    const T *mstart (size_t m) const
      { return &alm(index_l0(m)); }

    /*! Returns a constant reference to the a_lm data. */
    const mav<T,1> &Alms() const { return alm; }

    /*! Returns a reference to the a_lm data. */
    mav<T,1> &Alms() { return alm; }

    ptrdiff_t stride() const
      { return alm.stride(0); }

    /*! Adds all coefficients from \a other to the own coefficients. */
    void Add (const Alm &other)
      {
      MR_assert (conformable(other), "A_lm are not conformable");
      for (size_t m=0; m<alm.size(); ++m)
        alm.v(m) += other.alm(m);
      }
  };

#if 1
// the following struct is an adaptation of the algorithms found in
// https://github.com/MikaelSlevinsky/FastTransforms


struct ft_partial_sph_isometry_plan
  {
  static double Gy_index3(int l, int i, int j)
    {
    if (i+j == 2*l)
      {
      if (l+2 <= i && i <= 2*l)
        return (j+1)*(j+2);
      else if (0 <= i && i <= l-1)
        return -(j+1)*(j+2);
      else if (i==j+2) // j+1==l==i-1
        return 2*(j+1)*(j+2);
      else if (i==j)  // i==j==l
        return -2*(j+1)*(j+2);
      }
    else if (i+j == 2*l+2)
      {
      if (2 <= i && i <= l)
        return -(i-1)*i;
      else if (l+3 <= i && i <= 2*l+2)
        return (i-1)*i;
      }
    return 0.0;
    }

  static double Y_index(int l, int i, int j) // l>=0, i>=0, j>=i
    {
    auto r1 = Gy_index3(l, 2*l-i  , i)*Gy_index3(l, 2*l-i  , j);
    auto r2 = Gy_index3(l, 2*l-i+1, i)*Gy_index3(l, 2*l-i+1, j);
    auto r3 = Gy_index3(l, 2*l-i+2, i)*Gy_index3(l, 2*l-i+2, j);
    return (copysign(sqrt(abs(r1)),r1)
          + copysign(sqrt(abs(r2)),r2)
          + copysign(sqrt(abs(r3)),r3))
          *0.25/double((2*l+1)*(2*l+3));
    }
  static double Z_index(int l, int i, int j)
    {
    return (i==j) ? (j+1)*(2*l+1-j)/double((2*l+1)*(2*l+3)) : 0.0;
    }

  struct ft_symmetric_tridiagonal
    {
    vector<double> a, b;
    int n;

    ft_symmetric_tridiagonal(int N)
      : a(N), b(N-1), n(N) {}
    };

  class ft_symmetric_tridiagonal_symmetric_eigen
    {
    private:
      vector<double> A, B, C, lambda;
      int sign;

    public:
      int n;

    private:
      template<typename Tv, size_t N> int eval_helper
        (int jmin, const vector<double> &c, vector<double> &f) const
        {
        constexpr double eps = 0x1p-52;
        constexpr double floatmin = 0x1p-1022;

        if (n<1)
          {
          for (int j=jmin; j<n; ++j)
            f[j] = 0.0;
          return n;
          }
        constexpr size_t vlen=Tv::size();
        constexpr size_t step=vlen*N;
        int j=jmin;
        for (; j+int(step)<=n; j+=int(step))
          {
          array<Tv, N> vk, vkp1, nrm, X, fj;
          for (size_t i=0; i<N; ++i)
            {
            vk[i] = 1;
            vkp1[i] = 0;
            nrm[i] = 1;
            X[i] = Tv::loadu(&lambda[j+i*Tv::size()]);
            fj[i] = c[n-1];
            }
          for (int k=n-1; k>0; --k)
            {
            Tv maxnrm(0);
            for (size_t i=0; i<N; ++i)
              {
              auto vkm1 = (A[k]*X[i]+B[k])*vk[i] - C[k]*vkp1[i];
              vkp1[i] = vk[i];
              vk[i] = vkm1;
              nrm[i] += vkm1*vkm1;
              maxnrm = max(maxnrm, nrm[i]);
              fj[i] += vkm1*c[k-1];
              }
            if (any_of(maxnrm > eps/floatmin))
              for (size_t i=0; i<N; ++i)
                {
                nrm[i] = Tv(1.0)/sqrt(nrm[i]);
                vkp1[i] *= nrm[i];
                vk[i] *= nrm[i];
                fj[i] *= nrm[i];
                nrm[i] = 1.0;
                }
            }
          for (size_t i=0; i<N; ++i)
            for (size_t q=0; q<vlen; ++q)
              f[j+vlen*i+q] = fj[i][q]*copysign(1.0/sqrt(nrm[i][q]),sign*vk[i][q]);
          }
        return j;
        }

    public:
      ft_symmetric_tridiagonal_symmetric_eigen() {}

      ft_symmetric_tridiagonal_symmetric_eigen (const ft_symmetric_tridiagonal &T,
        const vector<double> &lambda_, const int sign_)
        : A(T.n), B(T.n), C(T.n), lambda(lambda_), sign(sign_), n(T.n)
        {
        if (n>1)
          {
          A[n-1] = 1/T.b[n-2];
          B[n-1] = -T.a[n-1]/T.b[n-2];
          }
        for (int i=n-2; i>0; i--)
          {
          A[i] = 1/T.b[i-1];
          B[i] = -T.a[i]/T.b[i-1];
          C[i] = T.b[i]/T.b[i-1];
          }
        }

      void eval (const vector<double> &x, vector<double> &y) const
        {
        int j = eval_helper<native_simd<double>,4>(0, x, y);
        j = eval_helper<native_simd<double>,2>(j, x, y);
        j = eval_helper<native_simd<double>,1>(j, x, y);
        eval_helper<simd<double,1>,1>(j, x, y);
        }
    };

  ft_symmetric_tridiagonal_symmetric_eigen F11, F21, F12, F22;
  int l;

  ft_partial_sph_isometry_plan(const int l_)
    : l(l_)
    {
    int n11 = l/2;
    ft_symmetric_tridiagonal Y11(n11);
    for (int i = 0; i < n11; i++)
      Y11.a[n11-1-i] = Y_index(l, 2*i+1, 2*i+1);
    for (int i = 0; i < n11-1; i++)
      Y11.b[n11-2-i] = Y_index(l, 2*i+1, 2*i+3);
    vector<double>lambda11(n11);
    for (int i = 0; i < n11; i++)
      lambda11[n11-1-i] = Z_index(l, 2*i+1, 2*i+1);
    int sign = (l%4)/2 == 1 ? 1 : -1;
    F11 = ft_symmetric_tridiagonal_symmetric_eigen(Y11, lambda11, sign);

    int n21 = (l+1)/2;
    ft_symmetric_tridiagonal Y21(n21);
    for (int i = 0; i < n21; i++)
      Y21.a[n21-1-i] = Y_index(l, 2*i, 2*i);
    for (int i = 0; i < n21-1; i++)
      Y21.b[n21-2-i] = Y_index(l, 2*i, 2*i+2);
    vector<double> lambda21(n21);
    for (int i = 0; i < n21; i++)
      lambda21[i] = Z_index(l, l+1-l%2+2*i, l+1-l%2+2*i);
    sign = ((l+1)%4)/2 == 1 ? -1 : 1;
    F21 = ft_symmetric_tridiagonal_symmetric_eigen(Y21, lambda21, sign);

    int n12 = (l+1)/2;
    ft_symmetric_tridiagonal Y12(n12);
    for (int i = 0; i < n12; i++)
      Y12.a[i] = Y_index(l, 2*i+l-l%2+1, 2*i+l-l%2+1);
    for (int i = 0; i < n12-1; i++)
      Y12.b[i] = Y_index(l, 2*i+l-l%2+1, 2*i+l-l%2+3);
    vector<double> lambda12(n12);
    for (int i = 0; i < n12; i++)
      lambda12[n12-1-i] = Z_index(l, 2*i, 2*i);
    F12 = ft_symmetric_tridiagonal_symmetric_eigen(Y12, lambda12, sign);

    int n22 = (l+2)/2;
    ft_symmetric_tridiagonal Y22(n22);
    for (int i = 0; i < n22; i++)
      Y22.a[i] = Y_index(l, 2*i+l+l%2, 2*i+l+l%2);
    for (int i = 0; i < n22-1; i++)
      Y22.b[i] = Y_index(l, 2*i+l+l%2, 2*i+l+l%2+2);
    vector<double> lambda22(n22);
    for (int i = 0; i < n22; i++)
      lambda22[i] = Z_index(l, l+l%2+2*i, l+l%2+2*i);
    sign = (l%4)/2 == 1 ? -1 : 1;
    F22 = ft_symmetric_tridiagonal_symmetric_eigen(Y22, lambda22, sign);
    }
  };


void xchg_yz(Alm<complex<double>> &alm, size_t nthreads)
  {
  auto lmax = alm.Lmax();
  MR_assert(lmax==alm.Mmax(), "lmax and mmax must be equal");

  if (lmax>0) // deal with l==1
    {
    auto t = -alm(1,0).real()/sqrt(2.);
    alm(1,0).real(-alm(1,1).imag()*sqrt(2.));
    alm(1,1).imag(t);
    }
  if (lmax<=1) return;
  execDynamic(lmax-1,nthreads,1,[&](ducc0::Scheduler &sched)
    {
    vector<double> tin(2*lmax+3), tout(2*lmax+3), tin2(2*lmax+3);
    while (auto rng=sched.getNext()) for(auto l=rng.lo+2; l<rng.hi+2; ++l)
      {
      ft_partial_sph_isometry_plan F(l);

      int mstart = 1+(l%2);
      for (int i=0; i<F.F11.n; ++i)
        tin[i] = alm(l,mstart+2*i).imag();
      F.F11.eval(tin, tout);
      for (int i=0; i<F.F11.n; ++i)
        alm(l,mstart+2*i).imag(tout[i]);

      mstart = l%2;
      for (int i=0; i<F.F22.n; ++i)
        tin[i] = alm(l,mstart+2*i).real();
      if (mstart==0)
        tin[0]/=sqrt(2.);
      F.F22.eval(tin, tout);
      if (mstart==0)
        tout[0]*=sqrt(2.);
      for (int i=0; i<F.F22.n; ++i)
        alm(l,mstart+2*i).real(tout[i]);

      mstart = 2-(l%2);
      for (int i=0; i<F.F21.n; ++i)
        tin[i] = alm(l,mstart+2*i).imag();

      mstart = 1-(l%2);
      for (int i=0; i<F.F12.n; ++i)
        tin2[i] = alm(l,mstart+2*i).real();
      if (mstart==0)
        tin2[0]/=sqrt(2.);
      F.F21.eval(tin,tout);
      if (mstart==0)
        tout[0]*=sqrt(2.);
      for (int i=0; i<F.F12.n; ++i)
        alm(l,mstart+2*i).real(tout[i]);

      F.F12.eval(tin2,tout);
      mstart = 2-(l%2);
      for (int i=0; i<F.F21.n; ++i)
        alm(l,mstart+2*i).imag(tout[i]);
      }
    });
  }

template<typename T> void rotate_alm (Alm<complex<T>> &alm,
  double psi, double theta, double phi, size_t nthreads)
  {
  auto lmax=alm.Lmax();
  MR_assert (alm.complete(), "rotate_alm: need complete A_lm set");

  if (theta!=0)
    {
    if (psi!=0)
      for (size_t m=0; m<=lmax; ++m)
        {
        auto exppsi = polar(1.,-psi*m);
        for (size_t l=m; l<=lmax; ++l)
          alm(l,m)*=exppsi;
        }
    xchg_yz(alm, nthreads);
    for (size_t m=0; m<=lmax; ++m)
      {
      auto exptheta = polar(1.,-theta*m);
      for (size_t l=m; l<=lmax; ++l)
        alm(l,m)*=exptheta;
      }
    xchg_yz(alm, nthreads);
    if (phi!=0)
      for (size_t m=0; m<=lmax; ++m)
        {
        auto expphi = polar(1.,-phi*m);
        for (size_t l=m; l<=lmax; ++l)
          alm(l,m)*=expphi;
        }
    }
  else
    if (phi+psi!=0)
      for (size_t m=0; m<=lmax; ++m)
        {
        auto expang = polar(1.,-(psi+phi)*m);
        for (size_t l=m; l<=lmax; ++l)
          alm(l,m) *= expang;
        }
  }
#endif
}

using detail_alm::Alm_Base;
using detail_alm::Alm;
#if 1
using detail_alm::rotate_alm;
#endif
}

#endif
