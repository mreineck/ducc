/*! \file alm.h
 *  Class for storing spherical harmonic coefficients.
 *
 *  Copyright (C) 2003-2020 Max-Planck-Society
 *  \author Martin Reinecke
 */

#ifndef MRUTIL_ALM_H
#define MRUTIL_ALM_H

#include "mr_util/infra/mav.h"
#include "mr_util/infra/error_handling.h"

namespace mr {

/*! Base class for calculating the storage layout of spherical harmonic
    coefficients. */
class Alm_Base
  {
  protected:
    size_t lmax, mmax, tval;

  public:
    /*! Returns the total number of coefficients for maximum quantum numbers
        \a l and \a m. */
    static size_t Num_Alms (size_t l, size_t m)
      {
      MR_assert(m<=l,"mmax must not be larger than lmax");
      return ((m+1)*(m+2))/2 + (m+1)*(l-m);
      }

    /*! Constructs an Alm_Base object with given \a lmax and \a mmax. */
    Alm_Base (size_t lmax_=0, size_t mmax_=0)
      : lmax(lmax_), mmax(mmax_), tval(2*lmax+1) {}

    /*! Returns the maximum \a l */
    size_t Lmax() const { return lmax; }
    /*! Returns the maximum \a m */
    size_t Mmax() const { return mmax; }

    /*! Returns an array index for a given m, from which the index of a_lm
        can be obtained by adding l. */
    size_t index_l0 (size_t m) const
      { return ((m*(tval-m))>>1); }

    /*! Returns the array index of the specified coefficient. */
    size_t index (size_t l, size_t m) const
      { return index_l0(m) + l; }

    /*! Returns \a true, if both objects have the same \a lmax and \a mmax,
        else  \a false. */
    bool conformable (const Alm_Base &other) const
      { return ((lmax==other.lmax) && (mmax==other.mmax)); }
  };

/*! Class for storing spherical harmonic coefficients. */
template<typename T> class Alm: public Alm_Base
  {
  private:
    mav<T,1> alm;

  public:
    /*! Constructs an Alm object with given \a lmax and \a mmax. */
    Alm (mav<T,1> &data, size_t lmax_, size_t mmax_)
      : Alm_Base(lmax_, mmax_), alm(data)
      { MR_assert(alm.size()==Num_Alms(lmax, mmax), "bad array size"); }
    Alm (const mav<T,1> &data, size_t lmax_, size_t mmax_)
      : Alm_Base(lmax_, mmax_), alm(data)
      { MR_assert(alm.size()==Num_Alms(lmax, mmax), "bad array size"); }
    Alm (size_t lmax_=0, size_t mmax_=0)
      : Alm_Base(lmax_,mmax_), alm ({Num_Alms(lmax,mmax)}) {}

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
      for (size_t m=0; m<=mmax; ++m)
        for (size_t l=m; l<=lmax; ++l)
          operator()(l,m)*=factor(l);
      }
    /*! \a a(l,m) *= \a factor[m] for all \a l,m. */
    template<typename T2> void ScaleM (const mav<T2,1> &factor)
      {
      MR_assert(factor.size()>size_t(mmax),
        "alm.ScaleM: factor array too short");
      for (size_t m=0; m<=mmax; ++m)
        for (size_t l=m; l<=lmax; ++l)
          operator()(l,m)*=factor(m);
      }
    /*! Adds \a num to a_00. */
    template<typename T2> void Add (const T2 &num)
      { alm.v(0)+=num; }

    /*! Returns a reference to the specified coefficient. */
    T &operator() (size_t l, size_t m)
      { return alm.v(index(l,m)); }
    /*! Returns a constant reference to the specified coefficient. */
    const T &operator() (size_t l, size_t m) const
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

}

#endif
