/*
 *  This file is part of libsharp2.
 *
 *  libsharp2 is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  libsharp2 is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with libsharp2; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/* libsharp2 is being developed at the Max-Planck-Institut fuer Astrophysik */

/*! \file sharp_ylmgen.h
 *  Code for efficient calculation of Y_lm(phi=0,theta)
 *
 *  Copyright (C) 2005-2019 Max-Planck-Society
 *  \author Martin Reinecke
 */

#ifndef SHARP2_YLMGEN_H
#define SHARP2_YLMGEN_H

#include <vector>
using std::vector;

enum { sharp_minscale=0, sharp_limscale=1, sharp_maxscale=1 };
static constexpr double sharp_fbig=0x1p+800,sharp_fsmall=0x1p-800;
static constexpr double sharp_ftol=0x1p-60;
static constexpr double sharp_fbighalf=0x1p+400;

struct sharp_ylmgen_dbl2 { double a, b; };

class sharp_Ylmgen
  {
  public:
    sharp_Ylmgen(size_t l_max, size_t m_max, size_t spin);

    /*! Prepares the object for the calculation at \a m. */
    void prepare(size_t m_);
    /*! Returns a vector with \a lmax+1 entries containing
        normalisation factors that must be applied to Y_lm values computed for
        \a spin. */
    static vector<double> get_norm(size_t lmax, size_t spin);
    /*! Returns a vectorwith \a lmax+1 entries containing
        normalisation factors that must be applied to Y_lm values computed for
        first derivatives. */
    static vector<double> get_d1norm(size_t lmax);

    /* for public use; immutable during lifetime */
    size_t lmax, mmax, s;
    vector<double> cf;
    vector<double> powlimit;

    /* for public use; will typically change after call to Ylmgen_prepare() */
    size_t m;

    vector<double> alpha;
    vector<sharp_ylmgen_dbl2> coef;

    /* used if s==0 */
    vector<double> mfac, eps;

    /* used if s!=0 */
    size_t sinPow, cosPow;
    bool preMinus_p, preMinus_m;
    vector<double> prefac;
    vector<int> fscale;

    size_t mlo, mhi;
  private:
    /* used if s==0 */
    vector<double> root, iroot;

    /* used if s!=0 */
    vector<double> flm1, flm2, inv;
  };

#endif
