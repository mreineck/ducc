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

/*! \file sharp_internal.h
 *  Internally used functionality for the spherical transform library.
 *
 *  Copyright (C) 2006-2020 Max-Planck-Society
 *  \author Martin Reinecke \author Dag Sverre Seljebotn
 */

#ifndef SHARP2_INTERNAL_H
#define SHARP2_INTERNAL_H

#include <complex>
#include <vector>
#include "ducc0/sharp/sharp.h"
#include "ducc0/infra/error_handling.h"
#include "ducc0/infra/mav.h"

namespace ducc0 {

namespace detail_sharp {

static constexpr int sharp_minscale=0, sharp_limscale=1, sharp_maxscale=1;
static constexpr double sharp_fbig=0x1p+800,sharp_fsmall=0x1p-800;
static constexpr double sharp_ftol=0x1p-60;
static constexpr double sharp_fbighalf=0x1p+400;

using std::complex;

class sharp_Ylmgen
  {
  public:
    struct dbl2 { double a, b; };
    sharp_Ylmgen(size_t l_max, size_t m_max, size_t spin);

    /*! Prepares the object for the calculation at \a m. */
    void prepare(size_t m_);
    /*! Returns a vector with \a lmax+1 entries containing
        normalisation factors that must be applied to Y_lm values computed for
        \a spin. */
    static std::vector<double> get_norm(size_t lmax, size_t spin);
    /*! Returns a vectorwith \a lmax+1 entries containing
        normalisation factors that must be applied to Y_lm values computed for
        first derivatives. */
    static std::vector<double> get_d1norm(size_t lmax);

    /* for public use; immutable during lifetime */
    size_t lmax, mmax, s;
    std::vector<double> cf;
    std::vector<double> powlimit;

    /* for public use; will typically change after call to Ylmgen_prepare() */
    size_t m;

    std::vector<double> alpha;
    std::vector<dbl2> coef;

    /* used if s==0 */
    std::vector<double> mfac, eps;

    /* used if s!=0 */
    size_t sinPow, cosPow;
    bool preMinus_p, preMinus_m;
    std::vector<double> prefac;
    std::vector<int> fscale;

    size_t mlo, mhi;
  private:
    /* used if s==0 */
    std::vector<double> root, iroot;

    /* used if s!=0 */
    std::vector<double> flm1, flm2, inv;
  };

class sharp_job
  {
  private:
    std::vector<std::any> alm;
    std::vector<std::any> map;

    void init_output();
    void alm2almtmp (size_t mi);
    void almtmp2alm (size_t mi);
    void ring2ringtmp (size_t iring, mav<double,2> &ringtmp);
    void ringtmp2ring (size_t iring, const mav<double,2> &ringtmp);
    void map2phase (size_t mmax, size_t llim, size_t ulim);
    void phase2map (size_t mmax, size_t llim, size_t ulim);
    mav<std::complex<double>,3> alloc_phase (size_t nm, size_t ntheta) const;
    void alloc_almtmp (size_t lmax);

  public:
    sharp_jobtype type;
    size_t spin;
    size_t flags;
    mav<complex<double>,3> phase;
    std::vector<double> norm_l;
    mav<complex<double>,2> almtmp;
    const sharp_geom_info &ginfo;
    const sharp_alm_info &ainfo;
    int nthreads;
    double time;
    uint64_t opcnt;

    sharp_job(sharp_jobtype type,
      size_t spin, const std::vector<std::any> &alm_,
      const std::vector<std::any> &map, const sharp_geom_info &geom_info,
      const sharp_alm_info &alm_info, size_t flags, int nthreads_);

    void set_phase(mav<std::complex<double>,3> &phase_);
    size_t nmaps() const { return 1+(spin>0); }
    size_t nalm() const { return (type==SHARP_ALM2MAP_DERIV1) ? 1 : (1+(spin>0)); }

    void execute();
  };

void inner_loop (sharp_job &job, const std::vector<bool> &ispair,
  const std::vector<double> &cth, const std::vector<double> &sth, size_t llim,
  size_t ulim, sharp_Ylmgen &gen, size_t mi, const std::vector<size_t> &mlim);

size_t sharp_max_nvec(size_t spin);

}

}

#endif
