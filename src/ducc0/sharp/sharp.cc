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

/*! \file sharp.cc
 *  Spherical transform library
 *
 *  Copyright (C) 2006-2021 Max-Planck-Society
 *  \author Martin Reinecke \author Dag Sverre Seljebotn
 */

#include <cmath>
#include <algorithm>
#include <memory>
#include "ducc0/math/math_utils.h"
#include "ducc0/math/fft1d.h"
#include "ducc0/sharp/sharp_internal.h"
#include "ducc0/sharp/sharp_almhelpers.h"
#include "ducc0/sharp/sharp_geomhelpers.h"
#include "ducc0/infra/threading.h"
#include "ducc0/infra/useful_macros.h"
#include "ducc0/infra/error_handling.h"
#include "ducc0/sharp/sht.h"

namespace ducc0 {

namespace detail_sharp {

using namespace std;

using dcmplx = complex<double>;
using fcmplx = complex<float>;

static size_t chunksize_min=500, nchunks_max=10;

static void get_chunk_info (size_t ndata, size_t nmult, size_t &nchunks, size_t &chunksize)
  {
  chunksize = (ndata+nchunks_max-1)/nchunks_max;
  if (chunksize>=chunksize_min) // use max number of chunks
    chunksize = ((chunksize+nmult-1)/nmult)*nmult;
  else // need to adjust chunksize and nchunks
    {
    nchunks = (ndata+chunksize_min-1)/chunksize_min;
    chunksize = (ndata+nchunks-1)/nchunks;
    if (nchunks>1)
      chunksize = ((chunksize+nmult-1)/nmult)*nmult;
    }
  nchunks = (ndata+chunksize-1)/chunksize;
  }

struct ringhelper
  {
  double phi0_;
  vector<dcmplx> shiftarr;
  size_t s_shift;
  unique_ptr<pocketfft_r<double>> plan;
  size_t length;
  bool norot;
  ringhelper() : length(0) {}
  void update(size_t nph, size_t mmax, double phi0)
    {
    norot = (abs(phi0)<1e-14);
    if (!norot)
      if ((mmax!=s_shift-1) || (!approx(phi0,phi0_,1e-12)))
      {
      shiftarr.resize(mmax+1);
      s_shift = mmax+1;
      phi0_ = phi0;
// FIXME: improve this by using sincos2pibyn(nph) etc.
      for (size_t m=0; m<=mmax; ++m)
        shiftarr[m] = dcmplx(cos(m*phi0),sin(m*phi0));
//      double *tmp=(double *) self->shiftarr;
//      sincos_multi (mmax+1, phi0, &tmp[1], &tmp[0], 2);
      }
    if (nph!=length)
      {
      plan.reset(new pocketfft_r<double>(nph));
      length=nph;
      }
    }
  DUCC0_NOINLINE void phase2ring (const sharp_geom_info &info, size_t iring,
    mav<double,1> &data, size_t mmax, const mav<dcmplx,1> &phase)
    {
    size_t nph = info.nph(iring);

    update (nph, mmax, info.phi0(iring));

    if (nph>=2*mmax+1)
      {
      if (norot)
        for (size_t m=0; m<=mmax; ++m)
          {
          data.v(2*m)=phase(m).real();
          data.v(2*m+1)=phase(m).imag();
          }
      else
        for (size_t m=0; m<=mmax; ++m)
          {
          dcmplx tmp = phase(m)*shiftarr[m];
          data.v(2*m)=tmp.real();
          data.v(2*m+1)=tmp.imag();
          }
      for (size_t m=2*(mmax+1); m<nph+2; ++m)
        data.v(m)=0.;
      }
    else
      {
      data.v(0)=phase(0).real();
      fill(&data.v(1),&data.v(nph+2),0.);

      size_t idx1=1, idx2=nph-1;
      for (size_t m=1; m<=mmax; ++m)
        {
        dcmplx tmp = phase(m);
        if(!norot) tmp*=shiftarr[m];
        if (idx1<(nph+2)/2)
          {
          data.v(2*idx1)+=tmp.real();
          data.v(2*idx1+1)+=tmp.imag();
          }
        if (idx2<(nph+2)/2)
          {
          data.v(2*idx2)+=tmp.real();
          data.v(2*idx2+1)-=tmp.imag();
          }
        if (++idx1>=nph) idx1=0;
        idx2 = (idx2==0) ? nph-1 : idx2-1;
        }
      }
    data.v(1)=data(0);
    plan->exec(&(data.v(1)), 1., false);
    }
  DUCC0_NOINLINE void ring2phase (const sharp_geom_info &info, size_t iring,
    mav<double,1> &data, size_t mmax, mav<dcmplx,1> &phase)
    {
    size_t nph = info.nph(iring);

    update (nph, mmax, -info.phi0(iring));

    plan->exec (&(data.v(1)), 1., true);
    data.v(0)=data(1);
    data.v(1)=data.v(nph+1)=0.;

    if (mmax<=nph/2)
      {
      if (norot)
        for (size_t m=0; m<=mmax; ++m)
          phase.v(m) = dcmplx(data(2*m), data(2*m+1));
      else
        for (size_t m=0; m<=mmax; ++m)
          phase.v(m) = dcmplx(data(2*m), data(2*m+1)) * shiftarr[m];
      }
    else
      {
      for (size_t m=0; m<=mmax; ++m)
        {
        auto idx=m%nph;
        dcmplx val;
        if (idx<(nph-idx))
          val = dcmplx(data(2*idx), data(2*idx+1));
        else
          val = dcmplx(data(2*(nph-idx)), -data(2*(nph-idx)+1));
        if (!norot)
          val *= shiftarr[m];
        phase.v(m)=val;
        }
      }
    }
  };

void sharp_job::init_output()
  {
  if (flags&SHARP_ADD) return;
  if (type == SHARP_MAP2ALM)
    for (size_t i=0; i<alm.size(); ++i)
      ainfo.clear_alm (alm[i]);
  else
    for (size_t i=0; i<map.size(); ++i)
      ginfo.clear_map(map[i]);
  }

DUCC0_NOINLINE void sharp_job::alm2almtmp (size_t mi, mav<dcmplx,2> &almtmp)
  {
  size_t nalm_ = nalm();
  size_t lmax = ainfo.lmax();
  if (type!=SHARP_MAP2ALM)
    {
    auto m=ainfo.mval(mi);
    auto lmin=(m<spin) ? spin : m;
    for (size_t i=0; i<nalm_; ++i)
      {
      auto sub = subarray<1>(almtmp, {0,i},{MAXIDX,0});
      ainfo.get_alm(mi, alm[i], sub);
      }
    for (auto l=m; l<lmin; ++l)
      for (size_t i=0; i<nalm_; ++i)
        almtmp.v(l,i) = 0;
    for (size_t i=0; i<nalm_; ++i)
      almtmp.v(lmax+1,i) = 0;
    if (spin>0)
      for (auto l=lmin; l<=lmax; ++l)
        for (size_t i=0; i<nalm_; ++i)
          almtmp.v(l,i) *= norm_l[l];
    }
  else
    for (size_t l=ainfo.mval(mi); l<(lmax+2); ++l)
      for (size_t i=0; i<nalm(); ++i)
        almtmp.v(l,i)=0;
  }

DUCC0_NOINLINE void sharp_job::almtmp2alm (size_t mi,
  mav<dcmplx,2> &almtmp)
  {
  if (type != SHARP_MAP2ALM) return;
  size_t lmax = ainfo.lmax();
  auto m=ainfo.mval(mi);
  auto lmin=(m<spin) ? spin : m;
  size_t nalm_ = nalm();
  if (spin>0)
    for (auto l=lmin; l<=lmax; ++l)
      for (size_t i=0; i<nalm_; ++i)
        almtmp.v(l,i) *= norm_l[l];
  for (size_t i=0; i<nalm_; ++i)
    ainfo.add_alm(mi, subarray<1>(almtmp, {0,i},{MAXIDX,0}), alm[i]);
  }

DUCC0_NOINLINE void sharp_job::ringtmp2ring (size_t iring,
  const mav<double,2> &ringtmp)
  {
  for (size_t i=0; i<nmaps(); ++i)
    ginfo.add_ring(flags&SHARP_USE_WEIGHTS, iring, subarray<1>(ringtmp, {i,1},{0,MAXIDX}), map[i]);
  }

DUCC0_NOINLINE void sharp_job::ring2ringtmp (size_t iring,
  mav<double,2> &ringtmp)
  {
  for (size_t i=0; i<nmaps(); ++i)
    {
    auto rtmp = subarray<1>(ringtmp, {i,1},{0,MAXIDX});
    ginfo.get_ring(flags&SHARP_USE_WEIGHTS, iring, map[i], rtmp);
    }
  }

//FIXME: set phase to zero if not SHARP_MAP2ALM?
DUCC0_NOINLINE void sharp_job::map2phase (size_t mmax, size_t llim, size_t ulim, mav<dcmplx,3> &phase)
  {
  if (type != SHARP_MAP2ALM) return;
  ducc0::execDynamic(ulim-llim, nthreads, 1, [&](ducc0::Scheduler &sched)
    {
    ringhelper helper;
    size_t rstride=ginfo.nphmax()+2;
    mav<double,2> ringtmp({nmaps(), rstride});

    while (auto rng=sched.getNext()) for(auto ith=rng.lo+llim; ith<rng.hi+llim; ++ith)
      {
      ring2ringtmp(ginfo.pair(ith).r1,ringtmp);
      for (size_t i=0; i<nmaps(); ++i)
        {
        auto rtmp = subarray<1>(ringtmp, {i,0}, {0,MAXIDX});
        auto ptmp = subarray<1>(phase, {2*(ith-llim), 0, i}, {0, MAXIDX, 0});
        helper.ring2phase (ginfo, ginfo.pair(ith).r1,rtmp,mmax,ptmp);
        }
      if (ginfo.pair(ith).r2!=~size_t(0))
        {
        ring2ringtmp(ginfo.pair(ith).r2,ringtmp);
        for (size_t i=0; i<nmaps(); ++i)
          {
          auto rtmp = subarray<1>(ringtmp, {i,0}, {0,MAXIDX});
          auto ptmp = subarray<1>(phase, {2*(ith-llim)+1, 0, i}, {0, MAXIDX, 0});
          helper.ring2phase (ginfo, ginfo.pair(ith).r2,rtmp,mmax,ptmp);
          }
        }
      }
    }); /* end of parallel region */
  }

DUCC0_NOINLINE void sharp_job::phase2map (size_t mmax, size_t llim, size_t ulim, const mav<dcmplx,3> &phase)
  {
  if (type == SHARP_MAP2ALM) return;
  ducc0::execDynamic(ulim-llim, nthreads, 1, [&](ducc0::Scheduler &sched)
    {
    ringhelper helper;
    size_t rstride=ginfo.nphmax()+2;
    mav<double,2> ringtmp({nmaps(), rstride});

    while (auto rng=sched.getNext()) for(auto ith=rng.lo+llim; ith<rng.hi+llim; ++ith)
      {
      for (size_t i=0; i<nmaps(); ++i)
        {
        auto rtmp = subarray<1>(ringtmp, {i,0}, {0,MAXIDX});
        auto ptmp = subarray<1>(phase, {2*(ith-llim), 0, i}, {0, MAXIDX, 0});
        helper.phase2ring (ginfo, ginfo.pair(ith).r1,rtmp,mmax,ptmp);
        }
      ringtmp2ring(ginfo.pair(ith).r1,ringtmp);
      if (ginfo.pair(ith).r2!=~size_t(0))
        {
        for (size_t i=0; i<nmaps(); ++i)
          {
          auto rtmp = subarray<1>(ringtmp, {i,0}, {0,MAXIDX});
          auto ptmp = subarray<1>(phase, {2*(ith-llim)+1, 0, i}, {0, MAXIDX, 0});
          helper.phase2ring (ginfo, ginfo.pair(ith).r2,rtmp,mmax,ptmp);
          }
        ringtmp2ring(ginfo.pair(ith).r2,ringtmp);
        }
      }
    }); /* end of parallel region */
  }

DUCC0_NOINLINE void sharp_job::execute()
  {
  size_t lmax = ainfo.lmax(),
         mmax = ainfo.mmax();

  norm_l = (type==SHARP_ALM2MAP_DERIV1) ?
     detail_sht::YlmBase::get_d1norm (lmax) :
     detail_sht::YlmBase::get_norm (lmax, spin);

/* clear output arrays if requested */
  init_output();

  size_t nchunks, chunksize;
  get_chunk_info(ginfo.npairs(), (spin==0) ? 128 : 64,
                 nchunks,chunksize);
//FIXME: needs to be changed to "nm"
  auto phase = mav<dcmplx,3>::build_noncritical({2*chunksize,mmax+1,nmaps()});
  detail_sht::YlmBase ylmbase(lmax,mmax,spin);
  detail_sht::SHT_mode mode = (type==SHARP_MAP2ALM) ? detail_sht::MAP2ALM : 
                             ((type==SHARP_ALM2MAP) ? detail_sht::ALM2MAP : detail_sht::ALM2MAP_DERIV1);
/* chunk loop */
  for (size_t chunk=0; chunk<nchunks; ++chunk)
    {
    size_t llim=chunk*chunksize, ulim=min(llim+chunksize,ginfo.npairs());
    vector<detail_sht::ringdata> rdata(ulim-llim);
    for (size_t i=0; i<ulim-llim; ++i)
      {
      double cth = ginfo.cth(ginfo.pair(i+llim).r1);
      double sth = ginfo.sth(ginfo.pair(i+llim).r1);
      auto mlim = detail_sht::get_mlim(lmax, spin, sth, cth);
      size_t idx = 2*i;
      size_t midx = 2*i+1;
      if (ginfo.pair(i+llim).r2==~size_t(0)) midx=idx;
      rdata[i] = { mlim, idx, midx, cth, sth };
      }

/* map->phase where necessary */
    map2phase(mmax, llim, ulim, phase);

    ducc0::execDynamic(ainfo.nm(), nthreads, 1, [&](ducc0::Scheduler &sched)
      {
      detail_sht::Ylmgen ylmgen(ylmbase);
      auto almtmp = mav<dcmplx,2>({lmax+2, nalm()});

      while (auto rng=sched.getNext()) for(auto mi=rng.lo; mi<rng.hi; ++mi)
        {
/* alm->alm_tmp where necessary */
        alm2almtmp(mi, almtmp);
        ylmgen.prepare(ainfo.mval(mi));
        detail_sht::inner_loop(mode, almtmp, phase, rdata, ylmgen, mi);

/* alm_tmp->alm where necessary */
        almtmp2alm(mi, almtmp);
        }
      }); /* end of parallel region */

/* phase->map where necessary */
    phase2map (mmax, llim, ulim, phase);
    } /* end of chunk loop */
  }

sharp_job::sharp_job (sharp_jobtype type_,
  size_t spin_, const vector<any> &alm_, const vector<any> &map_,
  const sharp_geom_info &geom_info, const sharp_alm_info &alm_info, size_t flags_, int nthreads_)
  : alm(alm_), map(map_), type(type_), spin(spin_), flags(flags_), ginfo(geom_info), ainfo(alm_info),
    nthreads(nthreads_)
  {
  if (type==SHARP_ALM2MAP_DERIV1) spin_=1;
  if (type==SHARP_MAP2ALM) flags|=SHARP_USE_WEIGHTS;
  if (type==SHARP_Yt) type=SHARP_MAP2ALM;
  if (type==SHARP_WY) { type=SHARP_ALM2MAP; flags|=SHARP_USE_WEIGHTS; }

  MR_assert(spin<=ainfo.lmax(), "bad spin");
  MR_assert(alm.size()==nalm(), "incorrect # of a_lm components");
  MR_assert(map.size()==nmaps(), "incorrect # of a_lm components");
  }

void sharp_execute (sharp_jobtype type, size_t spin, const vector<any> &alm,
  const vector<any> &map,
  const sharp_geom_info &geom_info, const sharp_alm_info &alm_info,
  size_t flags, int nthreads)
  {
  sharp_job job(type, spin, alm, map, geom_info, alm_info, flags, nthreads);
  job.execute();
  }

void sharp_set_chunksize_min(size_t new_chunksize_min)
  { chunksize_min=new_chunksize_min; }
void sharp_set_nchunks_max(size_t new_nchunks_max)
  { nchunks_max=new_nchunks_max; }

}}
