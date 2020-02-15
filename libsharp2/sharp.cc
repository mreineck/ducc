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

/*! \file sharp.c
 *  Spherical transform library
 *
 *  Copyright (C) 2006-2019 Max-Planck-Society
 *  \author Martin Reinecke \author Dag Sverre Seljebotn
 */

#include <cmath>
#include <atomic>
#include <memory>
#include "mr_util/math_utils.h"
#include "mr_util/fft1d.h"
#include "libsharp2/sharp_internal.h"
#include "libsharp2/sharp_almhelpers.h"
#include "libsharp2/sharp_geomhelpers.h"
#include "mr_util/threading.h"
#include "mr_util/useful_macros.h"
#include "mr_util/error_handling.h"
#include "mr_util/timers.h"

using namespace std;
using namespace mr;

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

MRUTIL_NOINLINE size_t sharp_get_mlim (size_t lmax, size_t spin, double sth, double cth)
  {
  double ofs=lmax*0.01;
  if (ofs<100.) ofs=100.;
  double b = -2*double(spin)*abs(cth);
  double t1 = lmax*sth+ofs;
  double c = double(spin)*spin-t1*t1;
  double discr = b*b-4*c;
  if (discr<=0) return lmax;
  double res=(-b+sqrt(discr))/2.;
  if (res>lmax) res=lmax;
  return size_t(res+0.5);
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
  MRUTIL_NOINLINE void phase2ring (const sharp_geom_info &info, size_t iring,
    double *data, size_t mmax, const dcmplx *phase, size_t pstride)
    {
    size_t nph = info.nph(iring);

    update (nph, mmax, info.phi0(iring));

    if (nph>=2*mmax+1)
      {
      if (norot)
        for (size_t m=0; m<=mmax; ++m)
          {
          data[2*m]=phase[m*pstride].real();
          data[2*m+1]=phase[m*pstride].imag();
          }
      else
        for (size_t m=0; m<=mmax; ++m)
          {
          dcmplx tmp = phase[m*pstride]*shiftarr[m];
          data[2*m]=tmp.real();
          data[2*m+1]=tmp.imag();
          }
      for (size_t m=2*(mmax+1); m<nph+2; ++m)
        data[m]=0.;
      }
    else
      {
      data[0]=phase[0].real();
      fill(data+1,data+nph+2,0.);

      size_t idx1=1, idx2=nph-1;
      for (size_t m=1; m<=mmax; ++m)
        {
        dcmplx tmp = phase[m*pstride];
        if(!norot) tmp*=shiftarr[m];
        if (idx1<(nph+2)/2)
          {
          data[2*idx1]+=tmp.real();
          data[2*idx1+1]+=tmp.imag();
          }
        if (idx2<(nph+2)/2)
          {
          data[2*idx2]+=tmp.real();
          data[2*idx2+1]-=tmp.imag();
          }
        if (++idx1>=nph) idx1=0;
        idx2 = (idx2==0) ? nph-1 : idx2-1;
        }
      }
    data[1]=data[0];
    plan->exec(&(data[1]), 1., false);
    }
  MRUTIL_NOINLINE void ring2phase (const sharp_geom_info &info, size_t iring,
    double *data, size_t mmax, dcmplx *phase, size_t pstride)
    {
    size_t nph = info.nph(iring);

    update (nph, mmax, -info.phi0(iring));

    plan->exec (&(data[1]), 1., true);
    data[0]=data[1];
    data[1]=data[nph+1]=0.;

    if (mmax<=nph/2)
      {
      if (norot)
        for (size_t m=0; m<=mmax; ++m)
          phase[m*pstride] = dcmplx(data[2*m], data[2*m+1]);
      else
        for (size_t m=0; m<=mmax; ++m)
          phase[m*pstride] =
            dcmplx(data[2*m], data[2*m+1]) * shiftarr[m];
      }
    else
      {
      for (size_t m=0; m<=mmax; ++m)
        {
        auto idx=m%nph;
        dcmplx val;
        if (idx<(nph-idx))
          val = dcmplx(data[2*idx], data[2*idx+1]);
        else
          val = dcmplx(data[2*(nph-idx)], -data[2*(nph-idx)+1]);
        if (!norot)
          val *= shiftarr[m];
        phase[m*pstride]=val;
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

MRUTIL_NOINLINE void sharp_job::alloc_phase (size_t nm, size_t ntheta, vector<dcmplx> &data)
  {
  if (type==SHARP_MAP2ALM)
    {
    s_m=2*nmaps();
    if (((s_m*16*nm)&1023)==0) nm+=3; // hack to avoid critical strides
    s_th=s_m*nm;
    }
  else
    {
    s_th=2*nmaps();
    if (((s_th*16*ntheta)&1023)==0) ntheta+=3; // hack to avoid critical strides
    s_m=s_th*ntheta;
    }
  data.resize(2*nmaps()*nm*ntheta);
  phase=data.data();
  }

void sharp_job::alloc_almtmp (size_t lmax, vector<dcmplx> &data)
  {
  data.resize(nalm()*(lmax+2));
  almtmp=data.data();
  }

MRUTIL_NOINLINE void sharp_job::alm2almtmp (size_t mi)
  {
  size_t nalm_ = nalm();
  size_t lmax = ainfo.lmax();
  if (type!=SHARP_MAP2ALM)
    {
    auto m=ainfo.mval(mi);
    auto lmin=(m<spin) ? spin : m;
    for (size_t i=0; i<nalm_; ++i)
      ainfo.get_alm(mi, alm[i], almtmp+i, nalm_);
    for (auto l=m; l<lmin; ++l)
      for (size_t i=0; i<nalm_; ++i)
        almtmp[nalm_*l+i] = 0;
    for (size_t i=0; i<nalm_; ++i)
      almtmp[nalm_*(lmax+1)+i] = 0;
    if (spin>0)
      for (auto l=lmin; l<=lmax; ++l)
        for (size_t i=0; i<nalm_; ++i)
          almtmp[nalm_*l+i] *= norm_l[l];
    }
  else
    for (size_t i=nalm_*ainfo.mval(mi); i<nalm_*(lmax+2); ++i)
      almtmp[i]=0;
  }

MRUTIL_NOINLINE void sharp_job::almtmp2alm (size_t mi)
  {
  if (type != SHARP_MAP2ALM) return;
  size_t lmax = ainfo.lmax();
  auto m=ainfo.mval(mi);
  auto lmin=(m<spin) ? spin : m;
  size_t nalm_ = nalm();
  if (spin>0)
    for (auto l=lmin; l<=lmax; ++l)
      for (size_t i=0; i<nalm_; ++i)
        almtmp[nalm_*l+i] *= norm_l[l];
  for (size_t i=0; i<nalm_; ++i)
    ainfo.add_alm(mi, almtmp+i, alm[i], nalm_);
  }

MRUTIL_NOINLINE void sharp_job::ringtmp2ring (size_t iring,
  const vector<double> &ringtmp, size_t rstride)
  {
  for (size_t i=0; i<nmaps(); ++i)
    ginfo.add_ring(flags&SHARP_USE_WEIGHTS, iring, &ringtmp[i*rstride+1], map[i]);
  }

MRUTIL_NOINLINE void sharp_job::ring2ringtmp (size_t iring,
  vector<double> &ringtmp, size_t rstride)
  {
  for (size_t i=0; i<nmaps(); ++i)
    ginfo.get_ring(flags&SHARP_USE_WEIGHTS, iring, map[i], &ringtmp[i*rstride+1]);
  }

//FIXME: set phase to zero if not SHARP_MAP2ALM?
MRUTIL_NOINLINE void sharp_job::map2phase (size_t mmax, size_t llim, size_t ulim)
  {
  if (type != SHARP_MAP2ALM) return;
  size_t pstride = s_m;
  mr::execDynamic(ulim-llim, 0, 1, [&](mr::Scheduler &sched)
    {
    ringhelper helper;
    size_t rstride=ginfo.nphmax()+2;
    vector<double> ringtmp(nmaps()*rstride);

    while (auto rng=sched.getNext()) for(auto ith=rng.lo+llim; ith<rng.hi+llim; ++ith)
      {
      size_t dim2 = s_th*(ith-llim);
      ring2ringtmp(ginfo.pair(ith).r1,ringtmp,rstride);
      for (size_t i=0; i<nmaps(); ++i)
        helper.ring2phase (ginfo, ginfo.pair(ith).r1,
          &ringtmp[i*rstride],mmax,&phase[dim2+2*i],pstride);
      if (ginfo.pair(ith).r2!=~size_t(0))
        {
        ring2ringtmp(ginfo.pair(ith).r2,ringtmp,rstride);
        for (size_t i=0; i<nmaps(); ++i)
          helper.ring2phase (ginfo, ginfo.pair(ith).r2,
            &ringtmp[i*rstride],mmax,&phase[dim2+2*i+1],pstride);
        }
      }
    }); /* end of parallel region */
  }

MRUTIL_NOINLINE void sharp_job::phase2map (size_t mmax, size_t llim, size_t ulim)
  {
  if (type == SHARP_MAP2ALM) return;
  size_t pstride = s_m;
  mr::execDynamic(ulim-llim, 0, 1, [&](mr::Scheduler &sched)
    {
    ringhelper helper;
    size_t rstride=ginfo.nphmax()+2;
    vector<double> ringtmp(nmaps()*rstride);

    while (auto rng=sched.getNext()) for(auto ith=rng.lo+llim; ith<rng.hi+llim; ++ith)
      {
      size_t dim2 = s_th*(ith-llim);
      for (size_t i=0; i<nmaps(); ++i)
        helper.phase2ring (ginfo, ginfo.pair(ith).r1,
          &ringtmp[i*rstride],mmax,&phase[dim2+2*i],pstride);
      ringtmp2ring(ginfo.pair(ith).r1,ringtmp,rstride);
      if (ginfo.pair(ith).r2!=~size_t(0))
        {
        for (size_t i=0; i<nmaps(); ++i)
          helper.phase2ring (ginfo, ginfo.pair(ith).r2,
            &ringtmp[i*rstride],mmax,&phase[dim2+2*i+1],pstride);
        ringtmp2ring(ginfo.pair(ith).r2,ringtmp,rstride);
        }
      }
    }); /* end of parallel region */
  }

MRUTIL_NOINLINE void sharp_job::execute()
  {
  mr::SimpleTimer timer;
  opcnt=0;
  size_t lmax = ainfo.lmax(),
         mmax = ainfo.mmax();

  norm_l = (type==SHARP_ALM2MAP_DERIV1) ?
     sharp_Ylmgen::get_d1norm (lmax) :
     sharp_Ylmgen::get_norm (lmax, spin);

/* clear output arrays if requested */
  init_output();

  size_t nchunks, chunksize;
  get_chunk_info(ginfo.npairs(),sharp_veclen()*sharp_max_nvec(spin),
                 nchunks,chunksize);
  vector<dcmplx> phasebuffer;
//FIXME: needs to be changed to "nm"
  alloc_phase(mmax+1,chunksize, phasebuffer);
  std::atomic<unsigned long long> a_opcnt(0);

/* chunk loop */
  for (size_t chunk=0; chunk<nchunks; ++chunk)
    {
    size_t llim=chunk*chunksize, ulim=min(llim+chunksize,ginfo.npairs());
    vector<bool> ispair(ulim-llim);
    vector<size_t> mlim(ulim-llim);
    vector<double> cth(ulim-llim), sth(ulim-llim);
    for (size_t i=0; i<ulim-llim; ++i)
      {
      ispair[i] = ginfo.pair(i+llim).r2!=~size_t(0);
      cth[i] = ginfo.cth(ginfo.pair(i+llim).r1);
      sth[i] = ginfo.sth(ginfo.pair(i+llim).r1);
      mlim[i] = sharp_get_mlim(lmax, spin, sth[i], cth[i]);
      }

/* map->phase where necessary */
    map2phase(mmax, llim, ulim);

    mr::execDynamic(ainfo.nm(), 0, 1, [&](mr::Scheduler &sched)
      {
      sharp_job ljob = *this;
      ljob.opcnt=0;
      sharp_Ylmgen generator(lmax,mmax,ljob.spin);
      vector<dcmplx> almbuffer;
      ljob.alloc_almtmp(lmax,almbuffer);

      while (auto rng=sched.getNext()) for(auto mi=rng.lo; mi<rng.hi; ++mi)
        {
/* alm->alm_tmp where necessary */
        ljob.alm2almtmp(mi);

        inner_loop (ljob, ispair, cth, sth, llim, ulim, generator, mi, mlim);

/* alm_tmp->alm where necessary */
        ljob.almtmp2alm(mi);
        }

      a_opcnt+=ljob.opcnt;
      }); /* end of parallel region */

/* phase->map where necessary */
    phase2map (mmax, llim, ulim);
    } /* end of chunk loop */

  opcnt = a_opcnt;
  time=timer();
  }

sharp_job::sharp_job (sharp_jobtype type_,
  size_t spin_, const vector<any> &alm_, const vector<any> &map_,
  const sharp_geom_info &geom_info, const sharp_alm_info &alm_info, size_t flags_)
  : alm(alm_), map(map_), type(type_), spin(spin_), flags(flags_), ginfo(geom_info), ainfo(alm_info),
    time(0.), opcnt(0)
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
  size_t flags, double *time, unsigned long long *opcnt)
  {
  sharp_job job(type, spin, alm, map, geom_info, alm_info, flags);

  job.execute();
  if (time!=nullptr) *time = job.time;
  if (opcnt!=nullptr) *opcnt = job.opcnt;
  }

void sharp_set_chunksize_min(size_t new_chunksize_min)
  { chunksize_min=new_chunksize_min; }
void sharp_set_nchunks_max(size_t new_nchunks_max)
  { nchunks_max=new_nchunks_max; }
