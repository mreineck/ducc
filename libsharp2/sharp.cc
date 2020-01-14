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
#include "mr_util/fft.h"
#include "libsharp2/sharp_ylmgen.h"
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

static int chunksize_min=500, nchunks_max=10;

static void get_chunk_info (int ndata, int nmult, int &nchunks, int &chunksize)
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
  int s_shift;
  unique_ptr<mr::detail_fft::rfftp<double>> plan;
  int length;
  int norot;
  ringhelper() : length(0) {}
  void update(int nph, int mmax, double phi0)
    {
    norot = (abs(phi0)<1e-14);
    if (!(norot))
      if ((mmax!=s_shift-1) || (!approx(phi0,phi0_,1e-12)))
      {
      shiftarr.resize(mmax+1);
      s_shift = mmax+1;
      phi0_ = phi0;
// FIXME: improve this by using sincos2pibyn(nph) etc.
      for (int m=0; m<=mmax; ++m)
        shiftarr[m] = dcmplx(cos(m*phi0),sin(m*phi0));
//      double *tmp=(double *) self->shiftarr;
//      sincos_multi (mmax+1, phi0, &tmp[1], &tmp[0], 2);
      }
    if (nph!=int(length))
      {
      plan.reset(new mr::detail_fft::rfftp<double>(nph));
      length=nph;
      }
    }
  MRUTIL_NOINLINE void phase2ring (const sharp_geom_info::Tring &info,
    double *data, int mmax, const dcmplx *phase, int pstride, int flags)
    {
    int nph = info.nph;

    update (nph, mmax, info.phi0);

    double wgt = (flags&SHARP_USE_WEIGHTS) ? info.weight : 1.;

    if (nph>=2*mmax+1)
      {
      if (norot)
        for (int m=0; m<=mmax; ++m)
          {
          data[2*m]=phase[m*pstride].real()*wgt;
          data[2*m+1]=phase[m*pstride].imag()*wgt;
          }
      else
        for (int m=0; m<=mmax; ++m)
          {
          dcmplx tmp = phase[m*pstride]*shiftarr[m];
          data[2*m]=tmp.real()*wgt;
          data[2*m+1]=tmp.imag()*wgt;
          }
      for (int m=2*(mmax+1); m<nph+2; ++m)
        data[m]=0.;
      }
    else
      {
      data[0]=phase[0].real()*wgt;
      fill(data+1,data+nph+2,0.);

      ptrdiff_t idx1=1, idx2=nph-1;
      for (int m=1; m<=mmax; ++m)
        {
        dcmplx tmp = phase[m*pstride]*wgt;
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
        if (--idx2<0) idx2=nph-1;
        }
      }
    data[1]=data[0];
    plan->exec(&(data[1]), 1., false);
    }
  MRUTIL_NOINLINE void ring2phase (const sharp_geom_info::Tring &info,
    double *data, int mmax, dcmplx *phase, int pstride, int flags)
    {
    int nph = info.nph;

    update (nph, mmax, -info.phi0);
    double wgt = (flags&SHARP_USE_WEIGHTS) ? info.weight : 1;

    plan->exec (&(data[1]), 1., true);
    data[0]=data[1];
    data[1]=data[nph+1]=0.;

    if (mmax<=nph/2)
      {
      if (norot)
        for (int m=0; m<=mmax; ++m)
          phase[m*pstride] = dcmplx(data[2*m], data[2*m+1]) * wgt;
      else
        for (int m=0; m<=mmax; ++m)
          phase[m*pstride] =
            dcmplx(data[2*m], data[2*m+1]) * shiftarr[m] * wgt;
      }
    else
      {
      for (int m=0; m<=mmax; ++m)
        {
        int idx=m%nph;
        dcmplx val;
        if (idx<(nph-idx))
          val = dcmplx(data[2*idx], data[2*idx+1]) * wgt;
        else
          val = dcmplx(data[2*(nph-idx)], -data[2*(nph-idx)+1]) * wgt;
        if (!norot)
          val *= shiftarr[m];
        phase[m*pstride]=val;
        }
      }
    }
  };

sharp_alm_info::sharp_alm_info (size_t lmax_, size_t nm_, ptrdiff_t stride_,
  const size_t *mval_, const ptrdiff_t *mstart)
  : lmax(lmax_), nm(nm_), mval(nm_), mvstart(nm), stride(stride_)
  {
  for (size_t mi=0; mi<nm; ++mi)
    {
    mval[mi] = mval_[mi];
    mvstart[mi] = mstart[mi];
    }
  }

sharp_alm_info::sharp_alm_info (size_t lmax_, size_t mmax, ptrdiff_t stride_,
  const ptrdiff_t *mstart)
  : lmax(lmax_), nm(mmax+1), mval(mmax+1), mvstart(mmax+1), stride(stride_)
  {
  for (size_t i=0; i<=mmax; ++i)
    {
    mval[i]=i;
    mvstart[i] = mstart[i];
    }
  }

ptrdiff_t sharp_alm_info::index (int l, int mi)
  {
  return mvstart[mi]+stride*l;
  }

sharp_geom_info::sharp_geom_info(size_t nrings, const size_t *nph, const ptrdiff_t *ofs,
  ptrdiff_t stride_, const double *phi0, const double *theta, const double *wgt)
  : ring(nrings), stride(stride_)
  {
  size_t pos=0;

  nphmax=0;

  for (size_t m=0; m<nrings; ++m)
    {
    ring[m].theta = theta[m];
    ring[m].cth = cos(theta[m]);
    ring[m].sth = sin(theta[m]);
    ring[m].weight = (wgt != nullptr) ? wgt[m] : 1.;
    ring[m].phi0 = phi0[m];
    ring[m].ofs = ofs[m];
    ring[m].nph = nph[m];
    if (nphmax<nph[m]) nphmax=nph[m];
    }
  sort(ring.begin(), ring.end(),[](const Tring &a, const Tring &b)
    { return (a.sth<b.sth); });
  while (pos<nrings)
    {
    pair.push_back(Tpair());
    pair.back().r1=pos;
    if ((pos<nrings-1) && approx(ring[pos].cth,-ring[pos+1].cth,1e-12))
      {
      if (ring[pos].cth>0)  // make sure northern ring is in r1
        pair.back().r2=pos+1;
      else
        {
        pair.back().r1=pos+1;
        pair.back().r2=pos;
        }
      ++pos;
      }
    else
      pair.back().r2=size_t(~0);
    ++pos;
    }

  sort(pair.begin(), pair.end(), [this] (const Tpair &a, const Tpair &b)
    {
    if (ring[a.r1].nph==ring[b.r1].nph)
    return (ring[a.r1].phi0 < ring[b.r1].phi0) ? true :
      ((ring[a.r1].phi0 > ring[b.r1].phi0) ? false :
        (ring[a.r1].cth>ring[b.r1].cth));
    return ring[a.r1].nph<ring[b.r1].nph;
    });
  }

/* This currently requires all m values from 0 to nm-1 to be present.
   It might be worthwhile to relax this criterion such that holes in the m
   distribution are permissible. */
static size_t sharp_get_mmax (const vector<size_t> &mval)
  {
  //FIXME: if gaps are allowed, we have to search the maximum m in the array
  auto nm=mval.size();
  vector<bool> mcheck(nm,false);
  for (auto m_cur : mval)
    {
    MR_assert(m_cur<nm, "not all m values are present");
    MR_assert(mcheck[m_cur]==false, "duplicate m value");
    mcheck[m_cur]=true;
    }
  return nm-1;
  }

MRUTIL_NOINLINE void sharp_geom_info::clear_map (double *map) const
  {
  for (const auto &r: ring)
    {
    if (stride==1)
      memset(&map[r.ofs],0,r.nph*sizeof(double));
    else
      for (size_t i=0;i<r.nph;++i)
        map[r.ofs+i*stride]=0;
    }
  }
MRUTIL_NOINLINE void sharp_geom_info::clear_map (float *map) const
  {
  for (const auto &r: ring)
    {
    if (stride==1)
      memset(&map[r.ofs],0,r.nph*sizeof(float));
    else
      for (size_t i=0;i<r.nph;++i)
        map[r.ofs+i*stride]=0;
    }
  }

MRUTIL_NOINLINE static void clear_alm (const sharp_alm_info *ainfo, void *alm,
  int flags)
  {
  for (size_t mi=0;mi<ainfo->nm;++mi)
    {
    auto m=ainfo->mval[mi];
    ptrdiff_t mvstart = ainfo->mvstart[mi];
    ptrdiff_t stride = ainfo->stride;
    if (flags&SHARP_DP)
      for (size_t l=m;l<=ainfo->lmax;++l)
        reinterpret_cast<dcmplx *>(alm)[mvstart+l*stride]=0.;
    else
      for (size_t l=m;l<=ainfo->lmax;++l)
        reinterpret_cast<fcmplx *>(alm)[mvstart+l*stride]=0.;
    }
  }

MRUTIL_NOINLINE void sharp_job::init_output()
  {
  if (flags&SHARP_ADD) return;
  if (type == SHARP_MAP2ALM)
    for (size_t i=0; i<nalm; ++i)
      clear_alm (ainfo,alm[i],flags);
  else
    for (size_t i=0; i<nmaps; ++i)
      (flags&SHARP_DP) ? ginfo->clear_map(reinterpret_cast<double *>(map[i]))
                       : ginfo->clear_map(reinterpret_cast<float  *>(map[i]));
  }

MRUTIL_NOINLINE void sharp_job::alloc_phase (size_t nm, size_t ntheta, vector<dcmplx> &data)
  {
  if (type==SHARP_MAP2ALM)
    {
    s_m=2*nmaps;
    if (((s_m*16*nm)&1023)==0) nm+=3; // hack to avoid critical strides
    s_th=s_m*nm;
    }
  else
    {
    s_th=2*nmaps;
    if (((s_th*16*ntheta)&1023)==0) ntheta+=3; // hack to avoid critical strides
    s_m=s_th*ntheta;
    }
  data.resize(2*nmaps*nm*ntheta);
  phase=data.data();
  }

void sharp_job::alloc_almtmp (size_t lmax, vector<dcmplx> &data)
  {
  data.resize(nalm*(lmax+2));
  almtmp=data.data();
  }

MRUTIL_NOINLINE void sharp_job::alm2almtmp (size_t lmax, size_t mi)
  {
  if (type!=SHARP_MAP2ALM)
    {
    auto ofs=ainfo->mvstart[mi];
    auto stride=ainfo->stride;
    auto m=ainfo->mval[mi];
    auto lmin=(m<spin) ? spin : m;
    if (spin==0)
      {
      if (flags&SHARP_DP)
        {
        for (auto l=m; l<lmin; ++l)
          for (size_t i=0; i<nalm; ++i)
            almtmp[nalm*l+i] = 0;
        for (auto l=lmin; l<=lmax; ++l)
          for (size_t i=0; i<nalm; ++i)
            almtmp[nalm*l+i] = reinterpret_cast<dcmplx **>(alm)[i][ofs+l*stride];
        for (size_t i=0; i<nalm; ++i)
          almtmp[nalm*(lmax+1)+i] = 0;
        }
      else
        {
        for (auto l=m; l<lmin; ++l)
          for (size_t i=0; i<nalm; ++i)
            almtmp[nalm*l+i] = 0;
        for (auto l=lmin; l<=lmax; ++l)
          for (size_t i=0; i<nalm; ++i)
            almtmp[nalm*l+i] = reinterpret_cast<fcmplx **>(alm)[i][ofs+l*stride];
        for (size_t i=0; i<nalm; ++i)
          almtmp[nalm*(lmax+1)+i] = 0;
        }
      }
    else
      {
      if (flags&SHARP_DP)
        {
        for (auto l=m; l<lmin; ++l)
          for (size_t i=0; i<nalm; ++i)
            almtmp[nalm*l+i] = 0;
        for (auto l=lmin; l<=lmax; ++l)
          for (size_t i=0; i<nalm; ++i)
            almtmp[nalm*l+i] = reinterpret_cast<dcmplx **>(alm)[i][ofs+l*stride]*norm_l[l];
        for (size_t i=0; i<nalm; ++i)
          almtmp[nalm*(lmax+1)+i] = 0;
        }
      else
        {
        for (auto l=m; l<lmin; ++l)
          for (size_t i=0; i<nalm; ++i)
            almtmp[nalm*l+i] = 0;
        for (auto l=lmin; l<=lmax; ++l)
          for (size_t i=0; i<nalm; ++i)
            almtmp[nalm*l+i] = dcmplx(reinterpret_cast<fcmplx **>(alm)[i][ofs+l*stride])*norm_l[l];
        for (size_t i=0; i<nalm; ++i)
          almtmp[nalm*(lmax+1)+i] = 0;
        }
      }
    }
  else
    for (size_t i=nalm*ainfo->mval[mi]; i<nalm*(lmax+2); ++i)
      almtmp[i]=0;
  }

MRUTIL_NOINLINE void sharp_job::almtmp2alm (size_t lmax, size_t mi)
  {
  if (type != SHARP_MAP2ALM) return;
  auto ofs=ainfo->mvstart[mi];
  auto stride=ainfo->stride;
  auto m=ainfo->mval[mi];
  auto lmin=(m<spin) ? spin : m;
  if (spin==0)
    {
    if (flags&SHARP_DP)
      for (auto l=lmin; l<=lmax; ++l)
        for (size_t i=0; i<nalm; ++i)
          ((dcmplx **)alm)[i][ofs+l*stride] += almtmp[nalm*l+i];
    else
      for (auto l=lmin; l<=lmax; ++l)
        for (size_t i=0; i<nalm; ++i)
          ((fcmplx **)alm)[i][ofs+l*stride] += fcmplx(almtmp[nalm*l+i]);
    }
  else
    {
    if (flags&SHARP_DP)
      for (auto l=lmin; l<=lmax; ++l)
        for (size_t i=0; i<nalm; ++i)
          ((dcmplx **)alm)[i][ofs+l*stride] += almtmp[nalm*l+i]*norm_l[l];
    else
      for (auto l=lmin; l<=lmax; ++l)
        for (size_t i=0; i<nalm; ++i)
          ((fcmplx **)alm)[i][ofs+l*stride] += fcmplx(almtmp[nalm*l+i]*norm_l[l]);
    }
  }

//virtual
void sharp_geom_info::add_ring(size_t iring, const double *ringtmp, double *map) const
  {
  double *MRUTIL_RESTRICT p1=&map[ring[iring].ofs];
  for (size_t m=0; m<ring[iring].nph; ++m)
    p1[m*stride] += ringtmp[m];
  }
//virtual
void sharp_geom_info::add_ring(size_t iring, const double *ringtmp, float *map) const
  {
  float *MRUTIL_RESTRICT p1=&map[ring[iring].ofs];
  for (size_t m=0; m<ring[iring].nph; ++m)
    p1[m*stride] += float(ringtmp[m]);
  }
//virtual
void sharp_geom_info::get_ring(size_t iring, const double *map, double *ringtmp) const
  {
  const double *MRUTIL_RESTRICT p1=&map[ring[iring].ofs];
  for (size_t m=0; m<ring[iring].nph; ++m)
    ringtmp[m] = p1[m*stride];
  }
//virtual
void sharp_geom_info::get_ring(size_t iring, const float *map, double *ringtmp) const
  {
  const float *MRUTIL_RESTRICT p1=&map[ring[iring].ofs];
  for (size_t m=0; m<ring[iring].nph; ++m)
    ringtmp[m] = p1[m*stride];
  }

MRUTIL_NOINLINE void sharp_job::ringtmp2ring (const sharp_geom_info &ginfo, size_t iring,
  const vector<double> &ringtmp, ptrdiff_t rstride)
  {
  if (flags & SHARP_DP)
    for (size_t i=0; i<nmaps; ++i)
      ginfo.add_ring(iring, &ringtmp[i*rstride+1], ((double  **)map)[i]);
  else
    for (size_t i=0; i<nmaps; ++i)
      ginfo.add_ring(iring, &ringtmp[i*rstride+1], ((float  **)map)[i]);
  }

MRUTIL_NOINLINE void sharp_job::ring2ringtmp (const sharp_geom_info &ginfo, size_t iring,
  vector<double> &ringtmp, ptrdiff_t rstride)
  {
  if (flags & SHARP_DP)
    for (size_t i=0; i<nmaps; ++i)
      ginfo.get_ring(iring, ((double  **)map)[i], &ringtmp[i*rstride+1]);
  else
    for (size_t i=0; i<nmaps; ++i)
      ginfo.get_ring(iring, ((float  **)map)[i], &ringtmp[i*rstride+1]);
  }

//FIXME: set phase to zero if not SHARP_MAP2ALM?
MRUTIL_NOINLINE void sharp_job::map2phase (size_t mmax, size_t llim, size_t ulim)
  {
  if (type != SHARP_MAP2ALM) return;
  int pstride = s_m;
  mr::execDynamic(ulim-llim, 0, 1, [&](mr::Scheduler &sched)
    {
    ringhelper helper;
    int rstride=ginfo->nphmax+2;
    vector<double> ringtmp(nmaps*rstride);

    while (auto rng=sched.getNext()) for(auto ith=rng.lo+llim; ith<rng.hi+llim; ++ith)
      {
      int dim2 = s_th*(ith-llim);
      ring2ringtmp(*ginfo, ginfo->pair[ith].r1,ringtmp,rstride);
      for (size_t i=0; i<nmaps; ++i)
        helper.ring2phase (ginfo->ring[ginfo->pair[ith].r1],
          &ringtmp[i*rstride],mmax,&phase[dim2+2*i],pstride,flags);
      if (ginfo->pair[ith].r2!=~size_t(0))
        {
        ring2ringtmp(*ginfo, ginfo->pair[ith].r2,ringtmp,rstride);
        for (size_t i=0; i<nmaps; ++i)
          helper.ring2phase (ginfo->ring[ginfo->pair[ith].r2],
            &ringtmp[i*rstride],mmax,&phase[dim2+2*i+1],pstride,flags);
        }
      }
    }); /* end of parallel region */
  }

MRUTIL_NOINLINE void sharp_job::phase2map (size_t mmax, size_t llim, size_t ulim)
  {
  if (type == SHARP_MAP2ALM) return;
  int pstride = s_m;
  mr::execDynamic(ulim-llim, 0, 1, [&](mr::Scheduler &sched)
    {
    ringhelper helper;
    int rstride=ginfo->nphmax+2;
    vector<double> ringtmp(nmaps*rstride);

    while (auto rng=sched.getNext()) for(auto ith=rng.lo+llim; ith<rng.hi+llim; ++ith)
      {
      int dim2 = s_th*(ith-llim);
      for (size_t i=0; i<nmaps; ++i)
        helper.phase2ring (ginfo->ring[ginfo->pair[ith].r1],
          &ringtmp[i*rstride],mmax,&phase[dim2+2*i],pstride,flags);
      ringtmp2ring(*ginfo, ginfo->pair[ith].r1,ringtmp,rstride);
      if (ginfo->pair[ith].r2!=~size_t(0))
        {
        for (size_t i=0; i<nmaps; ++i)
          helper.phase2ring (ginfo->ring[ginfo->pair[ith].r2],
            &ringtmp[i*rstride],mmax,&phase[dim2+2*i+1],pstride,flags);
        ringtmp2ring(*ginfo, ginfo->pair[ith].r2,ringtmp,rstride);
        }
      }
    }); /* end of parallel region */
  }

MRUTIL_NOINLINE void sharp_job::execute()
  {
  mr::SimpleTimer timer;
  opcnt=0;
  size_t lmax = ainfo->lmax,
         mmax = sharp_get_mmax(ainfo->mval);

  norm_l = (type==SHARP_ALM2MAP_DERIV1) ?
     sharp_Ylmgen::get_d1norm (lmax) :
     sharp_Ylmgen::get_norm (lmax, spin);

/* clear output arrays if requested */
  init_output();

  int nchunks, chunksize;
  get_chunk_info(ginfo->pair.size(),sharp_veclen()*sharp_max_nvec(spin),
                 nchunks,chunksize);
  vector<dcmplx> phasebuffer;
//FIXME: needs to be changed to "nm"
  alloc_phase(mmax+1,chunksize, phasebuffer);
  std::atomic<unsigned long long> a_opcnt(0);

/* chunk loop */
  for (int chunk=0; chunk<nchunks; ++chunk)
    {
    size_t llim=chunk*chunksize, ulim=min(llim+chunksize,ginfo->pair.size());
    vector<int> ispair(ulim-llim);
    vector<size_t> mlim(ulim-llim);
    vector<double> cth(ulim-llim), sth(ulim-llim);
    for (size_t i=0; i<ulim-llim; ++i)
      {
      ispair[i] = ginfo->pair[i+llim].r2!=~size_t(0);
      cth[i] = ginfo->ring[ginfo->pair[i+llim].r1].cth;
      sth[i] = ginfo->ring[ginfo->pair[i+llim].r1].sth;
      mlim[i] = sharp_get_mlim(lmax, spin, sth[i], cth[i]);
      }

/* map->phase where necessary */
    map2phase(mmax, llim, ulim);

    mr::execDynamic(ainfo->nm, 0, 1, [&](mr::Scheduler &sched)
      {
      sharp_job ljob = *this;
      ljob.opcnt=0;
      sharp_Ylmgen generator(lmax,mmax,ljob.spin);
      vector<dcmplx> almbuffer;
      ljob.alloc_almtmp(lmax,almbuffer);

      while (auto rng=sched.getNext()) for(auto mi=rng.lo; mi<rng.hi; ++mi)
        {
/* alm->alm_tmp where necessary */
        ljob.alm2almtmp(lmax, mi);

        inner_loop (ljob, ispair.data(), cth.data(), sth.data(), llim, ulim, generator, mi, mlim.data());

/* alm_tmp->alm where necessary */
        ljob.almtmp2alm(lmax, mi);
        }

      a_opcnt+=ljob.opcnt;
      }); /* end of parallel region */

/* phase->map where necessary */
    phase2map (mmax, llim, ulim);
    } /* end of chunk loop */

  opcnt = a_opcnt;
  time=timer();
  }

void sharp_job::build_common (sharp_jobtype type,
  size_t spin, void *alm, void *map, const sharp_geom_info &geom_info,
  const sharp_alm_info &alm_info, size_t flags)
  {
  if (type==SHARP_ALM2MAP_DERIV1) spin=1;
  if (type==SHARP_MAP2ALM) flags|=SHARP_USE_WEIGHTS;
  if (type==SHARP_Yt) type=SHARP_MAP2ALM;
  if (type==SHARP_WY) { type=SHARP_ALM2MAP; flags|=SHARP_USE_WEIGHTS; }

  MR_assert(spin<=alm_info.lmax, "bad spin");
  this->type = type;
  this->spin = spin;
  nmaps = (type==SHARP_ALM2MAP_DERIV1) ? 2 : ((spin>0) ? 2 : 1);
  nalm = (type==SHARP_ALM2MAP_DERIV1) ? 1 : ((spin>0) ? 2 : 1);
  ginfo = &geom_info;
  ainfo = &alm_info;
  this->flags = flags;
  time = 0.;
  opcnt = 0;
  this->alm=(void **)alm;
  this->map=(void **)map;
  }

void sharp_execute (sharp_jobtype type, int spin, void *alm, void *map,
  const sharp_geom_info &geom_info, const sharp_alm_info &alm_info,
  int flags, double *time, unsigned long long *opcnt)
  {
  sharp_job job;
  job.build_common (type, spin, alm, map, geom_info, alm_info, flags);

  job.execute();
  if (time!=nullptr) *time = job.time;
  if (opcnt!=nullptr) *opcnt = job.opcnt;
  }

void sharp_set_chunksize_min(int new_chunksize_min)
  { chunksize_min=new_chunksize_min; }
void sharp_set_nchunks_max(int new_nchunks_max)
  { nchunks_max=new_nchunks_max; }
