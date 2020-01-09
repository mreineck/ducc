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
#include <cstring>
#include <atomic>
#include <memory>
#include "mr_util/fft.h"
#include "libsharp2/sharp_ylmgen.h"
#include "libsharp2/sharp_internal.h"
#include "libsharp2/sharp_utils.h"
#include "libsharp2/sharp_almhelpers.h"
#include "libsharp2/sharp_geomhelpers.h"
#include "mr_util/threading.h"
#include "mr_util/useful_macros.h"
#include "mr_util/error_handling.h"
#include "mr_util/timers.h"

using namespace std;

using dcmplx = complex<double>;
using fcmplx = complex<float>;

static const double sqrt_one_half = 0.707106781186547572737310929369;
static const double sqrt_two = 1.414213562373095145474621858739;

static int chunksize_min=500, nchunks_max=10;

static void get_chunk_info (int ndata, int nmult, int *nchunks, int *chunksize)
  {
  *chunksize = (ndata+nchunks_max-1)/nchunks_max;
  if (*chunksize>=chunksize_min) // use max number of chunks
    *chunksize = ((*chunksize+nmult-1)/nmult)*nmult;
  else // need to adjust chunksize and nchunks
    {
    *nchunks = (ndata+chunksize_min-1)/chunksize_min;
    *chunksize = (ndata+(*nchunks)-1)/(*nchunks);
    if (*nchunks>1)
      *chunksize = ((*chunksize+nmult-1)/nmult)*nmult;
    }
  *nchunks = (ndata+(*chunksize)-1)/(*chunksize);
  }

MRUTIL_NOINLINE int sharp_get_mlim (int lmax, int spin, double sth, double cth)
  {
  double ofs=lmax*0.01;
  if (ofs<100.) ofs=100.;
  double b = -2*spin*fabs(cth);
  double t1 = lmax*sth+ofs;
  double c = (double)spin*spin-t1*t1;
  double discr = b*b-4*c;
  if (discr<=0) return lmax;
  double res=(-b+sqrt(discr))/2.;
  if (res>lmax) res=lmax;
  return (int)(res+0.5);
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
    norot = (fabs(phi0)<1e-14);
    if (!(norot))
      if ((mmax!=s_shift-1) || (!FAPPROX(phi0,phi0_,1e-12)))
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
  };

static int ringinfo_compare (const void *xa, const void *xb)
  {
  const sharp_ringinfo *a=(const sharp_ringinfo *)xa, *b=(const sharp_ringinfo *)xb;
  return (a->sth < b->sth) ? -1 : (a->sth > b->sth) ? 1 : 0;
  }
static int ringpair_compare (const void *xa, const void *xb)
  {
  const sharp_ringpair *a=(const sharp_ringpair *)xa, *b=(const sharp_ringpair *)xb;
//  return (a->r1.sth < b->r1.sth) ? -1 : (a->r1.sth > b->r1.sth) ? 1 : 0;
  if (a->r1.nph==b->r1.nph)
    return (a->r1.phi0 < b->r1.phi0) ? -1 :
      ((a->r1.phi0 > b->r1.phi0) ? 1 :
        (a->r1.cth>b->r1.cth ? -1 : 1));
  return (a->r1.nph<b->r1.nph) ? -1 : 1;
  }

void sharp_make_general_alm_info (int lmax, int nm, int stride, const int *mval,
  const ptrdiff_t *mstart, int flags, sharp_alm_info **alm_info)
  {
  sharp_alm_info *info = new sharp_alm_info;
  info->lmax = lmax;
  info->nm = nm;
  info->mval.resize(nm);
  info->mvstart.resize(nm);
  info->stride = stride;
  info->flags = flags;
  for (int mi=0; mi<nm; ++mi)
    {
    info->mval[mi] = mval[mi];
    info->mvstart[mi] = mstart[mi];
    }
  *alm_info = info;
  }

void sharp_make_alm_info (int lmax, int mmax, int stride,
  const ptrdiff_t *mstart, sharp_alm_info **alm_info)
  {
  vector<int> mval(mmax+1);
  for (int i=0; i<=mmax; ++i)
    mval[i]=i;
  sharp_make_general_alm_info (lmax, mmax+1, stride, mval.data(), mstart, 0, alm_info);
  }

ptrdiff_t sharp_alm_index (const sharp_alm_info *self, int l, int mi)
  {
  MR_assert(!(self->flags & SHARP_PACKED),
            "sharp_alm_index not applicable with SHARP_PACKED alms");
  return self->mvstart[mi]+self->stride*l;
  }

ptrdiff_t sharp_alm_count(const sharp_alm_info *self)
  {
  ptrdiff_t result=0;
  for (int im=0; im!=self->nm; ++im)
    {
    int m=self->mval[im];
    ptrdiff_t x=(self->lmax + 1 - m);
    if ((m!=0)&&((self->flags&SHARP_PACKED)!=0)) result+=2*x;
    else result+=x;
    }
  return result;
  }

void sharp_destroy_alm_info (sharp_alm_info *info)
  {
  delete info;
  }

void sharp_make_geom_info (int nrings, const int *nph, const ptrdiff_t *ofs,
  const int *stride, const double *phi0, const double *theta,
  const double *wgt, sharp_geom_info **geom_info)
  {
  sharp_geom_info *info = new sharp_geom_info;
  vector<sharp_ringinfo> infos(nrings);

  int pos=0;
  info->pair.resize(nrings);
  int npairs=0;
  info->nphmax=0;
  *geom_info = info;

  for (int m=0; m<nrings; ++m)
    {
    infos[m].theta = theta[m];
    infos[m].cth = cos(theta[m]);
    infos[m].sth = sin(theta[m]);
    infos[m].weight = (wgt != NULL) ? wgt[m] : 1.;
    infos[m].phi0 = phi0[m];
    infos[m].ofs = ofs[m];
    infos[m].stride = stride[m];
    infos[m].nph = nph[m];
    if (info->nphmax<nph[m]) info->nphmax=nph[m];
    }
  qsort(infos.data(),nrings,sizeof(sharp_ringinfo),ringinfo_compare);
  while (pos<nrings)
    {
    info->pair[npairs].r1=infos[pos];
    if ((pos<nrings-1) && FAPPROX(infos[pos].cth,-infos[pos+1].cth,1e-12))
      {
      if (infos[pos].cth>0)  // make sure northern ring is in r1
        info->pair[npairs].r2=infos[pos+1];
      else
        {
        info->pair[npairs].r1=infos[pos+1];
        info->pair[npairs].r2=infos[pos];
        }
      ++pos;
      }
    else
      info->pair[npairs].r2.nph=-1;
    ++pos;
    ++npairs;
    }

  qsort(info->pair.data(),npairs,sizeof(sharp_ringpair),ringpair_compare);
  info->pair.resize(npairs);
  }

ptrdiff_t sharp_map_size(const sharp_geom_info *info)
  {
  ptrdiff_t result = 0;
  for (int m=0; m<info->pair.size(); ++m)
    {
      result+=info->pair[m].r1.nph;
      result+=(info->pair[m].r2.nph>=0) ? (info->pair[m].r2.nph) : 0;
    }
  return result;
  }

void sharp_destroy_geom_info (sharp_geom_info *geom_info)
  {
  delete geom_info;
  }

/* This currently requires all m values from 0 to nm-1 to be present.
   It might be worthwhile to relax this criterion such that holes in the m
   distribution are permissible. */
static int sharp_get_mmax (const vector<int> &mval)
  {
  //FIXME: if gaps are allowed, we have to search the maximum m in the array
  auto nm=mval.size();
  vector<int> mcheck(nm,0);
  for (int i=0; i<nm; ++i)
    {
    int m_cur=mval[i];
    MR_assert((m_cur>=0) && (m_cur<nm), "not all m values are present");
    MR_assert(mcheck[m_cur]==0, "duplicate m value");
    mcheck[m_cur]=1;
    }
  return nm-1;
  }

MRUTIL_NOINLINE static void ringhelper_phase2ring (ringhelper &self,
  const sharp_ringinfo *info, double *data, int mmax, const dcmplx *phase,
  int pstride, int flags)
  {
  int nph = info->nph;

  self.update (nph, mmax, info->phi0);

  double wgt = (flags&SHARP_USE_WEIGHTS) ? info->weight : 1.;
  if (flags&SHARP_REAL_HARMONICS)
    wgt *= sqrt_one_half;

  if (nph>=2*mmax+1)
    {
    if (self.norot)
      for (int m=0; m<=mmax; ++m)
        {
        data[2*m]=phase[m*pstride].real()*wgt;
        data[2*m+1]=phase[m*pstride].imag()*wgt;
        }
    else
      for (int m=0; m<=mmax; ++m)
        {
        dcmplx tmp = phase[m*pstride]*self.shiftarr[m];
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

    int idx1=1, idx2=nph-1;
    for (int m=1; m<=mmax; ++m)
      {
      dcmplx tmp = phase[m*pstride]*wgt;
      if(!self.norot) tmp*=self.shiftarr[m];
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
  self.plan->exec(&(data[1]), 1., false);
  }

MRUTIL_NOINLINE static void ringhelper_ring2phase (ringhelper &self,
  const sharp_ringinfo *info, double *data, int mmax, dcmplx *phase,
  int pstride, int flags)
  {
  int nph = info->nph;
#if 1
  int maxidx = mmax; /* Enable this for traditional Healpix compatibility */
#else
  int maxidx = min(nph-1,mmax);
#endif

  self.update (nph, mmax, -info->phi0);
  double wgt = (flags&SHARP_USE_WEIGHTS) ? info->weight : 1;
  if (flags&SHARP_REAL_HARMONICS)
    wgt *= sqrt_two;

  self.plan->exec (&(data[1]), 1., true);
  data[0]=data[1];
  data[1]=data[nph+1]=0.;

  if (maxidx<=nph/2)
    {
    if (self.norot)
      for (int m=0; m<=maxidx; ++m)
        phase[m*pstride] = dcmplx(data[2*m], data[2*m+1]) * wgt;
    else
      for (int m=0; m<=maxidx; ++m)
        phase[m*pstride] =
          dcmplx(data[2*m], data[2*m+1]) * self.shiftarr[m] * wgt;
    }
  else
    {
    for (int m=0; m<=maxidx; ++m)
      {
      int idx=m%nph;
      dcmplx val;
      if (idx<(nph-idx))
        val = dcmplx(data[2*idx], data[2*idx+1]) * wgt;
      else
        val = dcmplx(data[2*(nph-idx)], -data[2*(nph-idx)+1]) * wgt;
      if (!self.norot)
        val *= self.shiftarr[m];
      phase[m*pstride]=val;
      }
    }

  for (int m=maxidx+1;m<=mmax; ++m)
    phase[m*pstride]=0.;
  }

MRUTIL_NOINLINE static void clear_map (const sharp_geom_info *ginfo, void *map,
  int flags)
  {
  if (flags & SHARP_NO_FFT)
    {
    for (int j=0;j<ginfo->pair.size();++j)
      {
      if (flags&SHARP_DP)
        {
        for (ptrdiff_t i=0;i<ginfo->pair[j].r1.nph;++i)
          ((dcmplx *)map)[ginfo->pair[j].r1.ofs+i*ginfo->pair[j].r1.stride]=0;
        for (ptrdiff_t i=0;i<ginfo->pair[j].r2.nph;++i)
          ((dcmplx *)map)[ginfo->pair[j].r2.ofs+i*ginfo->pair[j].r2.stride]=0;
        }
      else
        {
        for (ptrdiff_t i=0;i<ginfo->pair[j].r1.nph;++i)
          ((fcmplx *)map)[ginfo->pair[j].r1.ofs+i*ginfo->pair[j].r1.stride]=0;
        for (ptrdiff_t i=0;i<ginfo->pair[j].r2.nph;++i)
          ((fcmplx *)map)[ginfo->pair[j].r2.ofs+i*ginfo->pair[j].r2.stride]=0;
        }
      }
    }
  else
    {
    if (flags&SHARP_DP)
      {
      for (int j=0;j<ginfo->pair.size();++j)
        {
        double *dmap=(double *)map;
        if (ginfo->pair[j].r1.stride==1)
          memset(&dmap[ginfo->pair[j].r1.ofs],0,
            ginfo->pair[j].r1.nph*sizeof(double));
        else
          for (ptrdiff_t i=0;i<ginfo->pair[j].r1.nph;++i)
            dmap[ginfo->pair[j].r1.ofs+i*ginfo->pair[j].r1.stride]=0;
        if ((ginfo->pair[j].r2.nph>0)&&(ginfo->pair[j].r2.stride==1))
          memset(&dmap[ginfo->pair[j].r2.ofs],0,
            ginfo->pair[j].r2.nph*sizeof(double));
        else
          for (ptrdiff_t i=0;i<ginfo->pair[j].r2.nph;++i)
            dmap[ginfo->pair[j].r2.ofs+i*ginfo->pair[j].r2.stride]=0;
        }
      }
    else
      {
      for (int j=0;j<ginfo->pair.size();++j)
        {
        for (ptrdiff_t i=0;i<ginfo->pair[j].r1.nph;++i)
          ((float *)map)[ginfo->pair[j].r1.ofs+i*ginfo->pair[j].r1.stride]=0;
        for (ptrdiff_t i=0;i<ginfo->pair[j].r2.nph;++i)
          ((float *)map)[ginfo->pair[j].r2.ofs+i*ginfo->pair[j].r2.stride]=0;
        }
      }
    }
  }

MRUTIL_NOINLINE static void clear_alm (const sharp_alm_info *ainfo, void *alm,
  int flags)
  {
#define CLEARLOOP(real_t,body)             \
      {                                    \
        real_t *talm = (real_t *)alm;      \
          for (int l=m;l<=ainfo->lmax;++l) \
            body                           \
      }

  for (int mi=0;mi<ainfo->nm;++mi)
    {
      int m=ainfo->mval[mi];
      ptrdiff_t mvstart = ainfo->mvstart[mi];
      ptrdiff_t stride = ainfo->stride;
      if (!(ainfo->flags&SHARP_PACKED))
        mvstart*=2;
      if ((ainfo->flags&SHARP_PACKED)&&(m==0))
        {
        if (flags&SHARP_DP)
          CLEARLOOP(double, talm[mvstart+l*stride] = 0.;)
        else
          CLEARLOOP(float, talm[mvstart+l*stride] = 0.;)
        }
      else
        {
        stride*=2;
        if (flags&SHARP_DP)
          CLEARLOOP(double,talm[mvstart+l*stride]=talm[mvstart+l*stride+1]=0.;)
        else
          CLEARLOOP(float,talm[mvstart+l*stride]=talm[mvstart+l*stride+1]=0.;)
        }

#undef CLEARLOOP
    }
  }

MRUTIL_NOINLINE static void init_output (sharp_job *job)
  {
  if (job->flags&SHARP_ADD) return;
  if (job->type == SHARP_MAP2ALM)
    for (int i=0; i<job->nalm; ++i)
      clear_alm (job->ainfo,job->alm[i],job->flags);
  else
    for (int i=0; i<job->nmaps; ++i)
      clear_map (job->ginfo,job->map[i],job->flags);
  }

MRUTIL_NOINLINE static void alloc_phase (sharp_job *job, int nm, int ntheta)
  {
  if (job->type==SHARP_MAP2ALM)
    {
    job->s_m=2*job->nmaps;
    if (((job->s_m*16*nm)&1023)==0) nm+=3; // hack to avoid critical strides
    job->s_th=job->s_m*nm;
    }
  else
    {
    job->s_th=2*job->nmaps;
    if (((job->s_th*16*ntheta)&1023)==0) ntheta+=3; // hack to avoid critical strides
    job->s_m=job->s_th*ntheta;
    }
  job->phase=RALLOC(dcmplx,2*job->nmaps*nm*ntheta);
  }

static void dealloc_phase (sharp_job *job)
  { DEALLOC(job->phase); }

static void alloc_almtmp (sharp_job *job, int lmax)
  { job->almtmp=RALLOC(dcmplx,job->nalm*(lmax+2)); }

static void dealloc_almtmp (sharp_job *job)
  { DEALLOC(job->almtmp); }

MRUTIL_NOINLINE static void alm2almtmp (sharp_job *job, int lmax, int mi)
  {

#define COPY_LOOP(real_t, source_t, expr_of_x)              \
  {                                                         \
  for (int l=m; l<lmin; ++l)                                \
    for (int i=0; i<job->nalm; ++i)             \
      job->almtmp[job->nalm*l+i] = 0;           \
  for (int l=lmin; l<=lmax; ++l)                            \
    for (int i=0; i<job->nalm; ++i)             \
      {                                                     \
      source_t x = *(source_t *)(((real_t *)job->alm[i])+ofs+l*stride); \
      job->almtmp[job->nalm*l+i] = expr_of_x;   \
      }                                                     \
  for (int i=0; i<job->nalm; ++i)             \
    job->almtmp[job->nalm*(lmax+1)+i] = 0;           \
  }

  if (job->type!=SHARP_MAP2ALM)
    {
    ptrdiff_t ofs=job->ainfo->mvstart[mi];
    int stride=job->ainfo->stride;
    int m=job->ainfo->mval[mi];
    int lmin=(m<job->spin) ? job->spin : m;
    /* in the case of SHARP_REAL_HARMONICS, phase2ring scales all the
       coefficients by sqrt_one_half; here we must compensate to avoid scaling
       m=0 */
    double norm_m0=(job->flags&SHARP_REAL_HARMONICS) ? sqrt_two : 1.;
    if (!(job->ainfo->flags&SHARP_PACKED))
      ofs *= 2;
    if (!((job->ainfo->flags&SHARP_PACKED)&&(m==0)))
      stride *= 2;
    if (job->spin==0)
      {
      if (m==0)
        {
        if (job->flags&SHARP_DP)
          COPY_LOOP(double, double, x*norm_m0)
        else
          COPY_LOOP(float, float, x*norm_m0)
        }
      else
        {
        if (job->flags&SHARP_DP)
          COPY_LOOP(double, dcmplx, x)
        else
          COPY_LOOP(float, fcmplx, x)
        }
      }
    else
      {
      if (m==0)
        {
        if (job->flags&SHARP_DP)
          COPY_LOOP(double, double, x*job->norm_l[l]*norm_m0)
        else
          COPY_LOOP(float, float, x*job->norm_l[l]*norm_m0)
        }
      else
        {
        if (job->flags&SHARP_DP)
          COPY_LOOP(double, dcmplx, x*job->norm_l[l])
        else
          COPY_LOOP(float, fcmplx, x*float(job->norm_l[l]))
        }
      }
    }
  else
    memset (job->almtmp+job->nalm*job->ainfo->mval[mi], 0,
      job->nalm*(lmax+2-job->ainfo->mval[mi])*sizeof(dcmplx));

#undef COPY_LOOP
  }

MRUTIL_NOINLINE static void almtmp2alm (sharp_job *job, int lmax, int mi)
  {

#define COPY_LOOP(real_t, target_t, expr_of_x)               \
  for (int l=lmin; l<=lmax; ++l)                             \
    for (int i=0; i<job->nalm; ++i)              \
      {                                                      \
        dcmplx x = job->almtmp[job->nalm*l+i];   \
        *(target_t *)(((real_t *)job->alm[i])+ofs+l*stride) += expr_of_x; \
      }

  if (job->type != SHARP_MAP2ALM) return;
  ptrdiff_t ofs=job->ainfo->mvstart[mi];
  int stride=job->ainfo->stride;
  int m=job->ainfo->mval[mi];
  int lmin=(m<job->spin) ? job->spin : m;
  /* in the case of SHARP_REAL_HARMONICS, ring2phase scales all the
     coefficients by sqrt_two; here we must compensate to avoid scaling
     m=0 */
  double norm_m0=(job->flags&SHARP_REAL_HARMONICS) ? sqrt_one_half : 1.;
  if (!(job->ainfo->flags&SHARP_PACKED))
    ofs *= 2;
  if (!((job->ainfo->flags&SHARP_PACKED)&&(m==0)))
    stride *= 2;
  if (job->spin==0)
    {
    if (m==0)
      {
      if (job->flags&SHARP_DP)
        COPY_LOOP(double, double, x.real()*norm_m0)
      else
        COPY_LOOP(float, float, x.real()*norm_m0)
      }
    else
      {
      if (job->flags&SHARP_DP)
        COPY_LOOP(double, dcmplx, x)
      else
        COPY_LOOP(float, fcmplx, (fcmplx)x)
      }
    }
  else
    {
    if (m==0)
      {
      if (job->flags&SHARP_DP)
        COPY_LOOP(double, double, x.real()*job->norm_l[l]*norm_m0)
      else
        COPY_LOOP(float, fcmplx, (float)(x.real()*job->norm_l[l]*norm_m0))
      }
    else
      {
      if (job->flags&SHARP_DP)
        COPY_LOOP(double, dcmplx, x*job->norm_l[l])
      else
        COPY_LOOP(float, fcmplx, (fcmplx)(x*job->norm_l[l]))
      }
    }

#undef COPY_LOOP
  }

MRUTIL_NOINLINE static void ringtmp2ring (sharp_job *job, const sharp_ringinfo *ri,
  const vector<double> &ringtmp, int rstride)
  {
  if (job->flags & SHARP_DP)
    {
    double **dmap = (double **)job->map;
    for (int i=0; i<job->nmaps; ++i)
      {
      double *MRUTIL_RESTRICT p1=&dmap[i][ri->ofs];
      const double *MRUTIL_RESTRICT p2=&ringtmp[i*rstride+1];
      if (ri->stride==1)
        {
        if (job->flags&SHARP_ADD)
          for (int m=0; m<ri->nph; ++m)
            p1[m] += p2[m];
        else
          memcpy(p1,p2,ri->nph*sizeof(double));
        }
      else
        for (int m=0; m<ri->nph; ++m)
          p1[m*ri->stride] += p2[m];
      }
    }
  else
    {
    float  **fmap = (float  **)job->map;
    for (int i=0; i<job->nmaps; ++i)
      for (int m=0; m<ri->nph; ++m)
        fmap[i][ri->ofs+m*ri->stride] += (float)ringtmp[i*rstride+m+1];
    }
  }

MRUTIL_NOINLINE static void ring2ringtmp (sharp_job *job, const sharp_ringinfo *ri,
  vector<double> &ringtmp, int rstride)
  {
  if (job->flags & SHARP_DP)
    for (int i=0; i<job->nmaps; ++i)
      {
      double *MRUTIL_RESTRICT p1=&ringtmp[i*rstride+1],
             *MRUTIL_RESTRICT p2=&(((double *)(job->map[i]))[ri->ofs]);
      if (ri->stride==1)
        memcpy(p1,p2,ri->nph*sizeof(double));
      else
        for (int m=0; m<ri->nph; ++m)
          p1[m] = p2[m*ri->stride];
      }
  else
    for (int i=0; i<job->nmaps; ++i)
      for (int m=0; m<ri->nph; ++m)
        ringtmp[i*rstride+m+1] = ((float *)(job->map[i]))[ri->ofs+m*ri->stride];
  }

static void ring2phase_direct (sharp_job *job, const sharp_ringinfo *ri, int mmax,
  dcmplx *phase)
  {
  if (ri->nph<0)
    {
    for (int i=0; i<job->nmaps; ++i)
      for (int m=0; m<=mmax; ++m)
        phase[2*i+job->s_m*m]=0.;
    }
  else
    {
    MR_assert(ri->nph==mmax+1,"bad ring size");
    double wgt = (job->flags&SHARP_USE_WEIGHTS) ? (ri->nph*ri->weight) : 1.;
    if (job->flags&SHARP_REAL_HARMONICS)
      wgt *= sqrt_two;
    for (int i=0; i<job->nmaps; ++i)
      for (int m=0; m<=mmax; ++m)
        phase[2*i+job->s_m*m]= (job->flags & SHARP_DP) ?
          ((dcmplx *)(job->map[i]))[ri->ofs+m*ri->stride]*wgt :
          ((fcmplx *)(job->map[i]))[ri->ofs+m*ri->stride]*float(wgt);
    }
  }
static void phase2ring_direct (sharp_job *job, const sharp_ringinfo *ri, int mmax,
  dcmplx *phase)
  {
  if (ri->nph<0) return;
  MR_assert(ri->nph==mmax+1,"bad ring size");
  dcmplx **dmap = (dcmplx **)job->map;
  fcmplx **fmap = (fcmplx **)job->map;
  double wgt = (job->flags&SHARP_USE_WEIGHTS) ? (ri->nph*ri->weight) : 1.;
  if (job->flags&SHARP_REAL_HARMONICS)
    wgt *= sqrt_one_half;
  for (int i=0; i<job->nmaps; ++i)
    for (int m=0; m<=mmax; ++m)
      if (job->flags & SHARP_DP)
        dmap[i][ri->ofs+m*ri->stride] += wgt*phase[2*i+job->s_m*m];
      else
        fmap[i][ri->ofs+m*ri->stride] += (fcmplx)(wgt*phase[2*i+job->s_m*m]);
  }

//FIXME: set phase to zero if not SHARP_MAP2ALM?
MRUTIL_NOINLINE static void map2phase (sharp_job *job, int mmax, int llim, int ulim)
  {
  if (job->type != SHARP_MAP2ALM) return;
  int pstride = job->s_m;
  if (job->flags & SHARP_NO_FFT)
    {
    for (int ith=llim; ith<ulim; ++ith)
      {
      int dim2 = job->s_th*(ith-llim);
      ring2phase_direct(job,&(job->ginfo->pair[ith].r1),mmax,
        &(job->phase[dim2]));
      ring2phase_direct(job,&(job->ginfo->pair[ith].r2),mmax,
        &(job->phase[dim2+1]));
      }
    }
  else
    {
    mr::execDynamic(ulim-llim, 0, 1, [&](mr::Scheduler &sched)
      {
      ringhelper helper;
      int rstride=job->ginfo->nphmax+2;
      vector<double> ringtmp(job->nmaps*rstride);

      while (auto rng=sched.getNext()) for(auto ith=rng.lo+llim; ith<rng.hi+llim; ++ith)
        {
        int dim2 = job->s_th*(ith-llim);
        ring2ringtmp(job,&(job->ginfo->pair[ith].r1),ringtmp,rstride);
        for (int i=0; i<job->nmaps; ++i)
          ringhelper_ring2phase (helper,&(job->ginfo->pair[ith].r1),
            &ringtmp[i*rstride],mmax,&job->phase[dim2+2*i],pstride,job->flags);
        if (job->ginfo->pair[ith].r2.nph>0)
          {
          ring2ringtmp(job,&(job->ginfo->pair[ith].r2),ringtmp,rstride);
          for (int i=0; i<job->nmaps; ++i)
            ringhelper_ring2phase (helper,&(job->ginfo->pair[ith].r2),
             &ringtmp[i*rstride],mmax,&job->phase[dim2+2*i+1],pstride,job->flags);
          }
        }
      }); /* end of parallel region */
    }
  }

MRUTIL_NOINLINE static void phase2map (sharp_job *job, int mmax, int llim, int ulim)
  {
  if (job->type == SHARP_MAP2ALM) return;
  int pstride = job->s_m;
  if (job->flags & SHARP_NO_FFT)
    {
    for (int ith=llim; ith<ulim; ++ith)
      {
      int dim2 = job->s_th*(ith-llim);
      phase2ring_direct(job,&(job->ginfo->pair[ith].r1),mmax,
        &(job->phase[dim2]));
      phase2ring_direct(job,&(job->ginfo->pair[ith].r2),mmax,
        &(job->phase[dim2+1]));
      }
    }
  else
    {
    mr::execDynamic(ulim-llim, 0, 1, [&](mr::Scheduler &sched)
      {
      ringhelper helper;
      int rstride=job->ginfo->nphmax+2;
      vector<double> ringtmp(job->nmaps*rstride);

      while (auto rng=sched.getNext()) for(auto ith=rng.lo+llim; ith<rng.hi+llim; ++ith)
        {
        int dim2 = job->s_th*(ith-llim);
        for (int i=0; i<job->nmaps; ++i)
          ringhelper_phase2ring (helper,&(job->ginfo->pair[ith].r1),
            &ringtmp[i*rstride],mmax,&job->phase[dim2+2*i],pstride,job->flags);
        ringtmp2ring(job,&(job->ginfo->pair[ith].r1),ringtmp,rstride);
        if (job->ginfo->pair[ith].r2.nph>0)
          {
          for (int i=0; i<job->nmaps; ++i)
            ringhelper_phase2ring (helper,&(job->ginfo->pair[ith].r2),
              &ringtmp[i*rstride],mmax,&job->phase[dim2+2*i+1],pstride,job->flags);
          ringtmp2ring(job,&(job->ginfo->pair[ith].r2),ringtmp,rstride);
          }
        }
      }); /* end of parallel region */
    }
  }

MRUTIL_NOINLINE static void sharp_execute_job (sharp_job *job)
  {
  mr::timers::SimpleTimer timer;
  job->opcnt=0;
  int lmax = job->ainfo->lmax,
      mmax=sharp_get_mmax(job->ainfo->mval);

  job->norm_l = (job->type==SHARP_ALM2MAP_DERIV1) ?
     sharp_Ylmgen::get_d1norm (lmax) :
     sharp_Ylmgen::get_norm (lmax, job->spin);

/* clear output arrays if requested */
  init_output (job);

  int nchunks, chunksize;
  get_chunk_info(job->ginfo->pair.size(),sharp_veclen()*sharp_max_nvec(job->spin),
                 &nchunks,&chunksize);
//FIXME: needs to be changed to "nm"
  alloc_phase (job,mmax+1,chunksize);
  std::atomic<size_t> opcnt = 0;

/* chunk loop */
  for (int chunk=0; chunk<nchunks; ++chunk)
    {
    int llim=chunk*chunksize, ulim=min<int>(llim+chunksize,job->ginfo->pair.size());
    vector<int> ispair(ulim-llim);
    vector<int> mlim(ulim-llim);
    vector<double> cth(ulim-llim), sth(ulim-llim);
    for (int i=0; i<ulim-llim; ++i)
      {
      ispair[i] = job->ginfo->pair[i+llim].r2.nph>0;
      cth[i] = job->ginfo->pair[i+llim].r1.cth;
      sth[i] = job->ginfo->pair[i+llim].r1.sth;
      mlim[i] = sharp_get_mlim(lmax, job->spin, sth[i], cth[i]);
      }

/* map->phase where necessary */
    map2phase (job, mmax, llim, ulim);

    mr::execDynamic(job->ainfo->nm, 0, 1, [&](mr::Scheduler &sched)
      {
      sharp_job ljob = *job;
      ljob.opcnt=0;
      sharp_Ylmgen generator(lmax,mmax,ljob.spin);
      alloc_almtmp(&ljob,lmax);

      while (auto rng=sched.getNext()) for(auto mi=rng.lo; mi<rng.hi; ++mi)
        {
/* alm->alm_tmp where necessary */
        alm2almtmp (&ljob, lmax, mi);

        inner_loop (&ljob, ispair.data(), cth.data(), sth.data(), llim, ulim, generator, mi, mlim.data());

/* alm_tmp->alm where necessary */
        almtmp2alm (&ljob, lmax, mi);
        }

      dealloc_almtmp(&ljob);

      opcnt+=ljob.opcnt;
      }); /* end of parallel region */

/* phase->map where necessary */
    phase2map (job, mmax, llim, ulim);
    } /* end of chunk loop */

  dealloc_phase (job);
  job->opcnt = opcnt;
  job->time=timer();
  }

static void sharp_build_job_common (sharp_job *job, sharp_jobtype type,
  int spin, void *alm, void *map, const sharp_geom_info *geom_info,
  const sharp_alm_info *alm_info, int flags)
  {
  if (type==SHARP_ALM2MAP_DERIV1) spin=1;
  if (type==SHARP_MAP2ALM) flags|=SHARP_USE_WEIGHTS;
  if (type==SHARP_Yt) type=SHARP_MAP2ALM;
  if (type==SHARP_WY) { type=SHARP_ALM2MAP; flags|=SHARP_USE_WEIGHTS; }

  MR_assert((spin>=0)&&(spin<=alm_info->lmax), "bad spin");
  job->type = type;
  job->spin = spin;
  job->nmaps = (type==SHARP_ALM2MAP_DERIV1) ? 2 : ((spin>0) ? 2 : 1);
  job->nalm = (type==SHARP_ALM2MAP_DERIV1) ? 1 : ((spin>0) ? 2 : 1);
  job->ginfo = geom_info;
  job->ainfo = alm_info;
  job->flags = flags;
  if (alm_info->flags&SHARP_REAL_HARMONICS)
    job->flags|=SHARP_REAL_HARMONICS;
  job->time = 0.;
  job->opcnt = 0;
  job->alm=(void **)alm;
  job->map=(void **)map;
  }

void sharp_execute (sharp_jobtype type, int spin, void *alm, void *map,
  const sharp_geom_info *geom_info, const sharp_alm_info *alm_info,
  int flags, double *time, unsigned long long *opcnt)
  {
  sharp_job job;
  sharp_build_job_common (&job, type, spin, alm, map, geom_info, alm_info,
    flags);

  sharp_execute_job (&job);
  if (time!=NULL) *time = job.time;
  if (opcnt!=NULL) *opcnt = job.opcnt;
  }

void sharp_set_chunksize_min(int new_chunksize_min)
  { chunksize_min=new_chunksize_min; }
void sharp_set_nchunks_max(int new_nchunks_max)
  { nchunks_max=new_nchunks_max; }
