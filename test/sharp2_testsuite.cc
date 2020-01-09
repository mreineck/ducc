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

/*  \file sharp_testsuite.c
 *
 *  Copyright (C) 2012-2019 Max-Planck-Society
 *  \author Martin Reinecke
 */

#include <iostream>
#include <complex>
using std::complex;
#include "libsharp2/sharp.h"
#include "libsharp2/sharp_geomhelpers.h"
#include "libsharp2/sharp_almhelpers.h"
#include "mr_util/system.h"
#include "mr_util/error_handling.h"
#include "mr_util/threading.h"
#include "mr_util/math_utils.h"

using namespace std;
using namespace mr;

static void threading_status(void)
  {
  cout << "Threading: " << mr::max_threads() << " threads active." << endl;
  }

static void MPI_status(void)
  {
  cout << "MPI: not supported by this binary" << endl;
  }

static void sharp_announce (const string &name)
  {
  size_t m, nlen=name.length();
  cout << "\n+-";
  for (m=0; m<nlen; ++m) cout << "-";
  cout << "-+\n";
  cout << "| " << name << " |\n";
  cout << "+-";
  for (m=0; m<nlen; ++m) cout << "-";
  cout << "-+\n\n";
  cout << "Detected hardware architecture: " << sharp_architecture() << endl;
  cout << "Supported vector length: " << sharp_veclen() << endl;
  threading_status();
  MPI_status();
  cout << endl;
  }

static void sharp_module_startup (const string &name, int argc, int argc_expected,
  const string &argv_expected, int verbose)
  {
  if (verbose) sharp_announce (name);
  if (argc==argc_expected) return;
  if (verbose) cerr << "Usage: " << name << " " << argv_expected << endl;
  exit(1);
  }

typedef complex<double> dcmplx;

int ntasks, mytask;

static double drand (double min, double max, unsigned *state)
  {
  *state = (((*state) * 1103515245u) + 12345u) & 0x7fffffffu;
  return min + (max-min)*(*state)/(0x7fffffff+1.0);
  }

static void random_alm (dcmplx *alm, sharp_alm_info &helper, int spin, int cnt)
  {
#pragma omp parallel
{
  int mi;
#pragma omp for schedule (dynamic,100)
  for (mi=0;mi<helper.nm; ++mi)
    {
    int m=helper.mval[mi];
    unsigned state=1234567u*(unsigned)cnt+8912u*(unsigned)m; // random seed
    for (int l=m;l<=helper.lmax; ++l)
      {
      if ((l<spin)&&(m<spin))
        alm[helper.index(l,mi)] = 0.;
      else
        {
        double rv = drand(-1,1,&state);
        double iv = (m==0) ? 0 : drand(-1,1,&state);
        alm[helper.index(l,mi)] = dcmplx(rv,iv);
        }
      }
    }
} // end of parallel region
  }

static unsigned long long totalops (unsigned long long val)
  {
  return val;
  }

static double maxTime (double val)
  {
  return val;
  }

static double allreduceSumDouble (double val)
  {
  return val;
  }

static double totalMem()
  {
  return mr::getProcessInfo("VmHWM")*1024;
  }

static ptrdiff_t get_nalms(const sharp_alm_info &ainfo)
  {
  ptrdiff_t res=0;
  for (int i=0; i<ainfo.nm; ++i)
    res += ainfo.lmax-ainfo.mval[i]+1;
  return res;
  }

static ptrdiff_t get_npix(const sharp_geom_info &ginfo)
  {
  ptrdiff_t res=0;
  for (int i=0; i<ginfo.pair.size(); ++i)
    {
    res += ginfo.pair[i].r1.nph;
    if (ginfo.pair[i].r2.nph>0) res += ginfo.pair[i].r2.nph;
    }
  return res;
  }

static vector<double> get_sqsum_and_invert (dcmplx **alm, ptrdiff_t nalms, int ncomp)
  {
  vector<double> sqsum(ncomp);
  for (int i=0; i<ncomp; ++i)
    {
    sqsum[i]=0;
    for (ptrdiff_t j=0; j<nalms; ++j)
      {
      sqsum[i]+=norm(alm[i][j]);
      alm[i][j]=-alm[i][j];
      }
    }
  return sqsum;
  }

static void get_errors (dcmplx **alm, ptrdiff_t nalms, int ncomp, const vector<double> &sqsum,
  vector<double> &err_abs, vector<double> &err_rel)
  {
  long nalms_tot=nalms;

  err_abs.resize(ncomp);
  err_rel.resize(ncomp);
  for (int i=0; i<ncomp; ++i)
    {
    double sum=0, maxdiff=0, sumtot, sqsumtot, maxdifftot;
    for (ptrdiff_t j=0; j<nalms; ++j)
      {
      double sqr=norm(alm[i][j]);
      sum+=sqr;
      if (sqr>maxdiff) maxdiff=sqr;
      }
   maxdiff=sqrt(maxdiff);

    sumtot=sum;
    sqsumtot=sqsum[i];
    maxdifftot=maxdiff;
    sumtot=sqrt(sumtot/nalms_tot);
    sqsumtot=sqrt(sqsumtot/nalms_tot);
    err_abs[i]=maxdifftot;
    err_rel[i]=sumtot/sqsumtot;
    }
  }

static int good_fft_size(int n)
  {
  if (n<=6) return n;
  int bestfac=2*n;

  for (int f2=1; f2<bestfac; f2*=2)
    for (int f23=f2; f23<bestfac; f23*=3)
      for (int f235=f23; f235<bestfac; f235*=5)
        if (f235>=n) bestfac=f235;

  return bestfac;
  }

static void get_infos (const string &gname, int lmax, int &mmax, int &gpar1,
  int &gpar2, unique_ptr<sharp_geom_info> &ginfo, unique_ptr<sharp_alm_info> &ainfo, int verbose)
  {
  MR_assert(lmax>=0,"lmax must not be negative");
  if (mmax<0) mmax=lmax;
  MR_assert(mmax<=lmax,"mmax larger than lmax");

  verbose &= (mytask==0);
  if (verbose) cout << "lmax: " << lmax << ", mmax: " << mmax << endl;

  ainfo=sharp_make_triangular_alm_info(lmax,mmax,1);

  if (gname=="healpix")
    {
    if (gpar1<1) gpar1=lmax/2;
    if (gpar1==0) ++gpar1;
    ginfo=sharp_make_healpix_geom_info (gpar1, 1);
    if (verbose) cout << "HEALPix grid, nside=" << gpar1 << endl;
    }
  else if (gname=="gauss")
    {
    if (gpar1<1) gpar1=lmax+1;
    if (gpar2<1) gpar2=2*mmax+1;
    ginfo=sharp_make_gauss_geom_info (gpar1, gpar2, 0., 1, gpar2);
    if (verbose)
      cout << "Gauss-Legendre grid, nlat=" << gpar1 << ", nlon=" << gpar2 << endl;
    }
  else if (gname=="fejer1")
    {
    if (gpar1<1) gpar1=2*lmax+1;
    if (gpar2<1) gpar2=2*mmax+1;
    ginfo=sharp_make_fejer1_geom_info (gpar1, gpar2, 0., 1, gpar2);
    if (verbose)
      cout << "Fejer1 grid, nlat=" << gpar1 << ", nlon=" << gpar2 << endl;
    }
  else if (gname=="fejer2")
    {
    if (gpar1<1) gpar1=2*lmax+1;
    if (gpar2<1) gpar2=2*mmax+1;
    ginfo=sharp_make_fejer2_geom_info (gpar1, gpar2, 0., 1, gpar2);
    if (verbose)
      cout << "Fejer2 grid, nlat=" << gpar1 << ", nlon=" << gpar2 << endl;
    }
  else if (gname=="cc")
    {
    if (gpar1<1) gpar1=2*lmax+1;
    if (gpar2<1) gpar2=2*mmax+1;
    ginfo=sharp_make_cc_geom_info (gpar1, gpar2, 0., 1, gpar2);
    if (verbose)
      cout << "Clenshaw-Curtis grid, nlat=" << gpar1 << ", nlon=" << gpar2 << endl;
    }
  else if (gname=="smallgauss")
    {
    int nlat=gpar1, nlon=gpar2;
    if (nlat<1) nlat=lmax+1;
    if (nlon<1) nlon=2*mmax+1;
    gpar1=nlat; gpar2=nlon;
    ginfo=sharp_make_gauss_geom_info (nlat, nlon, 0., 1, nlon);
    ptrdiff_t npix_o=get_npix(*ginfo);
    size_t ofs=0;
    for (int i=0; i<ginfo->pair.size(); ++i)
      {
      sharp_ringpair &pair(ginfo->pair[i]);
      int pring=1+2*sharp_get_mlim(lmax,0,pair.r1.sth,pair.r1.cth);
      if (pring>nlon) pring=nlon;
      pring=good_fft_size(pring);
      pair.r1.nph=pring;
      pair.r1.weight*=nlon*1./pring;
      pair.r1.ofs=ofs;
      ofs+=pring;
      if (pair.r2.nph>0)
        {
        pair.r2.nph=pring;
        pair.r2.weight*=nlon*1./pring;
        pair.r2.ofs=ofs;
        ofs+=pring;
        }
      }
    if (verbose)
      {
      ptrdiff_t npix=get_npix(*ginfo);
      cout << "Small Gauss grid, nlat=" << nlat << ", npix=" << npix
           << ", savings=" << ((npix_o-npix)*100./npix_o) << "\%" << endl;
      }
    }
  else
    MR_fail("unknown grid geometry");
  }

static void check_sign_scale(void)
  {
  int lmax=50;
  int mmax=lmax;
  int nrings=lmax+1;
  int ppring=2*lmax+2;
  ptrdiff_t npix=(ptrdiff_t)nrings*ppring;
  auto tinfo = sharp_make_gauss_geom_info (nrings, ppring, 0., 1, ppring);

  /* flip theta to emulate the "old" Gaussian grid geometry */
  for (int i=0; i<tinfo->pair.size(); ++i)
    {
    const double pi=3.141592653589793238462643383279502884197;
    tinfo->pair[i].r1.cth=-tinfo->pair[i].r1.cth;
    tinfo->pair[i].r2.cth=-tinfo->pair[i].r2.cth;
    tinfo->pair[i].r1.theta=pi-tinfo->pair[i].r1.theta;
    tinfo->pair[i].r2.theta=pi-tinfo->pair[i].r2.theta;
    }

  auto alms = sharp_make_triangular_alm_info(lmax,mmax,1);
  ptrdiff_t nalms = ((mmax+1)*(mmax+2))/2 + (mmax+1)*(lmax-mmax);

  vector<double> bmap(2*npix);
  vector<double *>map({&bmap[0], &bmap[npix]});

  vector<dcmplx> balm(2*nalms,dcmplx(1.,1.));
  vector<dcmplx *>alm({&balm[0], &balm[nalms]});

  sharp_execute(SHARP_ALM2MAP,0,alm.data(),map.data(),*tinfo,*alms,SHARP_DP,
    nullptr,nullptr);
  MR_assert(approx(map[0][0     ], 3.588246976618616912e+00,1e-12),
    "error");
  MR_assert(approx(map[0][npix/2], 4.042209792157496651e+01,1e-12),
    "error");
  MR_assert(approx(map[0][npix-1],-1.234675107554816442e+01,1e-12),
    "error");

  sharp_execute(SHARP_ALM2MAP,1,alm.data(),map.data(),*tinfo,*alms,SHARP_DP,
    nullptr,nullptr);
  MR_assert(approx(map[0][0     ], 2.750897760535633285e+00,1e-12),
    "error");
  MR_assert(approx(map[0][npix/2], 3.137704477368562905e+01,1e-12),
    "error");
  MR_assert(approx(map[0][npix-1],-8.405730859837063917e+01,1e-12),
    "error");
  MR_assert(approx(map[1][0     ],-2.398026536095463346e+00,1e-12),
    "error");
  MR_assert(approx(map[1][npix/2],-4.961140548331700728e+01,1e-12),
    "error");
  MR_assert(approx(map[1][npix-1],-1.412765834230440021e+01,1e-12),
    "error");

  sharp_execute(SHARP_ALM2MAP,2,alm.data(),map.data(),*tinfo,*alms,SHARP_DP,
    nullptr,nullptr);
  MR_assert(approx(map[0][0     ],-1.398186224727334448e+00,1e-12),
    "error");
  MR_assert(approx(map[0][npix/2],-2.456676000884031197e+01,1e-12),
    "error");
  MR_assert(approx(map[0][npix-1],-1.516249174408820863e+02,1e-12),
    "error");
  MR_assert(approx(map[1][0     ],-3.173406200299964119e+00,1e-12),
    "error");
  MR_assert(approx(map[1][npix/2],-5.831327404513146462e+01,1e-12),
    "error");
  MR_assert(approx(map[1][npix-1],-1.863257892248353897e+01,1e-12),
    "error");

  sharp_execute(SHARP_ALM2MAP_DERIV1,1,alm.data(),map.data(),*tinfo,*alms,
    SHARP_DP,nullptr,nullptr);
  MR_assert(approx(map[0][0     ],-6.859393905369091105e-01,1e-11),
    "error");
  MR_assert(approx(map[0][npix/2],-2.103947835973212364e+02,1e-12),
    "error");
  MR_assert(approx(map[0][npix-1],-1.092463246472086439e+03,1e-12),
    "error");
  MR_assert(approx(map[1][0     ],-1.411433220713928165e+02,1e-12),
    "error");
  MR_assert(approx(map[1][npix/2],-1.146122859381925082e+03,1e-12),
    "error");
  MR_assert(approx(map[1][npix-1], 7.821618677689795049e+02,1e-12),
    "error");
  }

static void do_sht (sharp_geom_info &ginfo, sharp_alm_info &ainfo,
  int spin, vector<double> &err_abs, vector<double> &err_rel,
  double *t_a2m, double *t_m2a, unsigned long long *op_a2m,
  unsigned long long *op_m2a, size_t ntrans)
  {
  ptrdiff_t nalms = get_nalms(ainfo);
  int ncomp = (spin==0) ? 1 : 2;

  size_t npix = get_npix(ginfo);
  vector<double> bmap(ncomp*ntrans*npix, 0.);
  vector<double *>map(ncomp*ntrans);
  for (int i=0; i<ncomp*ntrans; ++i)
    map[i]=&bmap[i*npix];

  vector<dcmplx> balm(ncomp*ntrans*nalms);
  vector<dcmplx *>alm(ncomp*ntrans);
  for (int i=0; i<ncomp*ntrans; ++i)
    {
    alm[i] = &balm[i*nalms];
    random_alm(alm[i],ainfo,spin,i+1);
    }

  double tta2m, ttm2a;
  unsigned long long toa2m, tom2a;

  if (t_a2m!=nullptr) *t_a2m=0;
  if (op_a2m!=nullptr) *op_a2m=0;
  for (size_t itrans=0; itrans<ntrans; ++itrans)
    {
    sharp_execute(SHARP_ALM2MAP,spin,&alm[itrans*ncomp],&map[itrans*ncomp],ginfo,ainfo,
      SHARP_DP,&tta2m,&toa2m);
    if (t_a2m!=nullptr) *t_a2m+=maxTime(tta2m);
    if (op_a2m!=nullptr) *op_a2m+=totalops(toa2m);
    }
  auto sqsum=get_sqsum_and_invert(alm.data(),nalms,ntrans*ncomp);
  if (t_m2a!=nullptr) *t_m2a=0;
  if (op_m2a!=nullptr) *op_m2a=0;
  for (size_t itrans=0; itrans<ntrans; ++itrans)
    {
    sharp_execute(SHARP_MAP2ALM,spin,&alm[itrans*ncomp],&map[itrans*ncomp],ginfo,ainfo,
      SHARP_DP|SHARP_ADD,&ttm2a,&tom2a);
    if (t_m2a!=nullptr) *t_m2a+=maxTime(ttm2a);
    if (op_m2a!=nullptr) *op_m2a+=totalops(tom2a);
    }
  get_errors(alm.data(), nalms, ntrans*ncomp, sqsum, err_abs, err_rel);
  }

static void check_accuracy (sharp_geom_info &ginfo, sharp_alm_info &ainfo,
  int spin)
  {
  int ncomp = (spin==0) ? 1 : 2;
  vector<double> err_abs, err_rel;
  do_sht (ginfo, ainfo, spin, err_abs, err_rel, nullptr, nullptr,
    nullptr, nullptr, 1);
  for (int i=0; i<ncomp; ++i)
    MR_assert((err_rel[i]<1e-10) && (err_abs[i]<1e-10),"error");
  }

static void run(int lmax, int mmax, int nlat, int nlon, int spin)
  {
  unique_ptr<sharp_geom_info> ginfo;
  unique_ptr<sharp_alm_info> ainfo;
  get_infos ("gauss", lmax, mmax, nlat, nlon, ginfo, ainfo, 0);
  check_accuracy(*ginfo,*ainfo,spin);
  }

static void sharp_acctest(void)
  {
  if (mytask==0) sharp_module_startup("sharp_acctest",1,1,"",1);

  if (mytask==0) cout << "Checking signs and scales.\n";
  check_sign_scale();
  if (mytask==0) cout << "Passed.\n\n";

  if (mytask==0) cout << "Testing map analysis accuracy.\n";

  run(127, 127, 128, 256, 0);
  run(127, 127, 128, 256, 1);
  run(127, 127, 128, 256, 2);
  run(127, 127, 128, 256, 3);
  run(127, 127, 128, 256, 30);
  run(5, 0, 6, 1, 0);
  run(5, 0, 7, 2, 0);
  run(8, 8, 9, 17, 0);
  run(8, 8, 9, 17, 2);
  if (mytask==0) cout << "Passed.\n\n";
  }

static void sharp_test (int argc, const char **argv)
  {
  if (mytask==0) sharp_announce("sharp_test");
  MR_assert(argc>=8,"usage: grid lmax mmax geom1 geom2 spin [ntrans]");
  int lmax=atoi(argv[3]);
  int mmax=atoi(argv[4]);
  int gpar1=atoi(argv[5]);
  int gpar2=atoi(argv[6]);
  int spin=atoi(argv[7]);
  int ntrans=1;
  if (argc>=9) ntrans=atoi(argv[8]);

  if (mytask==0) cout << "Testing map analysis accuracy.\n";
  if (mytask==0) cout << "spin=" << spin << endl;
  if (mytask==0) cout << "ntrans=" << ntrans << endl;

  unique_ptr<sharp_geom_info> ginfo;
  unique_ptr<sharp_alm_info> ainfo;
  get_infos (argv[2], lmax, mmax, gpar1, gpar2, ginfo, ainfo, 1);

  int ncomp = (spin==0) ? 1 : 2;
  double t_a2m=1e30, t_m2a=1e30;
  unsigned long long op_a2m, op_m2a;
  vector<double> err_abs, err_rel;

  double t_acc=0;
  int nrpt=0;
  while(1)
    {
    ++nrpt;
    double ta2m2, tm2a2;
    do_sht (*ginfo, *ainfo, spin, err_abs, err_rel, &ta2m2, &tm2a2,
      &op_a2m, &op_m2a, ntrans);
    if (ta2m2<t_a2m) t_a2m=ta2m2;
    if (tm2a2<t_m2a) t_m2a=tm2a2;
    t_acc+=t_a2m+t_m2a;
    if (t_acc>2.)
      {
      if (mytask==0) cout << "Best of " << nrpt << " runs\n";
      break;
      }
    }

  if (mytask==0) cout << "wall time for alm2map: " << t_a2m << "s\n";
  if (mytask==0) cout << "Performance: " << 1e-9*op_a2m/t_a2m << "GFLOPs/s\n";
  if (mytask==0) cout << "wall time for map2alm: " << t_m2a << "s\n";
  if (mytask==0) cout << "Performance: " << 1e-9*op_m2a/t_m2a << "GFLOPs/s\n";

  if (mytask==0)
    for (int i=0; i<ntrans*ncomp; ++i)
      cout << "component " << i << ": rms " << err_rel[i] << ", maxerr " << err_abs[i] << endl;

  double iosize = ntrans*ncomp*(16.*get_nalms(*ainfo) + 8.*get_npix(*ginfo));
  iosize = allreduceSumDouble(iosize);

  double tmem=totalMem();
  if (mytask==0)
    cout << "\nMemory high water mark: " << tmem/(1<<20) << " MB\n";
  if (mytask==0)
    cout << "Memory overhead: " << (tmem-iosize)/(1<<20) << " MB ("
         << 100.*(1.-iosize/tmem) << "\% of working set)\n";

  double maxerel=0., maxeabs=0.;
  for (int i=0; i<ncomp; ++i)
    {
    if (maxerel<err_rel[i]) maxerel=err_rel[i];
    if (maxeabs<err_abs[i]) maxeabs=err_abs[i];
    }
  }

int main(int argc, const char **argv)
  {
  mytask=0; ntasks=1;

  MR_assert(argc>=2,"need at least one command line argument");
  auto mode = string(argv[1]);

  if (mode=="acctest")
    sharp_acctest();
  else if (mode=="test")
    sharp_test(argc,argv);
  else
    MR_fail("unknown command");

  return 0;
  }