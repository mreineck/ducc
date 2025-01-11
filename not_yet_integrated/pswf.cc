/*
  The code in this file was converted from specfun.f from scipy, which
  is in turn based on the work by Shanjie Zhang and Jianming Jin in
  their book "Computation of Special Functions", 1996, John Wiley &
  Sons, Inc.
 */

#include <cmath>
#include <vector>
#include <iostream>

double sdp_pswf_cipow(double base, int exp)
  {
  double result = 1;
  // This is never called with negative exponents. Avoids potential recursion.
  if (exp==1) return base;
  while (exp)
    {
    if ((exp & 1) != 0) result *= base;
    exp >>= 1;
    base *= base;
    }
  return result;
  }

static inline void pswf_sdmn(
        const int m,
        const int n,
        const double c,
        const double cv,
        const int kd,
        double* df
)
  {
/*       ===================================================== */
/*       Purpose: Compute the expansion coefficients of the */
/*                prolate and oblate spheroidal functions, dk */
/*       Input :  m  --- Mode parameter */
/*                n  --- Mode parameter */
/*                c  --- Spheroidal parameter */
/*                cv --- Characteristic value */
/*                KD --- Function code */
/*                       KD=1 for prolate; KD=-1 for oblate */
/*       Output:  DF(k) --- Expansion coefficients dk; */
/*                          DF(1), DF(2), ... correspond to */
/*                          d0, d2, ... for even n-m and d1, */
/*                          d3, ... for odd n-m */
/*       ===================================================== */

  int nm = int ((n-m) * .5 + c) + 25;
  if (c < 1e-10)
    {
    for (int i=0; i<nm; ++i)
      df[i] = 0.;
    df[(n-m) / 2] = 1.;
    return;
    }
  double cs = c * c * kd;
  int ip = 1;
  int k = 0;
  if (n-m == (n-m)/2<<1)
    ip = 0;
  std::vector<double> a(nm+2), d_(nm+2), g(nm+2);
  for (int i=0; i<nm+2; ++i)
    {
    if (ip==0)
      k = i << 1;
    if (ip==1)
      k = (i<<1) + 1;
    double dk0 = double (m + k);
    double dk1 = double (m + k + 1);
    double dk2 = double ((m + k) << 1);
    double d2k = double ((m << 1) + k);
    a[i] = (d2k+2.) * (d2k+1.) / ((dk2+3.) * (dk2+5.)) * cs;
    d_[i] = dk0 * dk1 + (dk0*2.*dk1 - m*2.*m - 1.) / ((dk2-1.) * (dk2+3.)) * cs;
    g[i] = k * (k-1.) / ((dk2-3.) * (dk2-1.)) * cs;
    }
  double fs = 1.;
  double f1 = 0.;
  double f0 = 1e-100;
  int kb = 0;
  df[nm] = 0.;
  double fl = 0.;
  for (k=nm; k>=1; --k)
    {
    double f = -((d_[k] - cv) * f0 + a[k] * f1) / g[k];
    if (std::abs(f) > std::abs(df[k]))
      {
      df[k-1] = f;
      f1 = f0;
      f0 = f;
      if (std::abs(f) > 1e100)
        {
        for (int k1=k; k1<=nm; ++k1)
          df[k1-1] *= 1e-100;
        f1 *= 1e-100;
        f0 *= 1e-100;
        }
      }
    else
      {
      kb = k;
      fl = df[k];
      f1 = 1e-100;
      double f2 = -(d_[0] - cv) / a[0] * f1;
      df[0] = f1;
      if (kb==1)
        fs = f2;
      else if (kb==2)
        {
        df[1] = f2;
        fs = -((d_[1] - cv) * f2 + g[1] * f1) / a[1];
        }
      else
        {
        df[1] = f2;
        for (int j=3; j<=kb+1; ++j)
          {
          double f = -((d_[j-2] - cv) * f2 + g[j-2] * f1) / a[j - 2];
          if (j <= kb)
            df[j-1] = f;
          if (std::abs(f) > 1e100)
            {
            for (int k1=0; k1<j; ++k1)
              df[k1] *= 1e-100;
            f *= 1e-100;
            f2 *= 1e-100;
            }
          f1 = f2;
          f2 = f;
          }
        fs = f;
        }
      break;
      }
    }

  double r1 = 1.;
  for (int j=m+ip+1; j<=(m+ip)<<1; ++j)
    r1 *= j;
  double su1 = df[0] * r1;
  for (k=2; k<=kb; ++k)
    {
    r1 = -r1 * (k + m + ip - 1.5) / (k-1.);
    su1 += r1 * df[k-1];
    }
  double su2 = 0.;
  double sw = 0.;
  for (k = kb + 1; k <= nm; ++k)
    {
    if (k!=1)
      r1 = -r1 * (k + m + ip - 1.5) / (k-1.);
    su2 += r1 * df[k-1];
    if (std::abs(sw-su2) < std::abs(su2) * 1e-14)
      break;
    sw = su2;
    }

  double r3 = 1.;
  for (int j=1; j <= (m+n+ip) / 2; ++j)
    r3 *= j + (n+m+ip) * .5;
  double r4 = 1.;
  for (int j=1; j <= (n-m-ip) / 2; ++j)
    r4 *= -4.*j;
  double s0 = r3 / (fl * (su1 / fs) + su2) / r4;
  for (int k=0; k<kb; ++k)
    df[k] *= fl/fs*s0;
  for (k=kb; k<nm; ++k)
    df[k] *= s0;
  }


static inline void pswf_sckb(
        int m,
        int n,
        double c,
        const double* df,
        double* ck
)
  {
  /*       ====================================================== */
  /*       Purpose: Compute the expansion coefficients of the */
  /*                prolate and oblate spheroidal functions */
  /*       Input :  m  --- Mode parameter */
  /*                n  --- Mode parameter */
  /*                c  --- Spheroidal parameter */
  /*                DF(k) --- Expansion coefficients dk */
  /*       Output:  CK(k) --- Expansion coefficients ck; */
  /*                          CK(1), CK(2), ... correspond to */
  /*                          c0, c2, ... */
  /*       ====================================================== */
// df[nm+1], ck[nm]
  if (c <= 1e-10) c = 1e-10;
  int nm = int ((n-m) * .5 + c) + 25;
  int ip = 1;
  if (n-m == (n-m)/2 << 1)
    ip = 0;
  double reg = 1.;
  if (m + nm > 80)
    reg = 1e-200;
  double fac = -sdp_pswf_cipow(.5, m);
  double sw = 0.;
  for (int k = 0; k <= nm-1; ++k)
    {
    fac = -fac;
    int i1 = (k<<1) + ip + 1;
    double r = reg;
    for (int i=i1; i<=i1 + (m<<1) - 1; ++i)
      r *= i;
    int i2 = k + m + ip;
    for (int i=i2; i<=i2+k-1; ++i)
      r *= i + .5;
    double sum = r * df[k];
    for (int i=k+1; i<=nm; ++i)
      {
      double d1 = i * 2. + ip;
      double d2 = m * 2. + d1;
      double d3 = i + m + ip - .5;
      r = r * d2 * (d2-1.) * i * (d3+k) / (d1 * (d1-1.) * (i-k) * d3);
      sum += r * df[i];
      if (std::abs(sw-sum) < std::abs(sum) * 1e-14)
        break;
      sw = sum;
      }

    double r1 = reg;
    for (int i=2; i<= m+k; ++i)
      r1 *= i;
    ck[k] = fac*sum/r1;
    }
  }

static inline void pswf_segv(
        const int m,
        const int n,
        const double c,
        const int kd,
        double* cv,
        double* eg
)
{
  std::vector<double> a(300), b(100), d_(300), e(300), f(300), g(300), h_(100);
  std::vector<double> cv0(100);

/*       ========================================================= */
/*       Purpose: Compute the characteristic values of spheroidal */
/*                wave functions */
/*       Input :  m  --- Mode parameter */
/*                n  --- Mode parameter */
/*                c  --- Spheroidal parameter */
/*                KD --- Function code */
/*                       KD=1 for Prolate; KD=-1 for Oblate */
/*       Output:  CV --- Characteristic value for given m, n and c */
/*                EG(L) --- Characteristic value for mode m and n' */
/*                          ( L = n' - m + 1 ) */
/*       ========================================================= */

  if (c < 1e-10)
    {
    for (int i=0; i<n-m+1; ++i)
      eg[i] = (i+m+1) * (i+m);
    *cv = eg[n-m];
    return;
    }
  int icm = (n-m+2)/2;
  int nm = int((n-m) * .5 + c) + 10;
  double cs = c*c*kd;
  int k = 0;
  for (int l=0; l<=1; ++l)
    {
    for (int i=1; i<=nm; ++i)
      {
      if (l==0)
        k = (i-1) << 1;
      if (l==1)
        k = (i<<1) - 1;
      double dk0 = double (m+k);
      double dk1 = double (m+k+1);
      double dk2 = double ((m+k) << 1);
      double d2k = double ((m<<1) + k);
      a[i-1] = (d2k+2.) * (d2k+1.) / ((dk2+3.) * (dk2+5.)) * cs;
      d_[i-1] = dk0*dk1 + (dk0*2.*dk1 - m*2.*m - 1.) / ((dk2-1.) * (dk2+3.)) * cs;
      g[i-1] = k * (k-1.) / ((dk2-3.) * (dk2-1.)) * cs;
      }
    for (k=1; k<nm; ++k)
      {
      e[k] = sqrt(a[k-1] * g[k]);
      f[k] = e[k] * e[k];
      }
    f[0] = e[0] = 0.;
    double xa = d_[nm-1] + std::abs(e[nm-1]);
    double xb = d_[nm-1] - std::abs(e[nm-1]);
    int nm1 = nm-1;
    for (int i=0; i<nm1; ++i)
      {
      double t = std::abs(e[i]) + std::abs(e[i+1]);
      xa = std::max(xa, d_[i]+t);
      xb = std::min(xb, d_[i]-t);
      }
    for (int i=0; i<icm; ++i)
      {
      b[i] = xa;
      h_[i] = xb;
      }
    for (k=1; k<=icm; ++k)
      {
      for (int k1=k; k1<=icm; ++k1)
        if (b[k1-1] < b[k-1])
          {
          b[k-1] = b[k1-1];
          break;
          }

      if (k!=1)
        if (h_[k-1] < h_[k-2])
          h_[k-1] = h_[k-2];

      double x1;
      while(true)
        {
        x1 = (b[k-1] + h_[k-1]) / 2.;
        cv0[k-1] = x1;
        if (std::abs((b[k-1] - h_[k-1]) / x1) < 1e-14)
          break;

        int j = 0;
        double s = 1.;
        for (int i=1; i<=nm; ++i)
          {
          if (s==0.)
            s += 1e-30;
          s = d_[i-1] - f[i-1]/s - x1;
          if (s<0.)
            ++j;
          }
        if (j<k)
          h_[k-1] = x1;
        else
          {
          b[k-1] = x1;
          if (j>=icm)
            b[icm-1] = x1;
          else
            {
            h_[j] = std::max(h_[j], x1);
            b[j-1] = std::min(b[j-1], x1);
            }
          }
        }

      cv0[k-1] = x1;
      if (l==0)
        eg[k*2-2] = cv0[k-1];
      if (l==1)
        eg[k*2-1] = cv0[k-1];
      }
    }
  *cv = eg[n-m];
  }

/**
 * @brief Evaluate PSWF at a specific point.
 *
 * Compute the prolate and oblate spheroidal angular functions of the first
 * kind and their derivatives.
 *
 * This function has been heavily specialised, as it it basically the
 * inner loop of PSWF generation.
 *
 * @param m Mode parameter, m = 0, 1, 2, ...
 * @param n Mode parameter, n = m, m + 1, ...
 * @param c Spheroidal parameter.
 * @param ck Expansion coefficients; CK(1), CK(2) ... correspond to c0, c2 ...
 * @param x Argument of angular function, |x| < 1.0
 * @return Angular function of the first kind.
 */
double sdp_pswf_aswfa(int m, int n, double c, const double* ck, double x)
  {
  const int nm = (int) ((n - m) / 2 + c) + 40;
  const int nm2 = nm / 2 - 2;
  const double x1 = 1.0 - x * x;
  const double a0 = (m == 0 && x1 == 0.0) ? 1.0 : pow(x1, m * 0.5);

  double su1 = ck[0];
  double x1p = x1;
  for (int k=1; k<=nm2; ++k, x1p*=x1)
    {
    const double r = ck[k] * x1p;
    su1 += r;
    const double t = r/su1;
    if (k >= 10 && abs(t) < 1e-14) break;
    }
  return ((n - m) % 2 == 0) ? (a0 * su1) : (a0 * x * su1);
  }

#include <vector>
#include <iostream>
#include <iomanip>

using namespace std;

int main()
  {
  int m=5;
  double c=0.8;
  vector<double> coeff(2000);
  vector<double> df(2000);
  
  int n = m+2;
  int kd = 1; // prolate
  double cv = 0.0, eg[200] = {0.0, 0.0};
  pswf_segv(m, n, c, kd, &cv, eg);
 // cout  << setprecision(15)<< cv<< endl;
  pswf_sdmn(m, n, c, cv, kd, df.data());
  pswf_sckb(m, n, c, df.data(), coeff.data());
  cout << setprecision(15) << sdp_pswf_aswfa(m, n, c, coeff.data(), 0.3) << endl;
  }
