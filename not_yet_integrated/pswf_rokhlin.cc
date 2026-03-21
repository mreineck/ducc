/* Routines for the evaluation of the prolate spheroidal wavefunction
   of order zero (Psi_0^c) inside [-1,1], for arbitrary frequency parameter c.
   They use a basis of Legendre polynomials.
   This is a collection of Fortran codes by Vladimir Rokhlin, specifically
   legeexps.f and prolcrea.f
   The originals may be found in src/common/specialfunctions/ of the DMK repo
   https://github.com/flatironinstitute/dmk
   They have been converted to C and repackaged by Libin Lu.
*/

#include <array>
#include <cmath>
#include <vector>

constexpr int PSWF_ERROR = 42;

class Prolate0Fun {
private:
  double c;
  std::vector<double> workdata; // Legendre coefficients
  std::vector<double> f1, f2;   // f1 = (j-1)/j; f2 = (2j-1)/j;
  int nterms;
  double xv0;

  static void prolcoef(double rlam, int k, double c, double &alpha, double &beta,
                       double &gamma) {
    double kf     = k;
    double alpha0 = kf * (kf - 1.) / ((2. * kf + 1.) * (2. * kf - 1.));
    double beta0  = ((kf + 1.) * (kf + 1.) / (2. * kf + 3.) + kf * kf / (2. * kf - 1.)) /
                   (2. * kf + 1.);
    double gamma0 = (kf + 1.) * (kf + 2.) / ((2. * kf + 1.) * (2. * kf + 3.));

    alpha = -c * c * alpha0;
    beta  = rlam - kf * (kf + 1.) - c * c * beta0;
    gamma = -c * c * gamma0;
  }

  static void prolmatr(std::vector<double> &as, std::vector<double> &bs,
                       std::vector<double> &cs, int n, double c, double rlam, int ifsymm,
                       int ifodd) {
    if (ifodd > 0) {
      for (int k = 1, k0 = 1; k0 <= n + 2; ++k, k0 += 2) {
        prolcoef(rlam, k0, c, as[k - 1], bs[k - 1], cs[k - 1]);

        if (ifsymm != 0) {
          if (k0 > 1)
            as[k - 1] *= std::sqrt((k0 + .5) / (k0 - 1.5));
          cs[k - 1] *= std::sqrt((k0 + .5) / (k0 + 2.5));
        }
      }
    } else {
      for (int k = 1, k0 = 0; k0 <= n + 2; ++k, k0 += 2) {
        prolcoef(rlam, k0, c, as[k - 1], bs[k - 1], cs[k - 1]);

        if (ifsymm != 0) {
          if (k0 != 0)
            as[k - 1] *= std::sqrt((k0 + .5) / (k0 - 1.5));
          cs[k - 1] *= std::sqrt((k0 + .5) / (k0 + 2.5));
        }
      }
    }
  }

  static void prolql1(int n, std::vector<double> &d, std::vector<double> &e) {
    if (n == 1) return;

    for (int i = 1; i < n; ++i)
      e[i - 1] = e[i];
    e[n - 1] = 0.0;

    for (int l = 0; l < n; ++l) {
      int j = 0;
      while (true) {
        int m;
        for (m = l; m < n - 1; ++m) {
          double tst1 = std::abs(d[m]) + std::abs(d[m + 1]);
          double tst2 = tst1 + std::abs(e[m]);
          if (tst2 == tst1) break;
        }

        if (m == l) break;
        if (j == 30) throw int(PSWF_ERROR);
        ++j;

        double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
        double r = std::sqrt(g * g + 1.0);
        g        = d[m] - d[l] + e[l] / (g + std::copysign(r, g));
        double s = 1.0;
        double c = 1.0;
        double p = 0.0;

        for (int i = m - 1; i >= l; --i) {
          double f = s * e[i];
          double b = c * e[i];
          r        = std::sqrt(f * f + g * g);
          e[i + 1] = r;
          if (r == 0.0) {
            d[i + 1] -= p;
            e[m] = 0.0;
            break;
          }
          s        = f / r;
          c        = g / r;
          g        = d[i + 1] - p;
          r        = (d[i] - g) * s + 2.0 * c * b;
          p        = s * r;
          d[i + 1] = g + p;
          g        = c * r - b;
        }

        if (r == 0.0) break;
        d[l] -= p;
        e[l] = g;
        e[m] = 0.0;
      }

      if (l == 0) continue;
      for (int i = l; i > 0; --i) {
        if (d[i] >= d[i - 1]) break;
        std::swap(d[i], d[i - 1]);
      }
    }
  }

  static void prolfact(std::vector<double> &a, const std::vector<double> &b,
                       const std::vector<double> &c, int n, std::vector<double> &u,
                       std::vector<double> &v, std::vector<double> &w) {
    // Eliminate down
    for (int i = 0; i < n - 1; ++i) {
      double d = c[i + 1] / a[i];
      a[i + 1] -= b[i] * d;
      u[i] = d;
    }

    // Eliminate up
    for (int i = n - 1; i > 0; --i) v[i] = b[i - 1] / a[i];

    // Scale the diagonal
    for (int i = 0; i < n; ++i) w[i] = 1. / a[i];
  }

  static void prolsolv(const std::vector<double> &u, const std::vector<double> &v,
                       const std::vector<double> &w, int n, std::vector<double> &rhs) {
    // Eliminate down
    for (int i = 0; i < n - 1; ++i) rhs[i + 1] -= u[i] * rhs[i];

    // Eliminate up
    for (int i = n - 1; i > 0; --i) rhs[i - 1] -= rhs[i] * v[i];

    // Scale
    for (int i = 0; i < n; ++i) rhs[i] *= w[i];
  }

  static void prolfun0(int n, double c, std::vector<double> &xk, double eps,
                       int &nterms) {
    double delta = 1.0e-8;

    xk.resize(n + 3);
    std::vector<double> as(n + 2), bs(n + 2), cs(n + 2), u(n + 2), v(n + 2), w(n + 2);
    prolmatr(as, bs, cs, n, c, 0., 1, -1);

    prolql1(n / 2, bs, as);

    double rlam = -bs[n / 2 - 1] + delta;

    std::fill(xk.begin(), xk.begin() + n, 1.0);

    prolmatr(as, bs, cs, n, c, rlam, 1, -1);

    prolfact(bs, cs, as, n / 2, u, v, w);

    int numit = 4;
    for (int ijk = 0; ijk < numit; ++ijk) {
      prolsolv(u, v, w, n / 2, xk);

      double d = 0;
      for (int j = 0; j < n / 2; ++j) d += xk[j] * xk[j];

      d = std::sqrt(d);
      for (int j = 0; j < n / 2; ++j) xk[j] /= d;

      double err = 0;
      for (int j = 0; j < n / 2; ++j) {
        err += (as[j] - xk[j]) * (as[j] - xk[j]);
        as[j] = xk[j];
      }
      err = std::sqrt(err);
    }

    for (int i = 0; i < n / 2; ++i) {
      if (std::abs(xk[i]) > eps) nterms = i + 1;
      xk[i] *= std::sqrt(i * 2 + .5);
    }

    xk.resize(nterms + 1);

    nterms *= 2;
  }

  static void prolps0i(double c, std::vector<double> &work, int &nterms) {
    static const std::array<int, 20> ns = {48,  64,  80,  92,  106, 120, 130,
                                           144, 156, 168, 178, 190, 202, 214,
                                           224, 236, 248, 258, 268, 280};

    double eps = 1.0e-16;
    int n      = static_cast<int>(c * 3) / 2;

    int i = static_cast<int>(c / 10);
    if (i < int(ns.size())) n = ns[i];

    work.resize(n + 3);

    prolfun0(n, c, work, eps, nterms);
  }

  double eval_raw(double x) const {
    int n       = nterms - 2;
    double pjm2 = 1.0;
    double pjm1 = x;

    double val = workdata[0];

    for (int j = 2; j <= n; j += 2) {
      pjm2 = f2[j] * x * pjm1 - f1[j] * pjm2;
      val += workdata[j / 2] * pjm2;
      pjm1 = f2[j + 1] * x * pjm2 - f1[j + 1] * pjm1;
    }
    return val;
  }

public:
  Prolate0Fun(double c_) : c(c_) {
    prolps0i(c, workdata, nterms);
    f1.resize(2 * workdata.size());
    f2.resize(2 * workdata.size());
    f1[0] = f2[0] = 0;
    for (size_t i = 1; i < f1.size(); ++i) {
      f1[i] = (i - 1.) / i;
      f2[i] = (2 * i - 1.) / i;
    }
    xv0 = 1. / eval_raw(0.);
  }

  double operator()(double x) const {
    if (std::abs(x) > 1) return 0.;
    return eval_raw(x) * xv0;
  }

  void multi_eval(const std::vector<double> &x, std::vector<double> &res) const {
    int n = nterms - 2;
    res.resize(x.size());

    constexpr int blksz = 4;
    size_t i            = 0;
    for (; i + blksz <= x.size(); i += blksz) {
      std::array<double, blksz> xx, pjm1, pjm2, val;

      for (int m = 0; m < blksz; ++m) {
        val[m]  = workdata[0];
        pjm2[m] = 1.0;
        xx[m]   = x[i + m];
        pjm1[m] = xx[m];
      }

      for (int j = 2; j <= n; j += 2) {
        for (int m = 0; m < blksz; ++m) {
          pjm2[m] = f2[j] * xx[m] * pjm1[m] - f1[j] * pjm2[m];
          val[m] += workdata[j / 2] * pjm2[m];
          pjm1[m] = f2[j + 1] * xx[m] * pjm2[m] - f1[j + 1] * pjm1[m];
        }
      }

      for (int m = 0; m < blksz; ++m) res[i + m] = val[m] * xv0;
    }

    for (; i < x.size(); ++i) res[i] = eval_raw(x[i]) * xv0;

    for (size_t j = 0; j < x.size(); ++j)
      if (std::abs(x[j]) > 1.) res[j] = 0.;
  }
};

#include <iostream>

using namespace std;

int main() {
  double sum = 0;
  Prolate0Fun fun(1.);
  vector<double> x{0.7, 0.7, 0.3, 0.8, 0.4, 0.1, 0.2, 0.3, 0.4}, res(x.size());
  for (size_t i = 0; i < 100000000; ++i) {
    fun.multi_eval(x, res);
    sum += res[1];
  }
  cout << sum << endl;
}
