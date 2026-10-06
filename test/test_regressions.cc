/* C++ regression checks for array handling, mathematical kernels, and
   HEALPix interpolation.

   The swap_axes and Wigner 3j checks use AddressSanitizer to detect invalid
   memory accesses (see run.sh).

   Usage: test_regressions <name>
   where <name> is one of: swap_axes, slice_wraparound, wigner3j_oob,
                           template_kernel, healpix_interpol
*/
#include <cmath>
#include <complex>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <new>
#include <string>
#include <vector>

#include "ducc0/infra/mav.h"
#include "ducc0/math/constants.h"
#include "ducc0/math/gridding_kernel.h"
#include "ducc0/math/wigner3j.h"
#include "ducc0/healpix/healpix_base.h"

using namespace ducc0;
using namespace std;

/* fmav_info::swap_axes (src/ducc0/infra/mav.h:326) and
   mav_info_proto::swap_axes (mav.h:536) assert `ax0<=ndim() && ax1<=ndim()`
   instead of `< ndim()`, so an axis index equal to ndim() passes validation
   and swaps one element past the end of the shape container (detected by
   ASAN). Correct behavior: reject invalid axis indices. */
static int test_swap_axes()
  {
  bool ok = true;
  fmav_info info(fmav_info::shape_t{2, 3});
  try
    {
    info.swap_axes(2, 0);  // axis index 2 is invalid for a 2D array
    cout << "fmav_info::swap_axes(2, 0) was accepted\n";
    ok = false;
    }
  catch (const exception &)
    { cout << "fmav_info::swap_axes(2, 0) rejected as expected\n"; }
  mav_info<2>::shape_t shp{2, 3};
  mav_info<2> info2(shp, {3, 1});
  try
    {
    info2.swap_axes(2, 0);
    cout << "mav_info<2>::swap_axes(2, 0) was accepted\n";
    ok = false;
    }
  catch (const exception &)
    { cout << "mav_info<2>::swap_axes(2, 0) rejected as expected\n"; }
  if (ok) cout << "PASS swap_axes\n";
  return ok ? 0 : 1;
  }

/* slice::size() (src/ducc0/infra/mav.h:171) computes
   `(min(shp,end)-beg+step-1)/step` in unsigned arithmetic, which wraps
   around when end<beg; the "bad subset" check in subdata() (mav.h:594)
   wraps the same way and passes. Correct behavior: reject invalid slices
   (end<beg with positive step, end>beg with negative step). */
static int test_slice_wraparound()
  {
  vmav<double, 2> arr({10, 10});
  bool ok = true;
  try
    {
    auto sub = arr.subarray<2>({slice(5, 2), slice(0, 4)});  // beg > end
    cout << "slice(5, 2) was accepted, sub shape: "
         << sub.shape(0) << " x " << sub.shape(1) << "\n";
    ok = false;
    }
  catch (const exception &)
    { cout << "slice(5, 2) rejected as expected\n"; }
  try
    {
    auto sub = arr.subarray<2>({slice(2, 5, -1), slice(0, 4)});  // end > beg, step < 0
    cout << "slice(2, 5, -1) was accepted, sub shape: "
         << sub.shape(0) << " x " << sub.shape(1) << "\n";
    ok = false;
    }
  catch (const exception &)
    { cout << "slice(2, 5, -1) rejected as expected\n"; }
  if (ok) cout << "PASS slice_wraparound\n";
  return ok ? 0 : 1;
  }

/* Wigner3j_direct::calc() (src/ducc0/math/wigner3j.h:191) evaluates
   `g[ofs-1]` for the EE/TE terms; the intended call pattern starts at
   ofs=0 (mcm.h: sum_wig_general), i.e. one double *before* the `g` table
   (detected by ASAN). Correct behavior: calc(0) must not access memory
   outside the table. */
static int test_wigner3j_oob()
  {
  using Tsimd = native_simd<double>;
  detail_wigner3j::Wigner3j_direct_tables<Tsimd> tables(64);
  detail_wigner3j::Wigner3j_direct<Tsimd> wd(tables);
  wd.prep<7>(10, 12);       // opmask TT|TE|EE
  auto res = wd.calc<7>(0);  // first offset used by sum_wig_general
  (void)res;
  cout << "PASS wigner3j_oob\n";  // only reached without an ASAN report
  return 0;
  }

/* TemplateKernel (src/ducc0/math/gridding_kernel.h): the degree of the input
   polynomial must be D-1 or D; this contract is asserted explicitly in
   transferCoeffs. Kernel evaluation must match the reference for the
   accepted degrees, and smaller degrees must be rejected. (Before the
   contract was asserted, a degree of D-2 left coefficient rows
   uninitialized and eval() read indeterminate memory; the buffer here is
   poisoned with 0xFF to keep catching that.) */
static int test_template_kernel()
  {
  constexpr size_t W = 8;
  using TK = TemplateKernel<W, native_simd<double>>;
  auto func = [](double v) { return exp(-v*v*9.); };
  detail_gridding_kernel::GLFullCorrection corr(W, func);
  bool ok = true;
  for (size_t deg : {W+3, W+2})  // d_input = D and D-1: must work
    {
    PolynomialKernel krn(W, deg, func, corr);
    alignas(alignof(TK)) static unsigned char raw[sizeof(TK)];
    memset(raw, 0xff, sizeof(raw));
    new (raw) TK(krn);
    double got = reinterpret_cast<TK *>(raw)->eval(0.37);
    reinterpret_cast<TK *>(raw)->~TK();
    double want = krn.eval(0.37);
    double relerr = abs(got-want)/max(1e-300, abs(want));
    if (!isfinite(got) || relerr > 1e-9)
      {
      cout << "degree " << deg << " (ofs=" << W+3+(W&1)-deg << "): eval="
           << got << ", expected " << want << "\n";
      ok = false;
      }
    else
      cout << "degree " << deg << " (ofs=" << W+3+(W&1)-deg << "): ok\n";
    }
    {  // d_input = D-2: must be rejected
    PolynomialKernel krn(W, W+1, func, corr);
    bool rejected = false;
    try { TK dummy(krn); (void)dummy; }
    catch (const exception &)
      { rejected = true; }
    if (rejected)
      cout << "degree " << W+1 << " (ofs=2): rejected as expected\n";
    else
      {
      cout << "degree " << W+1 << " (ofs=2) was accepted\n";
      ok = false;
      }
    }
  if (ok) cout << "PASS template_kernel\n";
  return ok ? 0 : 1;
  }

/* T_Healpix_Base::get_interpol (src/ducc0/healpix/healpix_base.cc:1246)
   wraps the computed azimuthal index only for i1<0 and i2>=nr, but never
   for i1>=nr. Since pointing does not normalize phi and get_interpol only
   validates theta, phi outside [0, 2pi) yields pixels from the wrong ring
   (or past the end of the map) while the weights look sane. Correct
   behavior: phi and phi+2pi describe the same direction and must give
   identical pixels and weights. */
static int test_healpix_interpol()
  {
  detail_healpix::T_Healpix_Base<int64_t> base(0, RING);
  double phi = 5.96463537;
  array<int64_t, 4> pix1, pix2;
  array<double, 4> wgt1, wgt2;
  base.get_interpol(pointing(1.03698646, phi), pix1, wgt1);
  base.get_interpol(pointing(1.03698646, phi + 2*pi), pix2, wgt2);
  if (pix1 != pix2)
    {
    cout << "phi        gives pixels " << pix1[0] << " " << pix1[1] << " "
         << pix1[2] << " " << pix1[3] << "\n";
    cout << "phi + 2pi  gives pixels " << pix2[0] << " " << pix2[1] << " "
         << pix2[2] << " " << pix2[3] << "\n";
    cout << "FAIL healpix_interpol\n";
    return 1;
    }
  cout << "PASS healpix_interpol\n";
  return 0;
  }

int main(int argc, char **argv)
  {
  if (argc != 2)
    {
    cerr << "usage: " << argv[0] << " <swap_axes|slice_wraparound|wigner3j_oob|template_kernel|healpix_interpol>\n";
    return 2;
    }
  string which = argv[1];
  if (which == "swap_axes") return test_swap_axes();
  if (which == "slice_wraparound") return test_slice_wraparound();
  if (which == "wigner3j_oob") return test_wigner3j_oob();
  if (which == "template_kernel") return test_template_kernel();
  if (which == "healpix_interpol") return test_healpix_interpol();
  cerr << "unknown test '" << which << "'\n";
  return 2;
  }
