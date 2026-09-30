/* Runtime proofs for bugs found during the 2026-09 bug hunt of ducc0.

   Each proof targets one confirmed bug and reports whether the bug could be
   reproduced. Compile with -fsanitize=address (see run.sh); two of the proofs
   (swap_axes, wigner3j) rely on ASAN to demonstrate the out-of-bounds access.

   Usage: bugproofs <name>
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

/* Bug 1: fmav_info::swap_axes (src/ducc0/infra/mav.h:326) and
   mav_info_proto::swap_axes (mav.h:536) assert `ax0<=ndim() && ax1<=ndim()`
   instead of `< ndim()`. Passing ax==ndim() passes validation and swaps one
   element past the end of the shape container. Expected: clean assertion
   failure. Actual (under ASAN): heap-buffer-overflow. */
static int proof_swap_axes()
  {
  fmav_info info(fmav_info::shape_t{2, 3});
  cout << "calling swap_axes(2, 0) on a 2D fmav_info (axis index 2 is invalid)\n";
  info.swap_axes(2, 0);
  cout << "no exception; shape now: " << info.shape(0) << " " << info.shape(1)
       << " (out-of-bounds swap happened)\n";
  // Same off-by-one in the fixed-shape variant; not ASAN-detectable
  // (intra-object overflow), but must also throw:
  mav_info<2>::shape_t shp{2, 3};
  mav_info<2> info2(shp, {3, 1});
  cout << "calling swap_axes(2, 0) on a 2D mav_info (axis index 2 is invalid)\n";
  info2.swap_axes(2, 0);
  cout << "no exception in mav_info either\n";
  return 0;
  }

/* Bug 2: slice::size() (src/ducc0/infra/mav.h:171) computes
   `(min(shp,end)-beg+step-1)/step` in unsigned arithmetic, which wraps around
   when end<beg. The "bad subset" safety check in subdata() (mav.h:594) wraps
   the same way and passes. Expected: MR_fail for an invalid slice.
   Actual: subarray() returns a view with a near-SIZE_MAX shape; any traversal
   is unbounded out-of-bounds access. */
static int proof_slice_wraparound()
  {
  vmav<double, 2> arr({10, 10});
  try
    {
    auto sub = arr.subarray<2>({slice(5, 2), slice(0, 4)});  // beg > end: invalid
    cout << "invalid slice (beg=5, end=2 on length-10 axis) accepted,"
         << " sub shape: " << sub.shape(0) << " " << sub.shape(1) << "\n";
    }
  catch (const exception &e)
    {
    cout << "rejected: " << e.what() << "\n";
    return 1;
    }
  try
    {
    auto sub = arr.subarray<2>({slice(2, 5, -1), slice(0, 4)});  // end > beg with negative step
    cout << "invalid negative-step slice accepted,"
         << " sub shape: " << sub.shape(0) << " " << sub.shape(1) << "\n";
    }
  catch (const exception &e)
    {
    cout << "rejected: " << e.what() << "\n";
    }
  return 0;
  }

/* Bug 3: Wigner3j_direct::calc() (src/ducc0/math/wigner3j.h:191) evaluates
   `g[ofs-1]` for the EE/TE terms. The only intended call pattern starts at
   ofs=0 (mcm.h: sum_wig_general), so this reads one double *before* the `g`
   table. The garbage is multiplied by `Jpmp*(Jpmp-1)` which is 0 at ofs=0,
   so the numerical result is usually still correct, but the read itself is
   out of bounds. Expected: no OOB access. Actual (under ASAN):
   heap-buffer-overflow READ of size 8, 0 bytes before the g buffer. */
static int proof_wigner3j_oob()
  {
  using Tsimd = native_simd<double>;
  detail_wigner3j::Wigner3j_direct_tables<Tsimd> tables(64);
  detail_wigner3j::Wigner3j_direct<Tsimd> wd(tables);
  cout << "prepping (el1, el2) = (10, 12), opmask = TT|TE|EE\n";
  wd.prep<7>(10, 12);
  cout << "calling calc(0) (the first offset used by sum_wig_general)\n";
  auto res = wd.calc<7>(0);
  cout << "returned " << res[0][0] << " without visible error"
       << " (the g[-1] read happened before this line)\n";
  return 0;
  }

/* Bug 4: TemplateKernel::transferCoeffs (src/ducc0/math/gridding_kernel.h:331)
   zeroes only `coeff[0 .. nvec_eval-1]` (one row) when the polynomial degree
   is smaller than the hardcoded degree D, but `ofs = D-d_input` can be >= 2,
   in which case rows 1..ofs-1 of the coefficient array are never written and
   eval() reads uninitialized memory. selectKernel() only produces degrees
   that keep ofs<=1, but TemplateKernel is public API and accepts any
   PolynomialKernel with degree <= D. The placement-new buffer is poisoned
   with 0xFF (NaN doubles) to make the uninitialized read deterministic. */
static int proof_template_kernel()
  {
  constexpr size_t W = 8;
  using TK = TemplateKernel<W, native_simd<double>>;
  auto func = [](double v) { return exp(-v*v*9.); };
  detail_gridding_kernel::GLFullCorrection corr(W, func);

  // control: degree W+2 -> ofs=1 -> correctly handled
  PolynomialKernel krn_ok(W, W+2, func, corr);
  alignas(alignof(TK)) static unsigned char raw_ok[sizeof(TK)];
  memset(raw_ok, 0xff, sizeof(raw_ok));
  new (raw_ok) TK(krn_ok);
  double got_ok = reinterpret_cast<TK *>(raw_ok)->eval(0.37);
  reinterpret_cast<TK *>(raw_ok)->~TK();
  cout << "control (degree W+2, ofs=1): eval=" << got_ok
       << ", reference=" << krn_ok.eval(0.37) << "\n";

  // bug: degree W+1 -> ofs=2 -> row 1 never written
  PolynomialKernel krn(W, W+1, func, corr);
  alignas(alignof(TK)) static unsigned char raw[sizeof(TK)];
  memset(raw, 0xff, sizeof(raw));
  new (raw) TK(krn);
  double got = reinterpret_cast<TK *>(raw)->eval(0.37);
  reinterpret_cast<TK *>(raw)->~TK();
  double want = krn.eval(0.37);
  cout << "degree W+1 (ofs=2): eval=" << got << ", reference=" << want << "\n";
  cout << "relative error: " << abs(got-want)/max(1e-300, abs(want)) << "\n";
  return 0;
  }

/* Bug 5: T_Healpix_Base::get_interpol (src/ducc0/healpix/healpix_base.cc:1246)
   wraps the computed azimuthal index only for i1<0 and i2>=nr, but never for
   i1>=nr. For phi outside [0, 2pi) (pointing does not normalize phi, and
   get_interpol only validates theta) the returned pixel numbers index into
   the following ring, or entirely past the end of the map, while the weights
   look perfectly sane. ang2pix/pix2ang handle any phi via fmodulo, so this is
   inconsistent within the same class. Expected: identical pixels for phi and
   phi+2pi. Actual: different (partially out-of-map) pixels. */
static int proof_healpix_interpol()
  {
  detail_healpix::T_Healpix_Base<int64_t> base(0, RING);
  double phi = 5.96463537;
  array<int64_t, 4> pix1, pix2;
  array<double, 4> wgt1, wgt2;
  base.get_interpol(pointing(1.03698646, phi), pix1, wgt1);
  base.get_interpol(pointing(1.03698646, phi + 2*pi), pix2, wgt2);
  cout << "phi          : pixels " << pix1[0] << " " << pix1[1] << " "
       << pix1[2] << " " << pix1[3] << "\n";
  cout << "phi + 2*pi   : pixels " << pix2[0] << " " << pix2[1] << " "
       << pix2[2] << " " << pix2[3] << "\n";
  cout << "weights phi  : " << wgt1[0] << " " << wgt1[1] << " "
       << wgt1[2] << " " << wgt1[3] << "\n";
  cout << "weights phi+2: " << wgt2[0] << " " << wgt2[1] << " "
       << wgt2[2] << " " << wgt2[3] << "\n";
  cout << "map has " << base.Npix() << " pixels\n";
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
  if (which == "swap_axes") return proof_swap_axes();
  if (which == "slice_wraparound") return proof_slice_wraparound();
  if (which == "wigner3j_oob") return proof_wigner3j_oob();
  if (which == "template_kernel") return proof_template_kernel();
  if (which == "healpix_interpol") return proof_healpix_interpol();
  cerr << "unknown proof '" << which << "'\n";
  return 2;
  }
