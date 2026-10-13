/* C++ unit tests for bugs found during the 2026-09 bug hunt of ducc0.

   Each test fails while its bug is present and passes once the bug is
   fixed, then remains as a regression test.

   The swap_axes and wigner3j tests detect out-of-bounds accesses via
   AddressSanitizer, so this binary must be built with -fsanitize=address
   (see run.sh).

   Usage: test_regressions --list or test_regressions <name>
*/
#include <algorithm>
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <cmath>
#include <complex>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <string>
#include <vector>

#include "ducc0/infra/mav.h"
#include "ducc0/infra/threading.h"
#include "ducc0/wgridder/wgridder_impl.h"
#include "ducc0/math/constants.h"
#include "ducc0/math/gridding_kernel.h"
#include "ducc0/math/math_utils.h"
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

/* fmav_info::subdata() must preserve a full slice of an empty dimension.
   Empty subviews are used when batched Python APIs select a transform whose
   point dimension is zero. */
static int test_subarray_empty_axis()
  {
  bool ok=true;
  double dummy=0;
  vmav<double,2> fixed(&dummy, {2,0}, {100,1});
  auto fixed_sub=fixed.subarray<1>({slice(1), slice()});
  if ((fixed_sub.shape(0)!=0) || (fixed_sub.size()!=0)
      || (fixed_sub.data()!=fixed.data()))
    ok=false;

  fmav_info info({2,0}, {100,1});
  vfmav<double> dynamic(&dummy, info);
  auto dynamic_sub=dynamic.subarray({slice(1), slice()});
  if ((dynamic_sub.shape(0)!=0) || (dynamic_sub.size()!=0)
      || (dynamic_sub.data()!=dynamic.data()))
    ok=false;

  if (ok) cout << "PASS subarray_empty_axis\n";
  else cout << "FAIL subarray_empty_axis\n";
  return ok ? 0 : 1;
  }

/* build_noncritical() must preserve zero-length axes in the returned view. */
static int test_build_noncritical_empty()
  {
  bool ok=true;
  auto check = [&ok](const auto &arr, size_t n0, size_t n1)
    {
    if ((arr.shape(0)!=n0) || (arr.shape(1)!=n1) || (arr.size()!=0))
      ok=false;
    };
  try
    {
    auto fixed_1d=vmav<double,1>::build_noncritical({0});
    if ((fixed_1d.shape(0)!=0) || (fixed_1d.size()!=0)) ok=false;
    auto fixed=vmav<double,2>::build_noncritical({2,0});
    check(fixed, 2, 0);
    auto fixed_uninit=vmav<double,2>::build_noncritical({0,2}, UNINITIALIZED);
    check(fixed_uninit, 0, 2);
    auto fixed_pagein=vmav<double,2>::build_noncritical({2,0}, PAGE_IN(1));
    check(fixed_pagein, 2, 0);
    auto fixed_padded=vmav<double,2>::build_noncritical({0,512}, UNINITIALIZED);
    check(fixed_padded, 0, 512);
    auto dynamic=vfmav<double>::build_noncritical({2,0});
    check(dynamic, 2, 0);
    auto dynamic_uninit=vfmav<double>::build_noncritical({0,2}, UNINITIALIZED);
    check(dynamic_uninit, 0, 2);
    auto dynamic_pagein=vfmav<double>::build_noncritical({0,2}, PAGE_IN(1));
    check(dynamic_pagein, 0, 2);
    }
  catch (const exception &e)
    {
    cout << "build_noncritical rejected an empty shape: " << e.what() << "\n";
    ok=false;
    }
  if (ok) cout << "PASS build_noncritical_empty\n";
  else cout << "FAIL build_noncritical_empty\n";
  return ok ? 0 : 1;
  }

/* The MAXIDX end marker represents an open end and must work with reverse
   slices as well as forward slices. */
static int test_reverse_slice_open_end()
  {
  vmav<double,1> arr({4});
  for (size_t i=0; i<4; ++i) arr(i)=double(i);
  bool ok=true;
  try
    {
    auto rev=arr.subarray<1>({slice(3, MAXIDX, -1)});
    if ((rev.shape(0)!=4) || (rev.stride(0)!=-1)) ok=false;
    for (size_t i=0; i<4; ++i)
      if (rev(i)!=double(3-i)) ok=false;
    }
  catch (const exception &e)
    {
    cout << "reverse open-end slice was rejected: " << e.what() << "\n";
    ok=false;
    }
  if (ok) cout << "PASS reverse_slice_open_end\n";
  else cout << "FAIL reverse_slice_open_end\n";
  return ok ? 0 : 1;
  }

/* Negative step magnitude must be computed without overflowing at PTRDIFF_MIN. */
static int test_slice_min_step()
  {
  vmav<double,1> arr({2});
  arr(0)=10.;
  arr(1)=20.;
  auto step=std::numeric_limits<ptrdiff_t>::min();
  bool ok=true;
  try
    {
    auto finite=arr.subarray<1>({slice(1, 0, step)});
    auto open=arr.subarray<1>({slice(1, MAXIDX, step)});
    if ((finite.size()!=1) || (finite(0)!=20.)) ok=false;
    if ((open.size()!=1) || (open(0)!=20.)) ok=false;
    }
  catch (const exception &e)
    {
    cout << "PTRDIFF_MIN slice failed: " << e.what() << "\n";
    ok=false;
    }
  if (ok) cout << "PASS slice_min_step\n";
  else cout << "FAIL slice_min_step\n";
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


/* isqrt (src/ducc0/math/math_utils.h): for 64-bit arguments where the
   result is near 2^32-1, (res+1)*(res+1) can overflow 64-bit arithmetic.
   isqrt must return the floor integer square root up to UINT64_MAX. */
static int test_isqrt()
  {
  bool ok = true;
  uint64_t max32 = 0xFFFFFFFFULL;
  uint64_t sq_max32 = max32 * max32;
  if (isqrt(sq_max32) != max32)
    {
    cout << "isqrt(" << sq_max32 << ") = " << isqrt(sq_max32)
         << ", expected " << max32 << "\n";
    ok = false;
    }
  if (isqrt(~0ULL) != max32)
    {
    cout << "isqrt(UINT64_MAX) = " << isqrt(~0ULL)
         << ", expected " << max32 << "\n";
    ok = false;
    }
  if (isqrt(sq_max32 - 1) != max32 - 1)
    {
    cout << "isqrt(" << sq_max32 - 1 << ") = " << isqrt(sq_max32 - 1)
         << ", expected " << max32 - 1 << "\n";
    ok = false;
    }
  if (ok) cout << "PASS isqrt\n";
  return ok ? 0 : 1;
  }

static int test_null_backed_empty_subarrays()
  {
  vmav<double,1> fixed(static_cast<double *>(nullptr), {0}, {1});
  auto fixed_sub=fixed.subarray<1>({slice()});
  if ((fixed_sub.shape(0)!=0) || (fixed_sub.size()!=0)
      || (fixed_sub.data()!=nullptr))
    return 1;

  fmav_info info({0}, {1});
  vfmav<double> dynamic(static_cast<double *>(nullptr), info);
  auto dynamic_sub=dynamic.subarray({slice()});
  if ((dynamic_sub.shape(0)!=0) || (dynamic_sub.size()!=0)
      || (dynamic_sub.data()!=nullptr))
    return 1;

  cout << "PASS null_backed_empty_subarrays\n";
  return 0;
  }

static int test_singleton_slice_reversed_sources()
  {
  double raw[2]={10., 20.};
  auto step=numeric_limits<ptrdiff_t>::min();

  vmav<double,1> fixed_reversed(raw+1, {2}, {-1});
  auto fixed_single=fixed_reversed.subarray<1>({slice(1, 0, step)});
  if ((fixed_single.size()!=1) || (fixed_single.stride(0)!=0)
      || (fixed_single(0)!=10.))
    return 1;

  fmav_info info({2}, {-1});
  vfmav<double> dynamic_reversed(raw+1, info);
  auto dynamic_single=dynamic_reversed.subarray({slice(1, 0, step)});
  if ((dynamic_single.size()!=1) || (dynamic_single.stride(0)!=0)
      || (dynamic_single(0)!=10.))
    return 1;

  cout << "PASS singleton_slice_reversed_sources\n";
  return 0;
  }

static int test_quickzero_empty_views()
  {
  vmav<float,2> empty_leading({0,2});
  vmav<float,2> empty_trailing({2,0});
  detail_gridder::quickzero(empty_leading, 1);
  detail_gridder::quickzero(empty_trailing, 1);
  cout << "PASS quickzero_empty_views\n";
  return 0;
  }

class TestThreadPool: public thread_pool
  {
  private:
    vector<thread> workers;
  public:
    ~TestThreadPool() override
      { for (auto &worker: workers) worker.join(); }
    size_t nthreads() const override { return 1; }
    size_t adjust_nthreads(size_t nthreads) const override
      { return min<size_t>(2, nthreads); }
    void submit(function<void()> work) override
      { workers.emplace_back(move(work)); }
  };

struct ActiveCallbackGuard
  {
  atomic<size_t> &active;
  explicit ActiveCallbackGuard(atomic<size_t> &active_): active(active_)
    { active.fetch_add(1); }
  ~ActiveCallbackGuard() { active.fetch_sub(1); }
  };

static int test_callback_exception_wait()
  {
  for (size_t throwing_thread: {size_t(0), size_t(1)})
    {
    TestThreadPool pool;
    ScopedUseThreadPool pool_guard(pool);
    atomic<size_t> started{0}, active{0};
    atomic<bool> other_done{false}, throwing_callback_ready{false};
    bool caught=false;
    try
      {
      execParallel(2, [&](Scheduler &sched)
        {
        ActiveCallbackGuard active_guard(active);
        size_t tid=sched.thread_num();
        started.fetch_or(size_t(1)<<tid);
        if (tid==throwing_thread)
          {
          while ((started.load() & (size_t(1)<<(1-tid)))==0)
            this_thread::yield();
          if (tid==1)
            while (!other_done.load()) this_thread::yield();
          throwing_callback_ready.store(true);
          throw runtime_error("callback failure");
          }
        if (throwing_thread==0)
          this_thread::sleep_for(chrono::milliseconds(20));
        other_done.store(true);
        });
      }
    catch (const runtime_error &e)
      { caught=string(e.what())=="callback failure"; }
    if (!caught || (started.load()!=3) || active.load()!=0
        || !other_done.load() || !throwing_callback_ready.load())
      return 1;
    }
  cout << "PASS callback_exception_wait\n";
  return 0;
  }

struct RegressionTest { const char *name; int (*run)(); };

static constexpr RegressionTest regression_tests[] =
  {
  {"swap_axes", test_swap_axes},
  {"slice_wraparound", test_slice_wraparound},
  {"wigner3j_oob", test_wigner3j_oob},
  {"template_kernel", test_template_kernel},
  {"healpix_interpol", test_healpix_interpol},
  {"subarray_empty_axis", test_subarray_empty_axis},
  {"null_backed_empty_subarrays", test_null_backed_empty_subarrays},
  {"build_noncritical_empty", test_build_noncritical_empty},
  {"reverse_slice_open_end", test_reverse_slice_open_end},
  {"slice_min_step", test_slice_min_step},
  {"singleton_slice_reversed_sources", test_singleton_slice_reversed_sources},
  {"quickzero_empty_views", test_quickzero_empty_views},
  {"isqrt", test_isqrt},
  {"callback_exception_wait", test_callback_exception_wait},
  };

int main(int argc, char **argv)
  {
  if ((argc==2) && (string(argv[1])=="--list"))
    {
    for (const auto &test: regression_tests) cout << test.name << "\n";
    return 0;
    }
  if (argc!=2)
    {
    cerr << "usage: " << argv[0] << " [--list|<test-name>]\n";
    return 2;
    }
  for (const auto &test: regression_tests)
    if (string(argv[1])==test.name) return test.run();
  cerr << "unknown test '" << argv[1] << "'\n";
  return 2;
  }
