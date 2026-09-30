#!/usr/bin/env bash
# Runs all bug proofs from the 2026-09 bug hunt and reports whether each bug
# could still be reproduced.
#
# Default mode: exit status is non-zero if a proof does NOT reproduce, i.e.
#   if a bug was fixed or a proof broke; use this while developing.
# --fail-on-confirmed: exit status is non-zero if any bug is still
#   reproducible; use this as a CI gate that goes green once all bugs are
#   fixed (matching the xfail(strict=True) tests in python/test).
#
# Requirements: g++ or clang++ with AddressSanitizer, bash, python3 with numpy
# and the ducc0 package installed (only for the python part, see bottom).
set -u
cd "$(dirname "$0")/.."

MODE=default
if [ "${1:-}" = "--fail-on-confirmed" ]; then MODE=fail-on-confirmed; fi

CXX="${CXX:-g++}"
CXXFLAGS="-std=c++17 -O1 -g -Isrc"
SRCS="src/ducc0/healpix/healpix_base.cc src/ducc0/healpix/healpix_tables.cc \
      src/ducc0/infra/mav.cc src/ducc0/infra/threading.cc \
      src/ducc0/math/gl_integrator.cc src/ducc0/math/space_filling.cc \
      src/ducc0/math/pointing.cc src/ducc0/math/geom_utils.cc \
      src/ducc0/infra/string_utils.cc src/ducc0/infra/system.cc"
OUT=/tmp/ducc0_bugproofs
mkdir -p "$OUT"

fail=0
conf=0
note() { printf '%s\n' "$*"; }

note "=== 1. fmav_info/mav_info_proto::swap_axes off-by-one (mav.h:326/536) ==="
$CXX $CXXFLAGS -fsanitize=address -o "$OUT/bugproofs" \
  test_cpp/test_bug_hunt.cc $SRCS -pthread 2>"$OUT/build.log" \
  || { note "BUILD FAILED"; cat "$OUT/build.log"; exit 1; }
if ASAN_OPTIONS=detect_leaks=0 "$OUT/bugproofs" swap_axes 2>&1 \
    | grep -q "heap-buffer-overflow"; then
  note "CONFIRMED: ASAN reports heap-buffer-overflow for swap_axes(ndim, 0)"
  conf=$((conf+1))
else
  note "NOT REPRODUCED (bug appears to be fixed)"; fail=1
fi

note ""
note "=== 2. slice::size() unsigned wraparound accepts invalid subarrays (mav.h:171) ==="
if "$OUT/bugproofs" slice_wraparound 2>&1 | grep -q "accepted"; then
  note "CONFIRMED: invalid slices are accepted with near-SIZE_MAX shapes"
  conf=$((conf+1))
else
  note "NOT REPRODUCED (bug appears to be fixed)"; fail=1
fi

note ""
note "=== 3. Wigner3j_direct::calc reads g[-1] at ofs=0 (wigner3j.h:191) ==="
if ASAN_OPTIONS=detect_leaks=0 "$OUT/bugproofs" wigner3j_oob 2>&1 \
    | grep -q "heap-buffer-overflow"; then
  note "CONFIRMED: ASAN reports an out-of-bounds read 8 bytes before the g table"
  conf=$((conf+1))
else
  note "NOT REPRODUCED (bug appears to be fixed)"; fail=1
fi

note ""
note "=== 4. TemplateKernel::transferCoeffs leaves coeff rows uninitialized (gridding_kernel.h:331) ==="
if "$OUT/bugproofs" template_kernel 2>&1 | grep -q "eval=-nan"; then
  note "CONFIRMED: degree < D-1 gives NaN kernel values (uninitialized read)"
  conf=$((conf+1))
else
  note "NOT REPRODUCED (bug appears to be fixed)"; fail=1
fi

note ""
note "=== 5. T_Healpix_Base::get_interpol wrong pixels for phi >= 2pi (healpix_base.cc:1246) ==="
out1=$("$OUT/bugproofs" healpix_interpol 2>&1)
p1=$(printf '%s\n' "$out1" | sed -n 's/phi          : pixels \(.*\)/\1/p')
p2=$(printf '%s\n' "$out1" | sed -n 's/phi + 2\*pi   : pixels \(.*\)/\1/p')
if [ -n "$p1" ] && [ "$p1" != "$p2" ]; then
  note "CONFIRMED: same direction, pixels '$p1' vs '$p2'"
  conf=$((conf+1))
else
  note "NOT REPRODUCED (bug appears to be fixed)"; fail=1
fi

note ""
note "=== 6. wgridder 2D path: (gridding?ms2d_in:ms2d_out)->shape() breaks custom buffers (wgridder_impl.h:719/1631) ==="
if $CXX $CXXFLAGS -c test_cpp/test_compilefail_wgridder_ternary.cc \
     -o "$OUT/wg_control.o" 2>"$OUT/wg_control.log"; then
  note "control (1-D buffer, fixed by b60cbec9): compiles"
else
  note "UNEXPECTED: control failed to compile"; cat "$OUT/wg_control.log"; fail=1
fi
if $CXX $CXXFLAGS -DBUGGY_2D -c test_cpp/test_compilefail_wgridder_ternary.cc \
     -o "$OUT/wg_buggy.o" 2>"$OUT/wg_buggy.log" \
   && [ ! -s "$OUT/wg_buggy.log" ]; then
  note "NOT REPRODUCED (2-D custom buffers compile now)"; fail=1
elif grep -q "conditional expression between distinct pointer types" "$OUT/wg_buggy.log"; then
  note "CONFIRMED: 2-D custom buffer fails with 'conditional expression between"
  note "           distinct pointer types' at wgridder_impl.h:719 and :1631"
  conf=$((conf+1))
else
  note "compile failed for a different reason, inspect manually"; fail=1
fi

note ""
note "=== 7. SphereInterpol legacy getPlane/updateAlm overloads are broken (sphere_interpol.h:539/640) ==="
if $CXX $CXXFLAGS -c test_cpp/test_compilefail_sphere_interpol.cc \
     -o "$OUT/si.o" 2>"$OUT/si.log"; then
  note "NOT REPRODUCED (legacy overloads compile now)"; fail=1
elif grep -q "no matching function for call to .*getPlane" "$OUT/si.log" \
   && grep -q "no matching function for call to .*updateAlm" "$OUT/si.log"; then
  note "CONFIRMED: both legacy overloads forward to nonexistent overloads"
  conf=$((conf+1))
else
  note "compile failed for a different reason, inspect manually"; fail=1
fi

note ""
note "=== 8. Python-level bugs (needs installed ducc0) ==="
if python3 -c "import ducc0" >/dev/null 2>&1; then
  python3 -m pytest python/test/test_bug_hunt.py -q -rx 2>&1 | tail -n 20
else
  note "ducc0 not installed, skipping (pip install . && re-run)"
fi

note ""
if [ "$MODE" = "fail-on-confirmed" ]; then
  if [ "$conf" -gt 0 ]; then
    note "$conf of 7 bug(s) still reproducible - failing."
    exit 1
  fi
  note "No bug reproducible anymore - all fixed."
  exit 0
fi
if [ "$fail" -eq 0 ]; then
  note "All C++ bug proofs reproduced."
else
  note "At least one bug proof did NOT reproduce - check above."
fi
exit $fail
