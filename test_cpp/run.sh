#!/usr/bin/env bash
# Builds and runs the C++ unit tests collected in test_cpp/.
#
# Each test fails while its bug is present and passes once the bug is
# fixed, then remains as a regression test.
# The swap_axes and wigner3j tests detect out-of-bounds accesses via
# AddressSanitizer; the test binary is therefore always built with
# -fsanitize=address.
#
# Exit status is non-zero if any test fails.
set -u
cd "$(dirname "$0")/.."

CXX="${CXX:-g++}"
CXXFLAGS="-std=c++17 -O1 -g -Isrc"
SRCS="src/ducc0/healpix/healpix_base.cc src/ducc0/healpix/healpix_tables.cc \
      src/ducc0/infra/mav.cc src/ducc0/infra/threading.cc \
      src/ducc0/math/gl_integrator.cc src/ducc0/math/space_filling.cc \
      src/ducc0/math/pointing.cc src/ducc0/math/geom_utils.cc \
      src/ducc0/infra/string_utils.cc src/ducc0/infra/system.cc"
OUT=/tmp/ducc0_cpptests
mkdir -p "$OUT"

note() { printf '%s\n' "$*"; }
npass=0
nfail=0

report() {  # $1: test name, $2: exit status, $3: log file
  if [ "$2" -eq 0 ]; then
    note "PASS  $1"
    npass=$((npass+1))
  else
    note "FAIL  $1"
    nfail=$((nfail+1))
    sed 's/^/      /' "$3"
  fi
  }

note "building test binary ..."
$CXX $CXXFLAGS -fsanitize=address -o "$OUT/bugtests" \
  test_cpp/test_bug_hunt.cc $SRCS -pthread 2>"$OUT/build.log" \
  || { note "BUILD FAILED"; cat "$OUT/build.log"; exit 1; }

for t in swap_axes slice_wraparound wigner3j_oob template_kernel healpix_interpol; do
  ASAN_OPTIONS=detect_leaks=0 "$OUT/bugtests" "$t" >"$OUT/$t.log" 2>&1
  report "$t" $? "$OUT/$t.log"
done

for t in test_wgridder_custom_buffer_1d test_wgridder_custom_buffer_2d \
         test_sphere_interpol_legacy_api; do
  note "compiling $t ..."
  $CXX $CXXFLAGS -c "test_cpp/$t.cc" -o "$OUT/$t.o" 2>"$OUT/$t.log"
  report "$t" $? "$OUT/$t.log"
done

note ""
note "$npass passed, $nfail failed"
[ "$nfail" -eq 0 ]
