#!/usr/bin/env bash
# Builds and runs the focused C++ regression and selection-policy tests.
set -euo pipefail
cd "$(dirname "$0")/.."

CXX="${CXX:-g++}"
CXXFLAGS=(-std=c++17 -O1 -g -Isrc -Ipython -DDUCC0_NAMESPACE=ducc0)
SOURCES=(
  src/ducc0/healpix/healpix_base.cc
  src/ducc0/healpix/healpix_tables.cc
  src/ducc0/infra/mav.cc
  src/ducc0/infra/threading.cc
  src/ducc0/math/gl_integrator.cc
  src/ducc0/math/space_filling.cc
  src/ducc0/math/pointing.cc
  src/ducc0/math/geom_utils.cc
  src/ducc0/infra/string_utils.cc
  src/ducc0/infra/system.cc
)
OUT="$(mktemp -d "${TMPDIR:-/tmp}/ducc0-cpp-tests.XXXXXX")"
trap 'rm -rf "$OUT"' EXIT

echo "Building regression tests ..."
"$CXX" "${CXXFLAGS[@]}" -fsanitize=address -fno-omit-frame-pointer \
  -o "$OUT/test_regressions" test/test_regressions.cc "${SOURCES[@]}" -pthread

for name in swap_axes slice_wraparound wigner3j_oob template_kernel healpix_interpol; do
  echo "Running $name ..."
  ASAN_OPTIONS=detect_leaks=0 "$OUT/test_regressions" "$name"
done

echo "Compiling C++ API checks ..."
"$CXX" "${CXXFLAGS[@]}" -c test/test_compile_api.cc -o "$OUT/test_compile_api.o"
