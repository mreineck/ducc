#!/usr/bin/env bash
# Build and run DUCC's C++ regression, compile-only API, and selector checks.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

CXX="${CXX:-g++}"
CXXFLAGS=(-std=c++17 -O1 -g -Isrc -DDUCC0_NAMESPACE=ducc0)
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
trap 'rm -rf -- "$OUT"' EXIT

failures=0

echo "Building ASan regression test binary with ${CXX} ..."
if "${CXX}" "${CXXFLAGS[@]}" -fsanitize=address -fno-omit-frame-pointer \
    -o "$OUT/test_regressions" test/test_regressions.cc "${SOURCES[@]}" \
    -pthread; then
  echo "PASS build regression test binary"
else
  status=$?
  echo "FAIL build regression test binary" >&2
  exit "$status"
fi

for name in swap_axes slice_wraparound wigner3j_oob template_kernel healpix_interpol; do
  if ASAN_OPTIONS=detect_leaks=0 "$OUT/test_regressions" "$name"; then
    echo "PASS regression $name"
  else
    echo "FAIL regression $name" >&2
    failures=1
  fi
done

echo "Compiling the combined API checks ..."
if "${CXX}" "${CXXFLAGS[@]}" -c test/test_compile_api.cc \
    -o "$OUT/test_compile_api.o"; then
  echo "PASS compile-only API checks"
else
  echo "FAIL compile-only API checks" >&2
  failures=1
fi

echo "Building the multiarch selector test ..."
if "${CXX}" -std=c++17 -Ipython test/test_multiarch.cc python/multiarch.cc \
    -o "$OUT/test_multiarch"; then
  echo "PASS build multiarch selector test"
  if "$OUT/test_multiarch"; then
    echo "PASS multiarch selector"
  else
    echo "FAIL multiarch selector" >&2
    failures=1
  fi
else
  echo "FAIL build multiarch selector test" >&2
  failures=1
fi

if ((failures)); then
  echo "C++ checks failed" >&2
  exit 1
fi
echo "All C++ checks passed"
