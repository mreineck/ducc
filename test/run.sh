#!/usr/bin/env bash
# Build and run DUCC's C++ regression, compile-only, and selector checks.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

CXX="${CXX:-g++}"
CXXFLAGS=(-std=c++23 -O1 -g -Isrc -DDUCC0_NAMESPACE=ducc0
          -fsanitize=address,undefined -fno-omit-frame-pointer)
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
)
OUT="$(mktemp -d "${TMPDIR:-/tmp}/ducc0-cpp-tests.XXXXXX")"
trap 'rm -rf -- "$OUT"' EXIT

failures=0

echo "Building ASan/UBSan regression test binary with ${CXX} ..."
if "${CXX}" "${CXXFLAGS[@]}" \
    -o "$OUT/test_regressions" test/test_regressions.cc "${SOURCES[@]}" \
    -pthread; then
  echo "PASS build regression test binary"
else
  status=$?
  echo "FAIL build regression test binary" >&2
  exit "$status"
fi

run_regression()
  {
  local name=$1
  local limiter=()
  if [[ "$name" == "callback_exception_wait" ]]; then
    if command -v timeout >/dev/null 2>&1; then
      limiter=(timeout 60)
    elif command -v gtimeout >/dev/null 2>&1; then
      limiter=(gtimeout 60)
    else
      echo "WARNING: no external timeout utility; running $name without a timeout" >&2
    fi
  fi
  ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 \
    "${limiter[@]}" "$OUT/test_regressions" "$name"
  }

if ! test_names=$("$OUT/test_regressions" --list); then
  echo "FAIL listing regression tests" >&2
  exit 1
fi
if [[ -z "$test_names" ]]; then
  echo "FAIL regression registry is empty" >&2
  exit 1
fi
while IFS= read -r name; do
  [[ -z "$name" ]] && continue
  if run_regression "$name"; then
    echo "PASS regression $name"
  else
    echo "FAIL regression $name" >&2
    failures=1
  fi
done <<< "$test_names"

echo "Compiling compile-only checks ..."
if "${CXX}" "${CXXFLAGS[@]}" -c test/test_compile.cc \
    -o "$OUT/test_compile.o"; then
  echo "PASS compile-only checks"
else
  echo "FAIL compile-only checks" >&2
  failures=1
fi

if ((failures)); then
  echo "C++ checks failed" >&2
  exit 1
fi
echo "All C++ checks passed"
