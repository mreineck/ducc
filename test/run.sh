#!/usr/bin/env bash
# Builds and runs the consolidated C++ regression, API, and selection tests.
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
failures=()
passed=0

record() {
  local name="$1"
  shift
  if "$@"; then
    echo "PASS $name"
    passed=$((passed+1))
  else
    echo "FAIL $name"
    failures+=("$name")
  fi
}

echo "Building regression test binary ..."
record "build regression tests" "$CXX" "${CXXFLAGS[@]}" \
  -fsanitize=address -fno-omit-frame-pointer \
  -o "$OUT/test_regressions" test/test_regressions.cc "${SOURCES[@]}" -pthread
if [[ -x "$OUT/test_regressions" ]]; then
  for name in swap_axes slice_wraparound wigner3j_oob template_kernel healpix_interpol; do
    record "regression $name" env ASAN_OPTIONS=detect_leaks=0 \
      "$OUT/test_regressions" "$name"
  done
fi

for source in \
    test/test_compile_api.cc \
    test/test_sphere_interpol_api.cc \
    test/test_wgridder_custom_buffer_1d.cc \
    test/test_wgridder_custom_buffer_2d.cc; do
  object="$OUT/$(basename "${source%.cc}").o"
  echo "Compiling $source ..."
  record "compile $source" "$CXX" "${CXXFLAGS[@]}" -c "$source" -o "$object"
done

record "build multiarch selection test" "$CXX" -std=c++17 -Ipython \
  test/test_multiarch.cc python/multiarch.cc -o "$OUT/test_multiarch"
if [[ -x "$OUT/test_multiarch" ]]; then
  record "multiarch selection policy" "$OUT/test_multiarch"
fi

echo "$passed passed, ${#failures[@]} failed"
if ((${#failures[@]})); then
  printf 'Failed C++ checks:\n'
  printf '  %s\n' "${failures[@]}"
  exit 1
fi
