/* SPDX-License-Identifier: BSD-3-Clause OR GPL-2.0-or-later */

/*
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its contributors may
  be used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
 *  This code is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This code is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this code; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/* Copyright (C) 2026 Max-Planck-Society, Albert M. W. Yau
   Authors: Martin Reinecke, Albert M. W. Yau */

#ifdef DUCC0_USE_NANOBIND
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
namespace py = nanobind;
#else
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif

#include <stdexcept>
#include <algorithm>

#ifdef DUCC0_MULTIARCH

// Currently we are only supporting multiarch on Linux x86_64.
// Double check that we are on such a system.
#if !(defined(__linux__) && defined(__x86_64__))
#error Multiarch is requested, but not supported on this system yet
#endif

#include <cpuid.h>

namespace ducc0_multiarch {

int configured_psabi_limit()
  {
  const char *value = std::getenv("DUCC0_MAX_PSABI_LEVEL");
  if (value == nullptr) return 4;

  char *end = nullptr;
  errno = 0;
  const long parsed = std::strtol(value, &end, 10);
  while (end != value && *end != '\0'
      && std::isspace(static_cast<unsigned char>(*end)))
    ++end;

  if (end == value || *end != '\0' || errno == ERANGE)
    throw std::runtime_error(
      "invalid DUCC0_MAX_PSABI_LEVEL: expected an integer");
  if (parsed < 1) return 1;
  if (parsed > 4) return 4;
  return static_cast<int>(parsed);
  }

namespace {

class x86_features
  {
  private:
    struct cpuid_result
      { std::uint32_t eax, ebx, ecx, edx; };

    static cpuid_result cpuid(std::uint32_t leaf, std::uint32_t subleaf)
      {
      unsigned int eax, ebx, ecx, edx;
      __cpuid_count(leaf, subleaf, eax, ebx, ecx, edx);
      return {eax, ebx, ecx, edx};
      }

    static bool has_bit(std::uint32_t value, unsigned bit)
      { return (value & (std::uint32_t(1) << bit)) != 0; }

    static std::uint64_t read_xcr(std::uint32_t index)
      {
      std::uint32_t eax, edx;
      __asm__ volatile(".byte 0x0f, 0x01, 0xd0"
        : "=a"(eax), "=d"(edx) : "c"(index));
      return (std::uint64_t(edx)<<32) | eax;
      }
    const cpuid_result leaf0 = cpuid(0, 0);
    const cpuid_result leaf1 = (leaf0.eax>=1) ? cpuid(1, 0) : cpuid_result{0,0,0,0};
    const cpuid_result leaf7 = (leaf0.eax>=7) ? cpuid(7, 0) : cpuid_result{0,0,0,0};
    const std::uint32_t high_leaf = 0x80000000;
    const cpuid_result leaf_high0 = cpuid(high_leaf, 0);
    const cpuid_result leaf_high1 = (leaf_high0.eax>=high_leaf+1)
      ? cpuid(high_leaf+1, 0) : cpuid_result{0,0,0,0};

  public:
    const bool sse3 = has_bit(leaf1.ecx, 0);
    const bool ssse3 = has_bit(leaf1.ecx, 9);
    const bool sse41 = has_bit(leaf1.ecx, 19);
    const bool sse42 = has_bit(leaf1.ecx, 20);
    const bool cx16 = has_bit(leaf1.ecx, 13);
    const bool lahf_sahf = has_bit(leaf_high1.ecx, 0);
    const bool popcnt = has_bit(leaf1.ecx, 23);
    const bool avx = has_bit(leaf1.ecx, 28);
    const bool xsave = has_bit(leaf1.ecx, 26);
    const bool osxsave = has_bit(leaf1.ecx, 27);
    const std::uint64_t xcr0 = (xsave && osxsave) ? read_xcr(0) : 0;
    const bool ymm_state = (xcr0 & 0x6)==0x6;
    const bool avx2 = has_bit(leaf7.ebx, 5);
    const bool bmi1 = has_bit(leaf7.ebx, 3);
    const bool bmi2 = has_bit(leaf7.ebx, 8);
    const bool f16c = has_bit(leaf1.ecx, 29);
    const bool fma = has_bit(leaf1.ecx, 12);
    const bool lzcnt = has_bit(leaf_high1.ecx, 5);
    const bool movbe = has_bit(leaf1.ecx, 22);
    const bool avx512f = has_bit(leaf7.ebx, 16);
    const bool avx512dq = has_bit(leaf7.ebx, 17);
    const bool avx512cd = has_bit(leaf7.ebx, 28);
    const bool avx512bw = has_bit(leaf7.ebx, 30);
    const bool avx512vl = has_bit(leaf7.ebx, 31);
    const bool zmm_state = (xcr0 & 0xe6)==0xe6;

    const bool usable_avx = avx && xsave && osxsave && ymm_state;
    const bool usable_avx2 = usable_avx && avx2;
    const bool usable_avx512 = usable_avx && avx512f && avx512dq
      && avx512cd && avx512bw && avx512vl && zmm_state;

    const bool usable_psabi_v2
      = cx16 && lahf_sahf && popcnt && sse3 && ssse3 && sse41 && sse42;
    const bool usable_psabi_v3
      = usable_psabi_v2 && usable_avx2 && bmi1 && bmi2 && f16c && fma && lzcnt && movbe;
    const bool usable_psabi_v4
      = usable_psabi_v3 && usable_avx512;

    const int psabi_level = usable_psabi_v4 ? 4 : (usable_psabi_v3 ? 3 : (usable_psabi_v2 ? 2 : 1));
  };

} // namespace

size_t selected_psabi_level()
  {
  auto lvl = std::min(x86_features().psabi_level, configured_psabi_limit());
  return (lvl>=4) ? 4 : ((lvl>=3) ? 3 : 1);
  }

} // namespace ducc0_multiarch

#endif

namespace {

void add_psabi_level(py::module_ &m, size_t psabi_level)
  {
  m.attr("misc").attr("psabi_level") = py::cpp_function(
    [psabi_level]() { return psabi_level; },
    R"doc(Return the active PSABI level for the current process.

For non-multiarch builds, this is always 0.
On multiarch x86_64 builds, this can be 1 (up to SSE2), 3 (up to AVX2),
or 4 (up to AVX512).

Returns
-------
int
    the PSABI level)doc");
  }

}

#ifdef DUCC0_MULTIARCH
namespace ducc0_v1 { void add_ducc0(py::module_ &m); }
namespace ducc0_v3 { void add_ducc0(py::module_ &m); }
namespace ducc0_v4 { void add_ducc0(py::module_ &m); }
#else
namespace ducc0 { void add_ducc0(py::module_ &m); }
#endif

#ifdef DUCC0_USE_NANOBIND
NB_MODULE(PKGNAME, m)
#else
PYBIND11_MODULE(PKGNAME, m, py::mod_gil_not_used())
#endif
  {
#define DUCC0_XSTRINGIFY(s) DUCC0_STRINGIFY(s)
#define DUCC0_STRINGIFY(s) #s
  m.attr("__version__") = DUCC0_XSTRINGIFY(PKGVERSION);
#undef DUCC0_STRINGIFY
#undef DUCC0_XSTRINGIFY
#ifdef DUCC0_USE_NANOBIND
  m.attr("__wrapper__") = "nanobind";
#else
  m.attr("__wrapper__") = "pybind11";
#endif

#ifdef DUCC0_MULTIARCH
  const auto psabi_level = ducc0_multiarch::selected_psabi_level();
  switch (psabi_level)
    {
    case 4: ducc0_v4::add_ducc0(m); break;
    case 3: ducc0_v3::add_ducc0(m); break;
    case 1: ducc0_v1::add_ducc0(m); break;
    default: throw std::runtime_error("no compatible DUCC0 multiarch profile");
    }
  add_psabi_level(m, psabi_level);
#else
  ducc0::add_ducc0(m);
  add_psabi_level(m, 0);
#endif
  }
