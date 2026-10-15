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

#include "multiarch.h"

namespace {

void add_cpu_info(py::module_ &m, bool multiarch,
                  ducc0_multiarch::profile_state state,
                  const std::vector<std::string> &features)
  {
  m.attr("misc").attr("cpu_info") = py::cpp_function(
    [multiarch, state, features]()
    {
    py::dict result;
    result["architecture"] = ducc0_multiarch::architecture_name();
    py::list feature_names;
    for (const auto &feature : features) feature_names.append(feature);
    result["features"] = feature_names;
    result["multiarch"] = multiarch;
    if (multiarch)
      {
      py::list compiled;
      for (int profile : ducc0_multiarch::compiled_profiles())
        compiled.append(ducc0_multiarch::profile_name(profile));
      result["compiled_profiles"] = compiled;

      py::list available;
      for (int profile : ducc0_multiarch::available_profiles(
             ducc0_multiarch::ducc_compiled_profiles_mask,
             state.host_psabi_level))
        available.append(ducc0_multiarch::profile_name(profile));
      result["available_profiles"] = available;
      result["configured_limit"] =
        ducc0_multiarch::profile_name(state.configured_limit);
      result["active_profile"] =
        ducc0_multiarch::profile_name(state.active_profile);
      }
    return result;
    },
    R"doc(Return information about the current CPU and DUCC dispatch configuration.

The ``features`` entry lists SIMD/vector features recognized by DUCC that are
available to the current process.

For multiarch builds, the result also reports the compiled and available
x86-64 profiles, the configured profile limit, and the profile selected for
the current process.

Returns
-------
dict
    CPU and dispatch information.)doc");
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
  const auto cpu = ducc0_multiarch::detect_cpu_capabilities();
  const auto state = ducc0_multiarch::current_profile_state(
    cpu.x86_psabi_level);
  switch (state.active_profile)
    {
    case 4:
      ducc0_v4::add_ducc0(m);
      break;
    case 3:
      ducc0_v3::add_ducc0(m);
      break;
    case 1:
      ducc0_v1::add_ducc0(m);
      break;
    default:
      throw std::runtime_error("no compatible DUCC0 multiarch profile");
    }
  add_cpu_info(m, true, state, cpu.features);
#else
  ducc0::add_ducc0(m);
  const auto cpu = ducc0_multiarch::detect_cpu_capabilities();
  add_cpu_info(m, false, {}, cpu.features);
#endif
  }
