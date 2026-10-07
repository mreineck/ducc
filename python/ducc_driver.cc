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
  const auto cpu_features = ducc0_multiarch::detect_cpu_features();
  const auto state = ducc0_multiarch::current_profile_state(
    cpu_features.host_psabi_level);
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
  add_cpu_info(m, true, state, cpu_features.names);
#else
  ducc0::add_ducc0(m);
  const auto cpu_features = ducc0_multiarch::detect_cpu_features();
  add_cpu_info(m, false, {}, cpu_features.names);
#endif
  }
