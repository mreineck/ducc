#ifdef DUCC0_USE_NANOBIND
#include <nanobind/nanobind.h>
namespace py = nanobind;
#else
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif

#include "multiarch.h"

namespace {

void add_cpu_info(py::module_ &m, bool multiarch, int usable_level,
                  int max_level, int selected_level)
  {
  m.attr("misc").attr("cpu_info") = py::cpp_function(
    [multiarch, usable_level, max_level, selected_level]()
    {
    py::dict result;
    result["architecture"] = ducc0_multiarch::architecture_name();
    result["multiarch"] = multiarch;
    if (multiarch)
      {
      py::list compiled;
      for (int level : ducc0_multiarch::compiled_levels())
        compiled.append(ducc0_multiarch::level_name(level));
      result["compiled_levels"] = compiled;

      py::list available;
      for (int level : ducc0_multiarch::available_levels(
             ducc0_multiarch::ducc_compiled_psabi_mask, usable_level))
        available.append(ducc0_multiarch::level_name(level));
      result["available_levels"] = available;
      result["max_level"] = ducc0_multiarch::level_name(max_level);
      result["selected_level"] = ducc0_multiarch::level_name(selected_level);
      }
    return result;
    });
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
  const int usable_level = ducc0_multiarch::usable_psabi_level();
  const int max_level = ducc0_multiarch::max_psabi_level();
  const int selected_level = ducc0_multiarch::select_psabi_level(
    usable_level, max_level, ducc0_multiarch::ducc_compiled_psabi_mask);
  if (selected_level >= 4) ducc0_v4::add_ducc0(m);
  else if (selected_level >= 3) ducc0_v3::add_ducc0(m);
  else ducc0_v1::add_ducc0(m);
  add_cpu_info(m, true, usable_level, max_level, selected_level);
#else
  ducc0::add_ducc0(m);
  add_cpu_info(m, false, 0, 0, 0);
#endif
  }
