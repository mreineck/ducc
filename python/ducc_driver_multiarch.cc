#ifdef DUCC0_USE_NANOBIND
#include <nanobind/nanobind.h>
#else
#include <pybind11/pybind11.h>
#endif
#include <string>
#include <cstdlib>

using namespace std;

#ifdef DUCC0_USE_NANOBIND
namespace py = nanobind;
#else
namespace py = pybind11;
#endif

namespace ducc0 { void add_ducc0(py::module_ &m); }
namespace ducc0_v3 { void add_ducc0(py::module_ &m); }
namespace ducc0_v4 { void add_ducc0(py::module_ &m); }

#ifdef DUCC0_USE_NANOBIND
NB_MODULE(PKGNAME, m)
#else
PYBIND11_MODULE(PKGNAME, m, py::mod_gil_not_used())
#endif
  {
#define DUCC0_XSTRINGIFY(s) DUCC0_STRINGIFY(s)
#define DUCC0_STRINGIFY(s) #s
#if defined (DUCC0_TARGET)
  MR_assert(cpu_supports(DUCC0_XSTRINGIFY(DUCC0_TARGET)),
    "required CPU feature not supported by this CPU");
#endif
  m.attr("__version__") = DUCC0_XSTRINGIFY(PKGVERSION);
#undef DUCC0_STRINGIFY
#undef DUCC0_XSTRINGIFY
#ifdef DUCC0_USE_NANOBIND
  m.attr("__wrapper__") = "nanobind";
#else
  m.attr("__wrapper__") = "pybind11";
#endif

  const auto *evar=getenv("DUCC0_PSABI");
  if (evar!=nullptr)
    {
    if (string(evar) == "v4") return ducc0_v4::add_ducc0(m);
    if (string(evar) == "v3") return ducc0_v3::add_ducc0(m);
    }
  ducc0::add_ducc0(m);
  }
