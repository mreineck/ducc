// IMPORTANT: always include "pybind_utils.h" first, even before standard headers
#include "ducc0/bindings/pybind_utils.h"

namespace DUCC0_NAMESPACE {

namespace detail_pymodule_fft { void add_fft(py::module_ &m); }

void add_ducc0(py::module_ &m)
  {
  detail_pymodule_fft::add_fft(m);
  m.def_submodule("misc");
  }

}
