// IMPORTANT: always include "pybind_utils.h" first, even before standard headers
#include "ducc0/bindings/pybind_utils.h"

namespace DUCC0_NAMESPACE {

namespace detail_pymodule_fft { void add_fft(py::module_ &m); }
namespace detail_pymodule_misc { void add_misc(py::module_ &m); }

void add_ducc0(py::module_ &m)
  {
  detail_pymodule_fft::add_fft(m);
  detail_pymodule_misc::add_misc(m);
  }

}
