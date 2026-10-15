#include "ducc0/fft/fftnd_impl.h"

namespace DUCC0_NAMESPACE{
namespace detail_fft {
#define T float
#include "ducc0/fft/fft_inst_inc.h"
#undef T
}
}
