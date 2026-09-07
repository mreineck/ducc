"""Types installed in wheels built with nanobind.

nanobind does not bind C++ ``long double`` parameters. The
``NoReturn`` aliases below make those overloads uncallable in this wheel while
allowing the public modules to share one canonical stub tree with pybind11.
"""

from typing import NoReturn, Sequence, Union

import numpy as np
from numpy.typing import NDArray

Real32Array = NDArray[np.float32]
Real64Array = NDArray[np.float64]
Float32Array = Real32Array
Float64Array = Real64Array
Complex32Array = NDArray[np.complex64]
Complex64Array = NDArray[np.complex128]
Int32Array = NDArray[np.int32]
Int64Array = NDArray[np.int64]
IntegerArray = Union[Int32Array, Int64Array]
UInt8Array = NDArray[np.uint8]
UInt64Array = NDArray[np.uint64]
BoolArray = NDArray[np.bool_]
LongFloatArray = NoReturn
LongComplexArray = NoReturn
RealArray = Union[Real32Array, Real64Array]
ComplexArray = Union[Complex32Array, Complex64Array]
NumericArray = Union[RealArray, ComplexArray]
Shape = Sequence[int]
