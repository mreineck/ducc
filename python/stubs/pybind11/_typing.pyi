"""Types installed in wheels built with pybind11."""

from typing import Sequence, Union

import numpy as np
from numpy.typing import NDArray

Real32Array = NDArray[np.float32]
Real64Array = NDArray[np.float64]
Float32Array = Real32Array
Float64Array = Real64Array
Complex32Array = NDArray[np.complex64]
Complex64Array = NDArray[np.complex128]
LongFloatArray = NDArray[np.longdouble]
LongComplexArray = NDArray[np.clongdouble]
Int32Array = NDArray[np.int32]
Int64Array = NDArray[np.int64]
IntegerArray = Union[Int32Array, Int64Array]
UInt8Array = NDArray[np.uint8]
UInt64Array = NDArray[np.uint64]
BoolArray = NDArray[np.bool_]
RealArray = Union[Real32Array, Real64Array]
ComplexArray = Union[Complex32Array, Complex64Array]
NumericArray = Union[RealArray, ComplexArray, LongFloatArray, LongComplexArray]
Shape = Sequence[int]
