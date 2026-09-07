"""Long-double typing checks for the pybind11 wheel."""

import numpy as np
import numpy.typing as npt
from typing_extensions import assert_type

import ducc0.fft

long_float: npt.NDArray[np.longdouble] = np.empty(4, dtype=np.longdouble)
long_complex: npt.NDArray[np.clongdouble] = np.empty(4, dtype=np.clongdouble)
assert_type(long_float, npt.NDArray[np.longdouble])
assert_type(long_complex, npt.NDArray[np.clongdouble])

assert_type(ducc0.fft.c2c(long_float), npt.NDArray[np.clongdouble])  # pyright: ignore[reportAssertTypeFailure]  # ty: ignore
assert_type(ducc0.fft.c2c(long_complex), npt.NDArray[np.clongdouble])  # pyright: ignore[reportAssertTypeFailure]  # ty: ignore
assert_type(ducc0.fft.r2c(long_float), npt.NDArray[np.clongdouble])  # pyright: ignore[reportAssertTypeFailure]  # ty: ignore
assert_type(ducc0.fft.c2r(long_complex), npt.NDArray[np.longdouble])  # pyright: ignore[reportAssertTypeFailure]  # ty: ignore
assert_type(ducc0.fft.r2r_fftpack(long_float, [0], True, True), npt.NDArray[np.longdouble])  # pyright: ignore[reportAssertTypeFailure]  # ty: ignore
assert_type(ducc0.fft.r2r_fftw(long_float, [0], True), npt.NDArray[np.longdouble])  # pyright: ignore[reportAssertTypeFailure]  # ty: ignore
assert_type(ducc0.fft.separable_hartley(long_float), npt.NDArray[np.longdouble])  # pyright: ignore[reportAssertTypeFailure]  # ty: ignore
assert_type(ducc0.fft.genuine_hartley(long_float), npt.NDArray[np.longdouble])  # pyright: ignore[reportAssertTypeFailure]  # ty: ignore
assert_type(ducc0.fft.separable_fht(long_float), npt.NDArray[np.longdouble])  # pyright: ignore[reportAssertTypeFailure]  # ty: ignore
assert_type(ducc0.fft.genuine_fht(long_float), npt.NDArray[np.longdouble])  # pyright: ignore[reportAssertTypeFailure]  # ty: ignore
assert_type(ducc0.fft.dct(long_float, 2), npt.NDArray[np.longdouble])  # pyright: ignore[reportAssertTypeFailure]  # ty: ignore
assert_type(ducc0.fft.dst(long_float, 2), npt.NDArray[np.longdouble])  # pyright: ignore[reportAssertTypeFailure]  # ty: ignore
assert_type(ducc0.fft.convolve_axis(long_float, long_float, 0, long_float), npt.NDArray[np.longdouble])  # pyright: ignore[reportAssertTypeFailure]  # ty: ignore
assert_type(ducc0.fft.convolve_axis(long_complex, long_complex, 0, long_complex), npt.NDArray[np.clongdouble])  # pyright: ignore[reportAssertTypeFailure]  # ty: ignore
