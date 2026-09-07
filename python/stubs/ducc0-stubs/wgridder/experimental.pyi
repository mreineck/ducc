# mypy: disable-error-code=overload-overlap
from typing import Optional, overload

from ducc0._typing import Complex32Array, Complex64Array, Float32Array, Float64Array, UInt64Array, UInt8Array
from ducc0.wgridder import dirty2vis as dirty2vis, vis2dirty as vis2dirty

@overload
def vis2dirty_bda(*, uvw: Float64Array, freqlist_id: UInt64Array, freqlist_nfreqs: UInt64Array, freqlist_freqs: Float64Array, vis: Complex32Array, wgt: Optional[Float32Array] = ..., npix_x: int = ..., npix_y: int = ..., pixsize_x: float, pixsize_y: float, epsilon: float, do_wgridding: bool = ..., nthreads: int = ..., verbosity: int = ..., mask: Optional[UInt8Array] = ..., flip_u: bool = ..., flip_v: bool = ..., flip_w: bool = ..., divide_by_n: bool = ..., dirty: Optional[Float32Array] = ..., sigma_min: float = ..., sigma_max: float = ..., center_x: float = ..., center_y: float = ..., allow_nshift: bool = ..., double_precision_accumulation: bool = ...) -> Float32Array: ...
@overload
def vis2dirty_bda(*, uvw: Float64Array, freqlist_id: UInt64Array, freqlist_nfreqs: UInt64Array, freqlist_freqs: Float64Array, vis: Complex64Array, wgt: Optional[Float64Array] = ..., npix_x: int = ..., npix_y: int = ..., pixsize_x: float, pixsize_y: float, epsilon: float, do_wgridding: bool = ..., nthreads: int = ..., verbosity: int = ..., mask: Optional[UInt8Array] = ..., flip_u: bool = ..., flip_v: bool = ..., flip_w: bool = ..., divide_by_n: bool = ..., dirty: Optional[Float64Array] = ..., sigma_min: float = ..., sigma_max: float = ..., center_x: float = ..., center_y: float = ..., allow_nshift: bool = ..., double_precision_accumulation: bool = ...) -> Float64Array: ...

@overload
def dirty2vis_bda(*, uvw: Float64Array, freqlist_id: UInt64Array, freqlist_nfreqs: UInt64Array, freqlist_freqs: Float64Array, dirty: Float32Array, wgt: Optional[Float32Array] = ..., pixsize_x: float, pixsize_y: float, epsilon: float, do_wgridding: bool = ..., nthreads: int = ..., verbosity: int = ..., mask: Optional[UInt8Array] = ..., flip_u: bool = ..., flip_v: bool = ..., flip_w: bool = ..., divide_by_n: bool = ..., vis: Optional[Complex32Array] = ..., sigma_min: float = ..., sigma_max: float = ..., center_x: float = ..., center_y: float = ..., allow_nshift: bool = ...) -> Complex32Array: ...
@overload
def dirty2vis_bda(*, uvw: Float64Array, freqlist_id: UInt64Array, freqlist_nfreqs: UInt64Array, freqlist_freqs: Float64Array, dirty: Float64Array, wgt: Optional[Float64Array] = ..., pixsize_x: float, pixsize_y: float, epsilon: float, do_wgridding: bool = ..., nthreads: int = ..., verbosity: int = ..., mask: Optional[UInt8Array] = ..., flip_u: bool = ..., flip_v: bool = ..., flip_w: bool = ..., divide_by_n: bool = ..., vis: Optional[Complex64Array] = ..., sigma_min: float = ..., sigma_max: float = ..., center_x: float = ..., center_y: float = ..., allow_nshift: bool = ...) -> Complex64Array: ...
