try:
    import jax
    have_jax = True
except ImportError:
    have_jax = False

from functools import partial
import numpy as np
import ducc0
    
import pytest
from numpy.testing import assert_, assert_allclose
pmp = pytest.mark.parametrize

if have_jax:

    from jax.interpreters import ad, mlir    
    
    for _name, _value in ducc0.jax.registrations().items():
        jax.lib.xla_client.register_custom_call_target(_name, _value, platform="cpu")
    
    def _from_id(objectid):
        import ctypes
        return ctypes.cast(objectid, ctypes.py_object).value
    
    def _get_prim(adjoint):
        return _prim_adjoint if adjoint else _prim_forward
    
    def _shape_res(state, adjoint):
        return state["shape_in" if adjoint else "shape_out"] 
    
    def _dtype_res(state, adjoint):
        return state["dtype_in" if adjoint else "dtype_out"] 
    
    def _exec_abstract(x, stateid, adjoint):
        state = _from_id(stateid)
        return (jax.core.ShapedArray(_shape_res(state, adjoint),
                                     _dtype_res(state, adjoint)), )
    
    def _lowering(ctx, x, *, platform="cpu", stateid, adjoint):
        import jaxlib
        state = _from_id(stateid)
        shape_in = _shape_res(state, not adjoint)
        dtype_in = _dtype_res(state, not adjoint)
        shape_out = _shape_res(state, adjoint)
        dtype_out = _dtype_res(state, adjoint)
        jaxtype_in = mlir.ir.RankedTensorType(x.type)
    
        dtype_out_mlir = mlir.dtype_to_ir_type(dtype_out)
        jaxtype_out = mlir.ir.RankedTensorType.get(shape_out, dtype_out_mlir)
        layout_in = tuple(range(len(shape_in) - 1, -1, -1))
        layout_out = tuple(range(len(shape_out) - 1, -1, -1))
    
        if platform == "cpu":
            return jaxlib.hlo_helpers.custom_call(
                platform + "_linop" + ("_adjoint" if adjoint else "_forward"),
                result_types=[jaxtype_out, ],
                operands=[mlir.ir_constant(stateid),  x],
                operand_layouts=[(), layout_in],
                result_layouts=[layout_out, ]
            ).results
        elif platform == "gpu":
            raise ValueError("No GPU support")
        raise ValueError(
            "Unsupported platform; this must be either 'cpu' or 'gpu'"
        )
    
    def _jvp(args, tangents, *, stateid, adjoint):
        prim = _get_prim(adjoint)
        res = prim.bind(args[0], stateid=stateid)
        return (res, jax.lax.zeros_like_array(res) if type(tangents[0]) is ad.Zero
                                                   else prim.bind(tangents[0], stateid=stateid))
    
    def _transpose(cotangents, args, *, stateid, adjoint):
        return (_get_prim(not adjoint).bind(cotangents[0], stateid=stateid), )
        
    def _batch(args, axes, *, stateid, adjoint):
        raise NotImplementedError("FIXME")
    
    def _make_prims():
        name = "ducc_linop_prim"
        global _prim_forward, _prim_adjoint
        _prim_forward = jax.core.Primitive(name+"_forward")
        _prim_adjoint = jax.core.Primitive(name+"_adjoint")
    
        for adjoint in (False, True):
            prim = _get_prim(adjoint)
            prim.multiple_results = True
            prim.def_impl(partial(jax.interpreters.xla.apply_primitive, prim))
            prim.def_abstract_eval(partial(_exec_abstract,adjoint=adjoint))
        
            for platform in ["cpu", "gpu"]:
                mlir.register_lowering(
                    prim,
                    partial(_lowering, platform=platform, adjoint=adjoint),
                    platform=platform)
    
            ad.primitive_jvps[prim] = partial(_jvp, adjoint=adjoint)
            ad.primitive_transposes[prim] = partial(_transpose, adjoint=adjoint)
            jax.interpreters.batching.primitive_batchers[prim] = partial(_batch, adjoint=adjoint)
    
    _make_prims()
    
    def _call(x, state, adjoint):
        return _get_prim(adjoint).bind(x, stateid=id(state))[0]
        
    
    class _Linop:
        @property
        def shape_in(self):
            return _shape_res(self._state, not self._adjoint)
    
        @property
        def shape_out(self):
            return _shape_res(self._state, self._adjoint)
    
        @property
        def dtype_in(self):
            return _dtype_res(self._state, not self._adjoint)
    
        @property
        def dtype_out(self):
            return _dtype_res(self._state, self._adjoint)
    
        @property
        def adjoint(self):
            return _Linop(self._state, not self._adjoint)
    
        def __init__(self, state, adjoint=False):
            self._state = state
            self._adjoint = adjoint
    
        def __call__(self, x):
            return _call(x, self._state, self._adjoint)
    
    
    def make_linop(**kwargs):
        import copy
        # somehow make sure that kwargs_clean only contains deep copies of
        # everything in kwargs that are not accessible from anywhere else.
        kwargs_clean = copy.deepcopy(kwargs)  # FIXME TODO
        return _Linop(kwargs_clean)
    
    
    def fht_operator(shape, dtype, axes, fct, nthreads):
        shape = tuple(shape)
        dtype = np.dtype(dtype)
        return make_linop(
            job="FHT",
            axes=tuple(axes),
            fct=float(fct),
            nthreads=int(nthreads),
            shape_in=shape,
            shape_out=shape,
            dtype_in=dtype,
            dtype_out=dtype)
          
    def sht2d_operator(lmax, mmax, ntheta, nphi, geometry, spin, dtype, nthreads):
        lmax = int(lmax)
        mmax = int(mmax)
        spin = int(spin)
        nalm = ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)
        ncomp = 1 if spin == 0 else 2
        dtype = np.dtype(dtype)
    
        tdict = { np.dtype(np.float32) : np.dtype(np.complex64),
                  np.dtype(np.float64) : np.dtype(np.complex128)}
        return make_linop(
            job="SHT2D",
            shape_in=(ncomp, nalm),
            shape_out=(ncomp, ntheta, nphi),
            dtype_in=tdict[dtype],
            dtype_out=dtype,
            lmax=lmax,
            mmax=mmax,
            spin=spin,
            geometry=str(geometry),
            nthreads=int(nthreads))
    
    def sht_healpix_operator(lmax, mmax, nside, spin, dtype, nthreads):
        lmax = int(lmax)
        mmax = int(mmax)
        nside = int(nside)
        spin = int(spin)
        nalm = ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)
        ncomp = 1 if spin == 0 else 2
        dtype = np.dtype(dtype)
    
        tdict = { np.dtype(np.float32) : np.dtype(np.complex64),
                  np.dtype(np.float64) : np.dtype(np.complex128)}
        return make_linop(
            job="SHT_Healpix",
            shape_in=(ncomp, nalm),
            shape_out=(ncomp, 12*nside**2),
            dtype_in=tdict[dtype],
            dtype_out=dtype,
            lmax=lmax,
            mmax=mmax,
            nside=nside,
            spin=spin,
            nthreads=int(nthreads))
    
    
    from jax import config
    config.update("jax_enable_x64", True)
   
def _assert_close(a, b, epsilon):
    assert_allclose(ducc0.misc.l2error(a, b), 0, atol=epsilon)

@pmp("shape_axes", (((100,),(0,)), ((10,17), (0,1)), ((10,17,3), (1,))))
@pmp("dtype", (np.float32, np.float64))
@pmp("nthreads", (1, 2))
def test_fht(shape_axes, dtype, nthreads):
    if not have_jax:
        pytest.skip()
    shape, axes = shape_axes
    myop = fht_operator(shape=shape, dtype=dtype, axes=axes, fct=1., nthreads=nthreads)
    rng = np.random.default_rng(42)
    a = (rng.random(shape)-0.5).astype(dtype)
    b1 = np.array(myop(a))
    b2 = ducc0.fft.genuine_fht(a, axes=axes, nthreads=nthreads)
    _assert_close(b1, b2, epsilon=1e-6 if dtype==np.float32 else 1e-14)
    b3 = np.array(myop.adjoint(a))
    _assert_close(b1, b3, epsilon=1e-6 if dtype==np.float32 else 1e-14)

