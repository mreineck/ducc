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
        shape_in2 = ctx.avals_in[0].shape
        dtype_in = _dtype_res(state, not adjoint)
        dtype_in2 =ctx.avals_in[0].dtype
        shape_out = _shape_res(state, adjoint)
        shape_out2 = ctx.avals_out[0].shape
        dtype_out = _dtype_res(state, adjoint)
        dtype_out2 =ctx.avals_out[0].dtype
        if (shape_in, dtype_in, shape_out, dtype_out) != (shape_in2, dtype_in2, shape_out2, dtype_out2):
            raise RuntimeError("bad input or output arrays")
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
        return _get_prim(adjoint).bind(x, stateid=id(state))
        
    
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

    def fht_operator(shape, dtype, axes, nthreads):
        def fhtfunc(inp, out, adjoint, state):
            ducc0.fft.genuine_fht(inp,
                                  out=out,
                                  axes=state["axes"],
                                  nthreads=state["nthreads"])

        shape = tuple(shape)
        dtype = np.dtype(dtype)
        return make_linop(
            func=fhtfunc,
            axes=tuple(axes),
            nthreads=int(nthreads),
            shape_in=shape,
            shape_out=shape,
            dtype_in=dtype,
            dtype_out=dtype)
   
    def sht2d_operator(lmax, mmax, ntheta, nphi, geometry, spin, dtype, nthreads):
        def sht2dfunc(inp, out, adjoint, state):
            if adjoint:
                ducc0.sht.adjoint_synthesis_2d(
                    lmax=state["lmax"],
                    mmax=state["mmax"],
                    spin=state["spin"],
                    map=inp,
                    alm=out,
                    nthreads=state["nthreads"],
                    geometry=state["geometry"])
            else:
                ducc0.sht.synthesis_2d(
                    lmax=state["lmax"],
                    mmax=state["mmax"],
                    spin=state["spin"],
                    map=out,
                    alm=inp,
                    nthreads=state["nthreads"],
                    geometry=state["geometry"])

        lmax = int(lmax)
        mmax = int(mmax)
        spin = int(spin)
        nalm = ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)
        ncomp = 1 if spin == 0 else 2
        dtype = np.dtype(dtype)
    
        tdict = { np.dtype(np.float32) : np.dtype(np.complex64),
                  np.dtype(np.float64) : np.dtype(np.complex128)}
        return make_linop(
            func=sht2dfunc,
            shape_in=(ncomp, nalm),
            shape_out=(ncomp, ntheta, nphi),
            dtype_in=tdict[dtype],
            dtype_out=dtype,
            lmax=lmax,
            mmax=mmax,
            spin=spin,
            geometry=str(geometry),
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
    from jax import jit
    shape, axes = shape_axes
    myop = fht_operator(shape=shape, dtype=dtype, axes=axes, nthreads=nthreads)
    op = jit(myop)
    op_adj = jit(myop.adjoint)
    rng = np.random.default_rng(42)
    a = (rng.random(shape)-0.5+1000).astype(dtype)
    b1 = np.array(op(a)[0])
    b2 = ducc0.fft.genuine_fht(a, axes=axes, nthreads=nthreads)
    _assert_close(b1, b2, epsilon=1e-6 if dtype==np.float32 else 1e-14)
    b3 = np.array(op_adj(a)[0])
    _assert_close(b1, b3, epsilon=1e-6 if dtype==np.float32 else 1e-14)

    from jax.test_util import check_grads
    # this seems to work for any order
    check_grads(op, (a,), order=1, modes=("fwd",), eps=1.)
    check_grads(op_adj, (a,), order=1, modes=("fwd",), eps=1.)
    # this works for order=1, but not for higher ones
    check_grads(op, (a,), order=1, modes=("rev",), eps=1.)
    check_grads(op_adj, (a,), order=1, modes=("rev",), eps=1.)

def nalm(lmax, mmax):
    return ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)

def random_alm(lmax, mmax, spin, ncomp, rng):
    res = rng.uniform(-1., 1., (ncomp, nalm(lmax, mmax))) \
     + 1j*rng.uniform(-1., 1., (ncomp, nalm(lmax, mmax)))
    # make a_lm with m==0 real-valued
    res[:, 0:lmax+1].imag = 0.
    ofs=0
    for s in range(spin):
        res[:, ofs:ofs+spin-s] = 0.
        ofs += lmax+1-s
    return res

@pmp("lmmax", ((10,10), (20, 5)))
@pmp("geometry", ("GL", "F1"))
@pmp("ntheta", (20,))
@pmp("nphi", (30,))
@pmp("spin", (0, 2))
@pmp("dtype", (np.float32, np.float64))
@pmp("nthreads", (1, 2))
def test_sht(lmmax, geometry, ntheta, nphi, spin, dtype, nthreads):
    if not have_jax:
        pytest.skip()
    from jax import jit
    lmax, mmax = lmmax
    ncomp = 1 if spin==0 else 2
    myop = sht2d_operator(lmax=lmax, mmax=mmax, ntheta=ntheta, nphi=nphi, geometry=geometry, spin=spin, dtype=dtype, nthreads=nthreads)
    op = jit(myop)
    op_adj = jit(myop.adjoint)
    rng = np.random.default_rng(42)
    tdict = { np.dtype(np.float32) : np.dtype(np.complex64),
              np.dtype(np.float64) : np.dtype(np.complex128)}
    alm0 = random_alm(lmax, mmax, spin, ncomp, rng).astype(tdict[np.dtype(dtype)])

    map1 = np.array(op(alm0)[0])
    map2 = ducc0.sht.synthesis_2d(alm=alm0, lmax=lmax, mmax=mmax, spin=spin, geometry=geometry, ntheta=ntheta, nphi=nphi, nthreads=nthreads)
    _assert_close(map1, map2, epsilon=1e-6 if dtype==np.float32 else 1e-14)

    map0 = (rng.random((ncomp, ntheta, nphi))-0.5).astype(dtype)
    alm1 = np.array(op_adj(map0)[0])
    alm2 = ducc0.sht.adjoint_synthesis_2d(map=map0, lmax=lmax, mmax=mmax, spin=spin, geometry=geometry, nthreads=nthreads)
    _assert_close(alm1, alm2, epsilon=1e-6 if dtype==np.float32 else 1e-14)

    from jax.test_util import check_grads
    # this seems to work for any order
    check_grads(op, (alm0,), order=1, modes=("fwd",), eps=1.)
    check_grads(op_adj, (map0,), order=1, modes=("fwd",), eps=1.)
    # this doesn"t seem to work at all
#    check_grads(op, (alm0,), order=1, modes=("rev",), eps=1.)
#    check_grads(op_adj, (map0,), order=1, modes=("rev",), eps=1.)

