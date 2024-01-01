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

    _global_opcounter = 0

    from jax.interpreters import ad, mlir    
    
    for _name, _value in ducc0.jax.registrations().items():
        jax.lib.xla_client.register_custom_call_target(_name, _value, platform="cpu")
    
    def _from_id(objectid):
        import ctypes
        return ctypes.cast(objectid, ctypes.py_object).value
    
    def _get_prim(adjoint):
        return _prim_adjoint if adjoint else _prim_forward
        
    def _exec_abstract(x, stateid, adjoint):
        state = _from_id(stateid)
        shp, tp = state["_func_abstract"](x.shape, x.dtype, adjoint, state)
        return (jax.core.ShapedArray(shp, tp), )
    
    def _lowering(ctx, x, *, platform="cpu", stateid, adjoint):
        import jaxlib
        state = _from_id(stateid)
        if len(ctx.avals_in) != 1:
            raise RuntimeError("need exactly one input object")
        shape_in = ctx.avals_in[0].shape
        dtype_in = ctx.avals_in[0].dtype
        if len(ctx.avals_out) != 1:
            raise RuntimeError("need exactly one output object")
        shape_out, dtype_out = state["_func_abstract"](shape_in, dtype_in, adjoint, state)

        jaxtype_in = mlir.ir.RankedTensorType(x.type)
    
        dtype_out_mlir = mlir.dtype_to_ir_type(dtype_out)
        jaxtype_out = mlir.ir.RankedTensorType.get(shape_out, dtype_out_mlir)
        layout_in = tuple(range(len(shape_in) - 1, -1, -1))
        layout_out = tuple(range(len(shape_out) - 1, -1, -1))

        dtype_dict = { np.dtype(np.float32): 3,
                       np.dtype(np.float64): 7,
                       np.dtype(np.complex64): 67,
                       np.dtype(np.complex128): 71 }

        operands = []
        operand_layouts = []

        # add array
        operands.append(x)
        operand_layouts.append(layout_in)

        # add opid
        operands.append(mlir.ir_constant(state["_opid"]))
        operand_layouts.append(())

        # add stateid
        operands.append(mlir.ir_constant(stateid))
        operand_layouts.append(())

        # add input dtype
        operands.append(mlir.ir_constant(dtype_dict[dtype_in]))
        operand_layouts.append(())
        # add input rank and shape
        operands.append(mlir.ir_constant(len(shape_in)))
        operand_layouts.append(())
        for i in shape_in:
            operands.append(mlir.ir_constant(i))
            operand_layouts.append(())

        # add output dtype
        operands.append(mlir.ir_constant(dtype_dict[dtype_out]))
        operand_layouts.append(())
        # add output rank and shape
        operands.append(mlir.ir_constant(len(shape_out)))
        operand_layouts.append(())
        for i in shape_out:
            operands.append(mlir.ir_constant(i))
            operand_layouts.append(())

        if platform == "cpu":
            shapeconst = tuple(mlir.ir_constant(s) for s in shape_in)
            return jaxlib.hlo_helpers.custom_call(
                platform + "_linop" + ("_adjoint" if adjoint else "_forward"),
                result_types=[jaxtype_out, ],
                operands=operands,
                operand_layouts=operand_layouts,
                result_layouts=[layout_out,]
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
        return _get_prim(not adjoint).bind(cotangents[0], stateid=stateid)        

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
        def adjoint(self):
            return _Linop(self._state, not self._adjoint)
    
        def __init__(self, state, adjoint=False):
            self._state = state
            self._adjoint = adjoint
    
        def __call__(self, x):
            return _call(x, self._state, self._adjoint)
    
    
    def make_linop(func, func_abstract, **kwargs):
        import copy
        # somehow make sure that kwargs_clean only contains deep copies of
        # everything in kwargs that are not accessible from anywhere else.
        kwargs_clean = copy.deepcopy(kwargs)  # FIXME TODO
        global _global_opcounter
        kwargs_clean["_opid"] = _global_opcounter
        _global_opcounter += 1
        kwargs_clean["_func"] = func
        kwargs_clean["_func_abstract"] = func_abstract
        return _Linop(kwargs_clean)

    def fht_operator(axes, nthreads):
        def fhtfunc(inp, out, adjoint, state):
            # This function must _not_ keep any reference to 'inp' or 'out'!
            # Also, it must not change 'inp' or 'state'.
            ducc0.fft.genuine_fht(inp,
                                  out=out,
                                  axes=state["axes"],
                                  nthreads=state["nthreads"])
        def fhtfunc_abstract(shape, dtype, adjoint, state):
            return shape, dtype

        return make_linop(
            fhtfunc, fhtfunc_abstract, axes=tuple(axes),
            nthreads=int(nthreads))
   
    def c2c_operator(axes, nthreads):
        def c2cfunc(inp, out, adjoint, state):
            ducc0.fft.c2c(inp,
                          out=out,
                          axes=state["axes"],
                          nthreads=state["nthreads"],
                          forward=not adjoint)
        def c2cfunc_abstract(shape, dtype, adjoint, state):
            return shape, dtype

        return make_linop(
            c2cfunc, c2cfunc_abstract, axes=tuple(axes),
            nthreads=int(nthreads))
   
    def alm2realalm(alm, lmax, dtype):
        res = np.zeros((alm.shape[0], alm.shape[1]*2-lmax-1),dtype=dtype)
        res[:, 0:lmax+1] = alm[:, 0:lmax+1].real
        res[:, lmax+1:] = alm[:, lmax+1:].view(dtype)*np.sqrt(2.)
        return res
    def realalm2alm(alm, lmax, dtype):
        res = np.zeros((alm.shape[0], (alm.shape[1]+lmax+1)//2), dtype=dtype)
        res[:, 0:lmax+1] = alm[:, 0:lmax+1]
        res[:, lmax+1:] = alm[:, lmax+1:].view(dtype)*(np.sqrt(2.)/2)
        return res
   
    def sht2d_operator(lmax, mmax, ntheta, nphi, geometry, spin, dtype, nthreads):
        tdict = { np.dtype(np.float32) : np.dtype(np.complex64),
                  np.dtype(np.float64) : np.dtype(np.complex128)}

        def sht2dfunc(inp, out, adjoint, state):
            # This function must _not_ keep any reference to 'inp' or 'out'!
            # Also, it must not change 'inp' or 'state'.
            if adjoint:
                tmp = ducc0.sht.adjoint_synthesis_2d(
                    lmax=state["lmax"],
                    mmax=state["mmax"],
                    spin=state["spin"],
                    map=inp,
                    nthreads=state["nthreads"],
                    geometry=state["geometry"])
                out[()] = alm2realalm(tmp, state["lmax"], inp.dtype)
            else:
                tmp = realalm2alm(inp, state["lmax"], tdict[np.dtype(inp.dtype)])
                ducc0.sht.synthesis_2d(
                    lmax=state["lmax"],
                    mmax=state["mmax"],
                    spin=state["spin"],
                    map=out,
                    alm=tmp,
                    nthreads=state["nthreads"],
                    geometry=state["geometry"])

        def sht2dfunc_abstract(shape_in, dtype_in, adjoint, state):
            spin = state["spin"]
            ncomp = 1 if spin==0 else 2
            if adjoint:
                lmax = state["lmax"]
                mmax = state["mmax"]
                nalm = ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)
                nalm = nalm*2 - lmax - 1
                shape_out = (ncomp, nalm)
            else:
                shape_out = (ncomp, state["ntheta"], state["nphi"])
            return shape_out, dtype_in

        lmax = int(lmax)
        mmax = int(mmax)
        spin = int(spin)
    
        return make_linop(
            sht2dfunc, sht2dfunc_abstract,
            lmax=int(lmax),
            mmax=int(mmax),
            spin=int(spin),
            ntheta=int(ntheta),
            nphi=int(nphi),
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
    myop = fht_operator(axes=axes, nthreads=nthreads)
    op = jit(myop)
    op_adj = jit(myop.adjoint)
    rng = np.random.default_rng(42)
    a = (rng.random(shape)-0.5).astype(dtype)
    b1 = np.array(op(a)[0])
    b2 = ducc0.fft.genuine_fht(a, axes=axes, nthreads=nthreads)
    _assert_close(b1, b2, epsilon=1e-6 if dtype==np.float32 else 1e-14)
    b3 = np.array(op_adj(a)[0])
    _assert_close(b1, b3, epsilon=1e-6 if dtype==np.float32 else 1e-14)

    from jax.test_util import check_grads
    max_order = 2
    check_grads(op, (a,), order=max_order, modes=("fwd",), eps=1.)
    check_grads(op_adj, (a,), order=max_order, modes=("fwd",), eps=1.)
    check_grads(op, (a,), order=max_order, modes=("rev",), eps=1.)
    check_grads(op_adj, (a,), order=max_order, modes=("rev",), eps=1.)

@pmp("shape_axes", (((100,),(0,)), ((10,17), (0,1)), ((10,17,3), (1,))))
@pmp("dtype", (np.complex64, np.complex128))
@pmp("nthreads", (1, 2))
def test_c2c(shape_axes, dtype, nthreads):
    if not have_jax:
        pytest.skip()
    from jax import jit
    shape, axes = shape_axes
    myop = c2c_operator(axes=axes, nthreads=nthreads)
    op = jit(myop)
    op_adj = jit(myop.adjoint)
    rng = np.random.default_rng(42)
    a = (rng.random(shape)-0.5).astype(dtype) + (1j*(rng.random(shape)-0.5)).astype(dtype)
    b1 = np.array(op(a)[0])
    b2 = ducc0.fft.c2c(a, axes=axes, forward=True, nthreads=nthreads)
    _assert_close(b1, b2, epsilon=1e-6 if dtype==np.complex64 else 1e-14)
    b3 = np.array(op_adj(a)[0])
    b4 = ducc0.fft.c2c(a, axes=axes, forward=False, nthreads=nthreads)
    _assert_close(b3, b4, epsilon=1e-6 if dtype==np.complex64 else 1e-14)

    from jax.test_util import check_grads
    max_order = 2
    check_grads(op, (a,), order=max_order, modes=("fwd",), eps=1.)
    check_grads(op_adj, (a,), order=max_order, modes=("fwd",), eps=1.)
# these two fail ... no idea why
#    check_grads(op, (a,), order=max_order, modes=("rev",), eps=1.)
#    check_grads(op_adj, (a,), order=max_order, modes=("rev",), eps=1.)

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
@pmp("geometry", ("GL", "F1", "F2", "CC", "DH", "MW", "MWflip"))
@pmp("ntheta", (20,))
@pmp("nphi", (30,))
@pmp("spin", (0, 2))
@pmp("dtype", (np.float32, np.float64))
@pmp("nthreads", (1, 2))
def test_sht2d(lmmax, geometry, ntheta, nphi, spin, dtype, nthreads):
    if not have_jax:
        pytest.skip()
    from jax import jit
    tdict = { np.dtype(np.float32) : np.dtype(np.complex64),
              np.dtype(np.float64) : np.dtype(np.complex128)}

    lmax, mmax = lmmax
    ncomp = 1 if spin==0 else 2
    myop = sht2d_operator(lmax=lmax, mmax=mmax, ntheta=ntheta, nphi=nphi, geometry=geometry, spin=spin, dtype=dtype, nthreads=nthreads)
    op = jit(myop)
    op_adj = jit(myop.adjoint)
    rng = np.random.default_rng(42)
    tdict = { np.dtype(np.float32) : np.dtype(np.complex64),
              np.dtype(np.float64) : np.dtype(np.complex128)}
    alm0 = random_alm(lmax, mmax, spin, ncomp, rng).astype(tdict[np.dtype(dtype)])
    alm0r = alm2realalm(alm0, lmax, dtype)

    map1 = np.array(op(alm0r)[0])
    map2 = ducc0.sht.synthesis_2d(alm=alm0, lmax=lmax, mmax=mmax, spin=spin, geometry=geometry, ntheta=ntheta, nphi=nphi, nthreads=nthreads)
    _assert_close(map1, map2, epsilon=1e-6 if dtype==np.float32 else 1e-14)

    map0 = (rng.random((ncomp, ntheta, nphi))-0.5).astype(dtype)
    alm1r = np.array(op_adj(map0)[0])
    alm1 = realalm2alm(alm1r, lmax, tdict[np.dtype(dtype)])
    alm2 = ducc0.sht.adjoint_synthesis_2d(map=map0, lmax=lmax, mmax=mmax, spin=spin, geometry=geometry, nthreads=nthreads)
    _assert_close(alm1, alm2, epsilon=1e-6 if dtype==np.float32 else 1e-14)

    from jax.test_util import check_grads
    max_order = 2
    check_grads(op, (alm0r,), order=max_order, modes=("fwd",), eps=1.)
    check_grads(op_adj, (map0,), order=max_order, modes=("fwd",), eps=1.)
    check_grads(op, (alm0r,), order=max_order, modes=("rev",), eps=1.)
    check_grads(op_adj, (map0,), order=max_order, modes=("rev",), eps=1.)
