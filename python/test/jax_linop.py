import jax

from functools import partial
import numpy as np
import ducc0

__all__ = ["make_linop"]

# incremented for every registered operator, strictly for uniqueness purposes
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

    # the values are explained in src/duc0/bindings/typecode.h
    dtype_dict = { np.dtype(np.float32): 3,
                   np.dtype(np.float64): 7,
                   np.dtype(np.complex64): 67,
                   np.dtype(np.complex128): 71 }

    # add array
    operands = [x]
    operand_layouts = [layout_in] + [()]*(7+len(shape_in)+len(shape_out))

    # add opid and stateid
    operands.append(mlir.ir_constant(state["_opid"]))
    operands.append(mlir.ir_constant(stateid))

    # add forward/adjoint mode
    operands.append(mlir.ir_constant(int(adjoint)))

    # add input dtype, rank, and shape
    operands.append(mlir.ir_constant(dtype_dict[dtype_in]))
    operands.append(mlir.ir_constant(len(shape_in)))
    operands += [mlir.ir_constant(i) for i in shape_in]

    # add output dtype, rank, and shape
    operands.append(mlir.ir_constant(dtype_dict[dtype_out]))
    operands.append(mlir.ir_constant(len(shape_out)))
    operands += [mlir.ir_constant(i) for i in shape_out]

    if platform == "cpu":
        shapeconst = tuple(mlir.ir_constant(s) for s in shape_in)
        return jaxlib.hlo_helpers.custom_call(
            platform + "_linop",
            result_types=[jaxtype_out, ],
            operands=operands,
            operand_layouts=operand_layouts,
            result_layouts=[layout_out]
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
    tmp = _get_prim(not adjoint).bind(cotangents[0].conj(), stateid=stateid)
    tmp[0] = tmp[0].conj()
    return tmp

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
