from typing import Optional

from ducc0._typing import ComplexArray
from ducc0.sht import (
    adjoint_analysis_2d as adjoint_analysis_2d,
    adjoint_synthesis as adjoint_synthesis,
    adjoint_synthesis_2d as adjoint_synthesis_2d,
    adjoint_synthesis_general as adjoint_synthesis_general,
    alm2leg as alm2leg,
    alm2leg_deriv1 as alm2leg_deriv1,
    analysis_2d as analysis_2d,
    get_gridweights as get_gridweights,
    leg2alm as leg2alm,
    leg2map as leg2map,
    map2leg as map2leg,
    maximum_safe_l as maximum_safe_l,
    pseudo_analysis as pseudo_analysis,
    pseudo_analysis_general as pseudo_analysis_general,
    synthesis as synthesis,
    synthesis_2d as synthesis_2d,
    synthesis_2d_deriv1 as synthesis_2d_deriv1,
    synthesis_deriv1 as synthesis_deriv1,
    synthesis_general as synthesis_general,
)

def alm2flm(alm: ComplexArray, spin: int, flm: Optional[ComplexArray] = ...) -> ComplexArray: ...
def flm2alm(flm: ComplexArray, spin: int, alm: Optional[ComplexArray] = ..., real: bool = ...) -> ComplexArray: ...
