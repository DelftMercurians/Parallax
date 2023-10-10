import equinox as eqx
from jaxtyping import Array, Float

from ._shapes import AbstractShape


class AbstractBody(eqx.Module, strict=True):
    mass: Float[Array, ""]
    inertia: Float[Array, ""]

    position: Float[Array, "2"]
    velocity: Float[Array, "2"]

    angle: Float[Array, ""]
    angular_velocity: Float[Array, "2"]

    shape: AbstractShape
