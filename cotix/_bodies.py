import equinox as eqx
import jax.numpy as jnp
from equinox import AbstractVar
from jaxtyping import Array, Float

from ._abstract_shapes import AbstractShape
from ._collisions import check_for_collision
from ._design_by_contract import class_invariant
from ._geometry_utils import HomogenuousTransformer
from ._shapes import Circle


@class_invariant
class AbstractBody(eqx.Module, strict=True):
    mass: AbstractVar[Float[Array, ""]]
    inertia: AbstractVar[Float[Array, ""]]

    position: AbstractVar[Float[Array, "2"]]
    velocity: AbstractVar[Float[Array, "2"]]

    angle: AbstractVar[Float[Array, ""]]
    angular_velocity: AbstractVar[Float[Array, "2"]]

    shape: AbstractVar[AbstractShape]
    _transform: AbstractVar[HomogenuousTransformer]

    def update_transform(self):
        """
        Construct a matrix that transforms shape in the correct configuration.
        While we could do it smarter, like cache the results somehow or whatever,
        we don't do that. Cuz, in theory, caching results is worse for complex shapes
        """

        return eqx.tree_at(
            lambda x: x._transform,
            self,
            replace=HomogenuousTransformer(position=self.position, angle=self.angle),
        )

    def get_transform(self):
        return eqx.error_if(
            self._transform,
            (~jnp.all(self._transform.angle == self.angle))
            | (~jnp.all(self._transform.position == self.position)),
            "Call update_transform on a Body, before "
            "doing any smart other things with it.",
        )

    def collides_with(self, other, key):
        return check_for_collision(
            self.shape,
            other.shape,
            key=key,
            trA=self.get_transform(),
            trB=other.get_transform(),
        )

    def set_mass(self, mass):
        return eqx.tree_at(lambda x: x.mass, self, mass)

    def set_inertia(self, inertia):
        return eqx.tree_at(lambda x: x.inertia, self, inertia)

    def set_position(self, position):
        new_self = eqx.tree_at(lambda x: x._transform, self, None)
        return eqx.tree_at(lambda x: x.position, new_self, position)

    def set_velocity(self, velocity):
        return eqx.tree_at(lambda x: x.velocity, self, velocity)

    def set_angle(self, angle):
        new_self = eqx.tree_at(lambda x: x._transform, self, None)
        return eqx.tree_at(lambda x: x.angle, new_self, angle)

    def set_angular_velocity(self, angular_velocity):
        return eqx.tree_at(lambda x: x.angular_velocity, self, angular_velocity)

    def set_shape(self, shape):
        return eqx.tree_at(lambda x: x.shape, self, shape)

    def __invariant__(self):
        return (
            # Checks for nans
            jnp.any(jnp.isnan(self.mass))
            | jnp.any(jnp.isnan(self.inertia))
            | jnp.any(jnp.isnan(self.position))
            | jnp.any(jnp.isnan(self.velocity))
            | jnp.any(jnp.isnan(self.angle))
            | jnp.any(jnp.isnan(self.angular_velocity))
            # Checks for some custom constraints
            | (self.shape is None)
            | jnp.any(self.mass < 0)
            | jnp.any(self.inertia < 0)
            | jnp.any(self.angle < -jnp.pi)
            | jnp.any(self.angle > jnp.pi)
        )


class Ball(AbstractBody, strict=True):
    mass: Float[Array, ""]
    inertia: Float[Array, ""]

    position: Float[Array, "2"]
    velocity: Float[Array, "2"]

    angle: Float[Array, ""]
    angular_velocity: Float[Array, "2"]

    shape: AbstractShape
    _transform: HomogenuousTransformer

    def __init__(self):
        self.mass = 1.0
        self.inertia = 1.0
        self.position = jnp.zeros((2,))
        self.velocity = jnp.zeros((2,))
        self.angle = 0
        self.angular_velocity = 0
        self.shape = Circle(0.05, jnp.zeros((2,)))
        self._transform = HomogenuousTransformer()
