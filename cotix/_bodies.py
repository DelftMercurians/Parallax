import equinox as eqx
import jax.numpy as jnp
from equinox import AbstractVar
from jaxtyping import Array, Float

from ._convex_shapes import Circle
from ._design_by_contract import class_invariant
from ._universal_shape import UniversalShape


@class_invariant
class AbstractBody(eqx.Module, strict=True):
    mass: AbstractVar[Float[Array, ""]]
    inertia: AbstractVar[Float[Array, ""]]

    position: AbstractVar[Float[Array, "2"]]
    velocity: AbstractVar[Float[Array, "2"]]

    angle: AbstractVar[Float[Array, ""]]
    angular_velocity: AbstractVar[Float[Array, ""]]

    shape: AbstractVar[UniversalShape]

    def update_transform(self):
        return eqx.tree_at(
            lambda x: x.shape,
            self,
            self.shape.update_transform(angle=self.angle, position=self.position),
        )

    def collides_with(self, other):
        return self.shape.collides_with(other.shape)

    def set_mass(self, mass: Float[Array, ""]):
        return eqx.tree_at(lambda x: x.mass, self, mass)

    def set_inertia(self, inertia: Float[Array, ""]):
        return eqx.tree_at(lambda x: x.inertia, self, inertia)

    def set_position(self, position: Float[Array, "2"]):
        return eqx.tree_at(lambda x: x.position, self, position)

    def set_velocity(self, velocity: Float[Array, "2"]):
        return eqx.tree_at(lambda x: x.velocity, self, velocity)

    def set_angle(self, angle: Float[Array, ""]):
        return eqx.tree_at(lambda x: x.angle, self, angle)

    def set_angular_velocity(self, angular_velocity: Float[Array, ""]):
        return eqx.tree_at(lambda x: x.angular_velocity, self, angular_velocity)

    def set_shape(self, shape: UniversalShape):
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
    angular_velocity: Float[Array, ""]

    shape: UniversalShape

    def __init__(self, mass, velocity, shape):
        self.mass = mass
        self.inertia = mass

        self.position = jnp.zeros((2,))
        self.velocity = velocity

        self.angle = jnp.array(0.0)
        self.angular_velocity = jnp.array(0.0)

        self.shape = shape

    @staticmethod
    def make_default():
        ball = Ball(
            jnp.array(1.0),
            jnp.zeros((2,)),
            UniversalShape(Circle(jnp.array(0.05), jnp.zeros((2,)))),
        )
        ball = (
            ball.set_mass(jnp.array(1.0))
            .set_inertia(jnp.array(1.0))
            .set_position(jnp.zeros((2,)))
            .set_velocity(jnp.zeros((2,)))
            .set_angle(jnp.array(0.0))
            .set_angular_velocity(jnp.array(0.0))
            .set_shape(UniversalShape(Circle(jnp.array(0.05), jnp.zeros((2,)))))
        )
        return ball
