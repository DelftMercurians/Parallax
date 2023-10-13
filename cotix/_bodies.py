import equinox as eqx
import jax.numpy as jnp
from equinox import AbstractVar
from jaxtyping import Array, Float

from ._abstract_shapes import AbstractShape
from ._collisions import check_for_collision
from ._geometry_utils import HomogenuousTransformer
from ._shapes import Circle


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
        matrix = jnp.array(
            [
                [jnp.cos(self.angle), -jnp.sin(self.angle), self.position[0]],
                [jnp.sin(self.angle), jnp.cos(self.angle), self.position[1]],
                [0.0, 0.0, 1.0],
            ]
        )

        return eqx.tree_at(lambda x: x._transform, self, HomogenuousTransformer(matrix))

    def get_transform(self):
        return eqx.error_if(
            self._transform,
            self._transform is None,
            "Call update_transform before doing anything with an updated object",
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
        return eqx.tree_at(lambda x: x.position, self, position)

    def set_velocity(self, velocity):
        return eqx.tree_at(lambda x: x.velocity, self, velocity)

    def set_angle(self, angle):
        return eqx.tree_at(lambda x: x.angle, self, angle)

    def set_angular_velocity(self, angular_velocity):
        return eqx.tree_at(lambda x: x.angular_velocity, self, angular_velocity)

    def set_shape(self, shape):
        return eqx.tree_at(lambda x: x.shape, self, shape)


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
