import equinox as eqx
import jax
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

    def get_transform(self):
        """
        Construct a matrix that transforms shape in the correct configuration.
        While we could do it smarter, like cache the results somehow or whatever,
        we don't do that. Cuz, in theory, caching results is worse for complex shapes

        So we return the homogenuous transform matrix
        """
        matrix = jnp.array(
            [
                [jnp.cos(self.angle), -jnp.sin(self.angle), self.position[0]],
                [jnp.sin(self.angle), jnp.cos(self.angle), self.position[1]],
                [0.0, 0.0, 1.0],
            ]
        )
        jax.debug.print("Transform matrix is {transform}", transform=matrix)
        return HomogenuousTransformer(matrix)

    def collides_with(self, other, key):
        return check_for_collision(
            self.shape,
            other.shape,
            key=key,
            trA=self.get_transform(),
            trB=other.get_transform(),
        )


class Ball(AbstractBody, strict=True):
    mass: Float[Array, ""]
    inertia: Float[Array, ""]

    position: Float[Array, "2"]
    velocity: Float[Array, "2"]

    angle: Float[Array, ""]
    angular_velocity: Float[Array, "2"]

    shape: AbstractShape

    def __init__(self):
        self.mass = 1.0
        self.inertia = 1.0
        self.position = jnp.zeros((2,))
        self.velocity = jnp.zeros((2,))
        self.angle = 0
        self.angular_velocity = 0
        self.shape = Circle(0.05, jnp.zeros((2,)))

    def move_to(self, target):
        return eqx.tree_at(lambda x: x.position, self, target)
