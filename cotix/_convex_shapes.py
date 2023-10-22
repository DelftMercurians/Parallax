import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float

from ._abstract_shapes import AbstractConvexShape, AbstractShape
from ._geometry_utils import order_clockwise


class Circle(AbstractConvexShape, strict=True):
    """
    Circular convex shape, has radius and a position.
    """

    radius: Float[Array, ""] = eqx.field(converter=jnp.asarray)
    position: Float[Array, "2"] = eqx.field(converter=jnp.asarray)

    def __init__(self, radius: Float[Array, ""], position: Float[Array, "2"]):
        self.radius = radius
        self.position = position

    def get_support(self, direction: Float[Array, "2"]):
        normalized_direction = direction / jnp.linalg.norm(direction)
        return normalized_direction * self.radius + self.position

    def get_center(self):
        return self.position


class Polygon(AbstractConvexShape, strict=True):
    """
    A **convex** polygon that has a bunch of vertices.
    """

    vertices: Float[Array, "size 2"]  # ordered clockwise

    def __init__(self, vertices: Float[Array, "size 2"]):
        self.vertices = order_clockwise(vertices)
        # TODO: error if passed vertices cannot form a convex polygon
        # TODO: error if not ordered after ordereing
        # TODO: error if two vertices

    def get_support(self, direction: Float[Array, "2"]) -> Float[Array, "2"]:
        dot_products = jax.lax.map(lambda x: jnp.dot(x, direction), self.vertices)
        return self.vertices.at[jnp.argmax(dot_products)].get()

    def get_center(self) -> Float[Array, "2"]:
        return jnp.mean(self.vertices, axis=0)


class AABB(AbstractConvexShape, strict=True):
    """
    Axis aligned bounding box described by min and max corner coordinates
    """

    min: Float[Array, "2"]
    max: Float[Array, "2"]

    def __init__(self, shape: AbstractShape | None = None):
        if isinstance(shape, AbstractShape):
            x_min = shape.get_support(jnp.array([-1.0, 0.0]))[0]
            y_min = shape.get_support(jnp.array([0.0, -1.0]))[1]
            x_max = shape.get_support(jnp.array([1.0, 0.0]))[0]
            y_max = shape.get_support(jnp.array([0.0, 1.0]))[1]

            x_max = eqx.error_if(x_max, x_max <= x_min, "AABB is invalid")
            y_max = eqx.error_if(y_max, y_max <= y_min, "AABB is invalid")

            self.min = jnp.stack([x_min, y_min])
            self.max = jnp.stack([x_max, y_max])
        else:
            self.max = self.min = jnp.zeros((2,))

    def get_support(self, direction: Float[Array, "2"]) -> Float[Array, "2"]:
        support_point = jnp.where(direction >= 0, self.max, self.min)
        return support_point

    def get_center(self):
        return (self.min + self.max) / 2.0

    @staticmethod
    def collides(a, b):
        is_first_below_second = a.max[1] < b.min[1]
        is_first_above_second = a.min[1] > b.max[1]
        is_first_left_second = a.max[0] < b.min[0]
        is_first_right_second = a.min[0] > b.max[0]

        return ~(
            is_first_below_second
            | is_first_left_second
            | is_first_above_second
            | is_first_right_second
        )

    @staticmethod
    def of_universal(shape):
        x_min = shape.get_global_support(jnp.array([-1.0, 0.0]))[0]
        y_min = shape.get_global_support(jnp.array([0.0, -1.0]))[1]
        x_max = shape.get_global_support(jnp.array([1.0, 0.0]))[0]
        y_max = shape.get_global_support(jnp.array([0.0, 1.0]))[1]

        x_max = eqx.error_if(x_max, x_max <= x_min, "AABB is invalid")
        y_max = eqx.error_if(y_max, y_max <= y_min, "AABB is invalid")

        aabb = AABB()
        aabb = eqx.tree_at(lambda x: x.min, aabb, jnp.stack([x_min, y_min]))
        aabb = eqx.tree_at(lambda x: x.max, aabb, jnp.stack([x_max, y_max]))
        return aabb
