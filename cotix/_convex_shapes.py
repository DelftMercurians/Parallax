import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float

from ._abstract_shapes import AbstractConvexShape
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

    @jax.jit
    def get_support(self, direction: Float[Array, "2"]):
        normalized_direction = direction / jnp.linalg.norm(direction)
        return normalized_direction * self.radius + self.position

    def contains(self, point, eps=1e-6):
        return jnp.sum((point - self.position) ** 2) <= (self.radius**2 + eps)

    def get_center(self):
        return self.position

    def move(self, delta: Float[Array, "2"]):
        return eqx.tree_at(lambda x: x.position, self, self.position + delta)


class AABB(AbstractConvexShape, strict=True):
    """
    Axis aligned bounding box described by min and max corner coordinates
    """

    upper: Float[Array, "2"]
    lower: Float[Array, "2"]

    def __init__(self, lower, upper):
        self.upper = upper
        self.lower = lower

    @jax.jit
    def get_support(self, direction: Float[Array, "2"]) -> Float[Array, "2"]:
        support_point = jnp.where(direction >= 0, self.upper, self.lower)
        return support_point

    def of(shape):
        x_min = shape.get_support(jnp.array([-1.0, 0.0]))[0]
        y_min = shape.get_support(jnp.array([0.0, -1.0]))[1]
        x_max = shape.get_support(jnp.array([1.0, 0.0]))[0]
        y_max = shape.get_support(jnp.array([0.0, 1.0]))[1]

        x_max = eqx.error_if(x_max, x_max <= x_min, "AABB is invalid")
        y_max = eqx.error_if(y_max, y_max <= y_min, "AABB is invalid")

        return AABB(jnp.stack([x_min, y_min]), jnp.stack([x_max, y_max]))

    def get_center(self):
        return (self.lower + self.upper) / 2.0

    def contains(self, point, eps=1e-6):
        return jnp.all((point >= self.lower - eps) & (point <= self.upper + eps))

    def move(self, delta: Float[Array, "2"]):
        new_self = eqx.tree_at(lambda x: x.upper, self, self.upper + delta)
        new_self = eqx.tree_at(lambda x: x.lower, new_self, new_self.lower + delta)
        return new_self


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
