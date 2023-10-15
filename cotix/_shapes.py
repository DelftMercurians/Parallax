import equinox as eqx
import jax
from jax import numpy as jnp, tree_util as jtu
from jaxtyping import Array, Float

from ._abstract_shapes import AbstractConvexShape, AbstractShape
from ._geometry_utils import order_clockwise


class CompositeShape(AbstractShape, strict=True):
    parts: list[AbstractShape]

    def get_support(self, direction):
        raise NotImplementedError

    def _get_center(self):
        return jtu.tree_reduce(
            lambda acc, shape: acc + shape.get_center(),
            self.parts,
            is_leaf=lambda node: isinstance(node, AbstractShape),
        )


class Circle(AbstractConvexShape, strict=True):
    radius: Float[Array, ""] = eqx.field(converter=jnp.asarray)
    position: Float[Array, "2"] = eqx.field(converter=jnp.asarray)

    def get_support(self, direction: Float[Array, "2"]):
        normalized_direction = direction / jnp.linalg.norm(direction)
        return normalized_direction * self.radius + self.position

    def _get_center(self):
        return self.position


class Polygon(AbstractConvexShape, strict=True):
    vertices: Float[Array, "size 2"]  # ordered clockwise

    def __init__(self, vertices: Float[Array, "size 2"]):
        self.vertices = order_clockwise(vertices)
        # TODO: error if passed vertices cannot form a convex polygon
        # TODO: error if not ordered after ordereing
        # TODO: error if two vertices

    def get_support(self, direction: Float[Array, "2"]) -> Float[Array, "2"]:
        dot_products = jax.lax.map(lambda x: jnp.dot(x, direction), self.vertices)
        return self.vertices.at[jnp.argmax(dot_products)].get()

    def _get_center(self) -> Float[Array, "2"]:
        return jnp.mean(self.vertices, axis=0)


class AABB(AbstractConvexShape, strict=True):
    min: Float[Array, "2"]
    max: Float[Array, "2"]

    def __init__(self, shape: AbstractShape):
        x_min = shape.get_support(jnp.array([-1.0, 0.0]))[0]
        y_min = shape.get_support(jnp.array([0.0, -1.0]))[1]
        x_max = shape.get_support(jnp.array([1.0, 0.0]))[0]
        y_max = shape.get_support(jnp.array([0.0, 1.0]))[1]

        x_max = eqx.error_if(x_max, x_max <= x_min, "AABB is invalid")
        y_max = eqx.error_if(y_max, y_max <= y_min, "AABB is invalid")

        self.min = jnp.stack([x_min, y_min])
        self.max = jnp.stack([x_max, y_max])

    def get_support(self, direction: Float[Array, "2"]) -> Float[Array, "2"]:
        support_point = jnp.where(direction >= 0, self.max, self.min)
        return support_point

    def _get_center(self):
        return (self.min + self.max) / 2.0
