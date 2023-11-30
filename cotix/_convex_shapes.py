import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float

from ._abstract_shapes import AbstractConvexShape
from ._geometry_utils import fast_normal, order_clockwise


class Circle(AbstractConvexShape):
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
        # direction = eqx.error_if(direction, jnp.all(direction == 0.0), "Nope")
        normalized_direction = direction / jnp.linalg.norm(direction)
        return normalized_direction * self.radius + self.position

    def contains(self, point, eps=1e-6):
        return jnp.sum((point - self.position) ** 2) <= ((self.radius + eps) ** 2)

    def get_center(self):
        return self.position

    def move(self, delta: Float[Array, "2"]):
        return eqx.tree_at(lambda x: x.position, self, self.position + delta)

    def transform(self, transformer):
        return Circle(
            radius=self.radius,
            position=self.position + transformer.shift(),
        )

    def draw(self, painter):
        painter.draw_circle(self.position, self.radius, (128, 128, 128))


class AABB(AbstractConvexShape):
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
        # direction = eqx.error_if(direction, jnp.all(direction == 0.0), "Nope")
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

    def get_edges(self):
        vs = jnp.array(
            [
                self.upper,
                [self.upper[0], self.lower[1]],
                self.lower,
                [self.lower[0], self.upper[1]],
            ]
        )
        return jnp.array(
            [[vs[0], vs[1]], [vs[1], vs[2]], [vs[2], vs[3]], [vs[3], vs[0]]]
        )

    def contains(self, point, eps=1e-6):
        return jnp.all((point >= self.lower - eps) & (point <= self.upper + eps))

    def move(self, delta: Float[Array, "2"]):
        new_self = eqx.tree_at(lambda x: x.upper, self, self.upper + delta)
        new_self = eqx.tree_at(lambda x: x.lower, new_self, new_self.lower + delta)
        return new_self

    def transform(self, transformer):
        return AABB(
            lower=self.lower + transformer.shift(),
            upper=self.upper + transformer.shift(),
        )

    def draw(self, painter):
        vs = jnp.array(
            [
                self.upper,
                [self.upper[0], self.lower[1]],
                self.lower,
                [self.lower[0], self.upper[1]],
            ]
        )
        painter.draw_polygon(vertices=vs, color=(128, 128, 128))


class AbstractPolygon(AbstractConvexShape):
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
        return jax.lax.cond(
            jnp.any(jnp.isnan(direction)),
            lambda: jnp.array([jnp.nan, jnp.nan]),
            lambda: self.vertices[jnp.argmax(dot_products)],
        )

    def get_center(self) -> Float[Array, "2"]:
        return jnp.mean(self.vertices, axis=0)

    def get_edges(self):
        return jnp.concatenate(
            [self.vertices, jnp.roll(self.vertices, shift=1, axis=0)], axis=1
        ).reshape((-1, 2, 2))

    def contains(self, point):
        edges = self.get_edges()
        dots = jax.lax.map(
            lambda edge: jnp.dot(point - edges[0], fast_normal(edges[0] - edges[1])),
            edges,
        )
        dots = jnp.sign(dots)
        return True

    def move(self, delta: Float[Array, "2"]):
        new_vertices = jax.lax.map(lambda x: x + delta, self.vertices)
        return type(self)(new_vertices)

    def transform(self, transformer):
        return type(self)(
            vertices=jax.lax.map(
                lambda x: transformer.forward_vector(x),
                self.vertices,
            ),
        )

    def draw(self, painter):
        painter.draw_polygon(self.vertices, (255, 255, 255))


class Polygon(AbstractPolygon):
    vertices: Float[Array, "size 2"]  # ordered clockwise

    def __init__(self, vertices: Float[Array, "size 2"]):
        self.vertices = order_clockwise(vertices)


class Polygon3(AbstractPolygon):
    vertices: Float[Array, "3 2"]  # ordered clockwise

    def __init__(self, vertices: Float[Array, "3 2"]):
        self.vertices = order_clockwise(vertices)


class Polygon4(AbstractPolygon):
    vertices: Float[Array, "4 2"]  # ordered clockwise

    def __init__(self, vertices: Float[Array, "4 2"]):
        self.vertices = order_clockwise(vertices)


class Polygon5(AbstractPolygon):
    vertices: Float[Array, "5 2"]  # ordered clockwise

    def __init__(self, vertices: Float[Array, "5 2"]):
        self.vertices = order_clockwise(vertices)


class Polygon6(AbstractPolygon):
    vertices: Float[Array, "6 2"]  # ordered clockwise

    def __init__(self, vertices: Float[Array, "6 2"]):
        self.vertices = order_clockwise(vertices)
