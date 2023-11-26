"""
A number of semi-private geometry utils for some simple and common operations
"""

import equinox as eqx
from jax import numpy as jnp, random as jr
from jaxtyping import Array, Float

from ._abstract_shapes import SupportFn


def is_point_in_triangle(pt, v1, v2, v3):
    """
    Checks whether a point is inside a two-dimensional triangle described by 3 points.
    """

    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(pt, v1, v2)
    d2 = sign(pt, v2, v3)
    d3 = sign(pt, v3, v1)

    has_neg = jnp.logical_or((d1 < 0), jnp.logical_or((d2 < 0), (d3 < 0)))
    has_pos = jnp.logical_or((d1 > 0), jnp.logical_or((d2 > 0), (d3 > 0)))

    return jnp.logical_not(jnp.logical_and(has_neg, has_pos))


def fast_normal(a):
    """
    Quickly computes some consistent normal of the vector.
    """
    return jnp.array([-a[1], a[0]])


def random_direction(key):
    """
    Generates a uniformly random normalized direction on a hypersphere (circle)
    """
    if key is None:
        return jnp.array([1.0, 0.0])

    # reference to why this works: https://mathworld.wolfram.com/HyperspherePointPicking.html
    x = jr.normal(key, (2,))
    return x / jnp.linalg.norm(x)


def minkowski_diff(
    a_support_fn: SupportFn, b_support_fn: SupportFn, direction: Float[Array, "2"]
):
    """
    Given two support functions, and a direction, computes minkowski difference.
    This is a really simple function, but relatively important, so I decided
    to isolate it.
    """
    return a_support_fn(direction) - b_support_fn(-direction)


def order_clockwise(vertices: Float[Array, "size 2"]) -> Float[Array, "size 2"]:
    """
    Orders a bunch of vertices clockwise around their center of mass.
    """
    relative_vertices = vertices - jnp.mean(vertices, axis=0)
    angles = jnp.arctan2(relative_vertices[:, 1], relative_vertices[:, 0])
    indices = jnp.argsort(angles, axis=0)
    return vertices[indices]


def perpendicular_vector(v):
    """Another function to quickly compute perpendicular vector."""
    return jnp.array([-v[1], v[0]])


def angle_between(v1, v2):
    v1_u = v1 / jnp.linalg.norm(v1)
    v2_u = v2 / jnp.linalg.norm(v2)
    return jnp.arccos(jnp.clip(jnp.dot(v1_u, v2_u), -1.0, 1.0))


class HomogenuousTransformer(eqx.Module, strict=True):
    """
    Allows to apply arbitrary affine transformations to passed vectors/directions
    """

    matrix: Float[Array, "3 3"]
    inv_matrix: Float[Array, "3 3"]

    position: Float[Array, "2"]
    angle: Float[Array, ""]

    def __init__(self, position=jnp.array([0.0, 0.0]), angle=jnp.array(0.0)):
        self.position = position
        self.angle = angle

        self.matrix = jnp.array(
            [
                [jnp.cos(self.angle), -jnp.sin(self.angle), self.position[0]],
                [jnp.sin(self.angle), jnp.cos(self.angle), self.position[1]],
                [0.0, 0.0, 1.0],
            ]
        )

        self.inv_matrix = jnp.linalg.inv(self.matrix)

    def inverse_direction(self, x):
        """Direction: from global coordinate system to local."""
        homo_dir = jnp.array([x[0], x[1], 0.0])
        transformed = self.inv_matrix @ homo_dir
        return jnp.array([transformed[0], transformed[1]])

    def forward_direction(self, x):
        """Direction: from local coordinate system to global."""
        homo_dir = jnp.array([x[0], x[1], 0.0])
        transformed = self.matrix @ homo_dir
        return jnp.array([transformed[0], transformed[1]])

    def inverse_vector(self, x):
        """Vector: from global coordinate system to local."""
        homo_dir = jnp.array([x[0], x[1], 1.0])
        transformed = self.inv_matrix @ homo_dir
        return jnp.array([transformed[0], transformed[1]]) / transformed[2]

    def forward_vector(self, x):
        """Vector: from local coordinate system to global"""
        homo_dir = jnp.array([x[0], x[1], 1.0])
        transformed = self.matrix @ homo_dir
        return jnp.array([transformed[0], transformed[1]]) / transformed[2]

    def shift(self):
        """Get the shift that is applied when moving from local to global"""
        return self.matrix[2, :2]
