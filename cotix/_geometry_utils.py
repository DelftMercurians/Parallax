import equinox as eqx
from jax import numpy as jnp, random as jr
from jaxtyping import Array, Float

from ._abstract_shapes import AbstractConvexShape


def is_point_in_triangle(pt, v1, v2, v3):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(pt, v1, v2)
    d2 = sign(pt, v2, v3)
    d3 = sign(pt, v3, v1)

    has_neg = jnp.logical_or((d1 < 0), jnp.logical_or((d2 < 0), (d3 < 0)))
    has_pos = jnp.logical_or((d1 > 0), jnp.logical_or((d2 > 0), (d3 > 0)))

    return jnp.logical_not(jnp.logical_and(has_neg, has_pos))


def fast_normal(a):
    return jnp.array([-a[1], a[0]])


def random_direction(key):
    if key is None:
        return jnp.array([1.0, 0.0])

    # reference to why this works: https://mathworld.wolfram.com/HyperspherePointPicking.html
    x = jr.normal(key, (2,))
    return x / jnp.linalg.norm(x)


def minkowski_diff(A: AbstractConvexShape, B: AbstractConvexShape, direction):
    return A._get_support(direction) - B._get_support(-direction)


def order_clockwise(vertices: Float[Array, "size 2"]) -> Float[Array, "size 2"]:
    relative_vertices = vertices - jnp.mean(vertices, axis=0)
    relative_vertices = eqx.error_if(
        relative_vertices,
        relative_vertices == jnp.zeros((2,)),
        "Encountered zero in order clockwise, cannot handle for now",
    )  # TODO: solve this
    angles = jnp.arctan2(relative_vertices[:, 1], relative_vertices[:, 0])
    indices = jnp.argsort(angles, axis=0)
    return vertices[indices]
