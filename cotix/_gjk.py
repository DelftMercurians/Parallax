import equinox as eqx
import jax
from jax import numpy as jnp

from ._geometry_utils import (
    fast_normal,
    is_point_in_triangle,
    minkowski_diff,
    random_direction,
)
from ._shapes import AbstractConvexShape


@eqx.filter_jit
def get_collision_simplex(A: AbstractConvexShape, B: AbstractConvexShape, key=None):
    # make the algorithm ''randomized'' -> probability of problems = 0
    # TODO: test weird cases
    initial_direction = random_direction(key)

    # setup simplex
    simplex = jnp.zeros((3, 2))
    # .at[0] inside jit is inplace -> no performance hit
    simplex = simplex.at[0].set(minkowski_diff(A, B, initial_direction))
    simplex = simplex.at[1].set(minkowski_diff(A, B, -simplex[0]))

    def reverse_simplex(simplex, direction):
        tmp = simplex[0]
        simplex = simplex.at[0].set(simplex[1])
        simplex = simplex.at[1].set(tmp)
        return (simplex, direction)

    def reverse_direction(simplex, direction):
        return (simplex, -direction)

    # arrange vertices clockwise
    # and make the direction a normal towrds origin
    direction = fast_normal(simplex[1] - simplex[0])
    simplex, direction = jax.lax.cond(
        jnp.dot(direction, -simplex[1]) > 0,
        reverse_simplex,
        reverse_direction,
        simplex,
        direction,
    )

    # the last point is computed only after we have correct direction
    c = minkowski_diff(A, B, direction)
    simplex = simplex.at[2].set(c)

    # lax while loop:
    # while cond_fn(x):
    #   x = body_fn(x)
    def body_fn(x):
        # unpack
        simplex, direction = x
        a, b, c = simplex

        ac_normal = fast_normal(c - a)
        cb_normal = fast_normal(b - c)

        # as the new direction choose the one that points towards the origin
        simplex, direction = jax.lax.cond(
            jnp.dot(ac_normal, -c) >= 0,
            lambda *_: (simplex.at[1].set(c), ac_normal),
            lambda *_: (simplex.at[0].set(c), cb_normal),
        )

        c = minkowski_diff(A, B, direction)  # get the point in the new direction
        simplex = simplex.at[2].set(c)

        # pack back to the same shape
        return (simplex, direction)

    def cond_fn(x):
        # unpack
        simplex, direction = x

        # c1: we were not able to go further than origin -> no collision
        c1 = jnp.dot(simplex[2], direction) <= 0

        # (c1 & c2): simplex contains the origin
        c2 = jnp.dot(fast_normal(simplex[2] - simplex[0]), -simplex[2]) < 0
        c3 = jnp.dot(fast_normal(simplex[1] - simplex[2]), -simplex[2]) < 0

        # return the overall stopping condition
        return ~(c1 | (c2 & c3))

    # use Equinox while loop, so that we can backpropagate through it (just in case)
    simplex, direction = eqx.internal.while_loop(
        cond_fn, body_fn, (simplex, direction), kind="checkpointed", max_steps=32
    )

    # if the simplex is valid, return simplex, otherwise return zeros
    value = jax.lax.cond(
        is_point_in_triangle(jnp.zeros((2,)), simplex[0], simplex[1], simplex[2]),
        lambda x: x,
        lambda x: jnp.zeros((3, 2)),
        simplex,
    )

    return value
