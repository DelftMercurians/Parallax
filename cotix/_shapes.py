import equinox as eqx
import jax
from jax import numpy as jnp, random as jr
from jaxtyping import Array, Float


class AbstractShape(eqx.Module):
    def _getFarthestPointInDirection(self, direction: Float[Array, "2"]):
        pass

    def _get_center(self):
        pass


class AbstractConvexShape(AbstractShape):
    ...


class Circle(AbstractConvexShape):
    radius: Float[Array, ""]
    position: Float[Array, "2"]

    def _getFarthestPointInDirection(self, direction: Float[Array, "2"]):
        return (
            direction / jnp.sqrt(jnp.sum(direction**2)) * self.radius + self.position
        )

    def _get_center(self):
        return self.position


def _is_point_in_triangle(pt, v1, v2, v3):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(pt, v1, v2)
    d2 = sign(pt, v2, v3)
    d3 = sign(pt, v3, v1)

    has_neg = jnp.logical_or((d1 < 0), jnp.logical_or((d2 < 0), (d3 < 0)))
    has_pos = jnp.logical_or((d1 > 0), jnp.logical_or((d2 > 0), (d3 > 0)))

    return jnp.logical_not(jnp.logical_and(has_neg, has_pos))


def _fast_normal(a):
    return jnp.array([-a[1], a[0]])


def _random_direction(key):
    x = jr.uniform(key, (2,)) - 0.5
    return x / jnp.sqrt(jnp.sum(x**2))


def _get_support(A: AbstractShape, B: AbstractShape, direction):
    return A._getFarthestPointInDirection(direction) - B._getFarthestPointInDirection(
        -direction
    )


@eqx.filter_jit
def _get_collision_simplex(A: AbstractShape, B: AbstractShape, key):
    initial_direction = _random_direction(key)

    simplex = jnp.zeros((3, 2))
    simplex = simplex.at[0].set(_get_support(A, B, initial_direction))
    simplex = simplex.at[1].set(_get_support(A, B, -simplex[0]))
    simplex = simplex.at[2].set(simplex[0])

    direction = _fast_normal(simplex[1] - simplex[0])

    def reverse_simplex(simplex, direction):
        tmp = simplex[0]
        simplex = simplex.at[0].set(simplex[1])
        simplex = simplex.at[1].set(tmp)
        return (simplex, direction)

    def reverse_direction(simplex, direction):
        return (simplex, -direction)

    # arrange vertices clockwise
    simplex, direction = jax.lax.cond(
        jnp.dot(direction, -simplex[1]) > 0,
        reverse_simplex,
        reverse_direction,
        simplex,
        direction,
    )

    def body_fn(x):
        simplex, direction = x
        a, b, c = simplex

        ac_normal = _fast_normal(c - a)
        cb_normal = _fast_normal(b - c)

        # as the new direction choose the one that is closest to the origin
        simplex, direction = jax.lax.cond(
            jnp.dot(ac_normal, -c) >= 0,
            lambda *_: (simplex.at[1].set(c), ac_normal),
            lambda *_: (simplex.at[0].set(c), cb_normal),
        )

        c = _get_support(A, B, direction)  # generate new point
        simplex = simplex.at[2].set(c)

        return (simplex, direction)

    def cond_fn(x):
        simplex, direction = x

        c1 = (
            jnp.dot(simplex[2], direction) <= 0
        )  # we were not able to go further than origin -> no collision
        c2 = jnp.dot(_fast_normal(simplex[2] - simplex[0]), -simplex[2]) < 0
        c3 = jnp.dot(_fast_normal(simplex[1] - simplex[2]), -simplex[2]) < 0
        # c1 and c2 -> simplex contains the origin
        return jnp.logical_not(jnp.logical_or(c1, jnp.logical_and(c2, c3)))

    c = _get_support(A, B, direction)
    simplex = simplex.at[2].set(c)

    simplex, direction = eqx.internal.while_loop(
        cond_fn, body_fn, (simplex, direction), kind="checkpointed", max_steps=32
    )

    value = jax.lax.cond(
        _is_point_in_triangle(jnp.zeros((2,)), simplex[0], simplex[1], simplex[2]),
        lambda x: x,
        lambda x: jnp.zeros((3, 2)),
        simplex,
    )
    return value
