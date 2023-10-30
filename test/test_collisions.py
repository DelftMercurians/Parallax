import jax
import pytest
from jax import numpy as jnp, random as jr

from cotix._bodies import Ball
from cotix._collisions import (
    check_for_collision_convex,
    compute_penetration_vector_convex,
)
from cotix._convex_shapes import AABB, Circle, Polygon
from cotix._universal_shape import UniversalShape


def test_circle_vs_circle():
    key = jr.PRNGKey(0)
    jr.PRNGKey(3)

    def f(key):
        key1, key2, key3, key4 = jr.split(key, 4)
        pa = jr.uniform(key1, (2,)) * 4 - 2
        pb = jr.uniform(key2, (2,)) * 4 - 2
        ra = jr.uniform(key3, ()) * 2 + 1e-2
        rb = jr.uniform(key4, ()) * 2 + 1e-2

        a = Circle(ra, pa)
        b = Circle(rb, pb)

        true_no_collision = jnp.sum((pa - pb) ** 2) > (ra + rb) ** 2
        true_shift = -jnp.sqrt(jnp.sum((pa - pb) ** 2)) + (ra + rb)

        dir = b.get_center() - a.get_center()
        res_collision, simplex = check_for_collision_convex(
            a.get_support, b.get_support, dir
        )
        res_no_collision = ~res_collision
        res_shift = compute_penetration_vector_convex(
            a.get_support, b.get_support, simplex
        )

        def _c1(_):
            return true_no_collision == res_no_collision

        def _c2(pa):
            first_cond = true_no_collision == res_no_collision

            # second condition is that it is approximately closest shift
            second_cond = (
                jnp.absolute(true_shift - jnp.linalg.norm(res_shift))
                / jnp.linalg.norm(true_shift)
                < 1e-2
            )
            return first_cond & second_cond

        return jax.lax.cond(true_no_collision, _c1, _c2, pa)

    N = 10000
    keys = jr.split(key, N)
    f = jax.jit(jax.vmap(f))
    out = f(keys)
    assert jnp.all(out)


def test_rect_vs_rect():
    key = jr.PRNGKey(0)

    def f(key, flag=jnp.array(False)):
        # todo: afaiu, this is only for axis aligned rectangles
        key1, key2, key3, key4, key = jr.split(key, 5)

        min1 = jr.uniform(key1, (2,)) * 4 - 2
        w1, h1 = jr.uniform(key2, (2,)) * 2

        a = Polygon(
            jnp.array(
                [
                    min1,
                    jnp.array([min1[0] + w1, min1[1]]),
                    jnp.array([min1[0] + w1, min1[1] + h1]),
                    jnp.array([min1[0], min1[1] + h1]),
                ]
            )
        )

        min2 = jr.uniform(key3, (2,)) * 4 - 2
        w2, h2 = jr.uniform(key4, (2,)) * 2 + 0.01

        b = Polygon(
            jnp.array(
                [
                    min2,
                    jnp.array([min2[0] + w2, min2[1]]),
                    jnp.array([min2[0] + w2, min2[1] + h2]),
                    jnp.array([min2[0], min2[1] + h2]),
                ]
            )
        )

        is_first_below_second = min1[1] + h1 < min2[1]
        is_first_above_second = min2[1] + h2 < min1[1]
        is_first_left_second = min1[0] + w1 < min2[0]
        is_first_right_second = min2[0] + w2 < min1[0]

        true_no_collision = (
            is_first_below_second
            | is_first_left_second
            | is_first_above_second
            | is_first_right_second
        )

        dir = b.get_center() - a.get_center()

        res_collision, simplex = check_for_collision_convex(
            a.get_support, b.get_support, dir
        )
        penetration = compute_penetration_vector_convex(
            a.get_support, b.get_support, simplex
        )
        penetration = jnp.absolute(penetration)

        c1 = res_collision != true_no_collision
        c2 = c1 & ((penetration[0] < 1e-5) | (penetration[1] < 1e-5))
        return jax.lax.cond(true_no_collision, lambda: c1, lambda: c2)

    N = 10000
    f = jax.vmap(f)
    out = f(jr.split(key, N))

    assert jnp.all(out)


def test_aabb_vs_aabb():
    key = jr.PRNGKey(0)

    def f(key, flag=jnp.array(False)):
        key1, key2, key3, key4, key = jr.split(key, 5)

        min1 = jr.uniform(key1, (2,)) * 4 - 2
        w1, h1 = jr.uniform(key2, (2,)) * 2
        max1 = min1 + jnp.array([w1, h1])
        shape1 = Polygon(
            jnp.array(
                [
                    min1,
                    jnp.array([min1[0] + w1, min1[1]]),
                    jnp.array([min1[0] + w1, min1[1] + h1]),
                    jnp.array([min1[0], min1[1] + h1]),
                ]
            )
        )

        aabb1 = AABB(shape1)

        center2 = jr.uniform(key3, (2,)) * 4 - 2
        r2 = jr.uniform(key4, ()) * 2 + 1e-2

        shape2 = Circle(r2, center2)

        aabb2 = AABB(shape2)

        is_first_below_second = max1[1] < center2[1] - r2
        is_first_above_second = center2[1] + r2 < min1[1]
        is_first_left_second = max1[0] < center2[0] - r2
        is_first_right_second = center2[0] + r2 < min1[0]

        true_no_collision = (
            is_first_below_second
            | is_first_left_second
            | is_first_above_second
            | is_first_right_second
        )

        dir = aabb2.get_center() - aabb1.get_center()

        res_collision, simplex = check_for_collision_convex(
            aabb1.get_support, aabb2.get_support, dir
        )
        penetration = compute_penetration_vector_convex(
            aabb1.get_support, aabb2.get_support, simplex
        )
        penetration = jnp.absolute(penetration)

        c1 = res_collision != true_no_collision
        c2 = c1 & ((penetration[0] < 1e-5) | (penetration[1] < 1e-5))
        return jax.lax.cond(true_no_collision, lambda: c1, lambda: c2)

    N = 10000
    f = jax.vmap(f)
    out = f(jr.split(key, N))

    assert jnp.all(out)


@pytest.mark.parametrize("pos", [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
def test_circle_circle_aligned_positive(pos):
    a = Circle(radius=jnp.array(1.0), position=jnp.array(pos))
    b = Circle(radius=jnp.array(0.2), position=jnp.zeros((2,)))
    dir = b.get_center() - a.get_center()

    res, simplex = check_for_collision_convex(
        a.get_support, b.get_support, initial_direction=dir
    )

    penetration = compute_penetration_vector_convex(
        a.get_support, b.get_support, simplex
    )

    assert jnp.linalg.norm(penetration - jnp.array(pos) * 0.2) < 1e-3


def test_circle_vs_circle_epa_direction():
    key = jr.PRNGKey(0)
    jr.PRNGKey(3)

    def f(key):
        key1, key2, key3, key4 = jr.split(key, 4)
        pa = jr.uniform(key1, (2,)) * 4 - 2
        pb = jr.uniform(key2, (2,)) * 4 - 2
        dist = jnp.sqrt(jnp.sum((pa - pb) ** 2))
        ra = jr.uniform(key3, minval=dist * 0.25, maxval=dist * 0.75)
        # guarantee a collision
        rb = jr.uniform(key4, minval=(dist - ra) * 1.05, maxval=(dist - ra) * 1.15)

        a = Circle(ra, pa)
        b = Circle(rb, pb)

        true_no_collision = jnp.sum((pa - pb) ** 2) > (ra + rb) ** 2
        -jnp.sqrt(jnp.sum((pa - pb) ** 2)) + (ra + rb)

        dir = b.get_center() - a.get_center()
        res_collision, simplex = check_for_collision_convex(
            a.get_support, b.get_support, dir
        )
        res_no_collision = ~res_collision
        epa_vector = compute_penetration_vector_convex(
            a.get_support, b.get_support, simplex
        )

        def angle_between(v1, v2):
            v1_u = v1 / jnp.linalg.norm(v1)
            v2_u = v2 / jnp.linalg.norm(v2)
            return jnp.arccos(jnp.clip(jnp.dot(v1_u, v2_u), -1.0, 1.0))

        def _c1(_):
            return true_no_collision == res_no_collision

        def _c2(pa):
            first_cond = true_no_collision == res_no_collision

            # second condition is that epa vector is in the right direction
            second_cond = angle_between(epa_vector, pa - pb) < 1e-2
            return first_cond & second_cond

        return jax.lax.cond(true_no_collision, _c1, _c2, pa)

    N = 10000
    keys = jr.split(key, N)
    f = jax.jit(jax.vmap(f))
    out = f(keys)
    if not jnp.all(out):
        print(f"failed: {jnp.sum(~out) / N}")
    assert jnp.all(out)


def test_ball_vs_ball_epa_direction():
    key = jr.PRNGKey(0)
    jr.PRNGKey(3)

    def f(key):
        key1, key2, key3, key4 = jr.split(key, 4)
        pa = jr.uniform(key1, (2,)) * 4 - 2
        pb = jr.uniform(key2, (2,)) * 4 - 2
        dist = jnp.sqrt(jnp.sum((pa - pb) ** 2))
        ra = jr.uniform(key3, minval=dist * 0.25, maxval=dist * 0.75)
        # guarantee a collision
        rb = jr.uniform(key4, minval=(dist - ra) * 1.05, maxval=(dist - ra) * 1.15)

        zero_position = jnp.zeros((2,))
        shape1 = Circle(ra, zero_position)
        shape2 = Circle(rb, zero_position)

        true_no_collision = jnp.sum((pa - pb) ** 2) > (ra + rb) ** 2
        -jnp.sqrt(jnp.sum((pa - pb) ** 2)) + (ra + rb)

        body1 = Ball(jnp.array(1.0), pa, jnp.zeros((2,)), UniversalShape(shape1))
        body2 = Ball(jnp.array(1.0), pb, jnp.zeros((2,)), UniversalShape(shape2))

        body1.shape.wrap_local_support(body1.shape.parts[0].get_support)
        body2.shape.wrap_local_support(body2.shape.parts[0].get_support)

        dir = pb - pa
        res_collision, simplex = check_for_collision_convex(
            body1.shape.get_global_support, body2.shape.get_global_support, dir
        )
        res_no_collision = ~res_collision
        epa_vector = compute_penetration_vector_convex(
            body1.shape.get_global_support, body2.shape.get_global_support, simplex
        )

        def angle_between(v1, v2):
            v1_u = v1 / jnp.linalg.norm(v1)
            v2_u = v2 / jnp.linalg.norm(v2)
            return jnp.arccos(jnp.clip(jnp.dot(v1_u, v2_u), -1.0, 1.0))

        def _c1(_):
            return true_no_collision == res_no_collision

        def _c2(pa):
            first_cond = true_no_collision == res_no_collision

            # second condition is that epa vector is in the right direction
            second_cond = angle_between(epa_vector, pa - pb) < 1e-2
            return first_cond & second_cond

        return jax.lax.cond(true_no_collision, _c1, _c2, pa)

    N = 10000
    keys = jr.split(key, N)
    f = jax.jit(jax.vmap(f))
    out = f(keys)
    if not jnp.all(out):
        print(f"failed: {jnp.sum(~out) / N}")
    assert jnp.all(out)
