import jax
from jax import numpy as jnp, random as jr

from cotix._collisions import check_for_collision, compute_penetration_vector
from cotix._shapes import AABB, Circle, Polygon


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

        res_collision, simplex = check_for_collision(a.get_support, b.get_support)
        res_no_collision = ~res_collision
        res_shift = compute_penetration_vector(a.get_support, b.get_support, simplex)

        def _c1(_):
            return true_no_collision == res_no_collision

        def _c2(pa):
            first_cond = true_no_collision == res_no_collision

            # second condition is that it is approximately closest shift
            second_cond = jnp.absolute(true_shift - jnp.linalg.norm(res_shift)) < 1e-2
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

        res_collision, simplex = check_for_collision(a.get_support, b.get_support)
        penetration = compute_penetration_vector(a.get_support, b.get_support, simplex)
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

        res_collision, simplex = check_for_collision(
            aabb1.get_support, aabb2.get_support
        )
        penetration = compute_penetration_vector(
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
