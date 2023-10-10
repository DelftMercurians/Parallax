import equinox as eqx
import jax
from jax import numpy as jnp, random as jr

from cotix._gjk import get_collision_simplex as gcs
from cotix._shapes import Circle, Polygon


def test_circle_vs_circle():
    key = jr.PRNGKey(0)

    @eqx.filter_jit
    def f(key):
        key1, key2, key3, key4 = jr.split(key, 4)
        pa = jr.uniform(key1, (2,)) * 4 - 2
        pb = jr.uniform(key2, (2,)) * 4 - 2
        ra = jr.uniform(key3, ()) * 2 + 1e-2
        rb = jr.uniform(key4, ()) * 2 + 1e-2

        a = Circle(ra, pa)
        b = Circle(rb, pb)

        true_no_collision = jnp.sum((pa - pb) ** 2) > (ra + rb) ** 2

        res_no_collision = jnp.all(gcs(a, b, key) == jnp.zeros((3, 2)))
        return true_no_collision == res_no_collision

    N = 100000
    f = jax.jit(jax.vmap(f))
    out = f(jr.split(key, N))
    assert jnp.all(out)


def test_rect_vs_rect():
    key = jr.PRNGKey(0)

    def f(key):
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
        w2, h2 = jr.uniform(key4, (2,)) * 2

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
        res_no_collision = jnp.all(gcs(a, b, key) == jnp.zeros((3, 2)))

        return res_no_collision == true_no_collision

    N = 100000
    f = jax.jit(jax.vmap(f))
    out = f(jr.split(key, N))
    assert jnp.all(out)
