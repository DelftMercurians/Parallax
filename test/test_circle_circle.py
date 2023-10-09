from jax import numpy as jnp, random as jr

from cotix._shapes import _get_collision_simplex as gcs, Circle


def test_circle_circle():
    key = jr.PRNGKey(0)
    for i in range(100):
        key1, key2, key3, key4, key = jr.split(key, 5)
        pa = jr.uniform(key1, (2,)) * 4 - 2
        pb = jr.uniform(key2, (2,)) * 4 - 2
        ra = jr.uniform(key3, ()) * 2 + 1e-2
        rb = jr.uniform(key4, ()) * 2 + 1e-2

        a = Circle(ra, pa)
        b = Circle(rb, pb)
        true_no_collision = jnp.sum((pa - pb) ** 2) > (ra + rb) ** 2

        res_no_collision = jnp.all(gcs(a, b, key) == jnp.zeros((3, 2)))
        assert res_no_collision == true_no_collision
