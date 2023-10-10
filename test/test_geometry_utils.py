import jax
from jax import numpy as jnp, random as jr

from cotix._geometry_utils import order_clockwise


def test_order_clockwise_consistency():
    """
    Test that the ordering is consistent if input shuffled.
    """
    key = jr.PRNGKey(42)
    vertices = jr.uniform(key, (100, 2))
    jitted_order = jax.jit(order_clockwise)
    sorted_starting = jitted_order(vertices)

    for i in range(100):
        key, _ = jr.split(key)
        shuffled = jr.permutation(key, vertices, axis=0, independent=False)
        sorted_again = jitted_order(shuffled)

        assert jnp.all(sorted_again == sorted_starting)
