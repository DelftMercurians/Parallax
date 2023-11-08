import jax
from jax import numpy as jnp, random as jr

from cotix._convex_shapes import Circle
from cotix._universal_shape import UniversalShape


def test_universal_shape_support_equivalence():
    circle = Circle(jnp.array(0.1), jnp.array([0.1, 0.2]))
    uni = UniversalShape(circle)

    random_dirs = jr.normal(jr.PRNGKey(42), (100, 2))
    supports1 = jax.vmap(uni.get_global_support)(random_dirs)
    supports2 = jax.vmap(circle.get_support)(random_dirs)

    assert jnp.all(supports1 == supports2)


def test_universal_shape_double_support_correctness():
    c1 = Circle(jnp.array(0.5), jnp.array([-10.0, 0.0]))
    c2 = Circle(jnp.array(1.0), jnp.array([1.0, 1.0]))
    uni = UniversalShape(c1, c2)

    assert jnp.all(
        uni.get_global_support(jnp.array([1.0, 0.0])) == jnp.array([2.0, 1.0])
    )
    assert jnp.all(
        uni.get_global_support(jnp.array([-1.0, 0.0])) == jnp.array([-10.5, 0.0])
    )
    assert jnp.all(
        uni.get_global_support(jnp.array([0.0, 1.0])) == jnp.array([1.0, 2.0])
    )
    assert jnp.all(
        uni.get_global_support(jnp.array([0.0, -1.0])) == jnp.array([-10.0, -0.5])
    )
