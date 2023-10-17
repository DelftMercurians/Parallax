import jax
from jax import numpy as jnp, random as jr

from cotix._convex_shapes import Circle, Polygon
from cotix._universal_shape import UniversalShape


def test_rect_support_vectors():
    rect1 = Polygon(
        jnp.array(
            [
                [-1, -1],
                [1, -1],
                [1, 1],
                [-1, 1],
            ],
            dtype=jnp.float32,
        )
    )
    assert jnp.all(rect1.get_center() == jnp.array([0.0, 0.0]))
    assert rect1.get_support(jnp.array([1.0, 0.0]))[0] == 1

    rect2 = Polygon(
        jnp.array(
            [
                [1, 1],
                [2, 1],
                [2, 2],
                [1, 2],
            ],
            dtype=jnp.float32,
        )
    )
    assert jnp.all(rect2.get_center() == jnp.array([1.5, 1.5]))
    assert rect2.get_support(jnp.array([1.0, 0.0]))[0] == 2
    assert rect2.get_support(jnp.array([0.0, -1.0]))[1] == 1


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


def test_universal_shape_contained_doesnt_change():
    circle = Circle(jnp.array(1.0), jnp.array([0.1, 0.2]))
    square = Polygon(jnp.array([[0.1, 0.3], [0.4, 0.3], [0.6, 0.2], [-0.2, 0.1]]))
    uni = UniversalShape(circle, square)

    random_dirs = jr.normal(jr.PRNGKey(42), (100, 2))
    supports1 = jax.vmap(uni.get_global_support)(random_dirs)
    supports2 = jax.vmap(circle.get_support)(random_dirs)

    assert jnp.all(supports1 == supports2)
