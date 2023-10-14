from jax import numpy as jnp

from cotix._shapes import Polygon


def test_rect_support_vectors():
    rect1 = Polygon(
        jnp.array(
            [
                [-1, -1],
                [1, -1],
                [1, 1],
                [-1, 1],
            ]
        )
    )
    assert jnp.all(rect1._get_center() == jnp.array([0, 0]))
    assert rect1.get_support(jnp.array([1, 0]))[0] == 1

    rect2 = Polygon(
        jnp.array(
            [
                [1, 1],
                [2, 1],
                [2, 2],
                [1, 2],
            ]
        )
    )
    assert jnp.all(rect2._get_center() == jnp.array([1.5, 1.5]))
    assert rect2.get_support(jnp.array([1, 0]))[0] == 2
    assert rect2.get_support(jnp.array([0, -1]))[1] == 1
