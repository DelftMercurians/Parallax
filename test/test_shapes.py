from jax import numpy as jnp

from cotix._geometry_utils import HomogenuousTransformer
from cotix._shapes import Polygon


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
    assert rect1._get_local_support(jnp.array([1.0, 0.0]))[0] == 1
    assert rect1._get_support(jnp.array([1.0, 0.0]), HomogenuousTransformer())[0] == 1

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
    assert rect2._get_local_support(jnp.array([1.0, 0.0]))[0] == 2
    assert rect2._get_support(jnp.array([1.0, 0.0]), HomogenuousTransformer())[0] == 2
    assert rect2._get_support(jnp.array([0.0, -1.0]), HomogenuousTransformer())[1] == 1
