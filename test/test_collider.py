from jax import numpy as jnp

from cotix._bodies import Ball
from cotix._colliders import NaiveCollider
from cotix._convex_shapes import AABB, Circle
from cotix._universal_shape import UniversalShape


def test_simple_world_broad_phase():
    a = Ball(
        jnp.array(1.0),
        jnp.zeros((2,)),
        UniversalShape(
            Circle(
                position=jnp.zeros(
                    2,
                ),
                radius=jnp.array(1.0),
            ),
            Circle(
                position=jnp.zeros(
                    2,
                ),
                radius=jnp.array(1.0),
            ),
            Circle(
                position=jnp.zeros(
                    2,
                ),
                radius=jnp.array(1.0),
            ),
        ),
    )
    b = Ball(
        jnp.array(1.0),
        jnp.zeros((2,)),
        UniversalShape(
            AABB(
                Circle(
                    position=jnp.ones(
                        2,
                    ),
                    radius=jnp.array(1.0),
                )
            ),
            AABB(
                Circle(
                    position=jnp.ones(
                        2,
                    ),
                    radius=jnp.array(1.0),
                )
            ),
            AABB(
                Circle(
                    position=jnp.ones(
                        2,
                    ),
                    radius=jnp.array(1.0),
                )
            ),
        ),
    )

    bodies = [a, b]

    collider = NaiveCollider()

    out = collider.broad_phase(bodies, limit=4)

    assert out[0].i == 0 and out[0].j == 1  # first collision is detected
    assert (
        (out[1].i == -1) and (out[2].i == -1) and (out[3].i == -1)
    )  # all other collisions are 'empty'


def test_simple_world_narrow_phase():
    a = Ball(
        jnp.array(1.0),
        jnp.zeros((2,)),
        UniversalShape(
            Circle(
                position=jnp.zeros(
                    2,
                ),
                radius=jnp.array(1.0),
            ),
            Circle(
                position=jnp.zeros(
                    2,
                ),
                radius=jnp.array(1.0),
            ),
            Circle(
                position=jnp.zeros(
                    2,
                ),
                radius=jnp.array(1.0),
            ),
        ),
    )
    b = Ball(
        jnp.array(1.0),
        jnp.zeros((2,)),
        UniversalShape(
            AABB(
                Circle(
                    position=jnp.ones(
                        2,
                    ),
                    radius=jnp.array(1.0),
                )
            ),
            AABB(
                Circle(
                    position=jnp.ones(
                        2,
                    ),
                    radius=jnp.array(1.0),
                )
            ),
            AABB(
                Circle(
                    position=jnp.ones(
                        2,
                    ),
                    radius=jnp.array(1.0),
                )
            ),
        ),
    )

    bodies = [a, b]

    collider = NaiveCollider()

    out = collider.total_phase(bodies, limit=3)

    assert out[0].i == 0 and out[0].j == 1  # first collision is detected
    assert (
        (out[1].i == -1) and (out[2].i == -1) and (out[3].i == -1)
    )  # all other collisions are 'empty'
