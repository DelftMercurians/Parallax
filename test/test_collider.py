import equinox as eqx
import jax
from jax import numpy as jnp, random as jr

from cotix._bodies import AnyBody
from cotix._colliders import NaiveCollider
from cotix._convex_shapes import AABB, Circle
from cotix._universal_shape import UniversalShape


def test_simple_world():
    a = AnyBody(
        position=jnp.zeros((2,)) + 1e-1,
        velocity=jnp.array([1.0, 0.0]),
        shape=UniversalShape(
            Circle(
                position=jnp.zeros(
                    2,
                )
                + 1e-1,
                radius=jnp.array(1.0),
            )
        ),
    )
    b = AnyBody(
        position=jnp.ones((2,)),
        shape=UniversalShape(
            AABB.of(
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

    eqx.filter_jit(collider.resolve)(bodies)

    assert True


def test_a_huge_chunk_of_balls():
    balls = []
    for i in range(40):
        balls.append(
            AnyBody(
                position=jnp.zeros((2,)) + 1e-1,
                velocity=jnp.array([1.0, 0.0]),
                shape=UniversalShape(
                    Circle(
                        position=jr.normal(
                            jr.PRNGKey(i),
                            (2,),
                        ),
                        radius=jnp.array(0.5),
                    )
                ),
            )
        )

    jax.config.update("jax_log_compiles", True)
    collider = NaiveCollider()
    eqx.filter_jit(collider.resolve)(balls)
    assert True


if __name__ == "__main__":
    test_simple_world()
    test_a_huge_chunk_of_balls()
