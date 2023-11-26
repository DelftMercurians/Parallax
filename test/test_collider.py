import equinox as eqx
from jax import numpy as jnp, random as jr

from cotix._bodies import AnyBody
from cotix._colliders import NaiveCollider
from cotix._convex_shapes import AABB, Circle
from cotix._physics_solvers import ExplicitEulerPhysics
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

    new_bodies = collider.resolve(bodies)

    # check that positions are sufficiently different
    assert jnp.linalg.norm(bodies[0].position - new_bodies[0].position) > 0.1
    assert jnp.linalg.norm(bodies[1].position - new_bodies[1].position) > 0.1


def test_a_huge_chunk_of_balls():
    # just test compilation speed: if this can be compiled less than a minute,
    # good enough ig :)
    balls = []
    for i in range(20):
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

    collider = NaiveCollider()
    eqx.filter_jit(collider.resolve)(balls)
    assert True


def test_two_ball_long():
    # two balls move towards each other, and collide
    # let's check that at some point they are moving in the opposite direction

    a = AnyBody(
        position=jnp.array([-1.86, 0.0]),
        velocity=jnp.array([1.0, 0.0]),
        elasticity=jnp.array(1.0),
        shape=UniversalShape(
            Circle(
                position=jnp.zeros(
                    2,
                ),
                radius=jnp.array(1.0),
            )
        ),
    )
    b = AnyBody(
        position=jnp.array([2.784, 0.0]),
        velocity=jnp.array([-1.51, 0.0]),
        elasticity=jnp.array(1.0),
        shape=UniversalShape(
            Circle(
                position=jnp.zeros(
                    2,
                ),
                radius=jnp.array(1.0),
            )
        ),
    )

    bodies = [a, b]
    physics_solver = ExplicitEulerPhysics()
    collider = NaiveCollider()

    for i in range(100):
        bodies, _ = eqx.filter_jit(physics_solver.step)(bodies, dt=1e-1)
        bodies = eqx.filter_jit(collider.resolve)(bodies)

    # check that positions are sufficiently different
    assert jnp.linalg.norm(bodies[0].position - bodies[1].position) > 5.0
    # check that first ball is moving to the left
    assert bodies[0].velocity[0] < -0.8
    # check that second ball is moving to the right
    assert bodies[1].velocity[0] > 0.8


if __name__ == "__main__":
    test_simple_world()
    test_a_huge_chunk_of_balls()
    test_two_ball_long()
