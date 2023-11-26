from jax import numpy as jnp

from cotix._bodies import AnyBody
from cotix._convex_shapes import Circle
from cotix._physics_solvers import ExplicitEulerPhysics
from cotix._universal_shape import UniversalShape


def test_simple_world():
    # creates a ball, moving to the right, and checks that it actually is moving
    # to the right
    a = AnyBody(
        position=jnp.zeros((2,)) + 1e-1,
        velocity=jnp.array([1.0, 0.0]),
        shape=UniversalShape(
            Circle(
                position=jnp.zeros(
                    2,
                ),
                radius=jnp.array(1.0),
            )
        ),
    )

    bodies = [a]

    solver = ExplicitEulerPhysics()

    for i in range(100):
        bodies, _ = solver.step(bodies, dt=1e-1)

    assert bodies[0].position[0] > 2e-1
    assert (bodies[0].position[1] < 1e-1 + 1e-2) & (bodies[0].position[1] > 1e-1 - 1e-2)
    assert (bodies[0].shape.parts[0].position[0] < 1e-2) & (
        bodies[0].shape.parts[0].position[0] > -1e-2
    )
    assert (bodies[0].shape.parts[0].position[1] < 1e-2) & (
        bodies[0].shape.parts[0].position[1] > -1e-2
    )
    assert jnp.linalg.norm(bodies[0].velocity[0] - 1.0) < 1e-2


if __name__ == "__main__":
    test_simple_world()
