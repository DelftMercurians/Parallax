import equinox as eqx
import jax
import pytest
from jax import numpy as jnp, random as jr

from cotix._colliders import RandomizedCollider
from cotix._lunar_lander import LunarLander
from cotix._physics_solvers import ExplicitEulerPhysics
from cotix._viz import Painter


@pytest.mark.skip
def test_lunar_lander():
    jax.config.update("jax_log_compiles", True)
    env = LunarLander()

    physics = ExplicitEulerPhysics()
    collider = RandomizedCollider()
    painter = Painter()

    @eqx.filter_jit
    def f(env, key):
        new_bodies, aux = physics.step(env.bodies, dt=1e-2)
        new_bodies = eqx.tree_at(
            lambda x: x[0].velocity,
            new_bodies,
            new_bodies[0].velocity + jnp.array([0.0, -0.001]),
        )
        new_bodies = collider.resolve(new_bodies, key)
        key, next_key = jr.split(key)
        env = eqx.tree_at(lambda x: x.bodies, env, new_bodies)
        painter.draw(env.bodies)
        return env, key

    key = jr.PRNGKey(0)
    for i in range(10000):
        env, key = f(env, key)


if __name__ == "__main__":
    test_lunar_lander()
