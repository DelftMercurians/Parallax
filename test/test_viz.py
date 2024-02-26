import equinox as eqx
import jax
import pytest
from jax import numpy as jnp, random as jr
from tqdm import tqdm as tqdm

from cotix._colliders import RandomizedCollider
from cotix._constraint_solvers import SimpleConstraintSolver
from cotix._lunar_lander import LunarLander
from cotix._physics_solvers import ExplicitEulerPhysics
from cotix._robocup import RoboCupEnv
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
            new_bodies[0].velocity + jnp.array([0.0, -0.002]),
        )

        def draw_log(log):
            pos = log.contact_point
            painter.draw_circle(pos, 0.1, (200, 0, 0))

        new_bodies = collider.resolve(new_bodies, key, draw_log)
        # new_bodies = constraintSolver.solve(new_bodies, env.constraints)
        key, next_key = jr.split(key)
        env = eqx.tree_at(lambda x: x.bodies, env, new_bodies)
        env = env.step()

        env.draw(painter)
        return env, key

    key = jr.PRNGKey(0)
    for i in range(10000):
        env, key = f(env, key)


def test_robocup_env():
    jax.config.update("jax_log_compiles", True)
    env = RoboCupEnv()

    physics = ExplicitEulerPhysics()
    collider = RandomizedCollider()
    constraintSolver = SimpleConstraintSolver(loops=1)
    painter = Painter()

    @eqx.filter_jit
    def f(env, key):
        new_bodies, aux = physics.step(env.bodies, dt=1e-2)
        new_bodies = collider.resolve(new_bodies, key)
        new_bodies = constraintSolver.solve(new_bodies, env.constraints)
        key, next_key = jr.split(key)
        env = eqx.tree_at(lambda x: x.bodies, env, new_bodies)
        painter.draw_env(env)
        return env, key

    key = jr.PRNGKey(0)
    for i in tqdm(range(3000)):
        env, key = f(env, key)


if __name__ == "__main__":
    test_lunar_lander()
