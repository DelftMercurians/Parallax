from typing import List

import equinox as eqx
import jax
import pytest
from jax import numpy as jnp, random as jr
from jaxtyping import Array, Float

from cotix._bodies import AbstractBody, Ball
from cotix._controls import AbstractControl
from cotix._convex_shapes import Circle
from cotix._envs import AbstractEnvironment, AbstractJudge
from cotix._universal_shape import UniversalShape
from cotix._worlds import AbstractWorld, AbstractWorldState


class SingleBallState(AbstractWorldState):
    bodies: List[AbstractBody]
    time: Float[Array, ""]

    def __init__(self, position):
        self.bodies = [
            Ball(
                jnp.array(1.0),
                jnp.zeros((2,)),
                UniversalShape(Circle(radius=jnp.array(1.0), position=jnp.zeros((2,)))),
            )
            .set_position(jnp.array([1.0, 0.0]))
            .set_velocity(jnp.zeros((2,)))
        ]
        self.time = jnp.array(0.0)


class SingleBallWorld(AbstractWorld):
    def forward(self, state, control_signal, dt):
        state = eqx.tree_at(
            lambda x: x.bodies[0].velocity,
            state,
            state.bodies[0].velocity * jnp.exp(-dt / 10.0)
            + control_signal * (1.0 - jnp.exp(-dt / 10.0)),
        )
        state = eqx.tree_at(
            lambda x: x.bodies[0].position,
            state,
            state.bodies[0].position + state.bodies[0].velocity * dt,
        )
        state = eqx.tree_at(lambda x: x.time, state, state.time + dt)
        return state


class SingleBallControl(AbstractControl):
    weights: Float[Array, "2 15"]
    biases: Float[Array, "15"]
    final_weights: Float[Array, "15 2"]
    final_biases: Float[Array, "2"]

    def __init__(self, params):
        self.weights = params[:30].reshape((2, 15))
        self.biases = params[30:45].reshape((15,))
        self.final_weights = params[60:90].reshape((15, 2))
        self.final_biases = params[90:92]

    def __call__(self, state: AbstractWorldState):
        # so we want a simple NN that transforms
        # time into a control signal; let's say we have 10 neurons
        def dense_prediction_fn(t):
            inner = jnp.tanh(state.bodies[0].position @ self.weights + self.biases)
            out = inner @ self.final_weights + self.final_biases
            return out

        return dense_prediction_fn, self


class SingleBallJudge(AbstractJudge):
    def __call__(self, state, control_signal):
        r1 = jnp.linalg.norm(state.bodies[0].velocity)  # reward for velocity
        r2 = -(
            (jnp.linalg.norm(state.bodies[0].position) - 1) ** 2
        )  # reward for staying on circle
        r3 = jnp.cross(
            state.bodies[0].position, state.bodies[0].velocity
        )  # reward for the correct direction
        r3 = r3 / (
            1e-6
            + jnp.linalg.norm(state.bodies[0].velocity)
            * jnp.linalg.norm(state.bodies[0].position)
        )
        return r1 + r2 + r3

    def is_done(self, state, control_signal):
        return state.time >= 100.0

    def end_reward(self, state, control_signal):
        return jnp.array(0.0)


class SingleBallEnvironment(AbstractEnvironment):
    world: SingleBallWorld
    state: SingleBallState
    control: SingleBallControl
    judge: SingleBallJudge

    def __init__(self, params):
        self.world = SingleBallWorld()
        self.state = SingleBallState(jnp.zeros((2,)))
        self.control = SingleBallControl(params)
        self.judge = SingleBallJudge()


@pytest.mark.order("last")
def test_ball_going_in_circles():
    best_params = jnp.zeros((150,))

    @jax.jit
    def f():
        def loop_body(carry, _):
            i, best_reward, best_params, key = carry

            def eval_params(params):
                env = SingleBallEnvironment(params)
                env, reward = env.eval(eval_period=100.0, num_NFEs=100)
                return reward

            keys = jr.split(key, 1000)
            params = jax.vmap(
                lambda key: best_params
                + jr.normal(key, (150,)) / ((i + 1.0) * jnp.sqrt(i + 1.0))
            )(keys)
            rewards = jax.vmap(eval_params)(params)

            index = jnp.argmax(rewards)
            best_params = params[index]

            best_reward, best_params = jax.lax.cond(
                rewards[index] > best_reward,
                lambda: (rewards[index], params[index]),
                lambda: (best_reward, best_params),
            )
            key, _ = jr.split(key, 2)

            return (i + 1.0, best_reward, best_params, key), None

        carry, _ = jax.lax.scan(
            loop_body,
            (jnp.array(0.0), -1e7, best_params, jr.PRNGKey(0)),
            None,
            length=50,
        )
        return carry[1], carry[2]

    best_reward, best_params = f()

    for n in [10, 25, 50, 100, 500, 1000, 5000, 10000, 50000]:
        env = SingleBallEnvironment(best_params)
        env, reward = env.eval(eval_period=100.0, num_NFEs=n)
        print(f"{n} NFEs: {reward}")
    print(
        "You can easily observe that the reward is 'good' only for "
        "particular number of NFEs, so no time-resolution invariance"
    )

    assert best_reward > 100


class ReallyAnnoyedJudge:
    def __call__(self, state, control_signal):
        return jnp.nan

    def end_reward(self, state, control_signal):
        return jnp.array(42.0)

    def is_done(self, state, control_signal):
        return True


def test_env_instant_exit():
    env = SingleBallEnvironment(jr.normal(jr.PRNGKey(0), (150,)))
    env = eqx.tree_at(lambda x: x.judge, env, ReallyAnnoyedJudge())
    env, reward = env.eval(eval_period=100.0, num_NFEs=100)
    assert reward == 42.0
