import equinox as eqx
import jax
from jax import numpy as jnp

from ._controls import AbstractControl, AbstractControlSignal
from ._worlds import AbstractWorld, AbstractWorldState


class AbstractJudge(eqx.Module):
    """
    Defines a reward in a standard fashion of:
    Reward = integrate_0^T reward(s(t), u(t)) dt + final_reward(s(t), u(t)).

    Besides, Judge defines (decides? idk) when we stop the evaluation.
    """

    def __call__(
        self, state: AbstractWorldState, control_signal: AbstractControlSignal
    ):
        raise NotImplementedError

    def is_done(self, state: AbstractWorldState, control_signal: AbstractControlSignal):
        raise NotImplementedError

    def end_reward(
        self, state: AbstractWorldState, control_signal: AbstractControlSignal
    ):
        raise NotImplementedError


class AbstractEnvironment(eqx.Module):
    world: eqx.AbstractVar[AbstractWorld]  # dynamics of the environment
    state: eqx.AbstractVar[AbstractWorldState]  # current state
    control: eqx.AbstractVar[AbstractControl]  # Control (densely)
    judge: eqx.AbstractVar[AbstractJudge]  # Gives rewards

    def eval(self, eval_period, num_NFEs, WFE_scale=10):
        def loop_body(carry, _):
            # unpack carry
            (state, control, time_per_NFE, reward), finished = carry

            # do a single NFE
            dense_control_approximation_fn, new_control = control(state)
            new_state = state
            new_control_signal = dense_control_approximation_fn(new_state)

            # check if we want to exit on zeroth iteration
            end_reward = jax.lax.cond(
                finished,
                lambda: reward,
                lambda: reward
                + +self.judge.end_reward(
                    state=new_state, control_signal=new_control_signal
                ),
            )
            possible_premature_out = (
                state,
                control,
                time_per_NFE,
                end_reward,
            )
            premature_out = possible_premature_out
            already_premature_outted = self.judge.is_done(
                state=new_state, control_signal=new_control_signal
            )

            # do a forward dynamics step a few times
            for _ in range(WFE_scale):
                # dt is fixed (for now at least, TODO)
                dt = time_per_NFE / float(WFE_scale)

                new_state = self.world.forward(
                    state=new_state, control_signal=new_control_signal, dt=dt
                )
                new_control_signal = dense_control_approximation_fn(new_state)

                # if judge thinks we are done, add the final step reward from judge,
                # and freeze the state
                # when using while loops this is done automatically,
                # but using while loops is a pain, so whatever.
                ending_reward = reward + self.judge.end_reward(
                    state=new_state, control_signal=new_control_signal
                )

                premature_out, already_premature_outted = jax.lax.cond(
                    self.judge.is_done(
                        state=new_state, control_signal=new_control_signal
                    )
                    & (~already_premature_outted),
                    lambda: (
                        (
                            new_state,
                            new_control,
                            time_per_NFE,
                            ending_reward,
                        ),
                        True,
                    ),  # if we want to exit -> update premature_out
                    lambda: (
                        premature_out,
                        already_premature_outted,
                    ),  # otherwise -> preserve it
                )

                # adding the reward for the step
                reward += (
                    self.judge(state=new_state, control_signal=new_control_signal) * dt
                )

            # restore frozen parameters, if we did freeze them
            actual_out = jax.lax.cond(
                already_premature_outted,
                lambda: (premature_out, True),
                lambda: ((new_state, new_control, time_per_NFE, reward), False),
            )
            # return new carry, and a thing to store (nothing)
            return actual_out, None

        ((end_state, _, _, reward), _), _ = jax.lax.scan(
            loop_body,
            ((self.state, self.control, eval_period / num_NFEs, jnp.array(0.0)), False),
            None,
            length=num_NFEs,
        )

        new_self = self
        new_self = eqx.tree_at(lambda x: x.state, self, end_state)
        # we don't update control, since it could have 'jumps' between evals
        # new_self = eqx.tree_at(lambda x: x.control, self, new_control)
        # a good question whether it actually makes sense to preserve something
        # except the state. TODO
        return new_self, reward
