import abc

import equinox as eqx
from jax import numpy as jnp, tree_util as jtu
from jaxtyping import Array, Float

from ._bodies import AbstractBody


class AbstractPhysicsSolver(eqx.Module, strict=True):
    @abc.abstractmethod
    def step(self, bodies, dt: Float[Array, ""]):
        raise NotImplementedError


class ExplicitEulerPhysics(AbstractPhysicsSolver, strict=True):
    def _single_body_step(self, body: AbstractBody, dt: Float[Array, ""]):
        new_body = body

        new_position = body.position + body.velocity * dt
        new_angle = body.angle + body.velocity * dt

        new_body = eqx.tree_at(lambda x: x.position, new_body, replace=new_position)
        new_body = eqx.tree_at(lambda x: x.angle, new_body, replace=new_angle)
        return new_body

    def step(self, bodies, dt=jnp.nan):
        dt = eqx.error_if(
            dt,
            jnp.isnan(dt),
            "You must provide dt; if you want to use "
            "adaptive step size - don't. If you have no idea what value "
            "to put as dt, put 1e-3: probs will be good enough.",
        )

        new_bodies = jtu.tree_map(
            lambda body: self._single_body_step(body, dt=dt),
            bodies,
            is_leaf=lambda x: isinstance(x, AbstractBody),
        )

        return new_bodies, self
