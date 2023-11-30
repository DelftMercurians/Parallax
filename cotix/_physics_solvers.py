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
        new_angle = body.angle + body.angular_velocity * dt

        new_body = new_body.set_position(new_position).set_angle(new_angle)
        return new_body

    def step(self, bodies, dt=jnp.nan):
        new_bodies = jtu.tree_map(
            lambda body: self._single_body_step(body, dt=dt),
            bodies,
            is_leaf=lambda x: isinstance(x, AbstractBody),
        )

        return new_bodies, self
