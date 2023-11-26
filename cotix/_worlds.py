from typing import List

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, Float

from ._bodies import AbstractBody
from ._colliders import AbstractCollider, NaiveCollider
from ._design_by_contract import class_invariant
from ._physics_solvers import AbstractPhysicsSolver, ExplicitEulerPhysics


class AbstractWorldState(eqx.Module, strict=True):
    """
    World state is separate from actual World, because I want it to.
    In practice though, I guess world describes a dynamics of the system:
    so it is simply a wrapper of a function, while state is actually used everywhere.
    So it seems adequate to separate them.
    """

    time: eqx.AbstractVar[Float[Array, ""]]
    bodies: eqx.AbstractVar[List[AbstractBody]]


class AbstractWorld(eqx.Module):
    """
    Simply forces the dynamics to be able to propagate forward. That is the only
    requirement for the world to work.

    **Important:** world is (sort of) stateless, it just describes the dynamics.
    Like, ok, it has some state, but this state is constant during forward dynamics
    eval, so we consider the world to be stateless, cuz state don't change, idk.
    """

    def forward(self, state: AbstractWorldState, control_signal, dt):
        raise NotImplementedError


class SimpleWorldState(AbstractWorldState, strict=True):
    time: Float[Array, ""]
    bodies: List[AbstractBody]

    def __init__(self, bodies):
        self.time = jnp.array(0.0)
        self.bodies = bodies


@class_invariant
class SimpleWorld(AbstractWorld):
    """
    Simple world is a world, which just 'works' in a naive fashion:
    resolves all the collisions, moves all the objects, etc,
    according to whatever laws we put into it. But only basic ones.
    So probably no friction, etc.
    """

    physics_solver: AbstractPhysicsSolver
    collider: AbstractCollider

    def __init__(
        self,
        physics_solver: AbstractPhysicsSolver = ExplicitEulerPhysics(),
        collider: AbstractCollider = NaiveCollider(),
    ):
        self.physics_solver = physics_solver
        self.collider = collider

    def forward(self, state: AbstractWorldState, control_signal, dt: Float[Array, ""]):
        new_state = control_signal.apply(state, dt)

        new_bodies = state.bodies
        new_bodies, new_physics_solver = self.physics_solver.step(new_bodies, dt=dt)
        new_bodies, new_collider = self.collider.resolve(new_bodies, dt=dt)
        new_state = eqx.apply_updates(state.bodies, new_bodies)

        return new_state

    def __invariant__(self):
        return False
