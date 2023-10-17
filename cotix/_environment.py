import abc

import equinox as eqx
from equinox import AbstractVar
from jax import numpy as jnp
from jaxtyping import Array, Float

from ._bodies import AbstractBody, Ball
from ._collision_detectors import AbstractCollisionDetector, ConvexCollisionDetector
from ._collision_solvers import AbstractCollisionSolver, NaiveCollisionSolver
from ._design_by_contract import classinvariant
from ._physics_solvers import AbstractPhysicsSolver, ExplicitEulerPhysics


@classinvariant
class AbstractEnvironment(eqx.Module, strict=True):
    bodies: AbstractVar[list[AbstractBody]]
    solver: AbstractVar[AbstractPhysicsSolver]
    collision_solver: AbstractVar[AbstractCollisionSolver]
    collision_detector: AbstractVar[AbstractCollisionDetector]

    @abc.abstractmethod
    def step(self, dt: Float[Array, ""]):
        """
        Does a step in the environment
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __invariant__(self):
        raise NotImplementedError


class SimpleEnvironment(AbstractEnvironment, strict=True):
    """
    Represents a 'fast' (in terms of development speed),
    and 'simple' (in terms of implementation complexity)
    environment, that simply has a ball, that you can control
    """

    bodies: list[Ball]
    physics_solver: AbstractPhysicsSolver
    collision_solver: AbstractCollisionSolver
    collision_detector: AbstractCollisionDetector

    def __init__(
        self,
        initial_position: Float[Array, "2"] = jnp.zeros((2,)),
        physics_solver: AbstractPhysicsSolver = ExplicitEulerPhysics,
        collision_solver: AbstractCollisionSolver = NaiveCollisionSolver,
        collision_detector: AbstractCollisionDetector = ConvexCollisionDetector,
    ):
        self.bodies = [Ball(radius=jnp.array(1.0), position=initial_position)]

        self.physics_solver = physics_solver.set_structure(self.bodies)
        self.collision_solver = collision_solver.set_structure(self.bodies)
        self.collision_detector = collision_detector.set_structure(self.bodies)

    def step(self, dt: Float[Array, ""]):
        # Update both physics solver (since it might have stored state)
        # And the bodies positions/velocities
        updates, new_physics_solver = self.physics_solver.step(self.bodies, dt=dt)
        new_bodies = eqx.apply_updates(self.bodies, updates)

        # Detect collisions, and resolve them
        detected_collisions, metadata = self.collision_detector.detect(new_bodies)

        # detected_collisions now contains exactly N (about 10) CollisionInfo's,
        # each one of CollisionInfo contains
        # (exists (true/false), body_path_1, body_path_2, penetration depth)
        # Metadata contains some random info,
        # like whether we detected all the collisions,
        # whether we used randomized algorithm, etc, so that collisions solver could
        # use these as hints to resolve collisions more efficiently
        new_collision_solver, new_bodies = self.collision_solver.resolve(
            new_bodies, detected_collisions, metadata=metadata
        )

        # self update
        # TODO: check that these things happen inplace: otherwise
        # the performance will degrade extremely
        new_self = self
        new_self = eqx.tree_at(
            lambda x: x.collision_solver, new_self, replace=new_collision_solver
        )
        new_self = eqx.tree_at(lambda x: x.bodies, new_self, replace=new_bodies)
        new_self = eqx.tree_at(
            lambda x: x.physics_solver, new_self, replace=new_physics_solver
        )
        return new_self

    def __invariant__(self):
        return False
