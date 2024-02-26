"""
Implements physics logic that resolves a simple elastic collision between two bodies.
"""

from typing import Tuple

import equinox as eqx
import jax.lax
from jax import numpy as jnp
from jaxtyping import Array, Float

from cotix._bodies import AbstractBody
from cotix._geometry_utils import perpendicular_vector

from ._contacts import ContactInfo


class CollisionResolutionExtraInfo(eqx.Module):
    """
    Additional information about the collision that is returned by _resolve_collision()
    """

    lever_arm1: Float[Array, ""]
    lever_arm2: Float[Array, ""]
    tangential_relative_velocity: Float[Array, ""]
    primary_col_impulse: Float[Array, ""]
    friction_impulse: Float[Array, ""]
    contact_point: Float[Array, "2"]

    def __init__(
        self,
        lever_arm1,
        lever_arm2,
        tangential_relative_velocity,
        primary_col_impulse,
        friction_impulse,
        contact_point,
    ):
        self.lever_arm1 = lever_arm1
        self.lever_arm2 = lever_arm2
        self.tangential_relative_velocity = tangential_relative_velocity
        self.primary_col_impulse = primary_col_impulse
        self.friction_impulse = friction_impulse
        self.contact_point = contact_point

    @staticmethod
    def make_default():
        z = jnp.array([0.0, 0.0], dtype=jnp.float32)
        return CollisionResolutionExtraInfo(z, z, z, z, z, z)


def resolve_collision(
    body1: AbstractBody, body2: AbstractBody, contact_info: ContactInfo
) -> Tuple[AbstractBody, AbstractBody, CollisionResolutionExtraInfo]:
    """
    Resolves a collision between two bodies.
    Potentially changes velocities, positions and angular velocities of the bodies.
    Contains a check that closest points of the bodies are moving apart.
      Does nothing if they are.
    """
    return jax.lax.cond(
        contact_info.isnan(),
        lambda: (body1, body2, CollisionResolutionExtraInfo.make_default()),
        lambda: resolve_collision_notnan(body1, body2, contact_info),
    )


def apply_impulse(body, impulse, point):
    arm = point - body.get_center_of_mass()
    torque = jnp.cross(arm, impulse)
    new_vel = body.velocity + impulse / body.mass
    new_ang_vel = body.angular_velocity + torque / body.inertia
    return body.set_velocity(new_vel).set_angular_velocity(new_ang_vel)


def resolve_collision_notnan(
    body1: AbstractBody, body2: AbstractBody, contact_info: ContactInfo
) -> Tuple[AbstractBody, AbstractBody, CollisionResolutionExtraInfo]:
    contact_point = contact_info.contact_point

    contact_point_velocity_body_1 = (
        body1.velocity
        + perpendicular_vector(contact_point - body1.get_center_of_mass())
        * body1.angular_velocity
    )

    contact_point_velocity_body_2 = (
        body2.velocity
        + perpendicular_vector(contact_point - body2.get_center_of_mass())
        * body2.angular_velocity
    )

    contact_point_relative_velocity = (
        contact_point_velocity_body_2 - contact_point_velocity_body_1
    )

    normal_direction = contact_info.penetration_vector / jnp.linalg.norm(
        contact_info.penetration_vector
    )

    contact_point_normal_velocity = jnp.dot(
        contact_point_relative_velocity, normal_direction
    )

    baumgarte_term = 0.3
    elasticity = jnp.minimum(body1.elasticity, body2.elasticity)
    r1 = contact_point - body1.get_center_of_mass()
    r2 = contact_point - body2.get_center_of_mass()
    lever_arm1 = jnp.sum(r1**2)
    lever_arm2 = jnp.sum(r2**2)
    ang = lever_arm1 / body1.inertia + lever_arm2 / body2.inertia

    normal_impulse_massless = (
        -(1.0 + elasticity) * contact_point_normal_velocity
        - baumgarte_term * jnp.linalg.norm(contact_info.penetration_vector) / 0.01
    )  # / dt is missing
    normal_impulse = normal_impulse_massless / (
        1.0 / body1.mass + 1.0 / body2.mass + ang
    )
    impulse_vec = normal_impulse * normal_direction

    # compute drag
    friction_coeff = (body1.friction_coefficient + body2.friction_coefficient) / 2
    vel_drag = (
        contact_point_relative_velocity
        + contact_point_normal_velocity * normal_direction
    )
    vel_drag_unit = vel_drag / jnp.linalg.norm(vel_drag)
    impulse_drag = -jnp.linalg.norm(vel_drag) / (
        1.0 / body1.mass + 1.0 / body2.mass + ang
    )
    impulse_drag = jnp.clip(impulse_drag, 0, normal_impulse * friction_coeff)
    impulse_d_vec = impulse_drag * vel_drag_unit

    impulse_vec = impulse_vec + impulse_d_vec

    # condition to apply new impulses:
    # if the bodies are moving apart, do nothing
    cond = jnp.dot(contact_info.penetration_vector, contact_point_relative_velocity) < 0

    new_body1, new_body2 = jax.lax.cond(
        cond,
        lambda: (body1, body2),
        lambda: (
            apply_impulse(body1, -impulse_vec, contact_point),
            apply_impulse(body2, impulse_vec, contact_point),
        ),
    )
    col = CollisionResolutionExtraInfo.make_default()
    col = eqx.tree_at(lambda x: x.contact_point, col, contact_point.astype(jnp.float32))
    return (new_body1, new_body2, col)
