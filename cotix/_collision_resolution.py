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


class ContactInfo(eqx.Module):
    """
    Additional information about the collision that is expected by _resolve_collision()
    """

    penetration_vector: Float[Array, "2"]
    contact_point: Float[Array, "2"]

    def __init__(self, penetration_vector, contact_point):
        self.penetration_vector = penetration_vector
        self.contact_point = contact_point


class CollisionResolutionExtraInfo(eqx.Module):
    lever_arm1: Float[Array, ""]
    lever_arm2: Float[Array, ""]
    tangential_relative_velocity: Float[Array, ""]
    primary_col_impulse: Float[Array, ""]
    friction_impulse: Float[Array, ""]

    def __init__(
        self,
        lever_arm1,
        lever_arm2,
        tangential_relative_velocity,
        primary_col_impulse,
        friction_impulse,
    ):
        self.lever_arm1 = lever_arm1
        self.lever_arm2 = lever_arm2
        self.tangential_relative_velocity = tangential_relative_velocity
        self.primary_col_impulse = primary_col_impulse
        self.friction_impulse = friction_impulse

    @staticmethod
    def make_default():
        z = jnp.array(0.0)
        return CollisionResolutionExtraInfo(z, z, z, z, z)


def _move_one_body(primary_body, secondary_body, split_portion, penetration_vector):
    # if one body has mass inf, the other body should move the entire distance,
    #   but np.inf/np.inf = nan
    primary_body = jax.lax.cond(
        secondary_body.mass == jnp.array(jnp.inf),
        lambda: primary_body.set_position(
            primary_body.position + split_portion * penetration_vector
        ),
        lambda: primary_body.set_position(
            primary_body.position
            + split_portion
            * penetration_vector
            * (secondary_body.mass / (primary_body.mass + secondary_body.mass))
        ),
    )
    return primary_body


def _split_bodies(
    body1: AbstractBody, body2: AbstractBody, penetration_vector: Float[Array, "2"]
) -> Tuple[AbstractBody, AbstractBody]:
    """
    Moves bodies apart so that they intersect less.
    Takes masses into account.
    """
    body1 = eqx.error_if(
        body1,
        (body1.mass == jnp.array(jnp.inf)) & (body2.mass == jnp.array(jnp.inf)),
        "Both bodies have infinite mass",
    )
    # it may be not good idea to split 100% of the penetration,
    #   but we can change that later
    split_portion = 1.0
    # apply translation to both bodies, taking their mass into account
    body1 = _move_one_body(body1, body2, split_portion, penetration_vector)
    body2 = _move_one_body(body2, body1, split_portion, -penetration_vector)
    return body1, body2


def _compute_lever_arms_and_tangential_velocity(
    change_of_basis, contact_point, body1, body2
) -> Tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """
    Auxiliary function that computes lever arms and tangential velocity.
    Lever arm is a scalar that determines rotational contribution
    of the primary impact force.
    Tangential velocity is a scalar that helps to compute frictional impulse.
    """
    # transform the received contact point to the new coordinate system
    contact_point = change_of_basis @ contact_point

    relative_contact_point1 = (
        contact_point - change_of_basis @ body1.get_center_of_mass()
    )
    relative_contact_point2 = (
        contact_point - change_of_basis @ body2.get_center_of_mass()
    )

    # lever arms can be negative, but i think it makes sense
    lever_arm1 = relative_contact_point1[1]
    lever_arm2 = relative_contact_point2[1]

    contact_point_speed1 = (
        jnp.dot(body1.velocity, change_of_basis[1])
        + relative_contact_point1[0] * body1.angular_velocity
    )
    contact_point_speed2 = (
        jnp.dot(body2.velocity, change_of_basis[1])
        + relative_contact_point2[0] * body2.angular_velocity
    )
    tangential_relative_velocity = contact_point_speed1 - contact_point_speed2
    return lever_arm1, lever_arm2, tangential_relative_velocity


def _resolve_collision(
    body1: AbstractBody, body2: AbstractBody, contact_info: ContactInfo
) -> Tuple[AbstractBody, AbstractBody, CollisionResolutionExtraInfo]:
    """
    Resolves a collision between two bodies.
    Potentially changes velocities, positions and angular velocities of the bodies.
    Contains a check that closest points of the bodies are moving apart.
      Does nothing if they are.
    """
    penetration_vector = contact_info.penetration_vector
    contact_point = contact_info.contact_point
    elasticity = body1.elasticity * body2.elasticity
    friction_coefficient = body1.friction_coefficient * body2.friction_coefficient

    # change coordinate system from (x, y) to (q, r)
    # where [0] is the line along epa_vector and [1] is perpendicular to it
    unit_collision_vector = penetration_vector / jnp.linalg.norm(penetration_vector)
    perpendicular = perpendicular_vector(unit_collision_vector)
    change_of_basis = jnp.array([unit_collision_vector, perpendicular])
    change_of_basis_inv = jnp.linalg.inv(change_of_basis)

    # everything below should be in the new coordinate system
    v1_col_basis = change_of_basis @ body1.velocity
    v2_col_basis = change_of_basis @ body2.velocity

    v_rel = v1_col_basis - v2_col_basis

    (
        lever_arm1,
        lever_arm2,
        tangential_relative_velocity,
    ) = _compute_lever_arms_and_tangential_velocity(
        change_of_basis, contact_point, body1, body2
    )

    # check that the collision has to be resolved
    point1_v = body1.velocity + body1.angular_velocity * lever_arm1
    point2_v = body2.velocity + body2.angular_velocity * lever_arm2
    # todo: this might not work with super concave bodies,
    #  when the center of mass is on the wrong side of the collision point
    velocities_away = jnp.dot(point1_v - point2_v, body1.position - body2.position) >= 0

    # called iff velocities_away is False
    def continue_resolution(body1, body2):
        # impulse computation is in accordance with
        # https://github.com/knyazer/RSS/blob/
        # 1246e03c5950a5549a128fbce97c7bd402f9bed7/engine/source/env/World.cpp#L87

        # inertia is kg * m^2, so this is kg^-1
        impulseFactor1 = (1 / body1.mass) + (lever_arm1**2) / body1.inertia
        impulseFactor2 = (1 / body2.mass) + (lever_arm2**2) / body2.inertia

        # here, we only care about the collision along the collision normal
        # units are kg * m / s
        primary_col_impulse = (
            (1 + elasticity) * v_rel[0] / (impulseFactor1 + impulseFactor2)
        )

        # v1 dot perpendicular tries to be the same as v2 dot perpendicular.
        #   clipped to be in the range [-friction_impulse_max, friction_impulse_max]
        friction_impulse_max = friction_coefficient * jnp.abs(primary_col_impulse)
        # this adds an impulse perpendicular to the collision normal
        friction_impulse = jnp.clip(
            # I am not too sure that the formula is correct here
            tangential_relative_velocity / (impulseFactor1 + impulseFactor2),
            -friction_impulse_max,
            friction_impulse_max,
        )

        col_impulse = -jnp.array([primary_col_impulse, friction_impulse])

        # jax.debug.print(
        #     "\nlever arms: {lever_arm1}, {lever_arm2}. \n"
        #     "center1: {center1}, center2: {center2}. "
        #     "contact_point: {contact_point}. \n"
        #     "global_supports {sup1}, {sup2}. "
        #     "collision_unit_vector {unit_collision_vector}. \n"
        #     "tangential_relative_velocity: {tangential_relative_velocity}. \n"
        #     "friction_impulse: {friction_impulse}. \n"
        #     "col_impulse in new basis: {col_impulse}. \n"
        #     "col_impulse original: {col_impulse_original}. \n",
        #     lever_arm1=lever_arm1,
        #     lever_arm2=lever_arm2,
        #     center1=body1.get_center_of_mass(),
        #     center2=body2.get_center_of_mass(),
        #     contact_point=contact_point,
        #     sup1=body1.shape.get_global_support(-unit_collision_vector),
        #     sup2=body2.shape.get_global_support(unit_collision_vector),
        #     unit_collision_vector=unit_collision_vector,
        #     tangential_relative_velocity=tangential_relative_velocity,
        #     friction_impulse=friction_impulse,
        #     col_impulse=col_impulse,
        #     col_impulse_original=change_of_basis_inv @ col_impulse,
        # )

        v1_new_col_basis = v1_col_basis + col_impulse / body1.mass
        v2_new_col_basis = v2_col_basis - col_impulse / body2.mass

        # col_impulse[0] is along collision normal
        body1 = body1.set_angular_velocity(
            body1.angular_velocity
            + (lever_arm1 * col_impulse[0]) / body1.inertia
            - col_impulse[1] / body1.inertia
        )
        body2 = body2.set_angular_velocity(
            body2.angular_velocity
            + (lever_arm2 * col_impulse[0]) / body2.inertia
            - col_impulse[1] / body2.inertia
        )
        # the rotational force is applied with the same sign to both bodies.
        #   this is because although the force acts in the opposite directions
        #   on the bodies, it contributes to rotation in the same direction,
        #   as the contact point is on the opposite sides of the center of mass

        v1_new = change_of_basis_inv @ v1_new_col_basis
        v2_new = change_of_basis_inv @ v2_new_col_basis

        body1 = body1.set_velocity(v1_new)
        body2 = body2.set_velocity(v2_new)

        body1, body2 = _split_bodies(body1, body2, penetration_vector)

        extra_info = CollisionResolutionExtraInfo(
            lever_arm1=lever_arm1,
            lever_arm2=lever_arm2,
            tangential_relative_velocity=tangential_relative_velocity,
            primary_col_impulse=primary_col_impulse,
            friction_impulse=friction_impulse,
        )
        return body1, body2, extra_info

    return jax.lax.cond(
        velocities_away,
        lambda: (body1, body2, CollisionResolutionExtraInfo.make_default()),
        lambda: continue_resolution(body1, body2),
    )
