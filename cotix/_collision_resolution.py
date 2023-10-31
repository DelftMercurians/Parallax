"""
Implements physics logic that resolves a simple elastic collision between two bodies.
"""


from typing import Tuple

import jax.lax
from jax import numpy as jnp
from jaxtyping import Array, Float

from cotix._bodies import AbstractBody
from cotix._geometry_utils import perpendicular_vector


def _split_bodies(
    body1: AbstractBody, body2: AbstractBody, epa_vector: Float[Array, "2"]
) -> Tuple[AbstractBody, AbstractBody]:
    # it may be not good idea to split 100% of the penetration,
    #   but we can change that later
    split_portion = 1.0
    # let's apply translation to both bodies, taking their mass into account
    body1 = body1.set_position(
        body1.position
        + split_portion * epa_vector * (body2.mass / (body1.mass + body2.mass))
    )
    body2 = body2.set_position(
        body2.position
        - split_portion * epa_vector * (body1.mass / (body1.mass + body2.mass))
    )
    return body1, body2


def _resolve_collision_checked(
    body1: AbstractBody, body2: AbstractBody, epa_vector: Float[Array, "2"]
) -> Tuple[AbstractBody, AbstractBody]:
    return jax.lax.cond(
        jnp.dot(body1.velocity - body2.velocity, epa_vector) >= 0.0,
        lambda: (body1, body2),
        lambda: _resolve_collision(body1, body2, epa_vector),
    )


def _resolve_collision(
    body1: AbstractBody, body2: AbstractBody, epa_vector: Float[Array, "2"]
) -> Tuple[AbstractBody, AbstractBody]:
    elasticity = body1.elasticity * body2.elasticity

    # change coordinate system from (x, y) to (q, r)
    # where [0] is the line along epa_vector and [1] is perpendicular to it
    unit_collision_vector = epa_vector / jnp.linalg.norm(epa_vector)
    perpendicular = perpendicular_vector(unit_collision_vector)
    change_of_basis = jnp.array([unit_collision_vector, perpendicular])
    change_of_basis_inv = jnp.linalg.inv(change_of_basis)

    # everything below should be in the new coordinate system
    change_of_basis @ unit_collision_vector
    perpendicular_new_basis = change_of_basis @ perpendicular
    v1_col_basis = change_of_basis @ body1.velocity
    v2_col_basis = change_of_basis @ body2.velocity

    v_rel = v1_col_basis - v2_col_basis

    # the contact point is set to be exactly between
    # furthest (penetrating) points of the bodies
    #   along the collision direction
    # get_global_support doesnt know about the new basis,
    # so we use a vector from the old basis
    contact_point = (
        body1.shape.get_global_support(-unit_collision_vector)
        + body2.shape.get_global_support(unit_collision_vector)
    ) / 2
    contact_point = change_of_basis @ contact_point

    relative_contact_point1 = (
        contact_point - change_of_basis @ body1.get_center_of_mass()
    )
    relative_contact_point2 = (
        contact_point - change_of_basis @ body2.get_center_of_mass()
    )

    lever_arm1 = jnp.dot(relative_contact_point1, perpendicular_new_basis)
    lever_arm2 = jnp.dot(relative_contact_point2, perpendicular_new_basis)

    # jax.debug.print(
    #     "\nrelative_contact_points: "
    #     "{relative_contact_point1}, {relative_contact_point2}. "
    #     "perpendicular: {perpendicular}, "
    #     "lever arms: {lever_arm1}, {lever_arm2}. \n"
    #     "center1: {center1}, center2: {center2}. "
    #     "contact_point: {contact_point}. \n"
    #     "global_supports {sup1}, {sup2}. "
    #     "collision_unit_vector {unit_collision_vector}. \n",
    #     relative_contact_point1=relative_contact_point1,
    #     relative_contact_point2=relative_contact_point2,
    #     perpendicular=perpendicular_new_basis,
    #     lever_arm1=lever_arm1,
    #     lever_arm2=lever_arm2,
    #     center1=body1.get_center_of_mass(),
    #     center2=body2.get_center_of_mass(),
    #     contact_point=change_of_basis_inv @ contact_point,
    #     sup1=body1.shape.get_global_support(-unit_collision_vector),
    #     sup2=body2.shape.get_global_support(unit_collision_vector),
    #     unit_collision_vector=unit_collision_vector,
    # )

    # impulse computation is in accordance with
    # https://github.com/knyazer/RSS/blob/
    # 1246e03c5950a5549a128fbce97c7bd402f9bed7/engine/source/env/World.cpp#L87

    # inertia is kg * m^2, so this is kg^-1
    impulseFactor1 = (1 / body1.mass) + (lever_arm1**2) / body1.inertia
    impulseFactor2 = (1 / body2.mass) + (lever_arm2**2) / body2.inertia

    # this is a vector because v_rel is a vector
    # units are kg * m / s
    col_impulse = -(1 + elasticity) * v_rel / (impulseFactor1 + impulseFactor2)

    v1_new_col_basis = v1_col_basis + col_impulse / body1.mass
    v2_new_col_basis = v2_col_basis - col_impulse / body2.mass

    # col_impulse[0] is along collision normal
    body1 = body1.set_angular_velocity(
        body1.angular_velocity + (lever_arm1 * col_impulse[0]) / body1.inertia
    )
    body2 = body2.set_angular_velocity(
        body2.angular_velocity - (lever_arm2 * col_impulse[0]) / body2.inertia
    )

    v1_new = change_of_basis_inv @ v1_new_col_basis
    v2_new = change_of_basis_inv @ v2_new_col_basis

    body1 = body1.set_velocity(v1_new)
    body2 = body2.set_velocity(v2_new)

    body1, body2 = _split_bodies(body1, body2, epa_vector)
    return body1, body2


def _1d_elastic_collision_velocities(m1, m2, u1, u2):
    v1 = ((m1 - m2) / (m1 + m2)) * u1 + ((2 * m2) / (m1 + m2)) * u2
    v2 = ((2 * m1) / (m1 + m2)) * u1 + ((m2 - m1) / (m1 + m2)) * u2
    return v1, v2
