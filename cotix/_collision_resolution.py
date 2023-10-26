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
    # lets apply translation to both bodies, taking their mass into account
    body1 = body1.set_position(
        body1.position + epa_vector * (body2.mass / (body1.mass + body2.mass))
    )
    body2 = body2.set_position(
        body2.position - epa_vector * (body1.mass / (body1.mass + body2.mass))
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

    v1_col_basis = change_of_basis @ body1.velocity
    v2_col_basis = change_of_basis @ body2.velocity

    v_rel = v1_col_basis - v2_col_basis
    col_impulse = -(1 + elasticity) / (body1.mass + body2.mass) * v_rel

    v1_new_col_basis = v1_col_basis + col_impulse / body1.mass
    v2_new_col_basis = v2_col_basis - col_impulse / body2.mass

    # the contact point is set to be exactly between
    # furthest (penetrating) points of the bodies along the collision direction
    contact_point = (
        body1.shape.get_global_support(unit_collision_vector)
        + body2.shape.get_global_support(-unit_collision_vector)
    ) / 2

    r1 = contact_point - body1.get_center_of_mass()
    r2 = contact_point - body2.get_center_of_mass()

    # ok so there is a cross product solution,
    # (which should work identically to the one below),
    # but i dont think it is intuitive,
    # so im gonna do it manually with the lever arm computation
    # # 2d cross product is v0.x * v1.y - v0.y * v1.x
    # omega1_new = body1.angular_velocity + inv_I1 * jnp.cross(r1, col_impulse)
    # omega2_new = body2.angular_velocity - inv_I2 * jnp.cross(r2, col_impulse)

    lever_arm1 = jnp.dot(r1, perpendicular)
    lever_arm2 = jnp.dot(r2, perpendicular)

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
