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
    # todo: we need to determine the elasticity of the collision.
    #  probably based on the body properties
    # todo: angular momentum

    elasticity = 1

    # change coordinate system from (x, y) to (q, r)
    # where q is the line along epa_vector and r is perpendicular to it
    unit_collision_vector = epa_vector / jnp.linalg.norm(epa_vector)
    perpendicular = perpendicular_vector(unit_collision_vector)
    change_of_basis = jnp.array([unit_collision_vector, perpendicular])
    change_of_basis_inv = jnp.linalg.inv(change_of_basis)

    v1q = jnp.dot(body1.velocity, unit_collision_vector)
    v1r = jnp.dot(body1.velocity, perpendicular)  # stays constant
    v2q = jnp.dot(body2.velocity, unit_collision_vector)
    v2r = jnp.dot(body2.velocity, perpendicular)  # stays constant

    v1q_new, v2q_new = _1d_elastic_collision_velocities(
        body1.mass, body2.mass, v1q, v2q
    )

    v1qr_new = jnp.array([v1q_new * elasticity, v1r])
    v2qr_new = jnp.array([v2q_new * elasticity, v2r])

    v1_new = jnp.matmul(change_of_basis_inv, v1qr_new)
    v2_new = jnp.matmul(change_of_basis_inv, v2qr_new)

    body1 = body1.set_velocity(v1_new)
    body2 = body2.set_velocity(v2_new)

    body1, body2 = _split_bodies(body1, body2, epa_vector)
    return body1, body2


def _1d_elastic_collision_velocities(m1, m2, u1, u2):
    v1 = ((m1 - m2) / (m1 + m2)) * u1 + ((2 * m2) / (m1 + m2)) * u2
    v2 = ((2 * m1) / (m1 + m2)) * u1 + ((m2 - m1) / (m1 + m2)) * u2
    return v1, v2
