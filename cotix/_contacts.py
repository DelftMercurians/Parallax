import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ._convex_shapes import AABB, Circle


class ContactInfo(eqx.Module):
    penetration_vector: Float[Array, "2"]
    contact_point: Float[Array, "2"]

    def __init__(self, penetration_vector, contact_point):
        self.penetration_vector = penetration_vector
        self.contact_point = contact_point

    @staticmethod
    def nan():
        return ContactInfo(jnp.zeros((2,)), jnp.array([jnp.nan, jnp.nan]))


def circle_vs_circle(a: Circle, b: Circle):
    delta = a.position - b.position
    distance = jnp.linalg.norm(delta)
    direction_between_shapes = jax.lax.cond(
        distance == 0.0, lambda: jnp.array([1.0, 0.0]), lambda: delta / distance
    )
    penetration_vector = direction_between_shapes * jnp.minimum(
        distance - (a.radius + b.radius), 0.0
    )
    contact_point = (
        b.position + direction_between_shapes * (b.radius - a.radius) + a.position
    ) / 2.0
    # check that centers lie from different sides of contact point
    # and if not, return the center that lies inside another circle
    contact_point = jax.lax.cond(
        jnp.dot(a.position - contact_point, b.position - contact_point) <= 0,
        lambda: contact_point,  # different sides
        lambda: jax.lax.cond(
            a.contains(b.position),  # same side
            lambda: b.position,  # b's center contained in a
            lambda: a.position,
        ),  # otherwise
    )

    return jax.lax.cond(
        distance <= a.radius + b.radius,
        lambda: ContactInfo(-penetration_vector, contact_point),
        lambda: ContactInfo.nan(),
    )


def aabb_vs_aabb(a: AABB, b: AABB, eps=1e-8):
    is_first_below_second = a.upper[1] <= b.lower[1]
    is_first_above_second = a.lower[1] >= b.upper[1]
    is_first_left_second = a.upper[0] <= b.lower[0]
    is_first_right_second = a.lower[0] >= b.upper[0]

    def estimate_contact():
        depths = jnp.array(
            [
                jnp.maximum(
                    a.upper[1] - b.lower[1], -eps
                ),  # eps here, so that 0 processed correctly
                jnp.maximum(b.upper[1] - a.lower[1], -eps),
                jnp.maximum(a.upper[0] - b.lower[0], -eps),
                jnp.maximum(b.upper[0] - a.lower[0], -eps),
            ]
        )
        dirs = jnp.array([[0, -1], [0, 1], [-1, 0], [1, 0]])

        index = jnp.argmin(depths)
        min_depth = jnp.clip(depths[index], a_min=0.0)
        penetration_vector = min_depth * dirs[index]
        min_upper = jnp.minimum(a.upper, b.upper)
        max_lower = jnp.maximum(a.lower, b.lower)
        return ContactInfo(penetration_vector, (min_upper + max_lower) / 2.0)

    return jax.lax.cond(
        ~(
            is_first_below_second
            | is_first_left_second
            | is_first_above_second
            | is_first_right_second
        ),
        lambda: estimate_contact(),
        lambda: ContactInfo.nan(),
    )


def circle_vs_aabb(a: Circle, b: AABB, eps=1e-6):
    disp = a.get_center() - b.get_center()
    clamp_disp = jnp.clip(disp, b.lower - b.get_center(), b.upper - b.get_center())
    ccp = (
        b.get_center() + clamp_disp
    )  # ccp = closest circle point, point on aabb that is closest to the circle
    ccp = eqx.error_if(
        ccp, ~b.contains(ccp), "Gm, closest point in the AABB is not in AABB. wut?"
    )

    vs = jnp.array(
        [
            b.lower,
            jnp.array([b.lower[0], b.upper[1]]),
            b.upper,
            jnp.array([b.upper[0], b.lower[1]]),
        ]
    )

    perfect_vertex = jnp.any(jnp.linalg.norm(vs - ccp, axis=1) < eps)

    def circle_dir_move():
        # move the aabb out of the circle, in the direction
        # that is not aligned with axes
        dir = ccp - a.position
        dir_norm = dir / jnp.linalg.norm(dir)  # TODO: check division by zero
        return ContactInfo(-(a.position + a.radius * dir_norm - ccp), ccp)

    def aligned_move():
        # now, we want to move aabb out of the circle moving only in one axis
        # while there is a hard way to do this, I am too lazy,
        # so I am just gonna try all 4 variants and see which one is the best
        dirs = jnp.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        a.position - ccp
        d_from_aabb = b.get_center() - ccp

        prods = jax.lax.map(lambda x: jnp.dot(x, d_from_aabb), dirs)
        jnp.argmax(prods)

        shifts = jnp.array(
            [
                a.position[1] + a.radius - b.lower[1],
                b.upper[1] - (a.position[1] - a.radius),
                a.position[0] + a.radius - b.lower[0],
                b.upper[0] - (a.position[0] - a.radius),
            ]
        )

        best_shift = jnp.argmin(shifts)
        return ContactInfo(-shifts[best_shift] * dirs[best_shift], ccp)

    return jax.lax.cond(
        a.contains(ccp),
        lambda: jax.lax.cond(perfect_vertex, circle_dir_move, aligned_move),
        lambda: ContactInfo.nan(),
    )
