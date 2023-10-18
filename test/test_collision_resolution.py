import jax
from jax import numpy as jnp, random as jr

from cotix._bodies import Ball
from cotix._collision_resolution import _resolve_collision
from cotix._collisions import (
    check_for_collision_convex,
    compute_penetration_vector_convex,
)
from cotix._convex_shapes import Circle
from cotix._universal_shape import UniversalShape


def test_circle_hits_circle_elastic():
    # this test is kinda weak: i dont test that the speeds are fully correct
    key = jr.PRNGKey(0)

    def f(key):
        key1, key2, key3, key4, key5, key6 = jr.split(key, 6)
        p1 = jr.uniform(key1, (2,)) * 4 - 2
        p2 = jr.uniform(key2, (2,)) * 4 - 2
        dist = jnp.sqrt(jnp.sum((p1 - p2) ** 2))
        r1 = jr.uniform(key3, minval=dist * 0.25, maxval=dist * 0.75)
        # guarantee a collision
        r2 = jr.uniform(key4, minval=(dist - r1) * 1.05, maxval=(dist - r1) * 1.15)

        shape1 = Circle(r1, p1)
        shape2 = Circle(r2, p2)

        # velocities into another ball
        v1 = jr.uniform(key5, (2,)) * 2 * (p2 - p1)
        v2 = jr.uniform(key6, (2,)) * 2 * (p1 - p2)

        body1 = Ball(jnp.array(1.0), v1, UniversalShape(shape1))
        body2 = Ball(jnp.array(1.0), v2, UniversalShape(shape2))

        dir = shape1.get_center() - shape2.get_center()

        res_first_collision, simplex = check_for_collision_convex(
            shape1.get_support, shape2.get_support, dir
        )
        penetration1 = compute_penetration_vector_convex(
            shape1.get_support, shape2.get_support, simplex
        )

        body1, body2 = _resolve_collision(body1, body2, penetration1)

        total_position1 = body1.shape.parts[0].position + body1.position
        total_position2 = body2.shape.parts[0].position + body2.position
        new_shape1 = Circle(body1.shape.parts[0].radius, total_position1)
        new_shape2 = Circle(body2.shape.parts[0].radius, total_position2)
        dir = new_shape1.get_center() - new_shape2.get_center()
        res_collision, simplex = check_for_collision_convex(
            new_shape1.get_support, new_shape2.get_support, dir
        )
        penetration2 = compute_penetration_vector_convex(
            new_shape1.get_support, new_shape2.get_support, simplex
        )

        # should be no collision and velocities away from another ball
        v1_away = jnp.dot(body1.velocity, total_position2 - total_position1) <= 0
        v2_away = jnp.dot(body2.velocity, total_position1 - total_position2) <= 0
        no_collision = jnp.logical_or(
            ~res_collision,
            jnp.linalg.norm(penetration2) < 1e-3 * jnp.linalg.norm(penetration1),
        )
        return no_collision & v1_away & v2_away & res_first_collision

    N = 1000
    f = jax.vmap(f)
    out = f(jr.split(key, N))

    assert jnp.all(out)
