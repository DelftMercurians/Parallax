import jax
from jax import numpy as jnp, random as jr

from cotix._bodies import Ball
from cotix._collision_resolution import _resolve_balls_collision
from cotix._collisions import check_for_collision, compute_penetration_vector
from cotix._shapes import Circle


def test_circle_hits_circle_elastic():
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

        body1 = Ball(1, v1, shape1)
        body2 = Ball(1, v2, shape2)

        res_first_collision, simplex = check_for_collision(shape1, shape2, key)
        penetration = compute_penetration_vector(shape1, shape2, simplex)

        body1, body2 = _resolve_balls_collision(body1, body2, penetration)

        # i really dont like to do it this way
        total_position1 = body1.shape.position + body1.position
        total_position2 = body2.shape.position + body2.position
        new_shape1 = Circle(body1.shape.radius, total_position1)
        new_shape2 = Circle(body2.shape.radius, total_position2)
        res_collision, simplex = check_for_collision(new_shape1, new_shape2, key)

        # should be no collision and velocities away from another ball
        v1_away = jnp.dot(body1.velocity, total_position2 - total_position1) <= 0
        v2_away = jnp.dot(body2.velocity, total_position1 - total_position2) <= 0
        return ~res_collision & v1_away & v2_away & res_first_collision

    N = 1000
    f = jax.vmap(f)
    out = f(jr.split(key, N))

    assert jnp.all(out)
