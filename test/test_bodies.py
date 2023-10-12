import equinox as eqx
from jax import numpy as jnp, random as jr

from cotix._bodies import Ball
from cotix._shapes import Circle


def test_ball_colliding():
    key = jr.PRNGKey(0)
    a = Ball().move_to(jnp.array([0.08, 0.0]))
    b = Ball().move_to(jnp.array([0.02, 0.0]))
    assert a.collides_with(b, key)[0] & b.collides_with(a, key)[0]

    a = a.move_to(jnp.array([1.0, 0.0]))
    assert (~b.collides_with(a, key)[0]) & (~a.collides_with(b, key)[0])


def test_misaligned_ball_rotating():
    key = jr.PRNGKey(0)
    a = Ball().move_to(jnp.array([0.0, 0.0]))
    a = eqx.tree_at(lambda x: x.shape, a, Circle(jnp.array([10.0, 0.0]), 0.05))
    a = eqx.tree_at(lambda x: x.angle, a, jnp.array(-jnp.pi / 2))

    b = Ball().move_to(jnp.array([0.0, 10.0]))

    assert b.collides_with(a, key)[0]
