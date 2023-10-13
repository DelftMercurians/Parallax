import equinox as eqx
from jax import numpy as jnp, random as jr

from cotix._bodies import Ball
from cotix._shapes import Circle


def test_ball_colliding():
    key = jr.PRNGKey(0)
    a = Ball().set_position(jnp.array([0.08, 0.0]))
    b = Ball().set_position(jnp.array([0.02, 0.0]))
    a = a.update_transform()
    b = b.update_transform()
    assert a.collides_with(b, key)[0] & b.collides_with(a, key)[0]

    a = a.set_position(jnp.array([1.0, 0.0]))
    a = a.update_transform()
    assert (~b.collides_with(a, key)[0]) & (~a.collides_with(b, key)[0])


def test_misaligned_ball_rotating():
    key = jr.PRNGKey(0)
    a = Ball().set_position(jnp.array([0.0, 0.0]))
    a = eqx.tree_at(lambda x: x.shape, a, Circle(0.05, jnp.array([10.0, 0.0])))
    a = eqx.tree_at(lambda x: x.angle, a, jnp.array(jnp.pi / 2))

    b = Ball().set_position(jnp.array([0.0, 10.0]))

    a = a.update_transform()
    b = b.update_transform()

    assert b.collides_with(a, key)[0] & a.collides_with(b, key)[0]


def test_misaligned_ball_rotating_jitted():
    @eqx.filter_jit
    def f():
        key = jr.PRNGKey(0)
        a = Ball().set_position(jnp.array([0.0, 0.0]))
        a = eqx.tree_at(lambda x: x.shape, a, Circle(0.05, jnp.array([10.0, 0.0])))
        a = eqx.tree_at(lambda x: x.angle, a, jnp.array(jnp.pi / 2))

        b = Ball().set_position(jnp.array([0.0, 10.0]))
        a = a.update_transform()
        b = b.update_transform()

        return b.collides_with(a, key)[0] & a.collides_with(b, key)[0]

    assert f()
