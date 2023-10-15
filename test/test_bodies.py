import equinox as eqx
from jax import numpy as jnp

from cotix._bodies import Ball
from cotix._convex_shapes import Circle
from cotix._universal_shape import UniversalShape


def test_ball_colliding():
    a = Ball().set_position(jnp.array([0.08, 0.0]))
    b = Ball().set_position(jnp.array([0.02, 0.0]))
    a = a.update_transform()
    b = b.update_transform()
    assert a.collides_with(b)[0] & b.collides_with(a)[0]

    a = a.set_position(jnp.array([1.0, 0.0]))
    a = a.update_transform()
    assert (~b.collides_with(a)[0]) & (~a.collides_with(b)[0])


def test_misaligned_ball_rotating():
    a = Ball().set_position(jnp.array([0.0, 0.0]))
    a = eqx.tree_at(
        lambda x: x.shape,
        a,
        UniversalShape(Circle(jnp.array(0.05), jnp.array([10.0, 0.0]))),
    )
    a = eqx.tree_at(lambda x: x.angle, a, jnp.array(jnp.pi / 2))

    b = Ball().set_position(jnp.array([0.0, 10.0]))

    a = a.update_transform()
    b = b.update_transform()

    assert b.collides_with(a)[0] & a.collides_with(b)[0]


def test_misaligned_ball_rotating_jitted():
    @eqx.filter_jit
    def f():
        a = Ball().set_position(jnp.array([0.0, 0.0]))
        a = eqx.tree_at(
            lambda x: x.shape,
            a,
            UniversalShape(Circle(jnp.array(0.05), jnp.array([10.0, 0.0]))),
        )
        a = eqx.tree_at(lambda x: x.angle, a, jnp.array(jnp.pi / 2))

        b = Ball().set_position(jnp.array([0.0, 10.0]))
        a = a.update_transform()
        b = b.update_transform()

        return b.collides_with(a)[0] & a.collides_with(b)[0]

    assert f()
