import beartype
import equinox as eqx
import jax.numpy as jnp
import jaxlib
import pytest

from cotix._bodies import Ball
from cotix._collisions import check_for_collision, compute_penetration_vector
from cotix._shapes import AABB, Circle, Polygon


typefailed = beartype.roar.BeartypeCallHintParamViolation
eqxfailed = (jaxlib.xla_extension.XlaRuntimeError, eqx.EquinoxTracetimeError)


def test_bad_types_shapes():
    with pytest.raises(typefailed):
        Circle(1, jnp.array([0.0, 0.0]))
    with pytest.raises(typefailed):
        Polygon(jnp.array([1.0, 2.0, 3.0]))
    with pytest.raises(typefailed):
        AABB("asdf")


def test_bad_types_collision():
    with pytest.raises(typefailed):
        check_for_collision(1, 2)
    with pytest.raises(typefailed):
        compute_penetration_vector("asdf", "lol")


def test_bad_types_bodies_with_invariant():
    with pytest.raises(eqxfailed):
        ball = Ball()
        ball = eqx.tree_at(lambda x: x.mass, ball, jnp.array([1, 2, 3]))
        # trigger invariant
        ball = ball.set_position(jnp.array([1.0, 2.0]))

    with pytest.raises(eqxfailed):
        ball = Ball()
        ball = eqx.tree_at(lambda x: x.velocity, ball, jnp.array(1.0))
        # trigger invariant
        ball = ball.set_position(jnp.array([1.0, 2.0]))

    with pytest.raises(eqxfailed):
        ball = Ball()
        ball = eqx.tree_at(lambda x: x.inertia, ball, jnp.nan)
        # trigger invariant
        ball = ball.set_position(jnp.array([1.0, 2.0]))
