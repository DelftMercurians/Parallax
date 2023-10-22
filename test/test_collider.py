import equinox as eqx
import jax
from jax import numpy as jnp, tree_util as jtu
from jaxtyping import Array, Int

from cotix._bodies import AbstractBody, Ball
from cotix._convex_shapes import Circle
from cotix._universal_shape import UniversalShape


class _BroadCollision(eqx.Module):
    i: Int[Array, ""]
    j: Int[Array, ""]


def test_simple_world_broad_phase():
    a = Ball(
        jnp.array(1.0),
        jnp.zeros((2,)),
        UniversalShape(
            Circle(
                position=jnp.zeros(
                    2,
                ),
                radius=jnp.array(1.0),
            )
        ),
    )
    b = Ball(
        jnp.array(1.0),
        jnp.zeros((2,)),
        UniversalShape(
            Circle(
                position=jnp.ones(
                    2,
                ),
                radius=jnp.array(1.0),
            )
        ),
    )

    bodies = [a, b]

    cond_fn = lambda x: isinstance(x, AbstractBody)
    only_shapes = eqx.filter(bodies, cond_fn, is_leaf=cond_fn)
    shapes, _ = jtu.tree_flatten_with_path(only_shapes, is_leaf=cond_fn)
    # shapes is now just a list, where first element is a PyTree representing path
    # and second element is the convex shape that is a part of shape.

    def second_map(shape_a, shape_b):
        # jax.debug.print("{a} <-> {b}\n\n", a=shape_a, b=shape_b)
        seq_key_a, body_a = shape_a
        seq_key_b, body_b = shape_b

        i = seq_key_a[0].idx
        j = seq_key_b[0].idx

        possibly_colliding = body_a.possibly_collides_with(body_b)

        # while doing condition here does not reduce execution time,
        # it allows us to not care in the future about resolving every collision
        # twice, or about resolving collision with ourselves.
        # This is just a convenient place to do filtering
        return jax.lax.cond(
            (i < j) & possibly_colliding,
            lambda: (True, i, j, body_a, body_b),
            lambda: (False, -1, -1, body_a, body_b),
        )

    def first_map(leaf):
        return jtu.tree_map(
            jtu.Partial(second_map, shape_b=leaf),
            shapes,
            is_leaf=lambda x: isinstance(x, tuple),
        )

    res = jtu.tree_map(first_map, shapes, is_leaf=lambda x: isinstance(x, tuple))
    res, _ = jtu.tree_flatten(res, is_leaf=lambda x: isinstance(x, tuple))

    res = eqx.filter(res, lambda leaf: leaf[0], is_leaf=lambda x: isinstance(x, tuple))
    res = jtu.tree_flatten(res, is_leaf=lambda x: isinstance(x, tuple))[0]
    eqx.tree_pprint(res)

    assert False
