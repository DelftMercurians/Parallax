from typing import List

import equinox as eqx
import jax
from jax import numpy as jnp, tree_util as jtu
from jaxtyping import Array, Float, Int

from ._bodies import AbstractBody
from ._convex_shapes import AABB
from ._utils import filter_scan


class AbstractCollider(eqx.Module):
    """
    Implements a collision-resolving something. Like, ehhh, idk.

    Effectively handles both collision detection and collision
    resolution, probably later we should separate it into two
    different abstractions. Cuz it's different. Though related.

    Like, since they sort of depend on each other, joining them
    together as a single entity makes sense. also because we almost
    never want to access one of them separetely (and when we do, we
    can just define another one).
    """

    def resolve(self, bodies: List[AbstractBody]):
        """
        Resolves some (meaning not all) collisions between provided bodies.
        Since it is JAX, and we want to avoid n^2 complexity,
        on every step we might resolve not all the collisions,
        but only some of them. Also, not giving a guarantee that
        we resolve all the collisions helps to avoid issues
        (aka having infinite loop) when we have unresolvable collisions.
        """
        raise NotImplementedError


class _BroadCollision(eqx.Module):
    i: Int[Array, ""]
    j: Int[Array, ""]


class _CollisionWithPenetration(eqx.Module):
    i: Int[Array, ""]
    j: Int[Array, ""]
    penetration_vector: Float[Array, "2"]


class _PostCollisionUpdate(eqx.Module):
    i: Int[Array, ""]
    j: Int[Array, ""]
    body_i: AbstractBody
    body_j: AbstractBody


class NaiveCollider(AbstractCollider):
    """
    Standard collider: use the most universal methods, to find collisions
    EPA and GJK, but they are the slowest. To resolve collisions,
    we firstly have a broad phase where we detect 4N collisions, and
    then we have an exact phase, where we detect just N collisions.
    """

    def broad_phase(self, bodies, N: int):
        res = [_BroadCollision(jnp.array(-1), jnp.array(-1))] * N

        def loop_body(carry, xs):
            collision_index, res_index, res = carry
            i = collision_index // len(bodies)
            j = jnp.mod(collision_index, len(bodies))
            jax.debug.print("{x}", x=(i, j))
            res, res_index = jax.lax.cond(
                AABB.collide(AABB(bodies[i]), AABB(bodies[j])),
                (
                    eqx.tree_at(
                        lambda r: r[res_index], res, replace=_BroadCollision(i, j)
                    ),
                    res_index + 1,
                ),
                (res[res_index], res_index),
            )
            return collision_index + 1, res_index

        (_, _, res), _ = filter_scan(
            loop_body, (jnp.array(0), jnp.array(0), res), None, length=len(bodies) ** 2
        )
        return res[:N]

    def exact_phase(self, bodies, collision_data, N: int):
        pass

    def resolve_penetration(self, collision_to_resolve, bodies):
        def resolve_collision_of_two_bodies(self, *args):
            raise NotImplementedError

        i, j = collision_to_resolve.i, collision_to_resolve.j
        penetration_vector = collision_to_resolve.penetration_vector
        new_body_a, new_body_b = resolve_collision_of_two_bodies(
            bodies[i], bodies[j], penetration_vector
        )
        return _PostCollisionUpdate(i, j, new_body_a, new_body_b)

    def resolve(self, bodies: List[AbstractBody]):
        broad_collisions = self.broad_phase(bodies, N=4 * len(bodies))
        exact_collisions = self._exact_phase(bodies, broad_collisions, N=len(bodies))

        updates = jtu.tree_map(
            lambda x: self.resolve_penetration(x, bodies),
            exact_collisions,
            is_leaf=isinstance(AbstractBody),
        )

        # now 'apply' updates, by setting correct bodies
        for upd in updates:
            bodies = bodies.at[upd.i].set(upd.body_i)
            bodies = bodies.at[upd.j].set(upd.body_j)
        return bodies
