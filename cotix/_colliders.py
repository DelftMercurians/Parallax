from typing import List

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Int

from ._bodies import AbstractBody
from ._collision_resolution import resolve_collision
from ._convex_shapes import AABB
from ._utils import make_pairs


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


@jax.jit
def _check_aabb_collision(a, b):
    c = (a[0] < b[0]) & AABB.collides(a[1], b[1])
    return jax.lax.cond(
        c,
        lambda: _BroadCollision(jnp.array(a[0]), jnp.array(b[0])),
        lambda: _BroadCollision(jnp.array(-1), jnp.array(-1)),
    )


@eqx.filter_jit
def resolve_idk(x, y):
    jax.debug.print("{x};{y}", x=x, y=y)


@eqx.filter_jit
def get_body(bodies, i):
    return bodies[int(i)]


class NaiveCollider(AbstractCollider):
    """
    Standard collider: use the most universal methods, to find collisions
    EPA and GJK, but they are the slowest. To resolve collisions,
    we firstly have a broad phase where we detect 4N collisions, and
    then we have an exact phase, where we detect just N collisions.
    """

    @eqx.filter_jit
    def broad_phase(self, bodies, limit: int):
        # map bodies to theirs aabbs, trace-time
        aabbs = [AABB()] * len(bodies)
        for i in range(len(bodies)):
            aabbs[i] = AABB.of_universal(bodies[i].shape)
        out = make_pairs(
            aabbs,
            _check_aabb_collision,
            _BroadCollision(jnp.array(-1), jnp.array(-1)),
            limit=limit,
        )

        return out

    @eqx.filter_jit
    def total_phase(self, bodies, limit: int):
        @jax.jit
        def _check_actual_collision(a, b):
            c = a[0] < b[0]
            return jax.lax.cond(
                c, lambda: a[1].penetrates_with(b[1]), lambda: jnp.zeros((2,))
            )

        out = make_pairs(bodies, _check_actual_collision, jnp.zeros((2,)), limit=limit)

        return out

    def resolve(self, bodies):
        initial_bodies = bodies
        length = len(bodies)

        broad_collisions = self.broad_phase(bodies, N=4 * length)
        penetrations = self.narrow_phase(bodies, broad_collisions, N=length)

        # yep, the reason we do a for-loop here and not jax tree map
        # is so that XLA maybe optimizes all memory shit inside it
        # cuz when we do jtu tree map, I think XLA does not need
        # extra-iteration optimizations, like it optimizes only inside the iteration
        for p in penetrations:
            new_body_a, new_body_b = resolve_collision(p.a, p.b, p.penetration_vector)
            # TODO: this cool piece of code might cause ~N copies of all the bodies,
            # which is like a loooot. Idk how to fix it though, so for now it is fine
            # btw, i am not sure if it actually causes them: maybe XLA is smart enough
            # to not copy bodies for every instruction (probably it is smart enough)
            bodies = bodies.set_at(p.i, new_body_a)
            bodies = bodies.set_at(p.j, new_body_b)

        out = bodies.to_pytree()
        # easy way to check if out and bodies are the same,
        # lol -> if the shapes are not the same,
        # JAX will commit suicide by itself
        return jax.lax.cond(True, lambda: out, lambda: initial_bodies)
