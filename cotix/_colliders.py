from typing import List

import equinox as eqx

from ._bodies import AbstractBody
from ._contacts import aabb_vs_aabb, circle_vs_aabb, circle_vs_circle
from ._convex_shapes import AABB, Circle


_contact_funcs = {
    (type(AABB), type(AABB)): aabb_vs_aabb,
    (type(Circle), type(Circle)): circle_vs_circle,
    (type(Circle), type(AABB)): circle_vs_aabb,
}


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


class NaiveCollider(AbstractCollider):
    """
    Naive collider is actually somewhat equivalent to the Brax collision resolution.

    We resolve collisions for every pair of bodies, and assume that body types are
    static
    so that we just have two for loops, and for every pair of bodies we do a collision
    resolve
    of course, this is great, but has the lowest performance possible:
    we are doing N^2 actual collision resolutions. Like, not just AABB checks or
    something,
    but real-life resolution, with like penetration vectors, contact points,
    and forces computations. This is usually undesirable. An interesting
    alternative is to
    do collision resolution in a randomized per-object-type-pair fashion, but that
    is a
    different collider that is to-be-implemented. TODO!
    """

    def resolve(self, bodies):
        for i, a in enumerate(bodies):
            for j, b in enumerate(bodies):
                # skip half of irrelevant checks
                if i <= j:
                    continue
                if (type(a), type(b)) in _contact_funcs:
                    _contact_funcs[(type(a), type(b))]
                elif (type(b), type(a)) in _contact_funcs:
                    _contact_funcs[(type(b), type(a))]
                else:
                    raise RuntimeError(
                        "Your done buddy, you are using illegal shapes "
                        "(write an issue on github about it or smth idk)"
                    )

                # now in the contact we have the contact :) Let's resolve it.
