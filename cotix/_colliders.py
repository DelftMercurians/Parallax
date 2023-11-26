from typing import List

import equinox as eqx
import jax
from jax import numpy as jnp, random as jr, tree_util as jtu

from ._bodies import AbstractBody, DynamicBody
from ._collision_resolution import resolve_collision
from ._contacts import aabb_vs_aabb, circle_vs_aabb, circle_vs_circle, ContactInfo
from ._convex_shapes import AABB, Circle


_contact_funcs = {
    (AABB, AABB): eqx.filter_jit(aabb_vs_aabb),
    (Circle, Circle): eqx.filter_jit(circle_vs_circle),
    (Circle, AABB): eqx.filter_jit(circle_vs_aabb),
}


resolve_collision_jitted = eqx.filter_jit(resolve_collision)


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
    @eqx.filter_jit  # this must be jitted because concrete JAX arrays are not hashable
    def resolve(self, bodies):
        # we are interested in a single contact per body
        # so we are going to join all 'same shape type' collisions with each other
        # so that the compilation time is "small"

        # so, we are going to have a dictionary that maps
        # (type1, type2) -> ((all bodies of type1), (all bodies of type2))
        # and then we are going to iterate over it, and for every body index
        # we are going to store the contact info for that body

        type_to_bodies = {}
        for i, body in enumerate(bodies):
            for j, body2 in enumerate(bodies):
                if i <= j:
                    continue
                for a in body.shape.parts:
                    for b in body2.shape.parts:
                        a = a.transform(body.shape._transformer)
                        b = b.transform(body2.shape._transformer)

                        type1 = type(a)
                        type2 = type(b)
                        # make sure that the order is the same as in _contact_funcs
                        if (type1, type2) in _contact_funcs.keys():
                            pass
                        elif (type2, type1) in _contact_funcs.keys():
                            type1, type2 = type2, type1
                        else:
                            raise RuntimeError(
                                "Your done buddy, you are using illegal shapes "
                                "(write an issue on github about it or smth idk)"
                            )

                        if (type1, type2) in type_to_bodies:
                            type_to_bodies[(type1, type2)][0].append((i, a))
                            type_to_bodies[(type1, type2)][1].append((j, b))
                        else:
                            type_to_bodies[(type1, type2)] = ([(i, a)], [(j, b)])

        # make sure there are no duplicates
        for key in type_to_bodies:
            type_to_bodies[key] = (
                list(set(type_to_bodies[key][0])),
                list(set(type_to_bodies[key][1])),
            )

        # and transform the list-of-structs into a struct-of-lists
        for key in type_to_bodies:
            list_of_body1, list_of_body2 = type_to_bodies[key]
            sof1 = jtu.tree_map(
                lambda *x: jnp.stack(x), *list_of_body1, is_leaf=eqx.is_array
            )
            sof2 = jtu.tree_map(
                lambda *x: jnp.stack(x), *list_of_body2, is_leaf=eqx.is_array
            )
            type_to_bodies[key] = (sof1, sof2)

        # now, we want to iterate over the dict, for every pair of types
        # we are going to do a contact check between body1 and body2
        # and then we are going to resolve the collision (only once)

        all_contacts = ContactInfo(
            penetration_vector=jnp.zeros((len(bodies), len(bodies), 2)),
            contact_point=jnp.zeros((len(bodies), len(bodies), 2)) * jnp.nan,
        )

        for key in type_to_bodies:
            bodies1, bodies2 = type_to_bodies[key]

            # let's vmap over the bodies
            def body2_loop(body2, body1):
                i, body1 = body1
                j, body2 = body2
                type1, type2 = type(body1), type(body2)

                # if the order is wrong, swap them
                if (type1, type2) not in _contact_funcs.keys():
                    type1, type2 = type2, type1
                    body1, body2 = body2, body1

                contacts = _contact_funcs[(type1, type2)](body1, body2)
                return (i, j, contacts)

            def body1_loop(body1):
                return eqx.filter_vmap(
                    jtu.Partial(body2_loop, body1=body1), in_axes=eqx.if_array(0)
                )(bodies2)

            current_contacts = eqx.filter_vmap(body1_loop, in_axes=eqx.if_array(0))(
                bodies1
            )

            # apply every update recorded in contact_points
            new_all_contact_points, _ = eqx.internal.scan(
                lambda data_arr, i: (
                    jtu.tree_map(
                        lambda leaf, to_set: leaf.at[
                            current_contacts[0][i], current_contacts[1][i]
                        ].set(to_set[i]),
                        data_arr,
                        current_contacts[2],
                        is_leaf=eqx.is_array,
                    ),
                    None,
                ),
                init=all_contacts,
                xs=jnp.arange(len(current_contacts)),
                kind="lax",
            )
            all_contacts = new_all_contact_points

        # almost done: now we have a SOA of constacts, and we want to resolve them
        # we are going to choose a random contact that is not nan along the first axis
        # and resolve it

        def choose_random_contact(arr, i):
            # count nans along the first axis
            is_bad = jnp.any(jnp.isnan(arr.contact_point), axis=-1)
            not_nans_count = jnp.sum(~is_bad)
            # probs is array of shape (arr[0], arr[1]) with probabilities of choosing
            probs = (~is_bad).astype(jnp.float32) / not_nans_count
            # choose a random contact
            contact_index = jax.lax.cond(
                not_nans_count == 0,
                lambda: 0,
                lambda: jr.choice(
                    jr.PRNGKey(0), jnp.arange(arr.contact_point.shape[0]), p=probs
                ),
            )

            out_element = jtu.tree_map(
                lambda x: x[contact_index], arr, is_leaf=eqx.is_array
            )

            return (i, contact_index, out_element)

        chosen_contacts = eqx.filter_vmap(
            choose_random_contact, in_axes=eqx.if_array(0)
        )(all_contacts, jnp.arange(len(bodies)))

        # now, we want to resolve the collision for every chosen contact
        # firstly, we need to transform bodies to ShapelessBodies (so that we can
        # vmap over resolution)

        shapeless_bodies = [DynamicBody(body) for body in bodies]

        # now, we need to transform shapeless_bodies to a SOA
        soa_shapeless = jtu.tree_map(
            lambda *x: jnp.stack(x), *shapeless_bodies, is_leaf=eqx.is_array
        )

        # after we got our SOA, we can do a scan over the chosen contacts, carrying
        # the SOA and resolving the collision for every contact
        def to_scan(soa, contact):
            i, j, contact = contact

            soa_i = jtu.tree_map(lambda x: x[i], soa, is_leaf=eqx.is_array)
            soa_j = jtu.tree_map(lambda x: x[j], soa, is_leaf=eqx.is_array)
            new_soa_i, new_soa_j, _ = resolve_collision_jitted(soa_i, soa_j, contact)
            new_soa = jtu.tree_map(
                lambda soa_leaf, new_value: soa_leaf.at[i].set(new_value),
                soa,
                new_soa_i,
                is_leaf=eqx.is_array,
            )
            new_soa = jtu.tree_map(
                lambda soa_leaf, new_value: soa_leaf.at[j].set(new_value),
                new_soa,
                new_soa_j,
                is_leaf=eqx.is_array,
            )

            return new_soa, None

        out, _ = eqx.internal.scan(
            to_scan, init=soa_shapeless, xs=chosen_contacts, kind="lax"
        )

        # And now, the final step: convert SOA of DynamicBodies to a list of
        # AbstractBodies (statically the same as we got on the input)
        # and return it

        for i in range(len(bodies)):
            bodies[i] = bodies[i].load(
                jtu.tree_map(lambda x: x[i], out, is_leaf=eqx.is_array)
            )

        return bodies
