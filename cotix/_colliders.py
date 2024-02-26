from typing import List

import equinox as eqx
import jax
from jax import numpy as jnp, random as jr, tree_util as jtu

from ._bodies import AbstractBody, DynamicBody
from ._collision_resolution import resolve_collision
from ._contacts import (
    aabb_vs_aabb,
    aabb_vs_polygon,
    circle_vs_aabb,
    circle_vs_circle,
    circle_vs_polygon,
    ContactInfo,
    polygon_vs_polygon,
)
from ._convex_shapes import AABB, Circle, Polygon, Polygon4, Polygon6


_contact_funcs = {
    (AABB, AABB): eqx.filter_jit(aabb_vs_aabb),
    (Circle, Circle): eqx.filter_jit(circle_vs_circle),
    (Circle, AABB): eqx.filter_jit(circle_vs_aabb),
    (Polygon, Polygon): eqx.filter_jit(polygon_vs_polygon),
    (AABB, Polygon): eqx.filter_jit(aabb_vs_polygon),
    (Circle, Polygon): eqx.filter_jit(circle_vs_polygon),
    (Circle, Polygon4): eqx.filter_jit(circle_vs_polygon),
    (Circle, Polygon6): eqx.filter_jit(circle_vs_polygon),
    (AABB, Polygon4): eqx.filter_jit(aabb_vs_polygon),
    (AABB, Polygon6): eqx.filter_jit(aabb_vs_polygon),
    (Polygon4, Polygon4): eqx.filter_jit(polygon_vs_polygon),
    (Polygon4, Polygon6): eqx.filter_jit(polygon_vs_polygon),
    (Polygon6, Polygon6): eqx.filter_jit(polygon_vs_polygon),
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
    @eqx.filter_jit
    def resolve(self, bodies):
        # simply resolve every collision that is present, in sequency
        pass


class RandomizedCollider(AbstractCollider):
    @eqx.filter_jit  # this must be jitted because concrete JAX arrays are not hashable
    def resolve(self, bodies, rkey, collision_callback=lambda x: None):
        # we are interested in a single contact per body
        # so we are going to join all 'same shape type' collisions with each other
        # so that the compilation time is "small"

        # so, we are going to have a dictionary that maps
        # (type1, type2) -> ((all bodies of type1), (all bodies of type2))
        # and then we are going to iterate over it, and for every body index
        # we are going to store the contact info for that body

        type_to_shapes = {}
        for i, body in enumerate(bodies):
            for j, body2 in enumerate(bodies):
                if i <= j:
                    continue
                for a in body.shape.parts:
                    a = a.transform(body.shape._transformer)
                    for b in body2.shape.parts:
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

                        if (type1, type2) in type_to_shapes:
                            type_to_shapes[(type1, type2)][0].append((i, a))
                            type_to_shapes[(type1, type2)][1].append((j, b))
                        else:
                            type_to_shapes[(type1, type2)] = ([(i, a)], [(j, b)])

        # make sure there are no duplicates
        for key in type_to_shapes:
            type_to_shapes[key] = (
                list(set(type_to_shapes[key][0])),
                list(set(type_to_shapes[key][1])),
            )

        # and transform the list-of-structs into a struct-of-lists
        for key in type_to_shapes:
            list_of_body1, list_of_body2 = type_to_shapes[key]
            sof1 = jtu.tree_map(
                lambda *x: jnp.stack(x), *list_of_body1, is_leaf=eqx.is_array
            )
            sof2 = jtu.tree_map(
                lambda *x: jnp.stack(x), *list_of_body2, is_leaf=eqx.is_array
            )
            type_to_shapes[key] = (sof1, sof2)

        # now, we want to iterate over the dict, for every pair of types
        # we are going to do a contact check between body1 and body2
        # and then we are going to resolve the collision (only once)

        all_contacts = ContactInfo(
            penetration_vector=jnp.zeros((len(bodies), len(bodies), 2)),
            contact_point=jnp.zeros((len(bodies), len(bodies), 2)) * jnp.nan,
        )

        skey = jr.split(rkey)[0]
        for type_key in type_to_shapes.keys():
            shapes1, shapes2 = type_to_shapes[type_key]
            N1 = len(shapes1[0])
            N2 = len(shapes2[0])

            # let's vmap over the shapes
            def shape2_loop(shape2, shape1):
                i, shape1 = shape1
                j, shape2 = shape2
                type1, type2 = type(shape1), type(shape2)

                # if the order is wrong, swap them
                if (type1, type2) not in _contact_funcs.keys():
                    type1, type2 = type2, type1
                    shape1, shape2 = shape2, shape1

                contacts = _contact_funcs[(type1, type2)](shape1, shape2)
                return (
                    i,
                    j,
                    jax.lax.cond(i < j, lambda: ContactInfo.nan(), lambda: contacts),
                )

            def shape1_loop(shape1):
                return eqx.filter_vmap(
                    jtu.Partial(shape2_loop, shape1=shape1), in_axes=eqx.if_array(0)
                )(shapes2)

            current_contacts = eqx.filter_vmap(shape1_loop, in_axes=eqx.if_array(0))(
                shapes1
            )

            skey = jr.split(skey)[0]

            # counts how many contacts we have for each pair of bodies
            counts = jnp.zeros((len(bodies), len(bodies)))

            # THIS IS QUADRATIC!
            def sc_2(carry, index, index2=None):
                index = (index, index2)
                i = current_contacts[0][index]
                j = current_contacts[1][index]
                assert i.size == 1
                assert j.size == 1
                contact = jtu.tree_map(
                    lambda x: x[index], current_contacts[2], is_leaf=eqx.is_array
                )

                return (
                    carry.at[i, j].add(1 - contact.isnan().astype(jnp.int32)),
                    None,
                )

            def sc_1(carry, index):
                return eqx.internal.scan(
                    jtu.Partial(sc_2, index2=index),
                    init=carry,
                    xs=jnp.arange(N1),
                    kind="lax",
                )

            counts, _ = eqx.internal.scan(
                sc_1, init=counts, xs=jnp.arange(N2), kind="lax"
            )

            def set_ax_2(data_arr, ind_and_key, ind2):
                ind1, key = ind_and_key
                index = (ind1, ind2)

                real_update = jtu.tree_map(
                    lambda x: x[index], current_contacts[2], is_leaf=eqx.is_array
                )
                g_upd_cond = ~real_update.isnan()

                counts[index]

                # we set the contact with probability 1 / (count + 1)
                p = jnp.array([0.5])

                key1, key2 = jr.split(key)
                cond1 = jr.bernoulli(key1, p[0], ())
                jr.bernoulli(key2, p[0], ())

                def setter(leaf, to_set):
                    i = current_contacts[0][index]
                    j = current_contacts[1][index]

                    assert i.size == 1
                    assert j.size == 1

                    out = leaf

                    out = jax.lax.cond(
                        cond1 & g_upd_cond,
                        lambda: leaf.at[i, j].set(to_set[index]),
                        lambda: out,
                    )
                    return out

                return (
                    jtu.tree_map(
                        setter, data_arr, current_contacts[2], is_leaf=eqx.is_array
                    ),
                    None,
                )

            def set_ax_1(data_arr, index_and_key):
                index, key = index_and_key
                return eqx.internal.scan(
                    jtu.Partial(set_ax_2, ind2=index),
                    init=data_arr,
                    xs=(jnp.arange(N1), jr.split(key, (N1,))),
                    kind="lax",
                )

            # apply every update recorded in contact_points
            new_all_contact_points, _ = eqx.internal.scan(
                set_ax_1,
                init=all_contacts,
                xs=(
                    jnp.arange(N2),
                    jr.split(skey, N2),
                ),
                kind="lax",
            )
            all_contacts = new_all_contact_points

        # almost done: now we have a SOA of constacts, and we want to resolve them
        # we are going to choose a random contact that is not nan along the first axis
        # and resolve it

        def choose_random_contact(arr, i, key):
            # count nans along the first axis
            is_bad = jnp.any(jnp.isnan(arr.contact_point), axis=-1)
            not_nans_count = jnp.sum(~is_bad)
            # probs is array of shape (arr[0], arr[1]) with probabilities of choosing
            probs = (~is_bad).astype(jnp.float32) / not_nans_count
            # choose a random contact
            contact_index = jax.lax.cond(
                not_nans_count == 0,
                lambda: i,
                lambda: jr.choice(key, jnp.arange(arr.contact_point.shape[0]), p=probs),
            )

            out_element = jtu.tree_map(
                lambda x: x[contact_index], arr, is_leaf=eqx.is_array
            )

            return (i, contact_index, out_element)

        chosen_contacts = eqx.filter_vmap(
            choose_random_contact, in_axes=eqx.if_array(0)
        )(all_contacts, jnp.arange(len(bodies)), jr.split(skey, len(bodies)))

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

            assert i.size == 1
            assert j.size == 1

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

            return jax.lax.cond(i == j, lambda: soa, lambda: new_soa), None

        out, _ = eqx.internal.scan(
            to_scan, init=soa_shapeless, xs=chosen_contacts, kind="lax"
        )

        # And now, the final step: convert SOA of DynamicBodies to a list of
        # AbstractBodies (statically the same as we got on the input)
        # and return it

        for i in range(len(bodies)):
            bodies[i] = (
                bodies[i]
                .load(jtu.tree_map(lambda x: x[i], out, is_leaf=eqx.is_array))
                .update_transform()
                # we need update transform here, and not in the load
                # since we want to allow illegal values when loading, e.g. booleans
            )

        return bodies
