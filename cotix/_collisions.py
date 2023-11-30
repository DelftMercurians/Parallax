"""
Implementations of EPA and GJK, and neat function-style interfaces to access them.
"""


import equinox as eqx
import jax
from jax import numpy as jnp, random as jr
from jaxtyping import Array, Float

from ._abstract_shapes import SupportFn
from ._geometry_utils import (
    fast_normal,
    is_point_in_triangle,
    minkowski_diff,
    random_direction,
)


def _get_collision_simplex(
    a_support_fn: SupportFn,
    b_support_fn: SupportFn,
    initial_direction: Float[Array, "2"],
):
    # TODO: test weird cases; use the direction between centers as the starting one

    # setup simplex
    simplex = jnp.zeros((3, 2))
    # .at[0] inside jit is inplace -> no performance hit
    simplex = simplex.at[0].set(
        minkowski_diff(a_support_fn, b_support_fn, initial_direction)
    )
    simplex = simplex.at[1].set(minkowski_diff(a_support_fn, b_support_fn, -simplex[0]))

    def reverse_simplex(simplex, direction):
        tmp = simplex[0]
        simplex = simplex.at[0].set(simplex[1])
        simplex = simplex.at[1].set(tmp)
        return (simplex, direction)

    def reverse_direction(simplex, direction):
        return (simplex, -direction)

    # arrange vertices clockwise
    # and make the direction a normal towrds origin
    direction = fast_normal(simplex[1] - simplex[0])
    simplex, direction = jax.lax.cond(
        jnp.dot(direction, -simplex[1]) > 0,
        reverse_simplex,
        reverse_direction,
        simplex,
        direction,
    )

    # the last point is computed only after we have correct direction
    c = minkowski_diff(a_support_fn, b_support_fn, direction)
    simplex = simplex.at[2].set(c)

    # lax while loop:
    # while cond_fn(x):
    #   x = body_fn(x)
    def body_fn(x):
        # unpack
        simplex, direction = x
        a, b, c = simplex

        ac_normal = fast_normal(c - a)
        cb_normal = fast_normal(b - c)

        # as the new direction choose the one that points towards the origin
        simplex, direction = jax.lax.cond(
            jnp.dot(ac_normal, -c) >= 0,
            lambda *_: (simplex.at[1].set(c), ac_normal),
            lambda *_: (simplex.at[0].set(c), cb_normal),
        )

        # get the point in the new direction
        c = minkowski_diff(a_support_fn, b_support_fn, direction)

        simplex = simplex.at[2].set(c)

        # pack back to the same shape
        return (simplex, direction)

    def cond_fn(x):
        # unpack
        simplex, direction = x

        # c1: we were not able to go further than origin -> no collision
        c1 = jnp.dot(simplex[2], direction) <= 0

        # (c1 & c2): simplex contains the origin
        c2 = jnp.dot(fast_normal(simplex[2] - simplex[0]), -simplex[2]) < 0
        c3 = jnp.dot(fast_normal(simplex[1] - simplex[2]), -simplex[2]) < 0

        # return the overall stopping condition
        return ~(c1 | (c2 & c3))

    # use Equinox while loop, so that we can backpropagate through it (just in case)
    simplex, direction = eqx.internal.while_loop(
        cond_fn, body_fn, (simplex, direction), kind="checkpointed", max_steps=32
    )

    # if the simplex is valid, return simplex, otherwise return zeros
    value = jax.lax.cond(
        is_point_in_triangle(jnp.zeros((2,)), simplex[0], simplex[1], simplex[2]),
        lambda x: x,
        lambda _: jnp.zeros((3, 2)),
        simplex,
    )

    return value


def _get_closest_minkowski_diff(
    a_support_fn: SupportFn,
    b_support_fn: SupportFn,
    simplex: Float[Array, "3 2"],
    solver_iterations,
) -> Float[Array, "2"]:
    # EPA (expanding polytope algorithm) forces us to have 'dynamic size' arrays.
    # JAX does not support dynamic size arrays. But we can cheat, as per usual, and just
    # use fixed size array, of let's say, 20 vertices, which will guarantee high enough
    # accuracy: at least ~pi/10 ~= 20 degrees, but in practice more like 1 / 2^(20/2)
    # and in practice, meaning 'when collision depth is small', and about 'deep'
    # we don't care, so it should be fiiiiiine
    # BTW, we are going to store edges, not vertices, since ordering is a bitch

    # TODO: handle correctly bad initial simplex (like, 3 points on one line, etc)
    solver_iterations = eqx.error_if(
        solver_iterations,
        solver_iterations < 3,
        "`solver_iterations` for the _get_closest_minkowski_diff must be"
        f"higher than 3, but was {solver_iterations}",
    )

    def displacement_to_origin(a, b):
        point = jnp.zeros((2,))
        length = jnp.sum((a - b) ** 2)

        def _f1():
            t = jnp.dot(point - b, a - b) / length

            t = jnp.clip(t, 0.0, 1.0)
            projection = b + t * (a - b)
            displacement = point - projection
            return jax.lax.cond(length == 0, lambda: -a, lambda: displacement)

        def _f2():
            return jnp.array([jnp.inf, jnp.inf])

        return jax.lax.cond(
            jnp.all((a == jnp.zeros_like(a)) & (b == jnp.zeros_like(b))), _f2, _f1
        )

    def get_closest_point_on_edge_to_point(a, b, point):
        length = jnp.sum((a - b) ** 2)

        def some_compute():
            t = jnp.dot(point - b, a - b) / length
            t = jnp.clip(t, 0.0, 1.0)
            projection = b + t * (a - b)
            displacement = point - projection
            return displacement

        return jax.lax.cond(length == 0.0, lambda: point - a, some_compute)

    def distance_to_origin(edge):
        return jnp.sum(displacement_to_origin(edge[0], edge[1]) ** 2)

    def get_closest_edge_to_origin(edges_l):
        distances_to_origin = jax.vmap(lambda x: distance_to_origin(x))(edges_l)
        edge_index = jnp.argmin(distances_to_origin)
        edge = edges_l[edge_index]
        return (edge, edge_index)

    @eqx.filter_jit
    def cond_fn(x):
        last_edge, new_point, bei, _, edges_l, prev_edge = x
        # if the edge is really small -> finish
        c1 = jnp.sum((last_edge[0] - last_edge[1]) ** 2) > 1e-9

        # we detect when the origins are ordered incorrectly, and stop
        # usually incorrect ordering happens after we hit numerical errors
        # that are big enough to break invariants
        c2 = jnp.cross(last_edge[0], last_edge[1]) >= 0

        normal = fast_normal(prev_edge[0] - prev_edge[1])
        normal = normal / jnp.linalg.norm(normal)
        d = jnp.dot(new_point, normal)
        edistance = jnp.linalg.norm(
            get_closest_point_on_edge_to_point(
                prev_edge[0], prev_edge[1], jnp.zeros((2,))
            )
        )
        c4 = (d - edistance > 1e-6) | (d <= 0)
        """"
        from matplotlib import pyplot as plt
        pp = order_clockwise(edges[:, 0, :])
        pp = list(filter(lambda x: not jnp.all(x == jnp.array([0, 0])), pp))
        pp.append(pp[0])
        pp = jnp.array(pp)
        plt.plot(pp[:, 0], pp[:, 1])
        e = prev_edge
        fn = fast_normal(e[0] - e[1])
        plt.plot([e[1][0], e[1][0] + fn[0]], [e[1][1], e[1][1] + fn[1]], 'r')
        plt.plot(e[:, 0], e[:, 1], 'g')
        plt.show()
        breakpoint()
        """
        final_c = c4 & (~jnp.any(jnp.isnan(last_edge))) & c1 & c2
        return final_c

    @eqx.filter_jit
    def body_fn(x):
        best_edge, _, best_edge_index, i, edges_l, _ = x

        # now we split edge that is closest to the origin into two,
        # taking the support point along normal as the third point
        normal = fast_normal(best_edge[0] - best_edge[1])
        normal = normal / jnp.linalg.norm(normal)  # TODO: remove this renormalization
        new_point = minkowski_diff(a_support_fn, b_support_fn, normal)

        # lets replace current edge with edge[0], new point:
        a = jnp.array([best_edge[0], new_point])
        b = jnp.array([new_point, best_edge[1]])

        def replac(edges_l):
            edges_l = edges_l.at[best_edge_index].set(a)
            edges_l = edges_l.at[i + 3].set(b)
            return edges_l

        edges_l = replac(edges_l)

        new_best_edge, new_best_edge_index = get_closest_edge_to_origin(edges_l)
        return new_best_edge, new_point, new_best_edge_index, i + 1, edges_l, best_edge

    edges = jnp.zeros((solver_iterations + 3, 2, 2))
    edges = edges.at[0].set(jnp.array([simplex[0], simplex[1]]))
    edges = edges.at[1].set(jnp.array([simplex[1], simplex[2]]))
    edges = edges.at[2].set(jnp.array([simplex[2], simplex[0]]))

    best_edge, index = get_closest_edge_to_origin(edges)

    #
    # with jax.disable_jit():
    #   best_edge, _, _, _, edges, prev_best_edge = eqx.internal.while_loop(
    #        cond_fn,
    #        body_fn,
    #        (best_edge, simplex[2], index, jnp.array(0), edges, edges[0]),
    #        kind="checkpointed",
    #        max_steps=solver_iterations,
    #    )
    x = (best_edge, simplex[2], index, jnp.array(0), edges, edges[0])
    # for i in range(solver_iterations):
    # if ~cond_fn(x):
    #    break
    #    x = jax.lax.cond(cond_fn(x), lambda: body_fn(x), lambda:x)
    # x = body_fn(x)
    x, _ = jax.lax.scan(
        lambda x, _: (jax.lax.cond(cond_fn(x), lambda: body_fn(x), lambda: x), _),
        x,
        None,
        length=solver_iterations,
    )
    best_edge, _, _, _, edges, prev_best_edge = x

    best_edge, _ = get_closest_edge_to_origin(edges)  # return best_point
    # print(edges)

    return get_closest_point_on_edge_to_point(
        best_edge[0], best_edge[1], jnp.zeros((2,))
    )


# Now, lets define the functions that are available to users
def check_for_collision_convex(
    support_a: SupportFn,
    support_b: SupportFn,
    initial_direction: Float[Array, "2"] = jnp.array([jnp.nan, jnp.nan]),
    key=None,
):
    """
    Applies GJK algorithm given two support functions (like real Callables),
    and optionally a starting direction.
    """
    if key is None:
        key = jr.PRNGKey(1)

    def rnd():
        return random_direction(key)

    def rnd_plus():
        return random_direction(key) * 0.1 + initial_direction * 0.9

    initial_direction = jax.lax.cond(
        jnp.any(jnp.isnan(initial_direction)), rnd, rnd_plus
    )
    simplex = _get_collision_simplex(support_a, support_b, initial_direction)
    area = jnp.cross(simplex[1] - simplex[0], simplex[2] - simplex[0])
    c = (
        jnp.all(simplex == jnp.zeros_like(simplex))
        | jnp.any(jnp.isnan(simplex))
        | (area == 0)
    )
    return jax.lax.cond(
        c,
        lambda: (False, jnp.nan * simplex),
        lambda: (True, simplex),
    )


def compute_penetration_vector_convex(
    support_a: SupportFn,
    support_b: SupportFn,
    simplex: Float[Array, "3 2"],
    solver_iterations=48,
):
    """
    Computes a conservative estimate of the minimum penetration
    vector using EPA algorithm.

    Minimum penetration vector is a vector of minimal length,
    shift by which separates the bodies.
    """
    penetration = _get_closest_minkowski_diff(
        support_a, support_b, simplex, solver_iterations
    )  # TODO: Decrease this value to 32
    return penetration
