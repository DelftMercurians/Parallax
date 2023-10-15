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
    A_support_fn: SupportFn,
    B_support_fn: SupportFn,
    initial_direction: Float[Array, "2"],
):
    # TODO: test weird cases; use the direction between centers as the starting one

    # setup simplex
    simplex = jnp.zeros((3, 2))
    # .at[0] inside jit is inplace -> no performance hit
    simplex = simplex.at[0].set(
        minkowski_diff(A_support_fn, B_support_fn, initial_direction)
    )
    simplex = simplex.at[1].set(minkowski_diff(A_support_fn, B_support_fn, -simplex[0]))

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
    c = minkowski_diff(A_support_fn, B_support_fn, direction)
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
        c = minkowski_diff(A_support_fn, B_support_fn, direction)

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
        lambda x: jnp.zeros((3, 2)),
        simplex,
    )

    return value


def _get_closest_minkowski_diff(
    A_support_fn: SupportFn,
    B_support_fn: SupportFn,
    simplex: Float[Array, "3 2"],
    solver_iterations,
):
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
            projection = b + t * (a - b)
            displacement = point - projection
            return displacement

        def _f2():
            return jnp.array([jnp.inf, jnp.inf])

        return jax.lax.cond(length < 1e-14, _f2, _f1)

    def get_closest_point_on_edge_to_point(a, b, point):
        length = jnp.sum((a - b) ** 2)

        t = jnp.dot(point - b, a - b) / length
        projection = b + t * (a - b)
        displacement = point - projection
        return displacement

    def distance_to_origin(edge):
        return jnp.sum(displacement_to_origin(edge[0], edge[1]) ** 2)

    def get_closest_edge_to_origin(edges):
        distances_to_origin = jax.vmap(lambda x: distance_to_origin(x))(edges)
        edge_index = jnp.argmin(distances_to_origin)
        edge = edges[edge_index]
        return (edge, edge_index)

    def cond_fn(x):
        last_edge, new_point, _, index, edges, prev_edge = x
        # if the edge is really small -> finish
        c1 = jnp.sum((last_edge[0] - last_edge[1]) ** 2) > 1e-9

        # we detect when the origins are ordered incorrectly, and stop
        # usually incorrect ordering happens after we hit numerical errors
        # that are big enough to break invariants
        c2 = (
            jnp.cross(last_edge[0], last_edge[1]) > 0
        )  # TODO: add more conditions, since this one is unstable af
        c3 = jnp.sum((last_edge - prev_edge) ** 2) > 1e-7
        return c1 & c2 & c3

    def body_fn(x):
        best_edge, _, best_edge_index, i, edges, _ = x

        # now we split edge that is closest to the origin into two,
        # taking the support point along normal as the third point
        normal = fast_normal(best_edge[0] - best_edge[1])
        normal = normal / (
            jnp.absolute(normal[0]) + jnp.absolute(normal[1])
        )  # TODO: remove this renormalization
        new_point = minkowski_diff(A_support_fn, B_support_fn, normal)
        # lets replace current edge with edge[0], new point:
        edges = edges.at[best_edge_index].set(jnp.array([best_edge[0], new_point]))
        # and insert in the end new_point, edge[1]
        edges = edges.at[i + 3].set(jnp.array([new_point, best_edge[1]]))

        new_best_edge, new_best_edge_index = get_closest_edge_to_origin(edges)
        return new_best_edge, new_point, new_best_edge_index, i + 1, edges, best_edge

    edges = jnp.zeros((solver_iterations, 2, 2))
    edges = edges.at[0].set(jnp.array([simplex[0], simplex[1]]))
    edges = edges.at[1].set(jnp.array([simplex[1], simplex[2]]))
    edges = edges.at[2].set(jnp.array([simplex[2], simplex[0]]))

    edge, index = get_closest_edge_to_origin(edges)

    best_edge, best_point, _, i, edges, prev_best_edge = eqx.internal.while_loop(
        cond_fn,
        body_fn,
        (edge, edge[0], index, jnp.array(0), edges, jnp.zeros((2, 2))),
        kind="checkpointed",
        max_steps=solver_iterations,
    )
    best_edge = prev_best_edge

    return get_closest_point_on_edge_to_point(
        best_edge[0], best_edge[1], jnp.zeros((2,))
    )


# Now, lets define the functions that are available to users
def check_for_collision_convex(
    support_A: SupportFn,
    support_B: SupportFn,
    initial_direction=random_direction(jr.PRNGKey(1)),
):
    simplex = _get_collision_simplex(support_A, support_B, initial_direction)
    return jax.lax.cond(
        jnp.all(simplex == jnp.zeros_like(simplex)),
        lambda: (False, jnp.nan * simplex),
        lambda: (True, simplex),
    )


def compute_penetration_vector_convex(
    support_A: SupportFn,
    support_B: SupportFn,
    simplex: Float[Array, "3 2"],
    solver_iterations=48,
):
    penetration = _get_closest_minkowski_diff(
        support_A, support_B, simplex, solver_iterations
    )  # TODO: Decrease this value to 32
    return penetration
