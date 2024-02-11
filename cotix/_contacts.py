import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Float

from ._collisions import check_for_collision_convex, compute_penetration_vector_convex
from ._convex_shapes import AABB, Circle


class ContactInfo(eqx.Module):
    penetration_vector: Float[Array, "2"]
    contact_point: Float[Array, "2"]

    def __init__(self, penetration_vector, contact_point):
        self.penetration_vector = penetration_vector
        self.contact_point = contact_point

    @staticmethod
    def nan():
        return ContactInfo(jnp.zeros((2,)), jnp.array([jnp.nan, jnp.nan]))

    def isnan(self):
        return jnp.any(jnp.isnan(self.contact_point))

    def invert(self):
        return ContactInfo(-self.penetration_vector, self.contact_point)


def circle_vs_circle(a: Circle, b: Circle):
    delta = a.position - b.position
    distance = jnp.linalg.norm(delta)
    direction_between_shapes = jax.lax.cond(
        distance == 0.0, lambda: jnp.array([1.0, 0.0]), lambda: delta / distance
    )
    penetration_vector = direction_between_shapes * jnp.minimum(
        distance - (a.radius + b.radius), 0.0
    )
    contact_point = (
        b.position + direction_between_shapes * (b.radius - a.radius) + a.position
    ) / 2.0
    # check that centers lie from different sides of contact point
    # and if not, return the center that lies inside another circle
    contact_point = jax.lax.cond(
        jnp.dot(a.position - contact_point, b.position - contact_point) <= 0,
        lambda: contact_point,  # different sides
        lambda: jax.lax.cond(
            a.contains(b.position),  # same side
            lambda: b.position,  # b's center contained in a
            lambda: a.position,
        ),  # otherwise
    )

    return jax.lax.cond(
        distance <= a.radius + b.radius,
        lambda: ContactInfo(-penetration_vector, contact_point),
        lambda: ContactInfo.nan(),
    )


def aabb_vs_aabb(a: AABB, b: AABB, eps=1e-8):
    is_first_below_second = a.upper[1] <= b.lower[1]
    is_first_above_second = a.lower[1] >= b.upper[1]
    is_first_left_second = a.upper[0] <= b.lower[0]
    is_first_right_second = a.lower[0] >= b.upper[0]

    def estimate_contact():
        depths = jnp.array(
            [
                jnp.maximum(
                    a.upper[1] - b.lower[1], -eps
                ),  # eps here, so that 0 processed correctly
                jnp.maximum(b.upper[1] - a.lower[1], -eps),
                jnp.maximum(a.upper[0] - b.lower[0], -eps),
                jnp.maximum(b.upper[0] - a.lower[0], -eps),
            ]
        )
        dirs = jnp.array([[0, -1], [0, 1], [-1, 0], [1, 0]])

        index = jnp.argmin(depths)
        min_depth = jnp.clip(depths[index], a_min=0.0)
        penetration_vector = min_depth * dirs[index]
        min_upper = jnp.minimum(a.upper, b.upper)
        max_lower = jnp.maximum(a.lower, b.lower)
        return ContactInfo(penetration_vector, (min_upper + max_lower) / 2.0)

    return jax.lax.cond(
        ~(
            is_first_below_second
            | is_first_left_second
            | is_first_above_second
            | is_first_right_second
        ),
        lambda: estimate_contact(),
        lambda: ContactInfo.nan(),
    )


def circle_vs_aabb(a: Circle, b: AABB, eps=1e-6):
    disp = a.get_center() - b.get_center()
    clamp_disp = jnp.clip(disp, b.lower - b.get_center(), b.upper - b.get_center())
    ccp = (
        b.get_center() + clamp_disp
    )  # ccp = closest circle point, point on aabb that is closest to the circle
    ccp = eqx.error_if(
        ccp, ~b.contains(ccp), "Gm, closest point in the AABB is not in AABB. wut?"
    )

    vs = jnp.array(
        [
            b.lower,
            jnp.array([b.lower[0], b.upper[1]]),
            b.upper,
            jnp.array([b.upper[0], b.lower[1]]),
        ]
    )

    perfect_vertex = jnp.any(jnp.linalg.norm(vs - ccp, axis=1) < eps)

    def circle_dir_move():
        # move the aabb out of the circle, in the direction
        # that is not aligned with axes
        dir = ccp - a.position
        dir_norm = dir / jnp.linalg.norm(dir)  # TODO: check division by zero
        return ContactInfo(-(a.position + a.radius * dir_norm - ccp), ccp)

    def aligned_move():
        # now, we want to move aabb out of the circle moving only in one axis
        # while there is a hard way to do this, I am too lazy,
        # so I am just gonna try all 4 variants and see which one is the best
        dirs = jnp.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        a.position - ccp
        d_from_aabb = b.get_center() - ccp

        prods = jax.lax.map(lambda x: jnp.dot(x, d_from_aabb), dirs)
        jnp.argmax(prods)

        shifts = jnp.array(
            [
                a.position[1] + a.radius - b.lower[1],
                b.upper[1] - (a.position[1] - a.radius),
                a.position[0] + a.radius - b.lower[0],
                b.upper[0] - (a.position[0] - a.radius),
            ]
        )

        best_shift = jnp.argmin(shifts)
        return ContactInfo(-shifts[best_shift] * dirs[best_shift], ccp)

    return jax.lax.cond(
        a.contains(ccp),
        lambda: jax.lax.cond(perfect_vertex, circle_dir_move, aligned_move),
        lambda: ContactInfo.nan(),
    )


def circle_vs_polygon(circle, polygon):
    # We can magically find the penetration vector by doing this (ultra slow though)
    exists, simplex = check_for_collision_convex(
        circle.get_support, polygon.get_support
    )
    penetration_vector = compute_penetration_vector_convex(
        circle.get_support, polygon.get_support, simplex, 128
    )

    # And then we just need to find a contact point. This is easy to do in linear time,
    # just look at all the edges one after another
    def edge_point_displacement(edge, point):
        a, b = edge
        length = jnp.sum((a - b) ** 2)

        def _f1():
            t = jnp.dot(point - b, a - b) / length
            t = jnp.clip(t, 0.0, 1.0)
            projection = b + t * (a - b)
            displacement = point - projection
            return displacement

        def _f2():
            return jnp.array([jnp.inf, jnp.inf])

        return jax.lax.cond(
            jnp.all((a == jnp.zeros((2,))) & (b == jnp.zeros((2,)))), _f2, _f1
        )

    edges = polygon.get_edges()
    disps = jax.lax.map(
        jtu.Partial(edge_point_displacement, point=circle.position), edges
    )
    dists = jnp.sum(disps**2, axis=1)
    minindex = jnp.argmin(dists)
    contact_point = circle.position + disps[minindex]
    contact_point = jax.lax.cond(
        dists[minindex] > (circle.radius**2),
        lambda: circle.position,
        lambda: contact_point,
    )
    return jax.lax.cond(
        exists,
        lambda: ContactInfo(penetration_vector, contact_point),
        lambda: ContactInfo.nan(),
    )


def _contact_from_edges(edges_a, vertices_a, in_a, edges_b, vertices_b, in_b):
    def edge_vs_edge(edge_a, edge_b):
        def crs(a, b):
            return a[0] * b[1] - b[0] * a[1]

        p = edge_a[0]
        r = edge_a[1] - edge_a[0]

        q = edge_b[0]
        s = edge_b[1] - edge_b[0]

        c = crs(r, s)

        t = crs((q - p), s) / c
        u = crs((q - p), r) / c

        return jax.lax.cond(
            jnp.all(c != 0.0) & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0),
            lambda: p + r * t,
            lambda: jnp.array([jnp.nan, jnp.nan]),
        )

    def check_edges_vs_edge(edge):
        return jax.lax.map(jtu.Partial(edge_vs_edge, edge_b=edge), edges_a)

    edge_intersections_list = jax.lax.map(check_edges_vs_edge, edges_b)
    edge_intersections_list = edge_intersections_list.reshape((-1, 2))
    # now, for every vertex that is inside the other shape, we add it to the avg_sum
    n = 0.0
    avg_sum = jnp.zeros((2,))
    for vertex in vertices_a:
        cond = in_b(vertex)
        avg_sum, n = jax.lax.cond(
            cond,
            lambda x: (x[0] + vertex, x[1] + 1),
            lambda x: (x[0], x[1]),
            (avg_sum, n),
        )
    for vertex in vertices_b:
        cond = in_a(vertex)
        avg_sum, n = jax.lax.cond(
            cond,
            lambda x: (x[0] + vertex, x[1] + 1),
            lambda x: (x[0], x[1]),
            (avg_sum, n),
        )

    # and, for every edge intersection that is not nan
    for intersection in edge_intersections_list:
        cond = jnp.any(jnp.isnan(intersection))
        avg_sum, n = jax.lax.cond(
            ~cond,
            lambda x: (x[0] + intersection, x[1] + 1),
            lambda x: (x[0], x[1]),
            (avg_sum, n),
        )

    # if there are no intersections, we return nan
    return jax.lax.cond(
        n > 0.0,
        lambda: avg_sum / n,
        lambda: jnp.array([jnp.nan, jnp.nan]),
    )


def aabb_vs_polygon(aabb, polygon):
    solver_iterations = min(48, 4 + len(polygon.vertices) + 1)
    exists, simplex = check_for_collision_convex(aabb.get_support, polygon.get_support)

    penetration_vector = compute_penetration_vector_convex(
        aabb.get_support, polygon.get_support, simplex, solver_iterations
    )

    contact_point = _contact_from_edges(
        aabb.get_edges(),
        aabb.get_vertices(),
        aabb.contains,
        polygon.get_edges(),
        polygon.get_vertices(),
        polygon.contains,
    )

    return jax.lax.cond(
        exists,
        lambda: ContactInfo(penetration_vector, contact_point),
        lambda: ContactInfo.nan(),
    )


def polygon_vs_polygon(poly_a, poly_b):
    solver_iterations = min(48, len(poly_a.vertices) + len(poly_b.vertices) + 1)
    exists, simplex = check_for_collision_convex(poly_a.get_support, poly_b.get_support)

    penetration_vector = compute_penetration_vector_convex(
        poly_a.get_support, poly_b.get_support, simplex, solver_iterations
    )

    contact_point = _contact_from_edges(
        poly_a.get_edges(),
        poly_a.get_vertices(),
        poly_a.contains,
        poly_b.get_edges(),
        poly_b.get_vertices(),
        poly_b.contains,
    )

    return jax.lax.cond(
        exists,
        lambda: ContactInfo(penetration_vector, contact_point),
        lambda: ContactInfo.nan(),
    )
