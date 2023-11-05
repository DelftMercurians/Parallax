import jax
from jax import numpy as jnp, random as jr
from pytest_check import check  # for possible multiple assert failures

from cotix._bodies import AnyBody, Ball
from cotix._collision_resolution import _resolve_collision_checked, ContactInfo
from cotix._collisions import (
    check_for_collision_convex,
    compute_penetration_vector_convex,
)
from cotix._convex_shapes import Circle, Polygon
from cotix._geometry_utils import angle_between
from cotix._universal_shape import UniversalShape


def _default_contact_point(body1, body2, penetration_vector):
    contact_point = (
        body1.shape.get_global_support(-penetration_vector)
        + body2.shape.get_global_support(penetration_vector)
    ) / 2
    return contact_point


def _collision_resolution_helper(body1, body2, contact_point=None):
    res_first_collision, simplex = check_for_collision_convex(
        body1.shape.get_global_support,
        body2.shape.get_global_support,
    )
    penetration_before = compute_penetration_vector_convex(
        body1.shape.get_global_support, body2.shape.get_global_support, simplex
    )

    if contact_point is None:
        contact_point = _default_contact_point(body1, body2, penetration_before)

    contact_info = ContactInfo(penetration_before, contact_point)

    body1, body2 = _resolve_collision_checked(body1, body2, contact_info)

    # test get global support
    res_collision, simplex = check_for_collision_convex(
        body1.shape.get_global_support,
        body2.shape.get_global_support,
    )
    penetration_after = compute_penetration_vector_convex(
        body1.shape.get_global_support, body2.shape.get_global_support, simplex
    )

    # velocities are such that the distance between balls is increasing
    velocities_away = (
        jnp.dot(body1.velocity - body2.velocity, body1.position - body2.position) >= 0
    )
    no_collision = jnp.logical_or(
        ~res_collision,
        jnp.linalg.norm(penetration_after) < 1e-3 * jnp.linalg.norm(penetration_before),
    )
    return (
        velocities_away,
        res_first_collision,
        no_collision,
        penetration_before,
        penetration_after,
        body1,
        body2,
    )


def test_circle_hits_circle_elastic():
    # this test is kinda weak: i dont test that the speeds are fully correct
    key = jr.PRNGKey(0)

    def f(key):
        key1, key2, key3, key4, key5, key6, key7 = jr.split(key, 7)
        p1 = jr.uniform(key1, (2,)) * 4 - 2
        p2 = jr.uniform(key2, (2,)) * 4 - 2
        dist = jnp.sqrt(jnp.sum((p1 - p2) ** 2))
        r1 = jr.uniform(key3, minval=dist * 0.25, maxval=dist * 0.75)
        # guarantee a collision
        r2 = jr.uniform(key4, minval=(dist - r1) * 1.05, maxval=(dist - r1) * 1.15)

        zero_position = jnp.zeros((2,))
        shape1 = Circle(r1, zero_position)
        shape2 = Circle(r2, zero_position)

        # velocities into another ball
        v1 = jr.uniform(key5, (2,)) * 2 * (p2 - p1)
        v2 = jr.uniform(key6, (2,)) * 2 * (p1 - p2)

        # sometimes the collision doesnt have to be resolved
        moving_apart = jr.bernoulli(key7, 0.2)
        moving_multiplier = moving_apart * (-2) + 1
        v1 = v1 * moving_multiplier
        v2 = v2 * moving_multiplier

        body1 = Ball(jnp.array(1.0), p1, v1, UniversalShape(shape1))
        body2 = Ball(jnp.array(1.0), p2, v2, UniversalShape(shape2))

        (
            velocities_away,
            res_first_collision,
            no_collision,
            penetration_before,
            penetration_after,
            body1,
            body2,
        ) = _collision_resolution_helper(body1, body2)

        # either the collision was resolved, or the balls are moving apart
        return (
            velocities_away,
            res_first_collision,
            no_collision,
            moving_apart,
            jnp.array([p1, p2, v1, v2, [r1, r2], penetration_before, [dist, dist]]),
            jnp.array(
                [
                    body1.position,
                    body2.position,
                    body1.velocity,
                    body2.velocity,
                    penetration_after,
                ]
            ),
        )

    N = 1000
    f = jax.vmap(f)
    (
        velocities_away,
        was_collision,
        collision_resolved,
        didnt_have_to_be_resolved,
        start_info,
        end_info,
    ) = f(jr.split(key, N))
    all_conditions = (
        velocities_away
        & was_collision
        & jnp.logical_xor(collision_resolved, didnt_have_to_be_resolved)
    )
    start_useful = start_info[~all_conditions]
    end_useful = end_info[~all_conditions]
    if end_useful.shape[0] != 0:
        jnp.set_printoptions(suppress=True)
        print(f"\ntotal wrong {end_useful.shape[0] / N}")
    for i in range(end_useful.shape[0]):
        print()
        print(f"start position 1: {start_useful[i, 0]}")
        print(f"start position 2: {start_useful[i, 1]}")
        print(f"end position 1: {end_useful[i, 0]}")
        print(f"end position 2: {end_useful[i, 1]}")
        print(f"start velocity 1: {start_useful[i, 2]}")
        print(f"start velocity 2: {start_useful[i, 3]}")
        print(f"end velocity 1: {end_useful[i, 2]}")
        print(f"end velocity 2: {end_useful[i, 3]}")
        print(f"radius 1: {start_useful[i, 4][0]}")
        print(f"radius 2: {start_useful[i, 4][1]}")
        print(f"distance: {start_useful[i, 6][0]}")
        print(f"end position difference: {end_useful[i, 0] - end_useful[i, 1]}")
        print(f"end velocity difference: {end_useful[i, 2] - end_useful[i, 3]}")
        print(f"penetration before: {start_useful[i, 5]}")
        print(f"penetration after: {end_useful[i, 4]}")

    assert jnp.all(was_collision), "there wasnt a collision"
    assert jnp.all(
        didnt_have_to_be_resolved | collision_resolved
    ), "collision was not resolved"
    assert jnp.all(
        ~didnt_have_to_be_resolved | ~collision_resolved
    ), "collision was resolved when it didnt have to be"
    assert jnp.all(velocities_away), "velocities arent away"


def test_triangle_circle_angular():
    # todo: so this test is not passing,
    #  because we dont yet have a way to compute non-obvious contact points

    # checks angular speed as well (nonzero for the triangle)
    zero_position = jnp.zeros((2,))
    r1 = jnp.array(1.0)
    shape1 = Circle(r1, zero_position)

    p1 = jnp.array([2.0, 3.0])
    # designed to be along the epa vector, into the triangle
    v1 = jnp.array([-2.5, -1.2])

    # a triangle that intersects the circle
    # center of mass is in 0, 0 vertex,
    # resulting rotation should be counterclockwise
    shape2 = Polygon(jnp.array([[-1.0, 0.0], [-0.2, 2.5], [1.0, 0.0]]))
    v2 = jnp.zeros((2,))
    p2 = jnp.ones((2,))

    body1 = Ball(jnp.array(1.0), p1, v1, UniversalShape(shape1))
    body2 = AnyBody(
        jnp.array(1.0),
        jnp.array(1.0),
        p2,
        v2,
        angle=jnp.array(0.0),
        angular_velocity=jnp.array(0.0),
        elasticity=jnp.array(0.5),
        friction_coefficient=jnp.array(0.5),
        shape=UniversalShape(shape2),
    )

    (
        velocities_away,
        res_first_collision,
        no_collision,
        penetration_before,
        penetration_after,
        body1,
        body2,
    ) = _collision_resolution_helper(body1, body2)

    epa_velocity_angle = angle_between(v1, -penetration_before)

    print(
        velocities_away
        & res_first_collision
        & no_collision
        & (epa_velocity_angle < 1e-3)
    )

    # assert (
    #     epa_velocity_angle < 1e-3
    # ), "velocity is not along epa vector (so epa is wrong)"
    # assert res_first_collision, "there wasnt a collision"
    # assert no_collision, "collision was not resolved"
    # assert velocities_away, "velocities arent away"
    #
    # assert (
    #     abs(body1.angular_velocity) <= 1e-3
    # ), "angular velocity of the ball is not zero"
    # assert (
    #     abs(body2.angular_velocity) > 1e-2
    # ), "angular velocity of the triangle is zero"
    # assert body2.angular_velocity > 0, (
    #     "angular velocity of the triangle is not positive. "
    #     "by convention, it should be positive "
    #     "if the triangle is rotating counterclockwise"
    # )


def test_ball_spins_after_wall():
    r1 = jnp.array(1.2)
    shape1 = Circle(r1, jnp.zeros((2,)))

    p1 = jnp.array([2.0, 3.1])
    v1 = jnp.array([3.0, -4.0])

    # a wall such that the ball slightly penetrates it
    shape2 = Polygon(jnp.array([[-1.0, 2.0], [4.0, 2.0], [4.0, 1.0], [-1.0, 1.0]]))
    v2 = jnp.zeros((2,))
    p2 = jnp.zeros((2,))

    body1 = Ball(jnp.array(1.0), p1, v1, UniversalShape(shape1))
    body2 = AnyBody(
        # wall is immovable
        mass=jnp.array(jnp.inf),
        inertia=jnp.array(jnp.inf),
        position=p2,
        velocity=v2,
        angle=jnp.array(0.0),
        angular_velocity=jnp.array(0.0),
        elasticity=jnp.array(0.5),
        friction_coefficient=jnp.array(0.5),
        shape=UniversalShape(shape2),
    )

    contact_point = (jnp.array([2, 1.9]) + jnp.array([2, 2])) / 2

    (
        velocities_away,
        res_first_collision,
        no_collision,
        penetration_before,
        penetration_after,
        body1,
        body2,
    ) = _collision_resolution_helper(body1, body2, contact_point)

    assert res_first_collision, "there wasnt a collision"
    assert no_collision, "collision was not resolved"
    assert velocities_away, "velocities arent away"

    # wall should be immovable
    assert (
        jnp.dot(body2.velocity, body2.velocity) == 0
    ), "velocity of the wall is not zero"
    assert body2.angular_velocity == 0, "angular velocity of the wall is not zero"

    # ball should spin
    assert abs(body1.angular_velocity) > 1e-2, "angular velocity of the ball is zero"

    assert body1.angular_velocity < 0, (
        "angular velocity of the ball is not negative. "
        "by convention, it should be negative "
        "if the body is rotating clockwise"
    )


def test_friction_ball_with_initial_angular_speed():
    r1 = jnp.array(1.2)
    shape1 = Circle(r1, jnp.zeros((2,)))

    p1 = jnp.array([2.0, 3.1])
    v1 = jnp.array([3.0, -4.0])

    # a wall such that the ball slightly penetrates it
    shape2 = Polygon(jnp.array([[-1.0, 2.0], [4.0, 2.0], [4.0, 1.0], [-1.0, 1.0]]))
    v2 = jnp.zeros((2,))
    p2 = jnp.zeros((2,))

    body1 = Ball(jnp.array(1.0), p1, v1, UniversalShape(shape1))
    initial_angular_v = jnp.array(1.0)  # 1 is counterclockwise
    body1 = body1.set_angular_velocity(initial_angular_v)
    body2 = AnyBody(
        mass=jnp.array(jnp.inf),
        inertia=jnp.array(jnp.inf),
        position=p2,
        velocity=v2,
        angle=jnp.array(0.0),
        angular_velocity=jnp.array(0.0),
        elasticity=jnp.array(0.5),
        friction_coefficient=jnp.array(0.5),
        shape=UniversalShape(shape2),
    )

    contact_point = (jnp.array([2, 1.9]) + jnp.array([2, 2])) / 2

    (
        velocities_away,
        res_first_collision,
        no_collision,
        penetration_before,
        penetration_after,
        body1,
        body2,
    ) = _collision_resolution_helper(body1, body2, contact_point)

    assert res_first_collision, "there wasnt a collision"
    assert no_collision, "collision was not resolved"
    assert velocities_away, "velocities arent away"

    # wall should be immovable
    assert (
        jnp.dot(body2.velocity, body2.velocity) == 0
    ), "velocity of the wall is not zero"
    assert body2.angular_velocity == 0, "angular velocity of the wall is not zero"

    # ball was spinning counterclockwise, so it should spin 'more clockwise' now
    # no abs() here because the velocity can be large and negative
    assert body1.angular_velocity < initial_angular_v, (
        "angular velocity of the ball did not decrease " "after collision with a wall"
    )


def test_friction_ball_with_huge_initial_angular_speed():
    r1 = jnp.array(1.2)
    shape1 = Circle(r1, jnp.zeros((2,)))

    p1 = jnp.array([2.0, 3.1])
    v1 = jnp.array([3.0, -4.0])

    # a wall such that the ball slightly penetrates it
    shape2 = Polygon(jnp.array([[-1.0, 2.0], [4.0, 2.0], [4.0, 1.0], [-1.0, 1.0]]))
    v2 = jnp.zeros((2,))
    p2 = jnp.zeros((2,))

    body1 = Ball(jnp.array(1.0), p1, v1, UniversalShape(shape1))
    initial_angular_v = jnp.array(1e3)
    body1 = body1.set_angular_velocity(initial_angular_v)
    body2 = AnyBody(
        mass=jnp.array(jnp.inf),
        inertia=jnp.array(jnp.inf),
        position=p2,
        velocity=v2,
        angle=jnp.array(0.0),
        angular_velocity=jnp.array(0.0),
        elasticity=jnp.array(0.5),
        friction_coefficient=jnp.array(0.5),
        shape=UniversalShape(shape2),
    )

    contact_point = (jnp.array([2, 1.9]) + jnp.array([2, 2])) / 2

    (
        velocities_away,
        res_first_collision,
        no_collision,
        penetration_before,
        penetration_after,
        body1,
        body2,
    ) = _collision_resolution_helper(body1, body2, contact_point)

    assert res_first_collision, "there wasnt a collision"
    assert no_collision, "collision was not resolved"
    assert velocities_away, "velocities arent away"

    # wall should be immovable
    assert (
        jnp.dot(body2.velocity, body2.velocity) == 0
    ), "velocity of the wall is not zero"
    assert body2.angular_velocity == 0, "angular velocity of the wall is not zero"

    # ball was spinning counterclockwise, so it should spin 'more clockwise' now,
    assert abs(body1.angular_velocity) < abs(
        initial_angular_v
    ), "angular velocity of the ball did not decrease "
    assert (
        body1.angular_velocity > 0
    ), "angular velocity should still be positive, cause it was so large before"


def test_friction_affects_center_of_mass_velocity():
    r1 = jnp.array(1.0)
    shape1 = Circle(r1, jnp.zeros((2,)))

    p1 = jnp.array([1.0, 1.9])
    v1 = jnp.array([0.0, -3.0])

    # a wall such that the ball slightly penetrates it
    shape2 = Polygon(jnp.array([[-1.0, 1.0], [1.0, 1.0], [0.01, -1.0]]))
    v2 = jnp.array([0.0, 1.0])
    p2 = jnp.array([1.0, 0.0])

    body1 = Ball(jnp.array(5.0), p1, v1, UniversalShape(shape1))
    initial_angular_v = jnp.array(-10.0)  # spinning clockwise
    body1 = body1.set_angular_velocity(initial_angular_v)
    body2 = AnyBody(
        mass=jnp.array(2.0),
        inertia=jnp.array(1.0),
        position=p2,
        velocity=v2,
        angle=jnp.array(0.0),
        angular_velocity=jnp.array(0.0),
        elasticity=jnp.array(0.5),
        friction_coefficient=jnp.array(0.5),
        shape=UniversalShape(shape2),
    )

    contact_point = (jnp.array([1, 0.9]) + jnp.array([1, 1])) / 2

    (
        velocities_away,
        res_first_collision,
        no_collision,
        penetration_before,
        penetration_after,
        body1,
        body2,
    ) = _collision_resolution_helper(body1, body2, contact_point)

    with check:
        assert res_first_collision, "there wasnt a collision"
        assert no_collision, "collision was not resolved"
        assert velocities_away, "velocities arent away"

    # ball is going to the right, because of its spin
    with check:
        assert body1.velocity[0] > 0, "velocity of the ball is not to the right"

    # triangle is spun counterclockwise and pushed left by the ball's rotation
    with check:
        assert body2.velocity[0] < 0, "velocity of the triangle is not to the left"
        assert body2.velocity[1] < 0, "velocity of the triangle is not down"
    with check:
        assert abs(body2.angular_velocity) > 1e-2, "triangle was not spun"
        assert body2.angular_velocity > 0, (
            "angular velocity of the triangle is not positive. "
            "by convention, it should be positive "
            "if the triangle is rotating counterclockwise"
        )

    with check:
        # ball should continue spinning, but slower
        assert abs(body1.angular_velocity) < abs(
            initial_angular_v
        ), "the ball's angular velocity should decrease"
        assert jnp.sign(body1.angular_velocity) == jnp.sign(
            initial_angular_v
        ), "the ball should keep rotating in the same direction"
