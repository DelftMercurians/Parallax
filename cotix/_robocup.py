import equinox as eqx
from jax import numpy as jnp, tree_util as jtu

from ._bodies import AnyBody
from ._constraints import AbstractConstraint
from ._convex_shapes import AABB, Circle
from ._universal_shape import UniversalShape


class RoboCupEnv(eqx.Module):
    bodies: list[AnyBody]
    colors: list[tuple]
    edge_colors: list[tuple]
    constraints: list[AbstractConstraint]

    def __init__(self):
        field_dim = (10.4, 7.4)
        play_area_dim = (9, 6)
        goal_dim = (0.2, 1)
        ball_radius = 0.022

        field_shape = AABB(
            -jnp.array(field_dim) / 2,
            jnp.array(field_dim) / 2,
        )

        play_area_shape = AABB(
            -jnp.array(play_area_dim) / 2,
            jnp.array(play_area_dim) / 2,
        )
        # Goal
        yellow_goal_bounding_shape = AABB(
            jnp.array([play_area_shape.lower[0] - goal_dim[0], -goal_dim[1] / 2]),
            jnp.array([play_area_shape.lower[0], goal_dim[1] / 2]),
        )

        # but actual goals are 3 aabbs with width of 0.01 that placed on the boundaries
        goal_width = 0.01
        yellow_goal_shape = UniversalShape(
            AABB(
                yellow_goal_bounding_shape.lower,
                yellow_goal_bounding_shape.lower + jnp.array([goal_width, goal_dim[1]]),
            ),
            AABB(
                yellow_goal_bounding_shape.lower - jnp.array([-goal_width, 0]),
                yellow_goal_bounding_shape.lower + jnp.array([goal_dim[0], goal_width]),
            ),
            AABB(
                yellow_goal_bounding_shape.upper - jnp.array([goal_dim[0], goal_width]),
                yellow_goal_bounding_shape.upper,
            ),
        )

        yellow_goal_body = AnyBody(
            shape=yellow_goal_shape,
            mass=jnp.array(jnp.inf),
            position=jnp.zeros((2,)),
            angle=jnp.array(0.0),
            velocity=jnp.zeros((2,)),
            angular_velocity=jnp.array(0.0),
            elasticity=jnp.array(0.5),
        )

        # now, the blue goal is just a reflection by y axis
        def reflect_aabb_by_y_axis(aabb):
            if not isinstance(aabb, AABB):
                return aabb
            return AABB(
                jnp.array([-aabb.upper[0], aabb.lower[1]]),
                jnp.array([-aabb.lower[0], aabb.upper[1]]),
            )

        blue_goal_shape = jtu.tree_map(
            reflect_aabb_by_y_axis,
            yellow_goal_shape,
            is_leaf=lambda x: isinstance(x, AABB),
        )

        blue_goal_body = AnyBody(
            shape=blue_goal_shape,
            mass=jnp.array(jnp.inf),
            position=jnp.zeros((2,)),
            angle=jnp.array(0.0),
            velocity=jnp.zeros((2,)),
            angular_velocity=jnp.array(0.0),
            elasticity=jnp.array(0.5),
        )

        # Now, let's talk about important areas (not bodies) of the field.
        # firstly, the whole field-ish area. This one is to just make sure that
        # objects don't 'escape' the field
        field_body = AnyBody(
            shape=UniversalShape(field_shape),
            mass=jnp.array(jnp.inf),
            position=jnp.zeros((2,)),
            angle=jnp.array(0.0),
            velocity=jnp.zeros((2,)),
            angular_velocity=jnp.array(0.0),
            is_area=True,
        )

        # the next is playable area. This one is about out zones and stuff, blah blah
        play_area_body = AnyBody(
            shape=UniversalShape(play_area_shape),
            mass=jnp.array(jnp.inf),
            position=jnp.zeros((2,)),
            angle=jnp.array(0.0),
            velocity=jnp.zeros((2,)),
            angular_velocity=jnp.array(0.0),
            is_area=True,
        )

        # in addition, let's add a ball
        ball_body = AnyBody(
            shape=UniversalShape(
                Circle(position=jnp.zeros((2,)), radius=jnp.array(ball_radius) * 3)
            ),
            mass=jnp.array(0.5),
            position=jnp.array([0.0, 0.0]),
            angle=jnp.array(0.0),
            velocity=jnp.array([1.0, 0.01]),
            angular_velocity=jnp.array(10.0),
            elasticity=jnp.array(1.0),
        )

        self.bodies = [
            field_body,
            play_area_body,
            yellow_goal_body,
            blue_goal_body,
            ball_body,
        ]

        # field is green, etc etc
        self.colors = [(0, 180, 0)] * 2 + [(255, 255, 0), (0, 128, 255), (255, 0, 0)]
        self.edge_colors = [(255, 255, 255)] * 2 + [
            (255, 255, 0),
            (0, 128, 255),
            None,
        ]

        self.constraints = []

    def draw(self, painter):
        for edge_color, color, body in zip(self.edge_colors, self.colors, self.bodies):
            shape = body.shape
            if color is not None:
                shape.draw(painter, color=color)
            else:
                shape.draw(painter)

            if edge_color is not None:
                shape.drawEdges(painter, color=edge_color)
        painter.next()
