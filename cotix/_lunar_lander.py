import equinox as eqx
from jax import numpy as jnp, random as jr

from ._bodies import AnyBody
from ._collision_resolution import apply_impulse
from ._convex_shapes import Polygon4, Polygon6
from ._geometry_utils import rotate
from ._universal_shape import UniversalShape


LANDER_POLY = [
    (-14, +17),
    (-17, 0),
    (-17, -10),
    (+17, -10),
    (+17, 0),
    (+14, +17),
]

LEG_AWAY = 24
LEG_DOWN = 8
LEG_W, LEG_H = 2, 8
LEG_ANGLE = -0.3


class LunarLander(eqx.Module, strict=True):
    bodies: list[AnyBody]

    def __init__(self, key=jr.PRNGKey(0)):
        lander_shape = Polygon6(jnp.array(LANDER_POLY) * 0.05)

        left_leg_shape = Polygon4(
            jnp.array(
                [
                    (-LEG_W, -LEG_H),
                    (+LEG_W, -LEG_H),
                    (+LEG_W, LEG_H),
                    (-LEG_W, LEG_H),
                ]
            )
        )

        # rotate the leg
        def rotate(vertices, angle):
            return jnp.dot(
                vertices,
                jnp.array(
                    [
                        [jnp.cos(angle), -jnp.sin(angle)],
                        [jnp.sin(angle), jnp.cos(angle)],
                    ]
                ),
            )

        left_leg_shape = eqx.tree_at(
            lambda x: x.vertices,
            left_leg_shape,
            rotate(left_leg_shape.vertices, LEG_ANGLE),
        )

        # scale the leg
        left_leg_shape = eqx.tree_at(
            lambda x: x.vertices,
            left_leg_shape,
            left_leg_shape.vertices * 0.05,
        )

        right_leg_shape = eqx.tree_at(
            lambda x: x.vertices,
            left_leg_shape,
            left_leg_shape.vertices * jnp.array([-1.0, 1.0]),
        )

        lander_center = jnp.array([0.0, 5.0])
        lander_left_leg = jnp.array([LEG_AWAY, -LEG_DOWN]) * 0.05 + lander_center
        lander_right_leg = jnp.array([-LEG_AWAY, -LEG_DOWN]) * 0.05 + lander_center

        lander = AnyBody(
            position=lander_center,
            velocity=jnp.array([0.0, 0.0]),
            angular_velocity=jnp.array(0.0),
            mass=jnp.array(30.0),
            inertia=jnp.array(30.0),
            angle=jnp.array(0.01),
            friction_coefficient=jnp.array(0.1),
            shape=UniversalShape(lander_shape),
        )

        right_leg = AnyBody(
            position=lander_right_leg,
            velocity=jnp.array([0.0, 0.0]),
            angular_velocity=jnp.array(0.0),
            inertia=jnp.array(1.0),
            friction_coefficient=jnp.array(0.1),
            shape=UniversalShape(right_leg_shape),
        )

        left_leg = AnyBody(
            position=lander_left_leg,
            velocity=jnp.array([0.0, 0.0]),
            angular_velocity=jnp.array(0.0),
            friction_coefficient=jnp.array(0.1),
            inertia=jnp.array(1.0),
            shape=UniversalShape(left_leg_shape),
        )

        # the ground is generated as a bunch of polygon4s that are
        # connected to each other, with random-ish heights inbetween 0 and 1
        k1, k2, k3, k4, k5 = jr.split(key, 5)
        heights = jr.uniform(k1, (8,), minval=-5.0, maxval=5.0)
        heights = heights.at[0].set(heights[0] * 10)
        heights = heights.at[3].set(-2.0)
        heights = heights.at[-4].set(-2.0)
        heights = heights.at[-1].set(heights[-1] * 10)

        positions = [
            -100,
            jr.uniform(k2, (), minval=-12.0, maxval=-9.0),
            jr.uniform(k3, (), minval=-8.0, maxval=-4.0),
            -2,
            2,
            jr.uniform(k4, (), minval=4.0, maxval=8.0),
            jr.uniform(k5, (), minval=9.0, maxval=12.0),
            100,
        ]
        polygons = []
        for i in range(len(heights) - 1):
            p1 = jnp.array([positions[i], heights[i]])
            p2 = jnp.array([positions[i], -10])
            p3 = jnp.array([positions[i + 1], heights[i + 1]])
            p4 = jnp.array([positions[i + 1], -10])
            polygons.append(Polygon4(jnp.array([p1, p2, p3, p4])))

        ground = AnyBody(
            position=jnp.array([0.0, 0.0]),
            mass=jnp.array(jnp.inf),
            inertia=jnp.array(jnp.inf),
            elasticity=jnp.array(0.1),
            friction_coefficient=jnp.array(0.1),
            shape=UniversalShape(*polygons),
        )

        self.bodies = [lander, right_leg, left_leg, ground]

    def step(self):
        lander_left_joint_1 = jnp.array([LEG_AWAY, -LEG_DOWN]) * 0.05
        lander_left_joint_1 = (
            rotate(lander_left_joint_1, self.bodies[0].angle) + self.bodies[0].position
        )

        lander_left_joint_2 = jnp.array([LEG_AWAY, -LEG_DOWN + 8]) * 0.05
        lander_left_joint_2 = (
            rotate(lander_left_joint_2, self.bodies[0].angle) + self.bodies[0].position
        )

        left_leg_joint_1 = self.bodies[2].position
        left_leg_joint_2 = self.bodies[2].position + rotate(
            jnp.array([0.0, 0.4]), self.bodies[2].angle
        )

        lander_right_joint_1 = jnp.array([-LEG_AWAY, -LEG_DOWN]) * 0.05
        lander_right_joint_1 = (
            rotate(lander_right_joint_1, self.bodies[0].angle) + self.bodies[0].position
        )

        lander_right_joint_2 = jnp.array([-LEG_AWAY, -LEG_DOWN + 8]) * 0.05
        lander_right_joint_2 = (
            rotate(lander_right_joint_2, self.bodies[0].angle) + self.bodies[0].position
        )

        right_leg_joint_1 = self.bodies[1].position
        right_leg_joint_2 = self.bodies[1].position + rotate(
            jnp.array([0.0, 0.4]), self.bodies[1].angle
        )

        def fixed_positional_constraint(
            body_and_contact1, body_and_contact2, impulse_fn
        ):
            delta_pos = body_and_contact1[1] - body_and_contact2[1]
            delta_vel = body_and_contact1[0].velocity_at(
                body_and_contact1[1]
            ) - body_and_contact2[0].velocity_at(body_and_contact2[1])
            impulse = impulse_fn(delta_pos, delta_vel)
            b1 = apply_impulse(body_and_contact1[0], -impulse, body_and_contact1[1])
            b2 = apply_impulse(body_and_contact2[0], impulse, body_and_contact2[1])
            return b1, b2

        def impulse_fn(dp, dv):
            return dp * 1.0 + dv * (jnp.linalg.norm(dv) + 0.1) * 0.05

        new_bodies = self.bodies
        lander, right_leg, left_leg = new_bodies[0], new_bodies[1], new_bodies[2]
        lander, left_leg = fixed_positional_constraint(
            (lander, lander_left_joint_1), (left_leg, left_leg_joint_1), impulse_fn
        )
        lander, left_leg = fixed_positional_constraint(
            (lander, lander_left_joint_2), (left_leg, left_leg_joint_2), impulse_fn
        )
        lander, right_leg = fixed_positional_constraint(
            (lander, lander_right_joint_1), (right_leg, right_leg_joint_1), impulse_fn
        )
        lander, right_leg = fixed_positional_constraint(
            (lander, lander_right_joint_2), (right_leg, right_leg_joint_2), impulse_fn
        )

        # lets also dump angular velocities of the legs by a lot
        right_leg = eqx.tree_at(
            lambda x: x.angular_velocity, right_leg, right_leg.angular_velocity * 0.95
        )
        left_leg = eqx.tree_at(
            lambda x: x.angular_velocity, left_leg, left_leg.angular_velocity * 0.95
        )

        new_bodies = eqx.tree_at(lambda x: x[0], new_bodies, lander)
        new_bodies = eqx.tree_at(lambda x: x[1], new_bodies, right_leg)
        new_bodies = eqx.tree_at(lambda x: x[2], new_bodies, left_leg)

        return eqx.tree_at(lambda cls: cls.bodies, self, new_bodies)

    def draw(self, painter):
        for body in self.bodies:
            body.draw(painter)
        painter.draw_line((-2, -1.8), (-2, -1.0), color=(255, 0, 0))
        painter.draw_line((2, -1.8), (2, -1.0), color=(255, 0, 0))
        painter.next()
