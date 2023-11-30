import equinox as eqx
from jax import numpy as jnp

from ._bodies import AnyBody
from ._convex_shapes import AABB, Polygon4, Polygon6
from ._universal_shape import UniversalShape


class LunarLander(eqx.Module, strict=True):
    bodies: list[AnyBody]

    def __init__(self):
        LANDER_POLY = [
            (-14, +17),
            (-17, 0),
            (-17, -10),
            (+17, -10),
            (+17, 0),
            (+14, +17),
        ]

        lander_shape = Polygon6(jnp.array(LANDER_POLY) * 0.05)

        LEG_AWAY = 24
        LEG_DOWN = 10
        LEG_W, LEG_H = 2, 8
        LEG_ANGLE = -0.3

        left_leg = Polygon4(
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

        left_leg = eqx.tree_at(
            lambda x: x.vertices,
            left_leg,
            rotate(left_leg.vertices, LEG_ANGLE),
        )

        # move the leg
        left_leg = eqx.tree_at(
            lambda x: x.vertices,
            left_leg,
            left_leg.vertices + jnp.array([LEG_AWAY, -LEG_DOWN]),
        )

        # scale the leg
        left_leg = eqx.tree_at(
            lambda x: x.vertices,
            left_leg,
            left_leg.vertices * 0.05,
        )

        right_leg = eqx.tree_at(
            lambda x: x.vertices,
            left_leg,
            left_leg.vertices * jnp.array([-1.0, 1.0]),
        )

        lander = AnyBody(
            position=jnp.array([0.0, 5.0]),
            velocity=jnp.array([0.0, -0.8]),
            angular_velocity=jnp.array(0.2),
            shape=UniversalShape(lander_shape, left_leg, right_leg),
        )

        ground = AnyBody(
            position=jnp.array([0.0, -5.0]),
            mass=jnp.array(jnp.inf),
            inertia=jnp.array(jnp.inf),
            elasticity=jnp.array(0.1),
            shape=UniversalShape(
                AABB(jnp.array([-100.0, -2.0]), jnp.array([100.0, 2.0]))
            ),
        )

        self.bodies = [lander, ground]
