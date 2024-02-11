"""
Abstractions and implementations necessary for Body manipulation
"""


import equinox as eqx
import jax.numpy as jnp
from equinox import AbstractVar
from jaxtyping import Array, Float

from ._universal_shape import UniversalShape


class AbstractBody(eqx.Module, strict=True):
    """
    An abstract Body, every implementation of the body must inherit from this class.
    Contains some common implemntations/properties.
    """

    is_area: AbstractVar[bool]

    mass: AbstractVar[Float[Array, ""]]
    inertia: AbstractVar[Float[Array, ""]]

    position: AbstractVar[Float[Array, "2"]]
    velocity: AbstractVar[Float[Array, "2"]]

    angle: AbstractVar[Float[Array, ""]]
    angular_velocity: AbstractVar[Float[Array, ""]]

    elasticity: AbstractVar[Float[Array, ""]]
    friction_coefficient: AbstractVar[Float[Array, ""]]

    shape: AbstractVar[UniversalShape]

    def update_transform(self):
        """
        Updates the transformation matrix of the body.
        Effectively syncrhonizes the 'physical state' of the body,
        with its 'shape' state: rotates and shifts the shape so that
        standard algorithms for shape intersection will work correctly.
        """
        return eqx.tree_at(
            lambda x: x.shape,
            self,
            self.shape.update_transform(angle=self.angle, position=self.position),
        )

    def collides_with(self, other):
        """Returns boolean, whether there is a collision with another body."""
        return self.shape.collides_with(other.shape)

    def penetrates_with(self, other):
        return self.shape.penetrates_with(other.shape)

    def possibly_collides_with(self, other):
        return self.shape.possibly_collides_with(other.shape)

    def set_mass(self, mass: Float[Array, ""]):
        """Sets the mass (a scalar) of the body."""
        return eqx.tree_at(lambda x: x.mass, self, mass)

    def set_inertia(self, inertia: Float[Array, ""]):
        """Sets inertia tensor (2d vector) of the body."""
        return eqx.tree_at(lambda x: x.inertia, self, inertia)

    def set_position(self, position: Float[Array, "2"], update_transform=True):
        """Sets position of the center of mass of the body."""
        tmp = eqx.tree_at(lambda x: x.position, self, position)
        if update_transform:
            return tmp.update_transform()
        else:
            return tmp

    def set_velocity(self, velocity: Float[Array, "2"]):
        """Sets velocity of the center of mass of the body."""
        return eqx.tree_at(lambda x: x.velocity, self, velocity)

    def set_angle(self, angle: Float[Array, ""], update_transform=True):
        """Sets the angle of rotation around its center of mass."""
        tmp = eqx.tree_at(lambda x: x.angle, self, angle)
        if update_transform:
            return tmp.update_transform()
        else:
            return tmp

    def set_angular_velocity(self, angular_velocity: Float[Array, ""]):
        """Sets the rate of change of bodys angle."""
        return eqx.tree_at(lambda x: x.angular_velocity, self, angular_velocity)

    def set_shape(self, shape: UniversalShape):
        """Replaces the shape with any other shape: not recommended to use."""
        return eqx.tree_at(lambda x: x.shape, self, shape)

    def get_center_of_mass(self):
        return self.position

    def get_mass_matrix(self):
        # it is a scalar since we are in 2d, so there is 1 axis of rotation
        return self.inertia

    def set_elasticity(self, elasticity: Float[Array, ""]):
        return eqx.tree_at(lambda x: x.elasticity, self, elasticity)

    def set_friction_coefficient(self, friction_coefficient: Float[Array, ""]):
        return eqx.tree_at(lambda x: x.friction_coefficient, self, friction_coefficient)

    def load(self, other):
        """
        Loads everything (except the shape) from another body
        """
        return (
            self.set_mass(other.mass)
            .set_inertia(other.inertia)
            .set_position(other.position, False)
            .set_velocity(other.velocity)
            .set_angle(other.angle, False)
            .set_angular_velocity(other.angular_velocity)
            .set_elasticity(other.elasticity)
            .set_friction_coefficient(other.friction_coefficient)
        )

    def draw(self, painter):
        self.shape.draw(painter)


class AnyBody(AbstractBody, strict=True):
    """
    A body with any shape. Useful for tests.
    """

    is_area: bool = eqx.field(static=True)

    mass: Float[Array, ""]
    inertia: Float[Array, ""]

    position: Float[Array, "2"]
    velocity: Float[Array, "2"]

    angle: Float[Array, ""]
    angular_velocity: Float[Array, ""]

    elasticity: Float[Array, ""]
    friction_coefficient: Float[Array, ""]

    shape: UniversalShape

    def __init__(
        self,
        mass=jnp.array(1.0),
        inertia=jnp.array(1.0),
        position=jnp.zeros((2,)),
        velocity=jnp.zeros((2,)),
        angle=jnp.zeros(()),
        angular_velocity=jnp.zeros(()),
        elasticity=jnp.array(1.0),
        friction_coefficient=jnp.array(1.0),
        is_area=False,
        shape=None,
    ):
        self.mass = mass.astype(jnp.float32)
        self.inertia = inertia.astype(jnp.float32)

        self.position = position.astype(jnp.float32)
        self.velocity = velocity.astype(jnp.float32)

        self.angle = angle.astype(jnp.float32)
        self.angular_velocity = angular_velocity.astype(jnp.float32)

        self.elasticity = elasticity.astype(jnp.float32)
        self.friction_coefficient = friction_coefficient.astype(jnp.float32)

        self.is_area = is_area

        self.shape = shape
        self.shape = self.shape.update_transform(
            angle=self.angle, position=self.position
        )


class DynamicBody(eqx.Module, strict=True):
    """
    A body without any shape. Not an abstract body,
    but is veeery useful for fast compilation,
    since it removes the only static component of the body.
    """

    mass: Float[Array, ""]
    inertia: Float[Array, ""]

    position: Float[Array, "2"]
    velocity: Float[Array, "2"]

    angle: Float[Array, ""]
    angular_velocity: Float[Array, ""]

    elasticity: Float[Array, ""]
    friction_coefficient: Float[Array, ""]

    def __init__(self, other: AbstractBody = None):
        if other is None:
            self.mass = jnp.array(1.0).astype(jnp.float32)
            self.inertia = jnp.array(1.0).astype(jnp.float32)
            self.position = jnp.zeros((2,)).astype(jnp.float32)
            self.velocity = jnp.zeros((2,)).astype(jnp.float32)
            self.angle = jnp.zeros(()).astype(jnp.float32)
            self.angular_velocity = jnp.zeros(()).astype(jnp.float32)
            self.elasticity = jnp.array(1.0).astype(jnp.float32)
            self.friction_coefficient = jnp.array(1.0).astype(jnp.float32)
            return

        self.mass = other.mass.astype(jnp.float32)
        self.inertia = other.inertia.astype(jnp.float32)

        self.position = other.position.astype(jnp.float32)
        self.velocity = other.velocity.astype(jnp.float32)

        self.angle = other.angle.astype(jnp.float32)
        self.angular_velocity = other.angular_velocity.astype(jnp.float32)

        self.elasticity = other.elasticity.astype(jnp.float32)
        self.friction_coefficient = other.friction_coefficient.astype(jnp.float32)

    def set_mass(self, mass: Float[Array, ""]):
        """Sets the mass (a scalar) of the body."""
        return eqx.tree_at(lambda x: x.mass, self, mass)

    def set_inertia(self, inertia: Float[Array, ""]):
        """Sets inertia tensor (2d vector) of the body."""
        return eqx.tree_at(lambda x: x.inertia, self, inertia)

    def set_position(self, position: Float[Array, "2"]):
        """Sets position of the center of mass of the body."""
        tmp = eqx.tree_at(lambda x: x.position, self, position)
        return tmp

    def set_velocity(self, velocity: Float[Array, "2"]):
        """Sets velocity of the center of mass of the body."""
        return eqx.tree_at(lambda x: x.velocity, self, velocity)

    def set_angle(self, angle: Float[Array, ""]):
        """Sets the angle of rotation around its center of mass."""
        tmp = eqx.tree_at(lambda x: x.angle, self, angle)
        return tmp

    def set_angular_velocity(self, angular_velocity: Float[Array, ""]):
        """Sets the rate of change of bodys angle."""
        return eqx.tree_at(lambda x: x.angular_velocity, self, angular_velocity)

    def set_shape(self, shape: UniversalShape):
        """Replaces the shape with any other shape: not recommended to use."""
        return eqx.tree_at(lambda x: x.shape, self, shape)

    def get_center_of_mass(self):
        return self.position

    def get_mass_matrix(self):
        # it is a scalar since we are in 2d, so there is 1 axis of rotation
        return self.inertia

    def set_elasticity(self, elasticity: Float[Array, ""]):
        return eqx.tree_at(lambda x: x.elasticity, self, elasticity)

    def set_friction_coefficient(self, friction_coefficient: Float[Array, ""]):
        return eqx.tree_at(lambda x: x.friction_coefficient, self, friction_coefficient)
