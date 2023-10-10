import equinox as eqx
from jax import numpy as jnp, tree_util as jtu
from jaxtyping import Array, Float


class AbstractShape(eqx.Module):
    parts: list[eqx.Module]

    def _get_center(self):
        return jtu.tree_reduce(
            lambda acc, shape: acc + shape._get_center(),
            self.parts,
            is_leaf=lambda node: isinstance(node, AbstractConvexShape),
        )


class AbstractConvexShape(eqx.Module):
    def _get_support(self, direction: Float[Array, "2"]):
        """
        Computes a support vector of a **convex** shape. Support vector is simply
        the farthest point in a particular direction. This does not include any
        rotations/shifts of the body, so them should be applied separately.

        **Arguments:**

        - `direction`: the direction (2D vector) along which to compute the support.

        **Returns:**

        a multiple of direction... TODO!

        """
        ...

    def _get_center(self):
        ...


class Circle(AbstractConvexShape):
    radius: Float[Array, ""]
    position: Float[Array, "2"]

    def _get_support(self, direction: Float[Array, "2"]):
        normalized_direction = direction / jnp.sqrt(jnp.sum(direction**2))
        return normalized_direction * self.radius + self.position

    def _get_center(self):
        return self.position
