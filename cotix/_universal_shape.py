"""
Contains only implementation for the Universal (or composite) shape.
"""

import equinox as eqx
import jax
from jax import numpy as jnp, tree_util as jtu
from jaxtyping import Array, Float

from ._abstract_shapes import AbstractShape, SupportFn
from ._collisions import check_for_collision_convex, compute_penetration_vector_convex
from ._geometry_utils import HomogenuousTransformer


class UniversalShape(eqx.Module, strict=True):
    """
    Universal shape defines a shape that contains arbitrary number (a list)
    of convex shapes. Any convex/concave shape can be represented using this class.

    In addition this class includes logic for conversion between global and local
    coordinate systems
    """

    parts: list[AbstractShape]
    _transformer: HomogenuousTransformer

    def __init__(self, *shapes: AbstractShape):
        self.parts = [*shapes]
        self._transformer = HomogenuousTransformer()

    def wrap_local_support(self, support_fn: SupportFn) -> SupportFn:
        """
        Given a support function in a local coordinate system (wrt center of the shape),
        transforms it into a global coordinate system (wrt center of the field)
        """

        def wrapper(direction: Float[Array, "2"]) -> Float[Array, "2"]:
            self._transformer.inverse_direction(direction)
            local_support = support_fn(direction)
            return self._transformer.forward_vector(local_support)

        return wrapper

    def get_global_support(self, direction: Float[Array, "2"]) -> Float[Array, "2"]:
        """
        Given a direction, computes a furthest point of
        the (possibly composite) shape in this direction
        """
        num_parts = len(self.parts)
        supports = jnp.zeros((num_parts, 2))
        for i in range(num_parts):
            supports = supports.at[i].set(
                self.wrap_local_support(self.parts[i].get_support)(direction)
            )

        dot_products = jax.lax.map(lambda sup: jnp.dot(sup, direction), supports)
        return supports[jnp.argmax(dot_products)]

    def update_transform(self, angle: Float[Array, ""], position: Float[Array, "2"]):
        """
        Updates the transformation between global and local coordinate systems
        """
        return eqx.tree_at(
            lambda x: x._transformer,
            self,
            HomogenuousTransformer(angle=angle, position=position),
        )

    def collides_with(self, other):
        """
        Returns true if we collide with another shape.
        """
        final_res = False
        final_simplex = jnp.zeros((3, 2))
        partA = self.parts[0]
        partB = other.parts[0]
        for first_part in self.parts:
            for second_part in other.parts:
                # find if there is a collision between two convex sub-shapes
                res, simplex = check_for_collision_convex(
                    self.wrap_local_support(first_part.get_support),
                    other.wrap_local_support(second_part.get_support),
                )
                final_simplex = jax.lax.cond(
                    (~final_res) & res,
                    lambda: (simplex, first_part, second_part),
                    lambda: (final_simplex, partA, partB),
                )
                final_res |= res
        return final_res, (final_simplex, partA, partB)

    def penetration_depth(self, other, metadata, solver_iterations=48):
        """
        This method 'fakes' a computation of penetration length. No,
        the length is truly computed, but just we only need metadata to compute it.
        Whatever....
        """
        return compute_penetration_vector_convex(
            metadata[1].get_support,
            metadata[2].get_support,
            metadata[0],
            solver_iterations,
        )

    def get_center(self):
        """
        Returns a geometric center of all the shapes in this universal shape:
        this value is not guaranteed to be anything meaningful, so use with care.
        """
        return jtu.tree_reduce(
            lambda acc, shape: acc + shape.get_center(),
            self.parts,
            is_leaf=lambda node: isinstance(node, AbstractShape),
        )
