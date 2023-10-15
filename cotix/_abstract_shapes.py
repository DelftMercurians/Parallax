import abc

import equinox as eqx
from beartype.typing import Callable
from jaxtyping import Array, Float


# type of the support function: takes a vector, returns a vector
# useful as a shorthand for type annotations
SupportFn = Callable[[Float[Array, "2"]], Float[Array, "2"]]


class AbstractShape(eqx.Module, strict=True):
    @abc.abstractmethod
    def get_support(self, direction):
        """
        Computes a support vector of a shape. Support vector is simply
        the farthest point in a particular direction. This does not include any
        rotations/shifts of the body, so they should be applied separately.

        **Arguments:**

        - `direction`: the direction (unnormalized) along which to compute the support.
        - `tr`: the transform (3x3 matrix in homogenuous coordinates) that
          represents position of the shape in space.
          Allows also to skew it, if you wish..

        **Returns:**

        Furthest point of the shape in the given direction.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def _get_center(self):
        """
        Returns an approximate central point of the shape. It is computed as middle of
        an axis-aligned bounding box.

        **Returns:**

        Center of the shape.

        """
        raise NotImplementedError


class AbstractConvexShape(AbstractShape, strict=True):
    ...
