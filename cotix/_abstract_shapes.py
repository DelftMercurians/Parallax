import abc

import equinox as eqx


class AbstractShape(eqx.Module, strict=True):
    def _get_support(self, direction, tr):
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
        local_direction = tr.inverse_direction(direction)
        local_support = self._get_local_support(local_direction)
        return tr.forward_vector(local_support)

    @abc.abstractmethod
    def _get_local_support(self, direction):
        raise NotImplementedError

    @abc.abstractmethod
    def get_center(self):
        """
        Returns an approximate central point of the shape. It is computed as middle of
        an axis-aligned bounding box.

        **Returns:**

        Center of the shape.

        """
        raise NotImplementedError


class AbstractConvexShape(AbstractShape, strict=True):
    ...
