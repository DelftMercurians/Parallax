import equinox as eqx


class AbstractCollisionSolver(eqx.Module, strict=True):
    """
    This class tries to resolve collisions, that are given already.
    """
