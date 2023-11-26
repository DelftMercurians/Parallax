import equinox as eqx

from ._worlds import AbstractWorldState


class AbstractControlSignal(eqx.Module, strict=True):
    """
    Has an apply method, that applies the control signal on a state,
    and, besides, just for typechecking we need this class.
    """

    def apply(self, state: AbstractWorldState, dt):
        raise NotImplementedError


class AbstractControl(eqx.Module):
    """
    Defines a control function. Generally, just forces you to
    have a __call__ method with a correct signature, that is all.

    BTW, call to a control should produce a 'dense' control function,
    not just a single value. That is because we live in continuous times,
    so we need to be able to apply control continuously in time.
    """

    def __call__(self, state: AbstractWorldState):
        raise NotImplementedError
