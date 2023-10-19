"""
Implements a few concepts from design by contract.
"""

import equinox as eqx
from jax import numpy as jnp


# This file contains implementation of design-by-contract patterns with equinox
# partly inspired by https://stackoverflow.com/questions/12151182/python-precondition-postcondition-for-member-function-how # noqa: E501


def pre_condition(condition):
    """
    Implements pre_condition that works under JAX transformations.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            fn_input = (args, kwargs)
            args, kwargs = eqx.error_if(
                fn_input,
                jnp.logical_not(condition(*fn_input[0], **fn_input[1])),
                "Pre condition failed",
            )

            retval = func(*args, **kwargs)
            return retval

        return wrapper

    return decorator


def post_condition(condition, provide_input=False):
    """
    Implements post condition that works under JAX transformation.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            retval = func(*args, **kwargs)
            if provide_input:
                retval = eqx.error_if(
                    retval,
                    jnp.logical_not(condition(retval, *args, **kwargs)),
                    "Post condition failed",
                )
            else:
                retval = eqx.error_if(
                    retval, jnp.logical_not(condition(retval)), "Post condition failed"
                )

            return retval

        return wrapper

    return decorator


def _check_all_annotations(cls):
    for x in type(cls).__annotations__.items():
        cls = eqx.error_if(
            cls,
            not isinstance(getattr(cls, x[0]), x[1]),
            f"{x[0]}={getattr(cls, x[0])} is not of type {x[1]}",
        )
    return cls


def _invariant_fn(cls):
    cls = _check_all_annotations(cls)
    user_inv = getattr(cls, "__invariant__")()
    return user_inv, cls


def _check_invariant(func):
    def wrapper(*args, **kwargs):
        cond, typechecked_self = _invariant_fn(args[0])
        checked_self = eqx.error_if(
            typechecked_self,
            cond,
            "Invariant failed! That is kinda bad. Probably nan"
            " or invalid value encountered in the checked class.",
        )
        retval = func(checked_self, *(args[1:]), **kwargs)
        return retval

    return wrapper


def class_invariant(cls):
    """
    Implements class invariant that works under JAX transformations.

    In fact, this one is really useful, since for every non-static method of
    a class it checks whether the invariant holds. Thus we can easily detect
    jnp.nans or invalid values 'early': before the runs of the function.

    Sadly, detecting them _after_ the function execution is much harder: we don't
    necessarily know if what the function returns is a valid object.
    """
    for name in dir(cls):
        if name.startswith("_"):
            continue
        setattr(cls, name, _check_invariant(getattr(cls, name)))
    return cls
