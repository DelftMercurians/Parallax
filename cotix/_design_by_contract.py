import equinox as eqx
from jax import numpy as jnp


# This file contains implementation of design-by-contract patterns with equinox
# partly inspired by https://stackoverflow.com/questions/12151182/python-precondition-postcondition-for-member-function-how # noqa: E501


def post_condition(condition):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retval = func(*args, **kwargs)
            retval = eqx.error_if(
                retval, jnp.logical_not(condition(retval)), "Post condition failed"
            )
            return retval

        return wrapper

    return decorator


def pre_condition(condition):
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
