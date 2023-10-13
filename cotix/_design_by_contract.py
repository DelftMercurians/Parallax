import equinox as eqx
from jax import numpy as jnp


# This file contains implementation of design-by-contract patterns with equinox
# partly inspired by https://stackoverflow.com/questions/12151182/python-precondition-postcondition-for-member-function-how # noqa: E501


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


def post_condition(condition, provide_input=False):
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


def _check_invariant(func, invariant):
    def wrapper(*args, **kwargs):
        checked_self = eqx.error_if(
            args[0],
            invariant(args[0]),
            "Invariant failed! That is kinda bad. Probably nan"
            " or invalid value encountered in the checked class.",
        )
        retval = func(checked_self, *(args[1:]), **kwargs)
        return retval

    return wrapper


def _check_all_annotations(instance):
    res = True
    for x in type(instance).__annotations__.items():
        print(f"checked {x}")
        res &= isinstance(getattr(instance, x[0]), x[1])
    return res


def class_invariant(cls):
    invariant = lambda instance: getattr(cls, "__invariant__")(
        instance
    ) & _check_all_annotations(instance)
    for name in dir(cls):
        if name.startswith("_"):
            continue
        setattr(cls, name, _check_invariant(getattr(cls, name), invariant))
    return cls
