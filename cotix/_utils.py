from typing import Callable

import equinox as eqx
import jax
from jax import lax, numpy as jnp, tree_util as jtu


@eqx.filter_jit
def filter_scan(f: Callable, init, xs, *args, **kwargs):
    """Same as lax.scan, but allows to have eqx.Module in carry"""
    init_dynamic_carry, static_carry = eqx.partition(init, eqx.is_array)

    def to_scan(dynamic_carry, x):
        carry = eqx.combine(dynamic_carry, static_carry)
        new_carry, out = f(carry, x)
        dynamic_new_carry, _ = eqx.partition(new_carry, eqx.is_array)
        return dynamic_new_carry, out

    out_carry, out_ys = lax.scan(to_scan, init_dynamic_carry, xs, *args, **kwargs)
    return eqx.combine(out_carry, static_carry), out_ys


@eqx.filter_jit
def filter_cond(pred, true_f: Callable, false_f: Callable, *args):
    """Same as lax.cond, but allows to return eqx.Module"""
    dynamic_true, static_true = eqx.partition(true_f(*args), eqx.is_array)
    dynamic_false, static_false = eqx.partition(false_f(*args), eqx.is_array)

    static_part = eqx.error_if(
        static_true,
        static_true != static_false,
        "Filtered conditional arguments should have the same static part",
    )

    dynamic_part = lax.cond(pred, lambda *_: dynamic_true, lambda *_: dynamic_false)
    return eqx.combine(dynamic_part, static_part)


def unfilter_select(pred, true_v, false_v):
    return jtu.tree_map(lambda a, b: lax.select(pred, a, b), true_v, false_v)


@eqx.filter_jit
def make_pairs(data, pair_fn, placeholder, limit, unsafe=False):
    # test (naively) for compilation-time explosion :)
    if not unsafe:
        if not (("jaxlib" in str(type(pair_fn))) or ("equinox" in str(type(pair_fn)))):
            raise Exception(
                """ Please, collide your head with a wall until you 
                    understand why you are not allowed to pass 
                    non-top-level-jitted
                    functions into make_pairs"""
            )
    # these things happen statically: effectively just code that is shoved
    # directly into XLA during tracing (compilation)
    evals = [placeholder] * len(data) * len(data)
    for i in range(len(data)):
        for j in range(len(data)):
            evals[i * len(data) + j] = pair_fn(
                (jnp.array(i), data[i]), (jnp.array(j), data[j])
            )

    return remove_placeholders(evals, placeholder, limit)


@eqx.filter_jit
def remove_placeholders(evals, placeholder, limit):
    """
    removes placeholders. Your job is to guarantee that: limit is static,
    placeholder has the same shape as all the things in the eval, eval is the list
    of pytrees of the placeholder structure.
    """
    evals = [
        *evals,
        placeholder,
    ]  # add placeholder at the end, for future reference and no conditions

    def simple_transpose(list_of_trees):
        return jax.tree_map(lambda *xs: list(xs), *list_of_trees)

    structure_of_arrays = simple_transpose(evals)

    # now, these things are (mostly) dynamic, so they happen during runtime,
    # using an actual values
    fully_arrayed = jtu.tree_map(
        lambda node: jnp.array(node),
        structure_of_arrays,
        is_leaf=lambda x: isinstance(x, list),
    )
    booleanized = jtu.tree_map(
        lambda node, comp: jax.lax.map(lambda ax: ~jnp.all(jnp.equal(ax, comp)), node),
        fully_arrayed,
        placeholder,
    )
    reduced = jtu.tree_reduce(
        lambda node, acc: node & acc, booleanized, jnp.array([True] * (len(evals)))
    )
    indexing = jnp.nonzero(
        reduced, size=limit, fill_value=len(evals) - 1
    )  # and here we use our end-placeholder as a fill value
    out_arrayed = jtu.tree_map(lambda node: node.at[indexing].get(), fully_arrayed)

    # and, again, static: extract the representation from our
    # struct-of-stacked-arrays into normal list-of-structs
    out_treed = [
        jtu.tree_map(lambda node: node.at[i].get(), out_arrayed) for i in range(limit)
    ]
    return out_treed
