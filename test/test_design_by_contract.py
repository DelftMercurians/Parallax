import equinox as eqx
import jax
import pytest
from jax import numpy as jnp

from cotix._design_by_contract import post_condition, pre_condition


def _test_function(f):
    wf = lambda a, b: f(jnp.array(a), jnp.array(b))

    wf(1, 2)
    wf(1, 0)
    wf(5, 7)

    # jax.jit throws ValueError, while eqx throws RuntimeError
    with pytest.raises((RuntimeError, ValueError)):
        wf(0, 0)
        wf(0.5, 0.49)
        wf(-42, -42)


@pytest.mark.filterwarnings("ignore")
def test_post_condition_no_jit():
    @post_condition(lambda out: out >= 5)
    def f(a, b):
        return 4 + jnp.abs(a) + jnp.abs(b)

    _test_function(f)


def test_post_condition_jax_jit():
    @jax.jit
    @post_condition(lambda out: out >= 5)
    def f(a, b):
        return 4 + jnp.abs(a) + jnp.abs(b)

    _test_function(f)


def test_post_condition_eqx_jit():
    @eqx.filter_jit
    @post_condition(lambda out: out >= 5)
    def f(a, b):
        return 4 + jnp.abs(a) + jnp.abs(b)

    _test_function(f)


@pytest.mark.filterwarnings("ignore")
def test_pre_condition_no_jit():
    @pre_condition(lambda a, b: a >= 1)
    def f(a, b):
        return 4 + jnp.abs(a) + jnp.abs(b)

    _test_function(f)


def test_pre_condition_jax_jit():
    @jax.jit
    @pre_condition(lambda a, b: a >= 1)
    def f(a, b):
        return 4 + jnp.abs(a) + jnp.abs(b)

    _test_function(f)


def test_pre_condition_eqx_jit():
    @eqx.filter_jit
    @pre_condition(lambda a, b: a >= 1)
    def f(a, b):
        return 4 + jnp.abs(a) + jnp.abs(b)

    _test_function(f)


@pytest.mark.filterwarnings("ignore")
def test_both_conditions_no_jit():
    @post_condition(lambda out: out >= 5)
    @pre_condition(lambda a, b: a >= 1)
    def f(a, b):
        return 4 + jnp.abs(a) + jnp.abs(b)

    _test_function(f)

    with pytest.raises((RuntimeError, ValueError)):
        f(jnp.array(0.0), jnp.array(10.0))


def test_both_conditions_eqx_jit():
    @eqx.filter_jit
    @post_condition(lambda out: out >= 5)
    @pre_condition(lambda a, b: a >= 1)
    def f(a, b):
        return 4 + jnp.abs(a) + jnp.abs(b)

    _test_function(f)

    with pytest.raises((RuntimeError, ValueError)):
        f(jnp.array(0.0), jnp.array(10.0))


def test_both_conditions_jax_jit():
    @jax.jit
    @post_condition(lambda out: out >= 5)
    @pre_condition(lambda a, b: a >= 1)
    def f(a, b):
        return 4 + jnp.abs(a) + jnp.abs(b)

    _test_function(f)

    with pytest.raises((RuntimeError, ValueError)):
        f(jnp.array(0.0), jnp.array(10.0))


def _test_function_2(f):
    wf = lambda x: f(jnp.array(x))

    wf(2.0)
    wf(-5.0)
    wf(15.0)

    with pytest.raises((RuntimeError, ValueError)):
        wf(0.0)
        wf(1.0)
        wf(0.7)


@pytest.mark.filterwarnings("ignore")
def test_post_with_input_no_jit():
    @post_condition((lambda out, inp: out > inp), provide_input=True)
    def f(x):
        return x * x

    _test_function_2(f)


def test_post_with_input_eqx_jit():
    @eqx.filter_jit
    @post_condition((lambda out, inp: out > inp), provide_input=True)
    def f(x):
        return x * x

    _test_function_2(f)


def test_post_with_input_jax_jit():
    @jax.jit
    @post_condition((lambda out, inp: out > inp), provide_input=True)
    def f(x):
        return x * x

    _test_function_2(f)
