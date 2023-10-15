import beartype
import equinox as eqx
import jax
import jaxlib
import pytest
from jax import numpy as jnp
from jaxtyping import Array, Float

from cotix._design_by_contract import class_invariant, post_condition, pre_condition


beardead = beartype.roar.BeartypeCallHintParamViolation
inverror = (eqx.EquinoxTracetimeError, TypeError, jaxlib.xla_extension.XlaRuntimeError)


@pytest.mark.parametrize("decorator", [lambda x: x, jax.jit, eqx.filter_jit])
@pytest.mark.parametrize(
    "condition",
    [
        post_condition((lambda out, *inp: out > inp[0] + inp[1]), provide_input=True),
        pre_condition(lambda a, b: a >= 1),
    ],
)
def test_parametrized_pre_and_post(decorator, condition):
    def f(a, b):
        return a**2 + b**2

    f = decorator(condition(f))

    f(jnp.array(2.0), jnp.array(0.0))
    f(jnp.array(1), jnp.array(1.5))
    f(jnp.array(5.0), jnp.array(0.0))
    f(jnp.array(10.0), jnp.array(-10.0))

    if "pre_condition" in condition.__qualname__:
        with pytest.raises((RuntimeError, ValueError)):
            f(jnp.array(0.0), jnp.array(5.0))
        with pytest.raises((RuntimeError, ValueError)):
            f(jnp.array(0.9), jnp.array(0))
    if "post_condition" in condition.__qualname__:
        with pytest.raises((RuntimeError, ValueError)):
            f(jnp.array(1.0), jnp.array(0.5))
        with pytest.raises((RuntimeError, ValueError)):
            f(jnp.array(1), jnp.array(1.0))
        with pytest.raises((RuntimeError, ValueError)):
            f(jnp.array(0.0), jnp.array(0.5))
        with pytest.raises((RuntimeError, ValueError)):
            f(jnp.array(0), jnp.array(0))


@class_invariant
@beartype.beartype
class _A(eqx.Module):
    x: Float[Array, ""]

    def g(self, x):
        return jnp.array([x.sum()]) + self.x

    def __invariant__(self):
        out = jnp.any(jnp.isnan(self.x)) | jnp.all(self.x < 0)
        return out


@pytest.mark.parametrize("decorator", [lambda x: x, jax.jit, eqx.filter_jit])
def test_invariant(decorator):
    @decorator
    def f():
        a = _A(jnp.array(1.0))
        a = eqx.tree_at(lambda x: x.x, a, jnp.array(5.0))
        return a.g(jnp.array([1, 2]))

    f()

    @decorator
    def f():
        a = _A(jnp.array(1.0))
        return a.g(jnp.array([1, 2]))

    f()

    with pytest.raises(inverror):

        @decorator
        def f():
            a = _A(jnp.array(1.0))
            a = eqx.tree_at(lambda x: x.x, a, jnp.nan)
            return a.g(jnp.array([1, 2]))

        f()

    with pytest.raises(inverror):

        @decorator
        def f():
            a = _A(jnp.array(1.0))
            a = eqx.tree_at(lambda x: x.x, a, 1)
            out = a.g(jnp.array([1, 2, 3]))
            return out

        f()

    with pytest.raises(inverror):

        @decorator
        def f():
            a = _A(jnp.array(1.0))
            a = eqx.tree_at(lambda x: x.x, a, jnp.array([1, 2, 3]))
            return a.g(jnp.array([1, 2, 3]))

        f()

    with pytest.raises(inverror):

        @decorator
        def f():
            a = _A(jnp.array(1.0))
            a = eqx.tree_at(lambda x: x.x, a, jnp.array(-10.0))
            return a.g(jnp.array([1, 2, 3]))

        f()
