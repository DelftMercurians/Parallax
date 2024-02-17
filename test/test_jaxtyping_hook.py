import beartype
import equinox as eqx
import jaxlib
import pytest

from cotix._collisions import (
    check_for_collision_convex,
    compute_penetration_vector_convex,
)


typefailed = beartype.roar.BeartypeCallHintParamViolation
eqxfailed = (jaxlib.xla_extension.XlaRuntimeError, eqx.EquinoxTracetimeError)


@pytest.mark.skip(reason="TODO: make jaxtyping great again")
def test_bad_types_collision():
    with pytest.raises(typefailed):
        check_for_collision_convex(1, 2)
    with pytest.raises(typefailed):
        compute_penetration_vector_convex("asdf", "lol")
