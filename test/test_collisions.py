import equinox as eqx
import jax
import pytest
from jax import numpy as jnp, random as jr, tree_util as jtu

from cotix._contacts import (
    _contact_from_edges,
    aabb_vs_aabb,
    aabb_vs_polygon,
    circle_vs_aabb,
    circle_vs_circle,
    polygon_vs_polygon,
)
from cotix._convex_shapes import AABB, Circle, Polygon


MAX_CALLS_PER_VMAP = 10_000
TESTS_PER_SCENARIO = 10_000_000

# Firstly, we define two extraordinarly convenient functions for testing invariants


def _test_with_seed(f, seed, N=TESTS_PER_SCENARIO, N_ratio=1.0):
    N = int(N_ratio * N)
    # and it tests a bunch of random configurations 'heavily'
    # and a bunch of them 'lightly'
    # where 'lightly' and 'heavily' qualify how many checks we conduct
    # in general, 'lightly' means that test time is ~1 eval while
    # 'heavily' means that test time is ~50 eval time, and also the
    # same amount of memory. While I could have reduced memory consumption
    # for the heavy testing, I don't really care.
    k1, k2 = jr.split(seed, 2)
    light_keys = jr.split(k1, 1 + (N // MAX_CALLS_PER_VMAP))
    heavy_keys = jr.split(k2, 1 + (N // MAX_CALLS_PER_VMAP))

    fl = jax.vmap(jtu.Partial(f, heavy=False))
    fh = jax.vmap(jtu.Partial(f, heavy=True))

    def wh(key):
        return fh(jr.split(key, MAX_CALLS_PER_VMAP // 50))

    def wl(key):
        return fl(jr.split(key, MAX_CALLS_PER_VMAP))

    out_light = jax.lax.map(wl, light_keys)
    out_heavy = jax.lax.map(wh, heavy_keys)

    if (not jnp.all(out_light)) or (not jnp.all(out_heavy)):
        print(f"Light tests: {out_light.size}; Heavy tests: {out_heavy.size}")
        print(
            f"Light failed: {jnp.count_nonzero(~out_light)}; "
            f"Heavy failed: {jnp.count_nonzero(~out_heavy)}"
        )

    # neat debugging thingy: automatically enter debugger if something failed
    # and enter it in the correct function call (since everything is vmapped ..
    #     simple eqx.debug.breakpoint_if does not work well)
    for wkey in light_keys[~jnp.all(out_light, axis=1)]:
        ikeys = jr.split(wkey, MAX_CALLS_PER_VMAP)
        iout = fl(ikeys)
        for key in ikeys[~iout]:
            f(key, heavy=False, debug=True)

    for wkey in heavy_keys[~jnp.all(out_heavy, axis=1)]:
        ikeys = jr.split(wkey, MAX_CALLS_PER_VMAP // 50)
        iout = fh(ikeys)
        for key in ikeys[~iout]:
            f(key, heavy=True, debug=True)

    # actual asserts
    assert jnp.all(out_light), "Light tests failed"
    assert jnp.all(out_heavy), "Heavy tests failed"


def _test_contact_info(f, a, b, heavy=True, debug=False, small_eps=1e-5):
    big_eps = 10 * small_eps

    info = f(a, b)
    # condition that should always hold true (unless no collision at all)
    contact_point_contained = a.contains(info.contact_point) & b.contains(
        info.contact_point
    )

    # check that we found no intersection from edges
    # (this is a bit of a hack, but it is a good sanity check)
    same_edging = True
    if (type(a) == Polygon or type(a) == AABB) and (
        type(b) == Polygon or type(b) == AABB
    ):
        edges_a = a.get_edges()
        edges_b = b.get_edges()
        contact_point = _contact_from_edges(edges_a, edges_b)
        same_edging = ~jnp.logical_xor(jnp.all(jnp.isnan(contact_point)), info.isnan())

    # 'resolve' the collision
    an = a.move(info.penetration_vector)
    new_info = f(an, b)

    # check that there is not penetration after we resolve the collision
    after_resolution_no_penetration = (
        jnp.linalg.norm(new_info.penetration_vector) < small_eps
    )

    no_shorter_resolution = True
    if heavy:
        # check that a shorter movement in any direction won't resolve a collision:
        # since we are 'guaranteeing' that penetration vector is the minimal one
        dirs_pen = jnp.linspace(0, 2 * jnp.pi, 20)
        length = jnp.clip(jnp.linalg.norm(info.penetration_vector) - big_eps, a_min=0.0)
        deltas = jnp.stack((jnp.cos(dirs_pen), jnp.sin(dirs_pen)), axis=1) * length

        def some_f(delta):
            moved = a.move(delta)
            out = f(moved, b)
            return out.penetration_vector

        penetrations_big = jax.lax.map(some_f, deltas)
        penetrations_big_cond = jnp.linalg.norm(penetrations_big, axis=1) > small_eps
        no_shorter_resolution = jnp.all(penetrations_big_cond) | (
            jnp.linalg.norm(info.penetration_vector) < 1.5 * small_eps
        )

    # check that we can obtain deep penetration if checking a bunch of directions
    exist_deep_enough = True
    if heavy:
        dirs = jnp.linspace(0, 2 * jnp.pi, 20)
        deltas = jnp.stack((jnp.cos(dirs), jnp.sin(dirs)), axis=1) * big_eps
        assert (
            deltas.shape[1] == 2
        ), "NOT A FAILURE OF A TEST, BUT A FAILURE OF A PRE-CONDITION"

        # get the corresponding penetration vectors for each of the moves
        penetrations = jax.vmap(lambda delta: f(an.move(delta), b).penetration_vector)(
            deltas
        )
        is_penetration_deep_enough = (
            jnp.linalg.norm(penetrations, axis=1) > big_eps * 0.5
        )

        assert is_penetration_deep_enough.shape == (
            20,
        ), "NOT A FAILURE OF A TEST, BUT A FAILURE OF A PRE-CONDITION"

        # and this is the final condition, which is true iff the objects are 'touching'
        # after resolving the collision
        exist_deep_enough = jnp.any(is_penetration_deep_enough)

    if debug:
        jax.debug.breakpoint()

    return (
        jnp.any(jnp.isnan(info.contact_point))
        | (
            exist_deep_enough
            & after_resolution_no_penetration
            & no_shorter_resolution
            & contact_point_contained
        )
    ) & same_edging


# and the actual tests follow ...


@pytest.mark.parametrize(
    "inp",
    [
        (
            AABB(
                lower=jnp.array([-0.72877413, 0.4557484]),
                upper=jnp.array([2.8047075, 1.2977818]),
            ),
            AABB(
                lower=jnp.array([0.54837185, -0.6289791]),
                upper=jnp.array([2.0380392, 2.3360252]),
            ),
        )
    ],
)
def test_aabb_vs_aabb_parametrized(inp):
    a, b = inp
    assert _test_contact_info(aabb_vs_aabb, a, b)


def test_aabb_vs_aabb_rand():
    def f(key, **kwargs):
        k1, k2, k3, k4, key = jr.split(key, 5)
        a = AABB(
            lower=jr.normal(k1, (2,)),
            upper=jr.normal(k1, (2,))
            + jr.uniform(k2, shape=(2,), minval=0.01, maxval=5.0),
        )
        b = AABB(
            lower=jr.normal(k3, (2,)),
            upper=jr.normal(k3, (2,))
            + jr.uniform(k4, shape=(2,), minval=0.01, maxval=5.0),
        )

        val = _test_contact_info(aabb_vs_aabb, a, b, **kwargs)
        return val

    _test_with_seed(f, jr.PRNGKey(0))


def test_circle_vs_circle_rand():
    def f(key, **kwargs):
        k1, k2, k3, k4, key = jr.split(key, 5)
        a = Circle(
            position=jr.normal(k1, (2,)),
            radius=jr.uniform(k2, shape=(), minval=0.01, maxval=5.0),
        )
        b = Circle(
            position=jr.normal(k3, (2,)),
            radius=jr.uniform(k4, shape=(), minval=0.01, maxval=5.0),
        )

        val = _test_contact_info(circle_vs_circle, a, b, **kwargs)
        return val

    _test_with_seed(f, jr.PRNGKey(1))


@pytest.mark.parametrize(
    "inp",
    [
        (
            Circle(jnp.array(4.808976), jnp.array([0.52343243, 0.38244677])),
            AABB(jnp.array([1.2948408, 1.4734308]), jnp.array([3.3397233, 6.3817973])),
        ),
        (
            Circle(jnp.array(1.0), jnp.array([1.0, 1.0])),
            AABB(jnp.array([-2.0, -2.0]), jnp.array([0.4, 0.7])),
        ),
        (
            Circle(jnp.array(5.0), jnp.zeros((2,))),
            AABB(jnp.array([-2.0, -2.0]), jnp.array([2.0, 2.0])),
        ),
        (
            Circle(jnp.array(3.7427633), jnp.array([-0.0277214, 1.0449156])),
            AABB(
                jnp.array([-0.6238362, -1.1297362]), jnp.array([1.3405488, -0.5544366])
            ),
        ),
        (
            Circle(jnp.array(0.5361439), jnp.array([-0.4457733, 0.5882554])),
            AABB(
                jnp.array([-0.44587463, -0.73396504]), jnp.array([0.0717122, 3.0028129])
            ),
        ),
        (
            Circle(jnp.array(1.0), jnp.zeros((2,))),
            AABB(jnp.array([-2.0, -2.0]), jnp.array([2.0, 2.0])),
        ),
        (
            Circle(jnp.array(0.01), jnp.array([0.0, 1.8])),
            AABB(jnp.array([-2.0, -2.0]), jnp.array([2.0, 2.0])),
        ),
        (
            Circle(jnp.array(1.0), jnp.array([0.1, 0.2])),
            AABB(jnp.array([-2.0, -2.3]), jnp.array([2.0, 2.0])),
        ),
        (
            Circle(jnp.array(1.0), jnp.array([-0.3, 0.05])),
            AABB(jnp.array([-2.1, -2.3]), jnp.array([2.0, 2.0])),
        ),
        (
            Circle(jnp.array(1.0), jnp.array([-0.12, -0.56])),
            AABB(jnp.array([-2.0, -2.0]), jnp.array([2.2, 2.3])),
        ),
    ],
)
def test_circle_vs_aabb_parametrized(inp):
    a = inp[0]
    b = inp[1]
    assert _test_contact_info(circle_vs_aabb, a, b, debug=False)


def test_circle_vs_aabb_rand():
    @eqx.filter_jit
    def f(key, **kwargs):
        k1, k2, k3, k4, key = jr.split(key, 5)
        a = Circle(
            position=jr.normal(k1, (2,)),
            radius=jr.uniform(k2, shape=(), minval=0.01, maxval=5.0),
        )
        b = AABB(
            lower=jr.normal(k3, (2,)),
            upper=jr.normal(k3, (2,))
            + jr.uniform(k4, shape=(2,), minval=0.01, maxval=5.0),
        )

        val = _test_contact_info(circle_vs_aabb, a, b, **kwargs)
        return val

    # since it is relatively slower to compute a collision between aabb and a circle
    # we are setting N_ratio to 0.2, which is probably small enough
    _test_with_seed(f, jr.PRNGKey(0), N_ratio=0.2)


"""
@pytest.mark.parametrize(
    "inp",
    [
        (
            Circle(jnp.array(2.3251324), jnp.array([-0.9682081, 0.6601708])),
            Polygon(jnp.array([[0.0, 1.0], [-0.5, -0.5], [0.5, -0.5]]))
        ),
        (
            Circle(jnp.array(0.9460892), jnp.array([-0.27002454, -0.07793757])),
            Polygon(jnp.array([[0.0, 1.0], [-0.5, -0.5], [0.5, -0.5]]))
        ),
        (
            Circle(jnp.array(1.8268951), jnp.array([-1.0627501, -0.18848151])),
            Polygon(jnp.array([[0.0, 1.0], [-0.5, -0.5], [0.5, -0.5]]))
        ),
        (
            Circle(jnp.array(1.9288263), jnp.array([0.96713173, -0.5197397])),
            Polygon(jnp.array([[0.0, 1.0], [-0.5, -0.5], [0.5, -0.5]]))
        ),
        (
            Circle(jnp.array(3.82366), jnp.array([0.89924145, 0.10981558])),
            Polygon(jnp.array([[0.0, 1.0], [-0.5, -0.5], [0.5, -0.5]]))
        ),
        (
            Circle(jnp.array(3.3993943), jnp.array([-0.5438884, -0.28964934])),
            Polygon(jnp.array([[0.0, 1.0], [-0.5, -0.5], [0.5, -0.5]]))
        ),
        (
            Circle(jnp.array(1.0), jnp.array([0.0, 0.0])),
            Polygon(jnp.array([[0.0, 0.9], [2.0, 3.0], [-2.0, 3.0]]))
        )
]
)
def test_circle_vs_polygon_parametrized(inp):
    a, b = inp

    assert _test_contact_info(circle_vs_polygon, a, b, debug=False)
"""

"""
def test_circle_vs_polygon_rand():
    @eqx.filter_jit
    def f(key, **kwargs):
        k1, k2, k3, k4, key = jr.split(key, 5)
        a = Circle(
            position=jr.normal(k1, (2,)),
            radius=jr.uniform(k2, shape=(), minval=0.05, maxval=5.0),
        )
        b = Polygon(jnp.array([[0.0, 1.0], [-0.5, -0.5], [0.5, -0.5]]))
        val = _test_contact_info(circle_vs_polygon, a, b, **kwargs, small_eps=1e-2)
        return val

    _test_with_seed(f, jr.PRNGKey(0), N_ratio=0.01)
"""

""""
TODO: make this pass i guess? Idk
def test_circle_vs_polygon_rand_2():
    @eqx.filter_jit
    def f(key, **kwargs):
        k1, k2, k3, k4, key = jr.split(key, 5)
        a = Circle(
            position=jr.normal(k1, (2,)),
            radius=jr.uniform(k2, shape=(), minval=0.05, maxval=5.0),
        )
        b = Polygon(jnp.array([[0.5, 0.5], [-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5]]))
        val = _test_contact_info(circle_vs_polygon, a, b, **kwargs, small_eps=1e-2)
        return val

    _test_with_seed(f, jr.PRNGKey(1), N_ratio=0.01)

"""


@pytest.mark.parametrize(
    "inp",
    [
        (
            AABB(
                lower=jnp.array([0.13606448, -2.6069396]),
                upper=jnp.array([0.39154065, 0.7682829]),
            ),
            Polygon(jnp.array([[0.5, 0.5], [-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5]])),
        ),
        (
            AABB(lower=jnp.array([-0.8, -0.8]), upper=jnp.array([-0.4, -0.4])),
            Polygon(jnp.array([[0.0, 1.0], [-0.5, -0.5], [0.5, -0.5]])),
        ),
    ],
)
def test_aabb_vs_polygon_parametrized(inp):
    assert _test_contact_info(aabb_vs_polygon, inp[0], inp[1])


def test_aabb_vs_polygon_rand():
    @eqx.filter_jit
    def f(key, **kwargs):
        k1, k2, k3, k4, key = jr.split(key, 5)
        a = AABB(
            lower=jr.normal(k3, (2,)),
            upper=jr.normal(k3, (2,))
            + jr.uniform(k4, shape=(2,), minval=0.01, maxval=5.0),
        )

        b = Polygon(jnp.array([[0.5, 0.5], [-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5]]))
        val = _test_contact_info(aabb_vs_polygon, a, b, **kwargs, small_eps=1e-3)
        return val

    _test_with_seed(f, jr.PRNGKey(2), N_ratio=0.1)


def test_aabb_vs_polygon_rand_2():
    @eqx.filter_jit
    def f(key, **kwargs):
        k1, k2, k3, k4, key = jr.split(key, 5)
        a = AABB(
            lower=jr.normal(k3, (2,)),
            upper=jr.normal(k3, (2,))
            + jr.uniform(k4, shape=(2,), minval=0.01, maxval=5.0),
        )

        b = Polygon(jnp.array([[0.3, 0.556], [-0.1, -0.2], [0.4, -0.3], [-0.8, 1.5]]))
        val = _test_contact_info(aabb_vs_polygon, a, b, **kwargs, small_eps=1e-3)
        return val

    _test_with_seed(f, jr.PRNGKey(3), N_ratio=0.1)


def test_polygon_vs_polygon_rand():
    @eqx.filter_jit
    def f(key, **kwargs):
        k1, k2, k3, k4, key = jr.split(key, 5)
        a = Polygon(jr.normal(k1, ((3, 2))))
        b = Polygon(jnp.array([[0.3, 0.556], [-0.1, -0.2], [0.4, -0.3], [-0.8, 1.5]]))
        val = _test_contact_info(polygon_vs_polygon, a, b, **kwargs, small_eps=1e-3)
        return val

    _test_with_seed(f, jr.PRNGKey(4), N_ratio=0.1)


def test_polygon_vs_polygon_rand_2():
    @eqx.filter_jit
    def f(key, **kwargs):
        k1, k2, k3, k4, key = jr.split(key, 5)
        a = Polygon(jr.normal(k2, ((6, 2))))
        b = Polygon(jnp.array([[0.3, 0.556], [-0.1, -0.2], [0.4, -0.3], [-0.8, 1.5]]))
        val = _test_contact_info(polygon_vs_polygon, a, b, **kwargs, small_eps=1e-3)
        return val

    _test_with_seed(f, jr.PRNGKey(5), N_ratio=0.1)
