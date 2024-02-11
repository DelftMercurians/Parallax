import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, Float, Int

from ._utils import soa_get


class AbstractConstraint(eqx.Module):
    """
    An abstract constraint definition.
    Has a list of bodies that it 'applies' from and to.
    Bodies are stored as indices into the list of bodies in the environment,
    so that jax can vmap well over them, when bodies are ShapelessBodies.

    Like, use the same ol' trick of removing shapes from bodies,
    thus making them homogeneous, and then applying whatever-you-want on top of them
    via a scan or something: makes things much faster, since now we can vmap, scan
    or whatever, since indices are available during runtime (indices are non static).
    """

    from_bodies: Int[Array, "n"]
    to_bodies: Int[Array, "m"]

    def __init__(self, from_bodies, to_bodies):
        self.from_bodies = from_bodies
        self.to_bodies = to_bodies

    def apply(self, bodies):
        """Apply the constraint on the SOA of shapeless bodies."""
        raise NotImplementedError()


class SoftPositionalConstraint(AbstractConstraint):
    """
    A soft position constraint.
    Kinda like a spring. Applies an 'impulse' to the bodies,
    such that they move towards the desired position.
    the impulse is scaled by dt, if it is passed.

    If the dt is not passed, we just complain, but still do work.

    In practice, this is more like a spring, than a real, hard constraint:
    it just applies some forces to some bodies.

    Attempts to combine all the bodies chosen positions into one point
    """

    from_bodies_position: Float[Array, "n 2"]
    to_bodies_position: Float[Array, "m 2"]
    stiffness: Float[Array, ""]

    def __init__(
        self,
        from_bodies,
        to_bodies,
        from_bodies_position,
        to_bodies_position,
        stiffness,
    ):
        self.from_bodies = from_bodies
        self.to_bodies = to_bodies
        self.from_bodies_position = from_bodies_position
        self.to_bodies_position = to_bodies_position
        self.stiffness = stiffness

    def apply(self, bodies, dt=1e-2):
        """
        Given a SOA of bodies, apply the constraint on the bodies,
        and return the new SOA.
        """

        assert len(self.from_bodies) == len(self.to_bodies)
        assert len(self.to_bodies) == len(self.from_bodies_position)
        assert self.stiffness.shape == ()

        # besides, let's make sure that the stiffness is more than zero,
        # (though allowed to be a nan)
        bodies = eqx.error_if(bodies, self.stiffness < 0.0, "stiffness < 0")

        # now, let's figure out the 'target' point of the resolution
        # it is the weighted by mass sum of all the absolute positions
        # and absolute positions can be computed from the ShapelessBodies by utilizing
        # the homogenuousTransformer
        indices = jnp.concatenate([self.from_bodies, self.to_bodies])
        rel_positions = jnp.concatenate(
            [self.from_bodies_position, self.to_bodies_position]
        )

        def rel_to_abs(index):
            body = soa_get(bodies, index)
            body_pos = body.position
            body_angle = body.angle - jnp.pi / 2
            rel_pos = rel_positions[index]

            assert rel_pos.shape == (2,)
            assert body_pos.shape == (2,)
            assert body_angle.shape == ()

            out = (
                jnp.array(
                    [
                        [jnp.cos(body_angle), -jnp.sin(body_angle)],
                        [jnp.sin(body_angle), jnp.cos(body_angle)],
                    ]
                )
                @ rel_pos
                + body_pos
            )
            return out

        absolute_positions = eqx.filter_vmap(rel_to_abs)(indices)
        masses = eqx.filter_vmap(lambda index: soa_get(bodies, index).mass)(indices)
        target_position = jnp.sum(
            absolute_positions * masses[:, None], axis=0
        ) / jnp.sum(masses)

        # now for every body, we need to compute the impulse, that is, force by dt
        # where force is stiffness * (target_position - body_position)

        def compute_impulse(index, reduced_index):
            body = soa_get(bodies, index)
            body_pos = body.position
            body_angle = body.angle

            assert body_pos.shape == (2,)
            assert body_angle.shape == ()

            force = self.stiffness * (
                target_position - absolute_positions[reduced_index]
            )
            return force * dt

        impulses = eqx.filter_vmap(compute_impulse)(indices, jnp.arange(len(indices)))
        impulses = eqx.error_if(
            impulses,
            jnp.linalg.norm(impulses.sum(axis=0)) > 1e-4,
            "constraint violated",
        )

        # apply the impulses only to the bodies that are specified in the constraint

        to_apply = jnp.zeros((bodies.mass.shape[0], 2))
        to_apply = to_apply.at[indices].set(impulses)

        assert to_apply.shape == bodies.velocity.shape

        bodies = eqx.tree_at(
            lambda x: x.velocity,
            bodies,
            bodies.velocity + to_apply,
        )

        return bodies
