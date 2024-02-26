from ._utils import lob_to_soa, soa_to_lob


class SimpleConstraintSolver:
    def __init__(self, loops=1):
        self.loops = loops

    def _solve(self, soa_bodies, constraints):
        for constraint in constraints:
            soa_bodies = constraint.apply(soa_bodies)
        return soa_bodies

    def solve(self, bodies, constraints):
        soa_bodies = lob_to_soa(bodies)
        for _ in range(self.loops):
            soa_bodies = self._solve(soa_bodies, constraints)
        return soa_to_lob(bodies, from_soa=soa_bodies)
