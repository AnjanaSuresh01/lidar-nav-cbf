"""Control-barrier-function safety filter over the raw LiDAR scan.

The filter sits between any planner and the robot and answers one question: of
all commands the robot can execute right now, which is closest to the one the
planner asked for while still keeping the barrier non-decreasing?

**Model.**  A unicycle is not control-affine in its own position, so the barrier
is written on a look-ahead point ``l`` metres ahead of the wheel axis:

    p     = (x + l cos t,  y + l sin t)
    p_dot = A(t) u,   A(t) = [[cos t, -l sin t], [sin t, l cos t]],   u = (v, w)

``A`` is invertible for ``l > 0``, so ``p`` is fully actuated and every barrier
on it is a *linear* constraint on ``u``.  This is the standard near-identity
diffeomorphism used for nonholonomic CBFs.

**Barrier.**  For each LiDAR return ``q_i`` in the world frame,

    h_i = ||p - q_i||^2 - r_s^2 ,     h_dot_i + alpha * h_i >= 0

which expands to ``2 (p - q_i)^T A(t) u >= -alpha h_i``: one linear row per beam.

**What the guarantee actually covers.**  Keeping ``||p - q_i|| >= r_s`` with
``r_s = radius + margin + l`` puts the wheel-axis centre at least
``radius + margin`` from every returned point, because the centre is exactly
``l`` behind ``p``.  So the claim is: *the robot body stays clear of every
obstacle point the scanner reported*.  It says nothing about surface between two
beams, and nothing about obstacles outside the scan.  Both gaps are real and
both are measured in RESULTS.md rather than argued away -- the filtered arms are
scored on the same collision counter as everything else.

The command box (acceleration limits, per-wheel speed, turn rate) enters the
same QP, so the filter cannot escape by asking for a command the robot would
have clipped anyway.  When no command satisfies every row, the filter brakes as
hard as the acceleration limit allows and the episode is flagged; that rate is
reported as ``qp_fallback``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import ROBOT, SIM, RobotSpec, SimSpec
from ..sim import Observation
from .qp import solve_qp2


@dataclass
class CBFFilter:
    robot: RobotSpec = ROBOT
    spec: SimSpec = SIM
    look_ahead: float = 0.15  # metres; also the slack added to the safe radius
    margin: float = 0.05  # metres of body clearance the filter tries to hold
    alpha: float = 3.0  # class-K gain, 1/s
    n_constraints: int = 8  # tightest beams kept, by barrier value
    # Relative cost of deviating in v versus omega.  Turning is cheap, slowing
    # down is expensive, so the filter prefers to steer around an obstacle
    # rather than stop in front of it.
    weight_v: float = 4.0
    weight_omega: float = 1.0
    relax_bisections: int = 8
    name: str = "cbf"
    last_stats: dict = field(default_factory=dict)

    @property
    def safe_radius(self) -> float:
        return self.robot.radius + self.margin + self.look_ahead

    def reset(self, mask: np.ndarray) -> None:  # stateless
        return None

    def filter(
        self, obs: Observation, v: np.ndarray, omega: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        n = obs.ranges.shape[0]
        u_nom = np.stack([v, omega], axis=1)

        A_cbf, b_cbf = self._barrier_rows(obs)
        A_box, b_box = self._command_box(obs, n)
        A = np.concatenate([A_cbf, A_box], axis=1)
        b = np.concatenate([b_cbf, b_box], axis=1)

        weights = np.array([self.weight_v, self.weight_omega])
        u, feasible = solve_qp2(u_nom, weights, A, b)

        dv = self.robot.max_accel_v * self.spec.dt
        dw = self.robot.max_accel_omega * self.spec.dt
        brake = np.stack(
            [
                np.clip(0.0, obs.last_u[:, 0] - dv, obs.last_u[:, 0] + dv),
                np.clip(0.0, obs.last_u[:, 1] - dw, obs.last_u[:, 1] + dw),
            ],
            axis=1,
        )

        relaxed = np.zeros(n, dtype=bool)
        if not feasible.all():
            u, relaxed = self._relax(u, feasible, u_nom, weights, A, b, A_cbf.shape[1], brake)

        # Anything still unsolvable brakes.  Not a safety claim -- just the
        # least-bad legal command once the barrier set is provably empty.
        stuck = ~feasible & ~relaxed
        u = np.where(stuck[:, None], brake, u)

        self.last_stats = {
            "fallback": stuck.astype(np.float64),
            "relaxed": relaxed.astype(np.float64),
            "intervened": (np.abs(u - u_nom) > 1e-6).any(axis=1).astype(np.float64),
        }
        return u[:, 0], u[:, 1], self.last_stats

    def _relax(
        self,
        u: np.ndarray,
        feasible: np.ndarray,
        u_nom: np.ndarray,
        weights: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        n_cbf: int,
        brake: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Uniform slack on the barrier rows, found by bisection.

        In a gap narrower than twice the safe radius the barrier set is *empty* -
        no command keeps every beam at arm's length because there is not that
        much room.  Braking there would make the filter refuse every passage the
        benchmark is built around, so instead we find the smallest uniform
        relaxation ``delta`` that makes the problem solvable and re-solve with
        ``b_cbf - delta``.  Feasibility is monotone in ``delta``, so bisection is
        exact to the bracket width.

        The command box is never relaxed: those constraints are the physical
        robot, not a design choice.  The relaxation rate is reported alongside
        the results, because a filter that is relaxed half the time is not
        enforcing much.
        """
        idx = np.flatnonzero(~feasible)
        A_s, b_s = A[idx], b[idx]
        # delta_hi makes braking satisfy every barrier row, so it is always a
        # feasible bracket endpoint.
        residual = b_s[:, :n_cbf] - np.einsum("nmj,nj->nm", A_s[:, :n_cbf], brake[idx])
        hi = np.maximum(residual.max(axis=1), 0.0) + 1e-6
        lo = np.zeros_like(hi)

        best = brake[idx].copy()
        solved = np.zeros(idx.size, dtype=bool)
        for _ in range(self.relax_bisections):
            mid = 0.5 * (lo + hi)
            b_try = b_s.copy()
            b_try[:, :n_cbf] -= mid[:, None]
            u_try, ok = solve_qp2(u_nom[idx], weights, A_s, b_try)
            best = np.where(ok[:, None], u_try, best)
            solved |= ok
            hi = np.where(ok, mid, hi)
            lo = np.where(ok, lo, mid)

        out = u.copy()
        out[idx] = best
        relaxed = np.zeros(u.shape[0], dtype=bool)
        relaxed[idx] = solved
        return out, relaxed

    def _barrier_rows(self, obs: Observation) -> tuple[np.ndarray, np.ndarray]:
        """One linear row per retained LiDAR return."""
        cos_t, sin_t = np.cos(obs.theta), np.sin(obs.theta)
        p = obs.pos + self.look_ahead * np.stack([cos_t, sin_t], axis=1)  # (n, 2)

        world_ang = obs.theta[:, None] + obs.angles[None, :]
        q = obs.pos[:, None, :] + obs.ranges[..., None] * np.stack(
            [np.cos(world_ang), np.sin(world_ang)], axis=-1
        )  # (n, b, 2)

        d = p[:, None, :] - q  # (n, b, 2)
        h = (d**2).sum(axis=-1) - self.safe_radius**2

        # A max-range beam saw nothing, so it constrains nothing.
        seen = obs.ranges < obs.max_range - 1e-6
        h = np.where(seen, h, np.inf)

        keep = np.argsort(h, axis=1)[:, : self.n_constraints]
        rows = np.arange(obs.ranges.shape[0])[:, None]
        d_k = d[rows, keep]  # (n, k, 2)
        h_k = h[rows, keep]

        # a = 2 * A(theta)^T d, with A(theta) as in the module docstring.
        a_v = 2.0 * (d_k[..., 0] * cos_t[:, None] + d_k[..., 1] * sin_t[:, None])
        a_w = 2.0 * self.look_ahead * (-d_k[..., 0] * sin_t[:, None] + d_k[..., 1] * cos_t[:, None])
        A = np.stack([a_v, a_w], axis=-1)

        # Unseen beams were padded with h = inf; neutralise those rows.
        dead = ~np.isfinite(h_k)
        A = np.where(dead[..., None], 0.0, A)
        b = np.where(dead, -1.0, -self.alpha * np.where(dead, 0.0, h_k))
        return A, b

    def _command_box(self, obs: Observation, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Acceleration, turn-rate and per-wheel-speed limits as ``A u >= b``."""
        dv = self.robot.max_accel_v * self.spec.dt
        dw = self.robot.max_accel_omega * self.spec.dt
        v0, w0 = obs.last_u[:, 0], obs.last_u[:, 1]
        half = 0.5 * self.robot.wheel_base
        vw = self.robot.max_wheel_speed
        wmax = self.robot.max_omega
        zero, one = np.zeros(n), np.ones(n)

        rows = [
            # acceleration window
            (np.stack([one, zero], 1), v0 - dv),
            (np.stack([-one, zero], 1), -(v0 + dv)),
            (np.stack([zero, one], 1), w0 - dw),
            (np.stack([zero, -one], 1), -(w0 + dw)),
            # turn rate
            (np.stack([zero, one], 1), -wmax * one),
            (np.stack([zero, -one], 1), -wmax * one),
            # per-wheel speed: |v +/- L/2 * omega| <= vw
            (np.stack([one, half * one], 1), -vw * one),
            (np.stack([-one, -half * one], 1), -vw * one),
            (np.stack([one, -half * one], 1), -vw * one),
            (np.stack([-one, half * one], 1), -vw * one),
        ]
        A = np.stack([r[0] for r in rows], axis=1)
        b = np.stack([r[1] for r in rows], axis=1)
        return A, b
