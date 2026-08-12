"""Dynamic Window Approach (Fox, Burgard & Thrun, 1997) over the LiDAR scan.

The classical baseline, and the same one BARN's original paper evaluates.  It is
given exactly the information every other planner gets: the scan and the goal
bearing.  Candidate trajectories are scored against obstacle points reconstructed
from the scan, not against the occupancy grid, so it has no map advantage.

DWA is also the teacher for the behaviour-cloned network, which is why it is
worth having a properly tuned one rather than a token implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import ROBOT, RobotSpec
from ..sim import Observation


@dataclass
class DWA:
    robot: RobotSpec = ROBOT
    dt: float = 0.1  # control interval, matches the simulator
    n_v: int = 5
    n_w: int = 15
    horizon_steps: int = 8
    horizon_dt: float = 0.2
    accel_v: float = 2.0  # m/s^2, sets the width of the dynamic window
    accel_w: float = 6.0  # rad/s^2
    w_progress: float = 1.0
    w_heading: float = 0.3
    w_clearance: float = 0.35
    w_velocity: float = 0.2
    clearance_cap: float = 0.5  # metres beyond which extra room stops helping
    safety_pad: float = 0.03  # metres of margin on top of the robot radius
    name: str = "dwa"

    def reset(self, mask: np.ndarray) -> None:  # stateless
        return None

    def act(self, obs: Observation) -> tuple[np.ndarray, np.ndarray]:
        n = obs.ranges.shape[0]
        v_cand, w_cand = self._window(obs)  # (n, K)
        traj, heads = self._rollout(v_cand, w_cand)  # (n, K, T, 2), (n, K, T)

        obstacles, valid = self._scan_points(obs)  # (n, B, 2), (n, B)
        d = np.linalg.norm(traj[:, :, :, None, :] - obstacles[:, None, None, :, :], axis=-1)
        d = np.where(valid[:, None, None, :], d, np.inf)
        clear = d.min(axis=(2, 3))  # (n, K)

        room = clear - self.robot.radius - self.safety_pad
        admissible = room > 0.0
        # A candidate is only admissible if the robot can still stop inside the
        # space it has; this is the braking condition from the original paper.
        admissible &= np.abs(v_cand) <= np.sqrt(2.0 * np.maximum(room, 0.0) * self.accel_v) + 1e-9
        # ...and if the wheels can actually produce it.  Sampling v and omega on
        # independent axes generates corner combinations outside the wheel-speed
        # diamond; scoring those would mean choosing a trajectory the robot then
        # does not drive, since the simulator scales the twist back.
        v_l, v_r = self.robot.twist_to_wheels(v_cand, w_cand)
        admissible &= np.maximum(np.abs(v_l), np.abs(v_r)) <= self.robot.max_wheel_speed + 1e-9

        goal_local = np.stack(
            [obs.goal_dist * np.cos(obs.heading_err), obs.goal_dist * np.sin(obs.heading_err)],
            axis=-1,
        )  # (n, 2)
        end = traj[:, :, -1, :]
        to_goal = np.arctan2(
            goal_local[:, None, 1] - end[..., 1], goal_local[:, None, 0] - end[..., 0]
        )
        bearing = np.abs(
            np.arctan2(np.sin(to_goal - heads[:, :, -1]), np.cos(to_goal - heads[:, :, -1]))
        )
        s_head = (np.pi - bearing) / np.pi

        # Distance actually closed over the horizon, normalised by the most the
        # robot could close.  Scoring bearing alone lets a candidate that merely
        # points at the goal beat one that drives towards it, and the robot then
        # sits still turning on the spot -- which is exactly what a clearance
        # term rewards it for doing.
        reach = self.robot.max_v * self.horizon_steps * self.horizon_dt
        closed = obs.goal_dist[:, None] - np.linalg.norm(goal_local[:, None, :] - end, axis=-1)
        s_progress = closed / reach

        s_clear = np.minimum(room, self.clearance_cap) / self.clearance_cap
        s_vel = np.maximum(v_cand, 0.0) / self.robot.max_v

        score = (
            self.w_progress * s_progress
            + self.w_heading * s_head
            + self.w_clearance * s_clear
            + self.w_velocity * s_vel
        )
        score = np.where(admissible, score, -np.inf)

        best = score.argmax(axis=1)
        rows = np.arange(n)
        v = v_cand[rows, best]
        w = w_cand[rows, best]

        # No admissible candidate: stop and rotate towards the roomier side.
        stuck = ~admissible.any(axis=1)
        if stuck.any():
            left = obs.ranges[:, obs.angles > 0].mean(axis=1)
            right = obs.ranges[:, obs.angles < 0].mean(axis=1)
            spin = np.where(left >= right, self.robot.max_omega, -self.robot.max_omega)
            v = np.where(stuck, 0.0, v)
            w = np.where(stuck, spin, w)
        return v, w

    def _window(self, obs: Observation) -> tuple[np.ndarray, np.ndarray]:
        v0, w0 = obs.last_u[:, 0], obs.last_u[:, 1]
        dv = self.accel_v * self.dt
        dw = self.accel_w * self.dt
        v_lo = np.maximum(v0 - dv, -0.5 * self.robot.max_v)  # reversing is allowed but slow
        v_hi = np.minimum(v0 + dv, self.robot.max_v)
        w_lo = np.maximum(w0 - dw, -self.robot.max_omega)
        w_hi = np.minimum(w0 + dw, self.robot.max_omega)
        frac_v = np.linspace(0.0, 1.0, self.n_v)
        frac_w = np.linspace(0.0, 1.0, self.n_w)
        vs = v_lo[:, None] + (v_hi - v_lo)[:, None] * frac_v[None, :]
        ws = w_lo[:, None] + (w_hi - w_lo)[:, None] * frac_w[None, :]
        v_cand = np.repeat(vs, self.n_w, axis=1)
        w_cand = np.tile(ws, (1, self.n_v))
        return v_cand, w_cand

    def _rollout(self, v: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Constant-twist arcs in the robot frame, starting at the origin."""
        t = np.arange(1, self.horizon_steps + 1) * self.horizon_dt  # (T,)
        theta = w[..., None] * t  # (n, K, T)
        small = np.abs(w[..., None]) < 1e-6
        r = np.where(small, 0.0, v[..., None] / np.where(small, 1.0, w[..., None]))
        x = np.where(small, v[..., None] * t, r * np.sin(theta))
        y = np.where(small, 0.0, r * (1.0 - np.cos(theta)))
        return np.stack([x, y], axis=-1), theta

    def _scan_points(self, obs: Observation) -> tuple[np.ndarray, np.ndarray]:
        """Beam endpoints in the robot frame, with max-range returns discarded."""
        valid = obs.ranges < obs.max_range - 1e-6
        pts = (
            obs.ranges[..., None]
            * np.stack([np.cos(obs.angles), np.sin(obs.angles)], axis=-1)[None, :, :]
        )
        return pts, valid
