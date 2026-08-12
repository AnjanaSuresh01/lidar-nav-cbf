"""Batched differential-drive simulator.

Every robot in the batch lives in its own map and they all step together, which
is what makes on-policy RL tractable on a laptop CPU: one vectorised ray cast
serves the whole batch.

The kinematics are the poster's, integrated in closed form rather than by Euler
steps.  A constant twist over one interval traces a circular arc, and using the
arc means a fast turn does not silently gain ground the robot never covered.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LIDAR, ROBOT, SIM, LidarSpec, RobotSpec, SimSpec
from .geometry import cast_rays, clearance
from .world import NavMap

SUBSTEPS = 4  # collision is checked at each, so a step cannot tunnel an obstacle

OUTCOME_RUNNING = 0
OUTCOME_SUCCESS = 1
OUTCOME_COLLISION = 2
OUTCOME_TIMEOUT = 3
OUTCOME_NAMES = {
    OUTCOME_SUCCESS: "success",
    OUTCOME_COLLISION: "collision",
    OUTCOME_TIMEOUT: "timeout",
}


@dataclass
class Observation:
    """What a planner is allowed to see: local sensing plus the goal bearing.

    No planner in this repository receives the map, the reference path or the
    geodesic field.  Those exist only for scoring and for shaping the RL reward
    during training.
    """

    ranges: np.ndarray  # (n, b) LiDAR distances, metres
    angles: np.ndarray  # (b,) beam angles in the robot frame
    goal_dist: np.ndarray  # (n,) straight-line distance to goal
    heading_err: np.ndarray  # (n,) bearing to goal in the robot frame, [-pi, pi]
    clearance: np.ndarray  # (n,) true distance to the nearest obstacle surface
    pos: np.ndarray  # (n, 2)
    theta: np.ndarray  # (n,)
    goal: np.ndarray  # (n, 2)
    last_u: np.ndarray  # (n, 2) previously commanded (v, omega)
    max_range: float


def integrate(
    pos: np.ndarray, theta: np.ndarray, v: np.ndarray, omega: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Exact unicycle integration of a constant twist over ``dt``."""
    theta_next = theta + omega * dt
    straight = np.abs(omega) < 1e-6
    r = np.where(straight, 0.0, v / np.where(straight, 1.0, omega))
    dx = np.where(straight, v * np.cos(theta) * dt, r * (np.sin(theta_next) - np.sin(theta)))
    dy = np.where(straight, v * np.sin(theta) * dt, -r * (np.cos(theta_next) - np.cos(theta)))
    return pos + np.stack([dx, dy], axis=-1), theta_next


def wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


class BatchSim:
    """N robots stepping in lockstep, one per map slot."""

    def __init__(
        self,
        maps: list[NavMap],
        n_envs: int,
        robot: RobotSpec = ROBOT,
        lidar: LidarSpec = LIDAR,
        spec: SimSpec = SIM,
        seed: int = 0,
        sequential: bool = False,
    ):
        if not maps:
            raise ValueError("need at least one map")
        shapes = {m.grid.shape for m in maps}
        if len(shapes) != 1:
            raise ValueError(f"all maps must share a grid shape, got {shapes}")
        self.maps = maps
        self.n = n_envs
        self.robot = robot
        self.lidar = lidar
        self.spec = spec
        self.res = maps[0].res
        self.rng = np.random.default_rng(seed)
        # Sequential assignment walks the map list in order and is what the
        # evaluation harness uses, so every planner sees the same maps in the
        # same order.  Random assignment is for training.
        self.sequential = sequential
        self._cursor = 0

        self.map_idx = np.zeros(n_envs, dtype=np.int64)
        self.grids = np.zeros((n_envs, *maps[0].grid.shape), dtype=bool)
        self.geodesic = np.zeros((n_envs, *maps[0].grid.shape), dtype=np.float64)
        self.goal = np.zeros((n_envs, 2))
        self.pos = np.zeros((n_envs, 2))
        self.theta = np.zeros(n_envs)
        self.last_u = np.zeros((n_envs, 2))
        self.steps = np.zeros(n_envs, dtype=np.int64)
        self.travelled = np.zeros(n_envs)
        self.min_clearance = np.full(n_envs, np.inf)
        self.froze = np.zeros(n_envs, dtype=bool)
        self.outcome = np.zeros(n_envs, dtype=np.int64)
        self._history = np.zeros((n_envs, spec.freeze_window, 2))
        self._obs_cache: Observation | None = None

        self.reset_all()

    # ---------------------------------------------------------------- resets

    def reset_all(self) -> Observation:
        self._cursor = 0
        self._reset_slots(np.arange(self.n))
        return self.observe()

    def _next_map(self) -> int:
        if self.sequential:
            idx = self._cursor % len(self.maps)
            self._cursor += 1
            return idx
        return int(self.rng.integers(len(self.maps)))

    def _reset_slots(self, slots: np.ndarray) -> None:
        for s in np.atleast_1d(slots):
            idx = self._next_map()
            nav = self.maps[idx]
            self.map_idx[s] = idx
            self.grids[s] = nav.grid
            self.geodesic[s] = np.where(np.isfinite(nav.geodesic), nav.geodesic, 1e3)
            self.goal[s] = nav.goal
            self.pos[s] = nav.start
            self.theta[s] = nav.start_theta
            self._history[s] = nav.start
        self.last_u[slots] = 0.0
        self.steps[slots] = 0
        self.travelled[slots] = 0.0
        self.min_clearance[slots] = np.inf
        self.froze[slots] = False
        self.outcome[slots] = OUTCOME_RUNNING
        self._obs_cache = None

    # ------------------------------------------------------------ perception

    def observe(self) -> Observation:
        """Sense the world.  Cached, because ray casting dominates the run time.

        The cache is invalidated by anything that moves a robot, so callers can
        ask for the observation as many times as is convenient without paying
        for it twice.
        """
        if self._obs_cache is not None:
            return self._obs_cache
        beam = self.lidar.angles
        world_angles = self.theta[:, None] + beam[None, :]
        ranges = cast_rays(
            self.grids,
            self.res,
            self.pos,
            world_angles,
            self.lidar.max_range,
            step=self.lidar.step,
            n_refine=self.lidar.n_refine,
        )
        delta = self.goal - self.pos
        goal_dist = np.linalg.norm(delta, axis=1)
        heading_err = wrap_angle(np.arctan2(delta[:, 1], delta[:, 0]) - self.theta)
        clear = clearance(self.grids, self.res, self.pos, reach=self.robot.radius + self.res)
        self._obs_cache = Observation(
            ranges=ranges,
            angles=beam,
            goal_dist=goal_dist,
            heading_err=heading_err,
            clearance=clear,
            pos=self.pos.copy(),
            theta=self.theta.copy(),
            goal=self.goal.copy(),
            last_u=self.last_u.copy(),
            max_range=self.lidar.max_range,
        )
        return self._obs_cache

    def set_pose(self, pos: np.ndarray | None = None, theta: np.ndarray | None = None) -> None:
        """Teleport robots and invalidate the observation cache.

        Assigning to ``sim.pos`` directly leaves a stale cached scan behind, so
        anything that repositions a robot outside of ``step`` goes through here.
        """
        if pos is not None:
            self.pos[:] = pos
        if theta is not None:
            self.theta[:] = theta
        self._obs_cache = None

    def geodesic_at(self, pos: np.ndarray | None = None) -> np.ndarray:
        """Nearest-cell lookup of the distance-to-goal field (training only)."""
        p = self.pos if pos is None else pos
        ny, nx = self.grids.shape[-2:]
        col = np.clip((p[:, 0] / self.res).astype(np.int64), 0, nx - 1)
        row = np.clip((p[:, 1] / self.res).astype(np.int64), 0, ny - 1)
        return self.geodesic[np.arange(self.n), row, col]

    # ------------------------------------------------------------------ step

    def step(self, v: np.ndarray, omega: np.ndarray) -> dict:
        """Advance one control interval and return per-environment outcomes.

        Commands are clipped to the wheel-speed envelope first, so a planner
        cannot win by asking for something the robot cannot do.
        """
        v, omega = self.clip_command(np.asarray(v, float), np.asarray(omega, float))
        self._obs_cache = None

        dt = self.spec.dt / SUBSTEPS
        collided = np.zeros(self.n, dtype=bool)
        for _ in range(SUBSTEPS):
            new_pos, new_theta = integrate(self.pos, self.theta, v, omega, dt)
            clear = clearance(self.grids, self.res, new_pos, reach=self.robot.radius + self.res)
            hit = clear <= self.robot.radius
            live = self.outcome == OUTCOME_RUNNING
            move = live & ~collided
            self.travelled += np.where(move, np.linalg.norm(new_pos - self.pos, axis=1), 0.0)
            self.pos = np.where(move[:, None], new_pos, self.pos)
            self.theta = np.where(move, new_theta, self.theta)
            self.min_clearance = np.where(
                move, np.minimum(self.min_clearance, clear), self.min_clearance
            )
            collided |= hit & move

        live = self.outcome == OUTCOME_RUNNING
        self.last_u = np.stack([v, omega], axis=1)
        self.steps += live

        # Freeze diagnostic: how far has the robot moved over the last window?
        slot = (self.steps - 1) % self.spec.freeze_window
        old = self._history[np.arange(self.n), slot]
        drift = np.linalg.norm(self.pos - old, axis=1)
        matured = self.steps > self.spec.freeze_window
        self.froze |= live & matured & (drift < self.spec.freeze_distance)
        self._history[np.arange(self.n), slot] = self.pos

        reached = np.linalg.norm(self.pos - self.goal, axis=1) < self.spec.goal_radius
        timeout = self.steps >= self.spec.max_steps

        self.outcome = np.where(live & collided, OUTCOME_COLLISION, self.outcome)
        self.outcome = np.where(
            (self.outcome == OUTCOME_RUNNING) & reached, OUTCOME_SUCCESS, self.outcome
        )
        self.outcome = np.where(
            (self.outcome == OUTCOME_RUNNING) & timeout, OUTCOME_TIMEOUT, self.outcome
        )
        return {
            "done": self.outcome != OUTCOME_RUNNING,
            "outcome": self.outcome.copy(),
            "collided": collided,
            "reached": reached,
        }

    def clip_command(self, v: np.ndarray, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project a twist onto what the robot can actually execute.

        Two limits, applied in order:

        1.  Acceleration: the command may not move further from the previous one
            than ``a * dt`` on either axis.
        2.  Wheel speed: the real limit is per wheel, not on ``v`` and ``omega``
            separately, so we scale the whole twist down until both wheels are
            legal rather than clipping each axis and silently changing the
            requested turn radius.

        Applied to every planner, so no arm can win by requesting the impossible.
        """
        dv = self.robot.max_accel_v * self.spec.dt
        dw = self.robot.max_accel_omega * self.spec.dt
        v = np.clip(v, self.last_u[:, 0] - dv, self.last_u[:, 0] + dv)
        omega = np.clip(omega, self.last_u[:, 1] - dw, self.last_u[:, 1] + dw)

        omega = np.clip(omega, -self.robot.max_omega, self.robot.max_omega)
        v_l, v_r = self.robot.twist_to_wheels(v, omega)
        worst = np.maximum(np.abs(v_l), np.abs(v_r))
        scale = np.minimum(1.0, self.robot.max_wheel_speed / np.maximum(worst, 1e-9))
        return v * scale, omega * scale

    # -------------------------------------------------------------- episodes

    def episode_records(self, slots: np.ndarray) -> list[dict]:
        return [
            {
                "map": self.maps[int(self.map_idx[s])].name,
                "map_index": int(self.map_idx[s]),
                "outcome": OUTCOME_NAMES[int(self.outcome[s])],
                "steps": int(self.steps[s]),
                "time": float(self.steps[s] * self.spec.dt),
                "travelled": float(self.travelled[s]),
                "min_clearance": float(self.min_clearance[s]),
                "froze": bool(self.froze[s]),
            }
            for s in np.atleast_1d(slots)
        ]
