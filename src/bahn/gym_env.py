"""Gymnasium interface over :class:`~bahn.sim.BatchSim`.

Two views of the same simulator:

*   :class:`BahnEnv` - a standard single-agent ``gymnasium.Env``, so the
    benchmark is usable with any RL library.
*   :class:`BahnVecEnv` - a Stable-Baselines3 ``VecEnv`` that steps the whole
    batch through one vectorised ray cast.  This is the one training uses; going
    through N independent copies of the single env would spend all its time in
    Python rather than in NumPy.

**Observation** (41 floats) is strictly local: the normalised scan, the goal in
polar robot-frame coordinates, and the previous command.  No map, no reference
path.  A policy trained here is a local planner in the same sense DWA is.

**Reward** uses the geodesic distance-to-goal field, which *is* privileged
information -- but only at training time, and only to score progress the policy
already made.  Shaping on straight-line distance instead turns every concave
obstacle into a reward wall the policy learns to press against; shaping on
geodesic distance removes that artefact without telling the policy anything at
run time.  This is the one place privileged information enters the project and
it is deliberately confined here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import LIDAR, ROBOT, SIM, LidarSpec, RobotSpec, SimSpec
from .sim import OUTCOME_COLLISION, OUTCOME_RUNNING, OUTCOME_SUCCESS, BatchSim, Observation
from .world import NavMap

OBS_EXTRA = 5  # goal distance, sin/cos of bearing, previous v and omega
V_MIN_FRAC = -0.3  # reversing is allowed, at 30 per cent of forward top speed


class RewardConfig:
    """Reward terms.  Kept small and few; heavy shaping buys a timid policy."""

    progress = 3.0  # per metre of geodesic distance closed
    step = -0.004  # per control interval, so dawdling costs something
    success = 5.0
    collision = -5.0
    proximity = -0.35  # scaled by how far inside `proximity_band` the robot is
    proximity_band = 0.45  # metres of clearance below which the penalty starts
    spin = -0.002  # per (rad/s)^2, discourages pirouetting in place


def encode_obs(obs: Observation, robot: RobotSpec, lidar: LidarSpec) -> np.ndarray:
    scan = np.clip(obs.ranges / lidar.max_range, 0.0, 1.0)
    goal = np.clip(obs.goal_dist / 10.0, 0.0, 1.0)
    return np.concatenate(
        [
            scan,
            goal[:, None],
            np.sin(obs.heading_err)[:, None],
            np.cos(obs.heading_err)[:, None],
            (obs.last_u[:, 0] / robot.max_v)[:, None],
            (obs.last_u[:, 1] / robot.max_omega)[:, None],
        ],
        axis=1,
    ).astype(np.float32)


def decode_action(action: np.ndarray, robot: RobotSpec) -> tuple[np.ndarray, np.ndarray]:
    """Map ``[-1, 1]^2`` onto the twist envelope."""
    a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    v_min = V_MIN_FRAC * robot.max_v
    v = v_min + 0.5 * (a[:, 0] + 1.0) * (robot.max_v - v_min)
    return v, a[:, 1] * robot.max_omega


def _spaces(lidar: LidarSpec):
    import gymnasium as gym

    dim = lidar.n_beams + OBS_EXTRA
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(dim,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    return obs_space, act_space


class NavCore:
    """Reward and termination logic, shared by both interfaces.

    Public so that the reward can be tested without importing an RL library:
    the guard in ``tests/test_reward.py`` protects against a bug that silently
    cost a whole training run, and it should keep running in CI whether or not
    torch is installed.
    """

    def __init__(
        self,
        maps: list[NavMap],
        n_envs: int,
        seed: int,
        robot: RobotSpec,
        lidar: LidarSpec,
        spec: SimSpec,
        sequential: bool,
    ):
        self.robot, self.lidar, self.spec = robot, lidar, spec
        self.sim = BatchSim(
            maps, n_envs, robot=robot, lidar=lidar, spec=spec, seed=seed, sequential=sequential
        )
        self.prev_geo = self.sim.geodesic_at()

    def obs(self) -> np.ndarray:
        return encode_obs(self.sim.observe(), self.robot, self.lidar)

    def advance(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        v, omega = decode_action(action, self.robot)
        pre = self.sim.observe()
        out = self.sim.step(v, omega)

        geo = self.sim.geodesic_at()
        r = RewardConfig
        # Clamp progress to what the robot could physically have covered in one
        # interval.  The geodesic field is continuous enough that this never
        # binds in normal operation; it is a guard so that any future
        # discontinuity in the field shows up as a capped reward rather than as
        # a policy that mysteriously refuses to move.
        cap = self.robot.max_v * self.spec.dt * 1.5
        reward = r.progress * np.clip(self.prev_geo - geo, -cap, cap) + r.step
        reward += r.spin * self.sim.last_u[:, 1] ** 2
        deficit = np.maximum(0.0, 1.0 - pre.clearance / r.proximity_band)
        reward += r.proximity * deficit
        reward += np.where(self.sim.outcome == OUTCOME_SUCCESS, r.success, 0.0)
        reward += np.where(self.sim.outcome == OUTCOME_COLLISION, r.collision, 0.0)
        self.prev_geo = geo

        done = out["done"]
        infos = [
            {
                "outcome": int(o),
                "TimeLimit.truncated": bool(o == 3),
            }
            for o in self.sim.outcome
        ]
        return reward, done, self.sim.outcome.copy(), infos

    def reset_slots(self, slots: np.ndarray) -> None:
        self.sim._reset_slots(slots)
        self.prev_geo = np.where(
            np.isin(np.arange(self.sim.n), slots), self.sim.geodesic_at(), self.prev_geo
        )


def make_env_class():
    """Built lazily so importing :mod:`bahn` does not require gymnasium."""
    import gymnasium as gym

    class BahnEnv(gym.Env):
        """Single-agent view.  Slower than the vectorised one; for interop."""

        metadata = {"render_modes": []}

        def __init__(
            self,
            maps: list[NavMap],
            seed: int = 0,
            robot: RobotSpec = ROBOT,
            lidar: LidarSpec = LIDAR,
            spec: SimSpec = SIM,
        ):
            self.observation_space, self.action_space = _spaces(lidar)
            self._core = NavCore(maps, 1, seed, robot, lidar, spec, sequential=False)

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            super().reset(seed=seed)
            self._core.reset_slots(np.array([0]))
            return self._core.obs()[0], {}

        def step(self, action):
            reward, done, outcome, infos = self._core.advance(np.asarray(action)[None, :])
            terminated = bool(outcome[0] in (OUTCOME_SUCCESS, OUTCOME_COLLISION))
            truncated = bool(done[0]) and not terminated
            return self._core.obs()[0], float(reward[0]), terminated, truncated, infos[0]

    return BahnEnv


def make_vec_env_class():
    from stable_baselines3.common.vec_env import VecEnv

    class BahnVecEnv(VecEnv):
        """N environments sharing one vectorised simulator."""

        def __init__(
            self,
            maps: list[NavMap],
            n_envs: int = 32,
            seed: int = 0,
            robot: RobotSpec = ROBOT,
            lidar: LidarSpec = LIDAR,
            spec: SimSpec = SIM,
        ):
            obs_space, act_space = _spaces(lidar)
            super().__init__(n_envs, obs_space, act_space)
            self._core = NavCore(maps, n_envs, seed, robot, lidar, spec, sequential=False)
            self._actions: np.ndarray | None = None

        def reset(self):
            self._core.reset_slots(np.arange(self.num_envs))
            return self._core.obs()

        def step_async(self, actions: np.ndarray) -> None:
            self._actions = actions

        def step_wait(self):
            reward, done, outcome, infos = self._core.advance(self._actions)
            finished = np.flatnonzero(done)
            if finished.size:
                # Only pay for the pre-reset observation when someone needs it.
                terminal = self._core.obs()
                for i in finished:
                    infos[i]["terminal_observation"] = terminal[i]
                self._core.reset_slots(finished)
            return self._core.obs(), reward.astype(np.float32), done, infos

        def close(self) -> None:
            return None

        def get_attr(self, attr_name: str, indices=None) -> list[Any]:
            return [getattr(self, attr_name, None)] * self.num_envs

        def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
            setattr(self, attr_name, value)

        def env_method(self, method_name: str, *args, indices=None, **kwargs) -> list[Any]:
            raise NotImplementedError("BahnVecEnv does not wrap per-environment objects")

        def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
            return [False] * self.num_envs

        @property
        def outcome(self) -> np.ndarray:
            return self._core.sim.outcome

    return BahnVecEnv


__all__ = [
    "OUTCOME_RUNNING",
    "NavCore",
    "RewardConfig",
    "decode_action",
    "encode_obs",
    "make_env_class",
    "make_vec_env_class",
]
