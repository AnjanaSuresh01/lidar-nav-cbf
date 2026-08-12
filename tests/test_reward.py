"""The RL reward signal.

These exist because of a bug that cost a full training run.  The geodesic field
was defined only on configuration-space free cells, so whenever the robot hugged
an obstacle -- a cell whose centre is not C-space free, even though the disc is
collision-free -- the progress term jumped by the unreachable sentinel.  The
resulting reward had a standard deviation of 615 and spikes to +/- 3000, and PPO
learned the only sane response: stand still and collect 0 per cent success.

Nothing about that was visible in a loss curve.  It is visible in these numbers.

They drive :class:`~bahn.gym_env.NavCore` directly rather than going through the
Stable-Baselines3 ``VecEnv`` wrapper, so the guard keeps running in CI on a
plain ``pip install -e ".[dev]"`` without pulling in torch.
"""

from __future__ import annotations

import numpy as np

from bahn.config import LIDAR, ROBOT, SIM
from bahn.gym_env import NavCore, RewardConfig
from bahn.world import barn_suite, split_suite

BOUND = RewardConfig.success + abs(RewardConfig.collision)  # largest legitimate step reward


def make_core(n_envs: int = 16) -> NavCore:
    maps, _ = split_suite(list(barn_suite(repetitions=3)), n_test=1)
    return NavCore(maps, n_envs, 0, ROBOT, LIDAR, SIM, sequential=False)


def rollout_rewards(steps: int = 250, n_envs: int = 16) -> tuple[np.ndarray, NavCore]:
    core = make_core(n_envs)
    rng = np.random.default_rng(0)
    rewards = []
    for _ in range(steps):
        a = rng.uniform(-1.0, 1.0, size=(n_envs, 2)).astype(np.float32)
        a[:, 0] = 1.0  # drive forward, so the robot actually meets obstacles
        reward, done, _, _ = core.advance(a)
        rewards.append(reward)
        finished = np.flatnonzero(done)
        if finished.size:
            core.reset_slots(finished)
    return np.concatenate(rewards), core


def test_geodesic_field_has_no_unreachable_cells_in_play():
    _, core = rollout_rewards(steps=60)
    assert (core.sim.geodesic >= 999).sum() == 0


def test_reward_stays_on_a_sane_scale():
    """A shaped step reward must not dwarf the terminal rewards it competes with."""
    r, _ = rollout_rewards()
    assert np.isfinite(r).all()
    assert np.abs(r).max() <= BOUND, f"single-step reward reached {np.abs(r).max():.1f}"
    assert r.std() < 3.0, f"reward std {r.std():.1f} is noise, not signal"


def test_progress_term_is_capped_at_what_the_robot_could_cover():
    """Cell quantisation at obstacle corners can shift the field by more than the
    robot moved; the cap keeps that from being read as real progress."""
    cap = ROBOT.max_v * SIM.dt * 1.5
    assert cap > ROBOT.max_v * SIM.dt  # never binds on legitimate motion
    r, _ = rollout_rewards(steps=120)
    ceiling = RewardConfig.progress * cap + RewardConfig.success + abs(RewardConfig.step)
    assert r.max() <= ceiling + 1e-6


def test_standing_still_is_worse_than_making_progress():
    """The whole point of the shaping: freezing must not be the optimal policy."""

    def mean_reward(v_action: float) -> float:
        core = make_core(n_envs=8)
        rewards = []
        for _ in range(40):
            a = np.tile([v_action, 0.0], (8, 1)).astype(np.float32)
            reward, _, _, _ = core.advance(a)
            rewards.append(reward)
        return float(np.mean(rewards))

    assert mean_reward(1.0) > mean_reward(-1.0)
