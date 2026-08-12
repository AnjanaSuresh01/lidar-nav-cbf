"""The RL reward signal.

These exist because of a bug that cost a full training run.  The geodesic field
was defined only on configuration-space free cells, so whenever the robot hugged
an obstacle -- a cell whose centre is not C-space free, even though the disc is
collision-free -- the progress term jumped by the unreachable sentinel.  The
resulting reward had a standard deviation of 615 and spikes to +/- 3000, and PPO
learned the only sane response: stand still and collect 0 per cent success.

Nothing about that was visible in a loss curve.  It is visible in these numbers.
"""

from __future__ import annotations

import numpy as np

from bahn.config import ROBOT, SIM
from bahn.gym_env import RewardConfig, make_vec_env_class
from bahn.world import barn_suite, split_suite

BOUND = 5.0 + abs(RewardConfig.collision)  # the largest legitimate single-step reward


def rollout_rewards(steps: int = 250, n_envs: int = 16):
    maps, _ = split_suite(list(barn_suite(repetitions=3)), n_test=1)
    env = make_vec_env_class()(maps, n_envs=n_envs, seed=0)
    env.reset()
    rng = np.random.default_rng(0)
    rewards = []
    for _ in range(steps):
        a = rng.uniform(-1.0, 1.0, size=(n_envs, 2)).astype(np.float32)
        a[:, 0] = 1.0  # drive forward, so the robot actually meets obstacles
        env.step_async(a)
        _, r, _, _ = env.step_wait()
        rewards.append(r)
    return np.concatenate(rewards), env


def test_geodesic_field_has_no_unreachable_cells_in_play():
    _, env = rollout_rewards(steps=60)
    assert (env._core.sim.geodesic >= 999).sum() == 0


def test_reward_stays_on_a_sane_scale():
    """A shaped step reward must not dwarf the terminal rewards it competes with."""
    r, _ = rollout_rewards()
    assert np.isfinite(r).all()
    assert np.abs(r).max() <= BOUND, f"single-step reward reached {np.abs(r).max():.1f}"
    assert r.std() < 3.0, f"reward std {r.std():.1f} is noise, not signal"


def test_progress_term_is_capped_at_what_the_robot_could_cover():
    """Cell-quantisation at obstacle corners can shift the field by more than the
    robot moved; the cap keeps that from being read as real progress."""
    cap = ROBOT.max_v * SIM.dt * 1.5
    assert cap > ROBOT.max_v * SIM.dt  # never binds on legitimate motion
    r, _ = rollout_rewards(steps=120)
    ceiling = RewardConfig.progress * cap + RewardConfig.success + abs(RewardConfig.step)
    assert r.max() <= ceiling + 1e-6


def test_standing_still_is_worse_than_making_progress():
    """The whole point of the shaping: freezing must not be the optimal policy."""
    maps, _ = split_suite(list(barn_suite(repetitions=3)), n_test=1)
    env = make_vec_env_class()(maps, n_envs=8, seed=0)
    env.reset()
    still = []
    for _ in range(40):
        env.step_async(np.tile([-1.0, 0.0], (8, 1)).astype(np.float32))
        _, r, _, _ = env.step_wait()
        still.append(r)

    env.reset()
    moving = []
    for _ in range(40):
        env.step_async(np.tile([1.0, 0.0], (8, 1)).astype(np.float32))
        _, r, _, _ = env.step_wait()
        moving.append(r)

    assert np.mean(moving) > np.mean(still)
