"""Kinematics, command limits and episode bookkeeping."""

from __future__ import annotations

import numpy as np
import pytest

from bahn.config import ROBOT, SIM
from bahn.sim import OUTCOME_COLLISION, OUTCOME_SUCCESS, BatchSim, integrate, wrap_angle

from .conftest import make_map


def a_map():
    return make_map(0.15, 3, 0)


def test_straight_line_integration():
    pos = np.zeros((1, 2))
    theta = np.zeros(1)
    p, t = integrate(pos, theta, np.array([1.0]), np.array([0.0]), 0.5)
    assert p[0] == pytest.approx([0.5, 0.0])
    assert t[0] == pytest.approx(0.0)


def test_constant_twist_traces_a_circle_of_radius_v_over_omega():
    v, w = 1.0, 2.0
    pos, theta = np.zeros((1, 2)), np.zeros(1)
    # A full revolution must return the robot exactly to its starting point.
    for _ in range(400):
        pos, theta = integrate(pos, theta, np.array([v]), np.array([w]), 2 * np.pi / w / 400)
    assert pos[0] == pytest.approx([0.0, 0.0], abs=1e-9)
    assert wrap_angle(theta)[0] == pytest.approx(0.0, abs=1e-9)


def test_circle_radius_is_v_over_omega():
    v, w = 0.8, 1.6
    pos, theta = np.zeros((1, 2)), np.zeros(1)
    quarter = (np.pi / 2) / w
    pos, theta = integrate(pos, theta, np.array([v]), np.array([w]), quarter)
    assert pos[0] == pytest.approx([v / w, v / w], abs=1e-9)


def test_wheel_round_trip():
    v = np.array([0.3, -0.2])
    w = np.array([1.1, -0.4])
    vl, vr = ROBOT.twist_to_wheels(v, w)
    v2, w2 = ROBOT.wheels_to_twist(vl, vr)
    assert v2 == pytest.approx(v)
    assert w2 == pytest.approx(w)


def test_acceleration_limit_is_enforced():
    sim = BatchSim([a_map()], n_envs=1)
    sim.step(np.array([10.0]), np.array([10.0]))
    assert sim.last_u[0, 0] <= ROBOT.max_accel_v * SIM.dt + 1e-9
    assert sim.last_u[0, 1] <= ROBOT.max_accel_omega * SIM.dt + 1e-9


def test_wheel_speed_limit_is_enforced_and_preserves_turn_radius():
    sim = BatchSim([a_map()], n_envs=1)
    sim.last_u[:] = [[1.0, 3.0]]  # pretend we are already moving fast
    v, w = sim.clip_command(np.array([1.0]), np.array([3.0]))
    vl, vr = ROBOT.twist_to_wheels(v, w)
    assert max(abs(vl[0]), abs(vr[0])) <= ROBOT.max_wheel_speed + 1e-9
    # Scaling the whole twist keeps v/omega, i.e. the requested arc.
    assert v[0] / w[0] == pytest.approx(1.0 / 3.0)


def test_collision_ends_the_episode_and_records_the_outcome():
    nav = a_map()
    sim = BatchSim([nav], n_envs=1)
    sim.pos[:] = nav.start
    sim.theta[:] = -np.pi / 2  # drive at the bottom wall
    for _ in range(SIM.max_steps):
        sim.step(np.array([1.0]), np.array([0.0]))
        if sim.outcome[0] != 0:
            break
    assert sim.outcome[0] == OUTCOME_COLLISION
    assert sim.min_clearance[0] <= ROBOT.radius


def test_reaching_the_goal_is_a_success():
    nav = a_map()
    sim = BatchSim([nav], n_envs=1)
    sim.pos[:] = nav.goal + np.array([0.1, 0.0])
    sim.step(np.array([0.0]), np.array([0.0]))
    assert sim.outcome[0] == OUTCOME_SUCCESS


def test_no_tunnelling_through_a_thin_obstacle():
    """Sub-stepping must catch a collision the end-of-interval pose would miss."""
    grid = np.zeros((10, 10), dtype=bool)
    grid[0, :] = True
    grid[-1, :] = True
    grid[4:6, 5] = True
    nav = make_map(0.15, 3, 0)
    nav.grid = grid
    nav.start = np.array([0.6, 1.2])
    nav.goal = np.array([2.2, 1.2])
    nav.geodesic = np.zeros_like(nav.grid, dtype=float)
    sim = BatchSim([nav], n_envs=1)
    for _ in range(40):
        sim.step(np.array([1.0]), np.array([0.0]))
        if sim.outcome[0] != 0:
            break
    assert sim.outcome[0] == OUTCOME_COLLISION


def test_freeze_flag_fires_for_a_stationary_robot():
    sim = BatchSim([a_map()], n_envs=1)
    for _ in range(SIM.freeze_window + 5):
        sim.step(np.array([0.0]), np.array([0.0]))
    assert sim.froze[0]


def test_freeze_flag_stays_clear_for_a_moving_robot():
    sim = BatchSim([a_map()], n_envs=1)
    for _ in range(SIM.freeze_window - 1):
        sim.step(np.array([1.0]), np.array([0.0]))
        if sim.outcome[0] != 0:
            break
    assert not sim.froze[0]


def test_observation_cache_is_invalidated_by_motion():
    sim = BatchSim([a_map()], n_envs=1)
    first = sim.observe()
    assert sim.observe() is first  # cached
    sim.step(np.array([1.0]), np.array([0.0]))
    assert sim.observe() is not first
