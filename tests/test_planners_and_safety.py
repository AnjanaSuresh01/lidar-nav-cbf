"""Planner conventions and the safety filter's actual claim.

The important test here is :func:`test_filter_prevents_collision_for_a_suicidal_planner`.
It runs a planner whose entire policy is "full speed ahead" and asserts that the
filter still keeps the robot off the walls -- which is the only claim the filter
makes that is worth anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from bahn.config import LIDAR, ROBOT
from bahn.planners.base import poster_encoding
from bahn.planners.dwa import DWA
from bahn.planners.reactive import ReactiveANN
from bahn.rollout import run_suite
from bahn.safety.cbf import CBFFilter
from bahn.sim import OUTCOME_COLLISION, BatchSim
from bahn.world import barn_suite

from .conftest import make_map


class FullSpeedAhead:
    """Worst plausible nominal planner: maximum forward speed, no steering."""

    name = "suicidal"

    def reset(self, mask):
        return None

    def act(self, obs):
        n = obs.ranges.shape[0]
        return np.full(n, ROBOT.max_v), np.zeros(n)


def a_map(fill=0.20, iters=3, index=0):
    return make_map(fill, iters, index)


def sim_with(nav, n=1):
    return BatchSim([nav], n_envs=n)


# ------------------------------------------------------------------ encoding


def test_poster_encoding_matches_the_stated_ranges():
    nav = a_map()
    obs = sim_with(nav).observe()
    steer, inv = poster_encoding(obs)
    assert (steer >= -0.5).all() and (steer <= 0.5).all()
    # d normalised to [0.1, 1] then inverted gives [1, 10].
    assert (inv >= 1.0 - 1e-9).all() and (inv <= 10.0 + 1e-9).all()


def test_poster_encoding_gives_closer_obstacles_more_weight():
    nav = a_map()
    obs = sim_with(nav).observe()
    _, inv = poster_encoding(obs)
    order = np.argsort(obs.ranges[0])
    assert np.all(np.diff(inv[0][order]) <= 1e-12)  # monotonically decreasing


def test_steering_term_is_zero_when_pointed_at_the_goal():
    nav = a_map()
    sim = sim_with(nav)
    sim.set_pose(theta=np.arctan2(*(nav.goal - nav.start)[::-1]))
    steer, _ = poster_encoding(sim.observe())
    assert steer[0] == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------------ planners


@pytest.mark.parametrize("planner", [ReactiveANN(), DWA()])
def test_planners_respect_the_command_envelope(planner):
    nav = a_map()
    sim = sim_with(nav, n=1)
    for _ in range(30):
        v, w = planner.act(sim.observe())
        assert np.isfinite(v).all() and np.isfinite(w).all()
        vl, vr = ROBOT.twist_to_wheels(v, w)
        assert max(abs(vl[0]), abs(vr[0])) <= ROBOT.max_wheel_speed + 1e-6
        sim.step(v, w)


def test_reactive_turns_away_from_a_one_sided_obstacle():
    """A wall close on the left must produce a right turn (negative omega)."""
    nav = a_map()
    sim = sim_with(nav)
    obs = sim.observe()
    obs.ranges[:] = LIDAR.max_range
    left = (obs.angles > 0.2) & (obs.angles < 1.2)
    obs.ranges[0, left] = 0.3
    obs.heading_err[:] = 0.0
    _, w = ReactiveANN().act(obs)
    assert w[0] < 0


def test_dwa_collides_less_than_the_hand_wired_reactive_controller():
    """DWA rejects candidate arcs that hit something; the reactive net has no
    such check. Collision rate is therefore a property of the two methods.

    Note what is *not* asserted here. An earlier version of this test claimed DWA
    must reach the goal more often, and once the reactive gains were tuned
    properly that became false on easy maps (0.625 vs 0.563 over these 16): a
    tuned Braitenberg vehicle simply barrels down an open corridor while DWA is
    cautious. That is a real result, not a misconfiguration, and it is why the
    benchmark stratifies by difficulty instead of reporting one mean.
    """
    maps = list(barn_suite(repetitions=2))[:16]
    reactive = run_suite(ReactiveANN(), maps, batch=8)
    dwa = run_suite(DWA(), maps, batch=8)
    rate = lambda rs, o: sum(r["outcome"] == o for r in rs) / len(rs)  # noqa: E731
    assert rate(dwa, "collision") < rate(reactive, "collision")


# -------------------------------------------------------------------- filter


def test_filter_leaves_a_safe_command_alone_in_open_space():
    grid = np.zeros((60, 60), dtype=bool)
    nav = a_map()
    nav.grid = grid
    nav.start = np.array([7.5, 7.5])
    nav.goal = np.array([9.0, 7.5])
    nav.geodesic = np.zeros_like(grid, dtype=float)
    sim = sim_with(nav)
    obs = sim.observe()
    v_in, w_in = np.array([0.2]), np.array([0.0])
    v, w, _ = CBFFilter().filter(obs, v_in, w_in)
    assert v[0] == pytest.approx(v_in[0], abs=1e-6)
    assert w[0] == pytest.approx(w_in[0], abs=1e-6)


def test_filter_slows_a_command_driving_into_a_wall():
    """Head-on wall at 0.6 m, nominal command full speed: the filter must brake.

    The geometry is explicit rather than borrowed from a generated map, because
    the whole point is that the wall has to be inside the scan for the filter to
    have anything to act on.
    """
    grid = np.zeros((40, 40), dtype=bool)
    grid[20, :] = True  # wall face at y = 5.0
    nav = a_map()
    nav.grid = grid
    nav.geodesic = np.zeros_like(grid, dtype=float)
    nav.start = np.array([5.0, 4.4])  # 0.6 m short of the wall
    nav.goal = np.array([5.0, 9.0])  # on the far side, so the nominal drives at it
    sim = sim_with(nav)
    sim.set_pose(theta=np.pi / 2)
    sim.last_u[:] = [[ROBOT.max_v, 0.0]]

    filt = CBFFilter()
    commanded = []
    for _ in range(6):
        v, w, _ = filt.filter(sim.observe(), np.array([ROBOT.max_v]), np.array([0.0]))
        commanded.append(float(v[0]))
        sim.step(v, w)

    assert commanded[0] < ROBOT.max_v, "filter passed a full-speed command into a wall"
    assert min(commanded) < 0.5 * ROBOT.max_v
    assert sim.outcome[0] != OUTCOME_COLLISION


@pytest.mark.parametrize("index", [0, 3, 7])
def test_filter_prevents_collision_for_a_suicidal_planner(index):
    nav = a_map(index=index)
    records = run_suite(FullSpeedAhead(), [nav], safety_filter=CBFFilter(), batch=1)
    assert records[0]["outcome"] != "collision"
    assert records[0]["min_clearance"] > ROBOT.radius


def test_unfiltered_suicidal_planner_does_collide():
    """Control for the test above: without the filter the same planner crashes."""
    nav = a_map()
    records = run_suite(FullSpeedAhead(), [nav], batch=1)
    assert records[0]["outcome"] == "collision"


def test_filter_never_returns_a_command_outside_the_envelope():
    nav = a_map(fill=0.30, iters=4)
    sim = sim_with(nav)
    filt = CBFFilter()
    rng = np.random.default_rng(0)
    for _ in range(80):
        obs = sim.observe()
        v_in = rng.uniform(-1.0, 1.0, size=1) * ROBOT.max_v
        w_in = rng.uniform(-1.0, 1.0, size=1) * ROBOT.max_omega
        v, w, stats = filt.filter(obs, v_in, w_in)
        dv = abs(v[0] - sim.last_u[0, 0])
        dw = abs(w[0] - sim.last_u[0, 1])
        assert dv <= ROBOT.max_accel_v * sim.spec.dt + 1e-6
        assert dw <= ROBOT.max_accel_omega * sim.spec.dt + 1e-6
        vl, vr = ROBOT.twist_to_wheels(v, w)
        assert max(abs(vl[0]), abs(vr[0])) <= ROBOT.max_wheel_speed + 1e-6
        if sim.outcome[0] != 0:
            break
        sim.step(v, w)


def test_safe_radius_accounts_for_the_look_ahead_offset():
    """The barrier is on a point ahead of the axle, so r_s must absorb the offset."""
    f = CBFFilter(look_ahead=0.15, margin=0.05)
    assert f.safe_radius == pytest.approx(ROBOT.radius + 0.05 + 0.15)
    # Holding the look-ahead point r_s from an obstacle leaves the body radius+margin.
    assert f.safe_radius - f.look_ahead == pytest.approx(ROBOT.radius + f.margin)
