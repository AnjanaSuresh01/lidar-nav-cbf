"""Map generation: determinism, solvability, and the difficulty gradient."""

from __future__ import annotations

import numpy as np
import pytest

from bahn.config import ROBOT
from bahn.geometry import clearance, free_space_mask
from bahn.world import barn_suite, split_suite

from .conftest import make_map


def test_same_parameters_give_the_same_map():
    a = make_map(0.25, 3, 4)
    b = make_map(0.25, 3, 4)
    assert np.array_equal(a.grid, b.grid)
    assert a.start == pytest.approx(b.start)
    assert a.goal == pytest.approx(b.goal)
    assert np.array_equal(a.path, b.path)


def test_different_parameters_give_different_maps():
    a = make_map(0.25, 3, 4)
    b = make_map(0.25, 3, 5)
    assert not np.array_equal(a.grid, b.grid)


@pytest.mark.parametrize("fill,iters", [(0.15, 2), (0.20, 3), (0.30, 4)])
def test_generated_maps_are_actually_solvable(fill, iters):
    """Every failure the benchmark reports must be the planner's, not the map's."""
    nav = make_map(fill, iters, 0)
    # The whole reference path must admit the robot disc.
    clear = clearance(nav.grid, nav.res, nav.path, reach=ROBOT.radius + 2 * nav.res)
    assert clear.min() > ROBOT.radius
    assert nav.path[0] == pytest.approx(nav.start)
    assert nav.path[-1] == pytest.approx(nav.goal)


def test_path_is_connected_in_steps_of_one_cell():
    nav = make_map(0.25, 3, 1)
    hops = np.linalg.norm(np.diff(nav.path, axis=0), axis=1)
    assert hops.max() <= nav.res * np.sqrt(2) + 1e-9


def test_geodesic_field_is_finite_everywhere():
    """A sentinel value anywhere the robot can stand poisons the reward signal.

    The robot's disc is often collision-free in a cell whose centre is not
    C-space free; if those cells kept an infinite distance-to-goal, the progress
    term would jump by the sentinel every time the robot hugged an obstacle.
    """
    nav = make_map(0.30, 3, 0)
    assert np.isfinite(nav.geodesic).all()


def test_geodesic_field_decreases_along_the_reference_path():
    """Following A* must monotonically close geodesic distance to the goal.

    The field is *deliberately* discontinuous across obstacles -- that is what
    makes it geodesic rather than Euclidean -- so there is no grid-wide Lipschitz
    property to assert.  What must hold is that it decreases along a path the
    robot can actually drive.
    """
    nav = make_map(0.25, 3, 0)
    cells = (nav.path / nav.res).astype(int)
    along = nav.geodesic[cells[:, 1], cells[:, 0]]
    assert np.all(np.diff(along) < 1e-9)
    assert along[-1] == pytest.approx(0.0, abs=1e-9)


def test_geodesic_field_agrees_with_the_reference_path_length():
    nav = make_map(0.20, 3, 2)
    row = int(nav.start[1] / nav.res)
    col = int(nav.start[0] / nav.res)
    assert nav.geodesic[row, col] == pytest.approx(nav.path_length, rel=1e-6)


def test_corridor_walls_are_present():
    nav = make_map(0.15, 2, 0)
    assert nav.grid[0].all()
    assert nav.grid[-1].all()


def test_start_and_goal_are_collision_free():
    nav = make_map(0.30, 4, 3)
    mask = free_space_mask(nav.grid, nav.res, ROBOT.radius)
    for p in (nav.start, nav.goal):
        assert mask[int(p[1] / nav.res), int(p[0] / nav.res)]


def test_difficulty_increases_with_fill_percentage():
    """The suite has to span a difficulty range or the benchmark says nothing."""
    easy = [make_map(0.15, 3, i) for i in range(6)]
    hard = [make_map(0.30, 3, i) for i in range(6)]
    mean = lambda ms, k: np.mean([m.difficulty[k] for m in ms])  # noqa: E731
    assert mean(hard, "closest_obstacle") < mean(easy, "closest_obstacle")
    assert mean(hard, "avg_visibility") < mean(easy, "avg_visibility")
    assert mean(hard, "dispersion") > mean(easy, "dispersion")
    assert mean(hard, "characteristic_dimension") < mean(easy, "characteristic_dimension")


def test_suite_size_and_split_are_disjoint():
    maps = list(barn_suite(repetitions=3))
    assert len(maps) == 12 * 3
    train, test = split_suite(maps, n_test=1)
    assert not ({m.name for m in train} & {m.name for m in test})
    # Splitting by repetition index keeps every parameter combination in both.
    assert {(m.params.fill_pct, m.params.smooth_iters) for m in train} == {
        (m.params.fill_pct, m.params.smooth_iters) for m in test
    }
