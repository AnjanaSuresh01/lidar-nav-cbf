"""Ray casting and clearance against hand-computable geometry."""

from __future__ import annotations

import numpy as np
import pytest

from bahn.geometry import cast_rays, clearance, free_space_mask

RES = 0.25


def corridor(ny=10, nx=10) -> np.ndarray:
    """Empty interior with walls on the top and bottom rows."""
    g = np.zeros((ny, nx), dtype=bool)
    g[0, :] = True
    g[-1, :] = True
    return g


def test_ray_hits_wall_at_known_distance():
    grid = corridor()
    origin = np.array([[1.25, 1.125]])  # centre of row 4, well clear of both walls
    # Straight up: the wall's lower face is at y = 9 * RES = 2.25.
    up = cast_rays(grid, RES, origin, np.array([[np.pi / 2]]), max_range=5.0)
    assert up[0, 0] == pytest.approx(2.25 - 1.125, abs=2e-3)
    # Straight down: the wall's upper face is at y = RES.
    down = cast_rays(grid, RES, origin, np.array([[-np.pi / 2]]), max_range=5.0)
    assert down[0, 0] == pytest.approx(1.125 - 0.25, abs=2e-3)


def test_ray_out_of_map_hits_the_boundary():
    grid = np.zeros((4, 4), dtype=bool)
    origin = np.array([[0.5, 0.5]])
    d = cast_rays(grid, RES, origin, np.array([[0.0]]), max_range=5.0)
    assert d[0, 0] == pytest.approx(0.5, abs=2e-3)  # right edge at x = 4 * 0.25


def test_ray_returns_max_range_when_nothing_is_hit():
    grid = np.zeros((200, 200), dtype=bool)
    origin = np.array([[25.0, 25.0]])
    d = cast_rays(grid, RES, origin, np.array([[0.0]]), max_range=3.0)
    assert d[0, 0] == pytest.approx(3.0)


def test_refinement_beats_the_marching_step():
    """The bisection must actually tighten the coarse quantisation."""
    grid = np.zeros((20, 20), dtype=bool)
    grid[:, 12] = True
    origin = np.array([[0.13, 1.0]])  # deliberately off a step boundary
    truth = 12 * RES - 0.13
    coarse = cast_rays(grid, RES, origin, np.array([[0.0]]), 5.0, step=0.2, n_refine=0)
    fine = cast_rays(grid, RES, origin, np.array([[0.0]]), 5.0, step=0.2, n_refine=8)
    assert abs(fine[0, 0] - truth) < abs(coarse[0, 0] - truth)
    assert fine[0, 0] == pytest.approx(truth, abs=1e-3)


def test_clearance_matches_closed_form_for_one_obstacle():
    # 20x20 at 0.25 m spans [0, 5]; the query points sit far enough inside that
    # the map boundary (which also counts as occupied) is never the nearest thing.
    grid = np.zeros((20, 20), dtype=bool)
    grid[8, 8] = True  # square spanning [2.0, 2.25] in both axes

    face = np.array([[1.5, 2.125]])  # level with the square, 0.5 m to its left
    assert clearance(grid, RES, face, reach=1.0)[0] == pytest.approx(0.5)

    diag = np.array([[1.5, 1.5]])  # below-left of the corner at (2.0, 2.0)
    assert clearance(grid, RES, diag, reach=1.0)[0] == pytest.approx(np.hypot(0.5, 0.5))


def test_clearance_treats_the_map_boundary_as_an_obstacle():
    grid = np.zeros((20, 20), dtype=bool)
    pt = np.array([[0.4, 2.5]])
    assert clearance(grid, RES, pt, reach=1.0)[0] == pytest.approx(0.4)


def test_clearance_saturates_at_reach():
    grid = np.zeros((40, 40), dtype=bool)
    pt = np.array([[5.0, 5.0]])
    assert clearance(grid, RES, pt, reach=0.6)[0] == pytest.approx(0.6)


def test_free_space_mask_excludes_cells_the_robot_cannot_occupy():
    grid = corridor(ny=6, nx=6)
    mask = free_space_mask(grid, RES, radius=0.22)
    assert not mask[0].any() and not mask[-1].any()  # the walls themselves
    # Row 1 centres sit 0.125 m from the wall face, inside a 0.22 m radius.
    assert not mask[1].any()
    assert mask[3].any()


def test_batched_grid_matches_looping_over_single_grids():
    rng = np.random.default_rng(0)
    grids = rng.random((4, 12, 12)) < 0.2
    origins = rng.uniform(0.6, 2.4, size=(4, 2))
    angles = rng.uniform(-np.pi, np.pi, size=(4, 9))
    batched = cast_rays(grids, RES, origins, angles, max_range=3.0)
    for i in range(4):
        one = cast_rays(grids[i], RES, origins[i : i + 1], angles[i : i + 1], max_range=3.0)
        assert np.allclose(batched[i], one[0])
