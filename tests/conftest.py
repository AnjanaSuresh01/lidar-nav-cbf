from __future__ import annotations

import pytest

from bahn.world import MapParams, generate_map


def make_map(fill: float = 0.20, iters: int = 3, index: int = 0, max_attempts: int = 200):
    """A solvable map for the given parameters.

    ``generate_map`` returns None when a draw happens to be unsolvable; the
    retry loop lives in ``barn_suite``.  Tests need the same behaviour without
    building the whole suite.
    """
    for attempt in range(max_attempts):
        nav = generate_map(MapParams(fill, iters, index, attempt))
        if nav is not None:
            return nav
    raise AssertionError(f"no solvable map for fill={fill} iters={iters} index={index}")


@pytest.fixture
def nav_map():
    return make_map()
