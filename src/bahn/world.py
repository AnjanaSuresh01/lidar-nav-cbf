"""BARN-specification map generation, reference paths and difficulty metrics.

The generator follows the recipe in *Benchmarking Metric Ground Navigation*
(Perille, Truong, Xiao, Stone; arXiv:2008.13315), which is the map source used
by the BARN Challenge at ICRA:

    30 x 30 grid, initial fill in {0.15, 0.20, 0.25, 0.30}, smoothing iterations
    in {2, 3, 4}, fill threshold 5, clear threshold 1, 8-connected neighbourhood,
    start and goal drawn from free cells on opposite edges, flood fill to discard
    unsolvable layouts, A* for the reference path.  12 parameter combinations
    x 25 repetitions = 300 environments.

Two deliberate departures, both because we simulate in 2-D rather than Gazebo:

*   Cell size is 0.25 m here, so a map spans 7.5 x 7.5 m.  With a 0.22 m robot
    radius a one-cell gap is impassable and a two-cell gap has 6 cm of clearance
    either side, which puts the interesting part of the difficulty range inside
    the dataset rather than off the end of it.
*   The top and bottom rows are forced occupied, forming the corridor walls that
    the Gazebo BARN worlds have.  Without them a planner can drive around the
    obstacle field instead of through it and every map scores the same.
*   Three free columns are appended at each end as approach lanes.  The paper
    draws start and goal from free cells on the left and right edges, but with a
    walled border no edge cell is ever collision-free for a 0.22 m disc; the
    lanes give the robot somewhere legal to stand while keeping the obstacle
    field itself exactly 30 x 30.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .config import ROBOT
from .geometry import cast_rays, clearance, free_space_mask

GRID_N = 30
RESOLUTION = 0.25
PAD_COLS = 3  # free approach lanes at each end
LANE_COL = 1  # start/goal sit in this column of the lane, inset from the wall
FILL_PCTS = (0.15, 0.20, 0.25, 0.30)
SMOOTH_ITERS = (2, 3, 4)
FILL_THRESHOLD = 5
CLEAR_THRESHOLD = 1
REPETITIONS = 25


@dataclass(frozen=True)
class MapParams:
    fill_pct: float
    smooth_iters: int
    index: int
    attempt: int = 0

    @property
    def seed(self) -> int:
        """Deterministic seed, so the 300-map suite is byte-identical anywhere."""
        key = (
            int(round(self.fill_pct * 100)) * 1_000_000
            + self.smooth_iters * 100_000
            + self.index * 100
            + self.attempt
        )
        return key

    @property
    def name(self) -> str:
        return f"f{int(round(self.fill_pct * 100)):02d}_s{self.smooth_iters}_{self.index:02d}"


@dataclass
class NavMap:
    """One benchmark environment."""

    params: MapParams
    grid: np.ndarray  # (ny, nx) bool, True = occupied
    res: float
    start: np.ndarray  # (2,) world metres
    goal: np.ndarray  # (2,)
    start_theta: float
    path: np.ndarray  # (k, 2) A* reference path in world metres
    geodesic: np.ndarray  # (ny, nx) collision-free distance-to-goal, metres
    difficulty: dict[str, float] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.params.name

    @property
    def extent(self) -> tuple[float, float]:
        ny, nx = self.grid.shape
        return nx * self.res, ny * self.res

    @property
    def path_length(self) -> float:
        """Length of the A* reference path: the shortest-path term in SPL."""
        return float(np.linalg.norm(np.diff(self.path, axis=0), axis=1).sum())

    def optimal_time(self, max_v: float) -> float:
        """BARN's optimal traversal time: reference path length at top speed."""
        return self.path_length / max_v


def _smooth(grid: np.ndarray) -> np.ndarray:
    """One cellular-automaton pass; cells outside the grid count as free."""
    padded = np.zeros((grid.shape[0] + 2, grid.shape[1] + 2), dtype=np.int32)
    padded[1:-1, 1:-1] = grid
    counts = np.zeros_like(grid, dtype=np.int32)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            counts += padded[1 + dr : 1 + dr + grid.shape[0], 1 + dc : 1 + dc + grid.shape[1]]
    out = grid.copy()
    out[counts >= FILL_THRESHOLD] = True
    out[counts <= CLEAR_THRESHOLD] = False
    return out


def _raw_grid(params: MapParams, n: int = GRID_N) -> np.ndarray:
    rng = np.random.default_rng(params.seed)
    grid = rng.random((n, n)) < params.fill_pct
    for _ in range(params.smooth_iters):
        grid = _smooth(grid)
    lane = np.zeros((n, PAD_COLS), dtype=bool)
    grid = np.hstack([lane, grid, lane])
    grid[0, :] = True  # corridor walls
    grid[-1, :] = True
    return grid


def _astar(cfree: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list | None:
    """8-connected A* over C-space free cells; Euclidean cost and heuristic."""
    ny, nx = cfree.shape
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def h(node: tuple[int, int]) -> float:
        return float(np.hypot(node[0] - goal[0], node[1] - goal[1]))

    open_heap = [(h(start), 0.0, start)]
    came: dict[tuple[int, int], tuple[int, int]] = {}
    best = {start: 0.0}
    while open_heap:
        _, g, node = heapq.heappop(open_heap)
        if node == goal:
            path = [node]
            while node in came:
                node = came[node]
                path.append(node)
            return path[::-1]
        if g > best.get(node, np.inf):
            continue
        for dr, dc in moves:
            nxt = (node[0] + dr, node[1] + dc)
            if not (0 <= nxt[0] < ny and 0 <= nxt[1] < nx) or not cfree[nxt]:
                continue
            # Do not cut a diagonal between two occupied cells.
            if dr and dc and not (cfree[node[0] + dr, node[1]] or cfree[node[0], node[1] + dc]):
                continue
            step = float(np.hypot(dr, dc))
            cand = g + step
            if cand < best.get(nxt, np.inf) - 1e-12:
                best[nxt] = cand
                came[nxt] = node
                heapq.heappush(open_heap, (cand + h(nxt), cand, nxt))
    return None


def geodesic_field(cfree: np.ndarray, goal_cell: tuple[int, int], res: float) -> np.ndarray:
    """Shortest collision-free distance from every free cell to the goal, in metres.

    Used *only* to shape the RL reward during training.  Rewarding straight-line
    progress instead makes a concave obstacle look like a reward wall, and the
    policy learns to press into it; rewarding geodesic progress removes that
    artefact without giving the policy any extra information at run time, since
    the observation stays local (LiDAR plus goal bearing).

    Dijkstra runs over configuration-space free cells, but the field is then
    **extended to every cell** by nearest-neighbour lookup plus the Euclidean gap.
    That extension is not cosmetic.  A robot whose disc is collision-free
    routinely sits in a cell whose *centre* is not C-space free -- that is what
    hugging an obstacle looks like on a grid -- so leaving those cells at
    infinity makes the progress term jump by the sentinel value whenever the
    robot passes close to anything.  Measured before this was fixed: 3.1 per cent
    of lookups hit the sentinel, giving a reward with standard deviation 615 and
    spikes to +/- 3000, and PPO correctly learned that the safest thing to do
    with such a signal is to stand still.
    """
    dist = np.full(cfree.shape, np.inf)
    dist[goal_cell] = 0.0
    heap = [(0.0, goal_cell)]
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    ny, nx = cfree.shape
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist[node] + 1e-12:
            continue
        for dr, dc in moves:
            nxt = (node[0] + dr, node[1] + dc)
            if not (0 <= nxt[0] < ny and 0 <= nxt[1] < nx) or not cfree[nxt]:
                continue
            if dr and dc and not (cfree[node[0] + dr, node[1]] or cfree[node[0], node[1] + dc]):
                continue
            cand = d + float(np.hypot(dr, dc)) * res
            if cand < dist[nxt] - 1e-12:
                dist[nxt] = cand
                heapq.heappush(heap, (cand, nxt))

    reachable = np.isfinite(dist)
    if not reachable.all():
        gap, idx = ndimage.distance_transform_edt(
            ~reachable, return_distances=True, return_indices=True
        )
        dist = dist[idx[0], idx[1]] + gap * res
    return dist


def _cell_centres(cells: list, res: float) -> np.ndarray:
    arr = np.asarray(cells, dtype=np.float64)
    return np.stack([(arr[:, 1] + 0.5) * res, (arr[:, 0] + 0.5) * res], axis=1)


def difficulty_metrics(grid: np.ndarray, res: float, path: np.ndarray) -> dict[str, float]:
    """The five BARN difficulty metrics, averaged over the reference path.

    Definitions follow the paper.  Two need an interpretation the paper leaves
    implicit, noted here so the numbers are reproducible rather than merely
    plausible:

    *   ``dispersion`` counts occupied/free alternations around a 16-ray scan.
        We evaluate occupancy at a fixed 1.0 m radius, so it measures how many
        distinct obstacle groups surround the robot.
    *   ``characteristic_dimension`` is the smallest width through the cell: for
        each of 8 axes at 22.5 degrees we sum the two opposing ray distances and
        take the minimum over axes.
    """
    n = len(path)
    scan_range = 5.0

    near = clearance(grid, res, path, reach=scan_range)

    ang8 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    vis = cast_rays(grid, res, path, np.tile(ang8, (n, 1)), max_range=scan_range)

    ang16 = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    ring = path[:, None, :] + 1.0 * np.stack([np.cos(ang16), np.sin(ang16)], axis=-1)[None]
    ring_occ = clearance(grid, res, ring, reach=res) <= 0.0
    alternations = (ring_occ != np.roll(ring_occ, 1, axis=1)).sum(axis=1)

    ang16_axis = np.linspace(0, np.pi, 8, endpoint=False)
    both = np.concatenate([ang16_axis, ang16_axis + np.pi])
    axis_rays = cast_rays(grid, res, path, np.tile(both, (n, 1)), max_range=scan_range)
    widths = axis_rays[:, :8] + axis_rays[:, 8:]

    straight = float(np.linalg.norm(path[-1] - path[0]))
    arc = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())

    return {
        "closest_obstacle": float(near.mean()),
        "avg_visibility": float(vis.mean()),
        "dispersion": float(alternations.mean()),
        "characteristic_dimension": float(widths.min(axis=1).mean()),
        "tortuosity": arc / straight if straight > 0 else float("nan"),
    }


def generate_map(params: MapParams, res: float = RESOLUTION) -> NavMap | None:
    """Build one environment, or return None if it is unsolvable.

    Solvability is checked in configuration space: obstacles are inflated by the
    robot radius and a flood fill must connect a free cell on the left edge to
    one on the right edge.  This guarantees a collision-free path exists, so
    every failure the benchmark reports is the planner's, not the map's.
    """
    grid = _raw_grid(params)
    cfree = free_space_mask(grid, res, ROBOT.radius)

    start_col = LANE_COL
    goal_col = cfree.shape[1] - 1 - LANE_COL
    starts = np.flatnonzero(cfree[:, start_col])
    goals = np.flatnonzero(cfree[:, goal_col])
    if starts.size == 0 or goals.size == 0:
        return None

    rng = np.random.default_rng(params.seed + 7)
    start_cell = (int(rng.choice(starts)), start_col)
    goal_cell = (int(rng.choice(goals)), goal_col)

    cells = _astar(cfree, start_cell, goal_cell)
    if cells is None:
        return None

    path = _cell_centres(cells, res)
    nav = NavMap(
        params=params,
        grid=grid,
        res=res,
        start=path[0].copy(),
        goal=path[-1].copy(),
        start_theta=0.0,  # facing into the corridor; the path direction is not leaked
        path=path,
        geodesic=geodesic_field(cfree, goal_cell, res),
    )
    nav.difficulty = difficulty_metrics(grid, res, path)
    return nav


def barn_suite(
    repetitions: int = REPETITIONS,
    max_attempts: int = 200,
    res: float = RESOLUTION,
) -> Iterator[NavMap]:
    """Yield the 300-map suite: 12 parameter combinations x 25 repetitions.

    Unsolvable draws are discarded and the seed advanced, exactly as the paper
    describes, so the suite is a deterministic function of the parameters alone.
    """
    for fill in FILL_PCTS:
        for iters in SMOOTH_ITERS:
            for index in range(repetitions):
                for attempt in range(max_attempts):
                    nav = generate_map(MapParams(fill, iters, index, attempt), res=res)
                    if nav is not None:
                        yield nav
                        break
                else:
                    raise RuntimeError(
                        f"no solvable map for fill={fill} iters={iters} index={index} "
                        f"after {max_attempts} attempts"
                    )


def split_suite(maps: list[NavMap], n_test: int = 10) -> tuple[list[NavMap], list[NavMap]]:
    """Split by repetition index so train and test share no map.

    The last ``n_test`` repetition indices of every parameter combination are
    held out, which keeps the difficulty distribution of the two splits matched
    while guaranteeing no test map was ever seen during training.
    """
    cutoff = max(m.params.index for m in maps) + 1 - n_test
    train = [m for m in maps if m.params.index < cutoff]
    test = [m for m in maps if m.params.index >= cutoff]
    return train, test
