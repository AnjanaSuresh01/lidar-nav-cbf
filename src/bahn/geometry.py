"""Vectorised grid geometry: LiDAR ray casting and exact disc clearance.

Both functions are written to operate on a batch of robots at once, because the
RL rollout runs many environments in lockstep and ray casting is the inner loop
of the whole project.

Grid convention throughout: cell ``(row, col)`` covers world coordinates
``[col*res, (col+1)*res) x [row*res, (row+1)*res)``.  Anything outside the grid
counts as occupied, so the map boundary is a wall.

``grid`` may be a single ``(ny, nx)`` map or a stack of ``(n, ny, nx)`` maps, one
per robot.  The stacked form is what the RL rollout uses: every environment in
the batch sits in a different map and they all step together.
"""

from __future__ import annotations

import numpy as np


def _lookup(grid: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Occupancy at integer cell coordinates; out of bounds reads as occupied."""
    ny, nx = grid.shape[-2:]
    inside = (rows >= 0) & (rows < ny) & (cols >= 0) & (cols < nx)
    r = np.clip(rows, 0, ny - 1)
    c = np.clip(cols, 0, nx - 1)
    if grid.ndim == 2:
        occ = grid[r, c]
    else:
        n = grid.shape[0]
        if rows.shape[0] != n:
            raise ValueError(f"batched grid of {n} maps needs leading axis {n}, got {rows.shape}")
        idx = np.arange(n).reshape((n,) + (1,) * (rows.ndim - 1))
        occ = grid[idx, r, c]
    return np.where(inside, occ, True)


def cast_rays(
    grid: np.ndarray,
    res: float,
    origins: np.ndarray,
    angles: np.ndarray,
    max_range: float,
    step: float = 0.05,
    n_refine: int = 5,
) -> np.ndarray:
    """March rays through an occupancy grid and return hit distances.

    Args:
        grid: ``(ny, nx)`` boolean array, True where occupied.
        res: metres per cell.
        origins: ``(n, 2)`` ray origins in world coordinates.
        angles: ``(n, b)`` absolute ray angles in the world frame.
        max_range: beams that hit nothing return exactly this.
        step: coarse marching step in metres.
        n_refine: bisection steps applied to the coarse hit, each halving the
            residual quantisation error.

    Returns:
        ``(n, b)`` distances in metres, clipped to ``max_range``.

    The coarse march finds the first *sample* inside an obstacle, so it can
    overshoot the true surface by up to ``step``.  The bisection then brackets
    the surface to ``step / 2**n_refine`` (0.05 m -> 1.6 mm at the defaults),
    which matters because the safety filter differentiates these readings.
    """
    origins = np.asarray(origins, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)
    if origins.ndim != 2 or origins.shape[1] != 2:
        raise ValueError(f"origins must be (n, 2), got {origins.shape}")
    if angles.shape[0] != origins.shape[0]:
        raise ValueError(f"angles {angles.shape} does not batch with origins {origins.shape}")

    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=-1)  # (n, b, 2)

    n_steps = int(np.ceil(max_range / step)) + 1
    ts = np.arange(n_steps, dtype=np.float64) * step  # (k,)
    pts = origins[:, None, None, :] + ts[None, None, :, None] * dirs[:, :, None, :]
    occ = _occupied(grid, res, pts)  # (n, b, k)

    # First occupied sample along each beam; beams that never hit get n_steps.
    any_hit = occ.any(axis=-1)
    first = np.where(any_hit, occ.argmax(axis=-1), n_steps)

    lo = np.maximum(first - 1, 0) * step  # last known-free sample
    hi = np.minimum(first, n_steps - 1) * step  # first known-occupied sample
    for _ in range(n_refine):
        mid = 0.5 * (lo + hi)
        mid_pts = origins[:, None, :] + mid[..., None] * dirs
        mid_occ = _occupied(grid, res, mid_pts)
        hi = np.where(mid_occ, mid, hi)
        lo = np.where(mid_occ, lo, mid)
    dist = 0.5 * (lo + hi)

    return np.where(any_hit, np.minimum(dist, max_range), max_range)


def _occupied(grid: np.ndarray, res: float, pts: np.ndarray) -> np.ndarray:
    """Occupancy lookup for world points; out of bounds counts as occupied."""
    col = np.floor(pts[..., 0] / res).astype(np.int64)
    row = np.floor(pts[..., 1] / res).astype(np.int64)
    return _lookup(grid, row, col)


def clearance(
    grid: np.ndarray,
    res: float,
    points: np.ndarray,
    reach: float,
) -> np.ndarray:
    """Exact distance from each point to the nearest obstacle surface.

    Obstacles are axis-aligned squares, so the point-to-obstacle distance is
    computed in closed form rather than read off a distance transform: a
    transform measures centre-to-centre distance and is off by up to half a cell
    (0.125 m at our resolution), which is comparable to the clearances that
    decide whether a run collides.

    Only cells within ``reach`` metres are examined, so distances above
    ``reach`` saturate.  Callers that only need "is this a collision" should
    pass ``reach`` a little above the robot radius.

    Args:
        points: ``(..., 2)`` world coordinates.

    Returns:
        Array of shape ``points.shape[:-1]``; ``reach`` where nothing is near.
    """
    pts = np.asarray(points, dtype=np.float64)
    k = int(np.ceil(reach / res))
    offsets = np.arange(-k, k + 1)
    dr, dc = np.meshgrid(offsets, offsets, indexing="ij")
    dr, dc = dr.ravel(), dc.ravel()  # (m,)

    col0 = np.floor(pts[..., 0] / res).astype(np.int64)
    row0 = np.floor(pts[..., 1] / res).astype(np.int64)
    rows = row0[..., None] + dr  # (..., m)
    cols = col0[..., None] + dc

    occ = _lookup(grid, rows, cols)

    # Distance from the point to the axis-aligned square of each candidate cell.
    x0 = cols * res
    y0 = rows * res
    px = pts[..., 0][..., None]
    py = pts[..., 1][..., None]
    ddx = np.maximum.reduce([x0 - px, np.zeros_like(x0, dtype=np.float64), px - (x0 + res)])
    ddy = np.maximum.reduce([y0 - py, np.zeros_like(y0, dtype=np.float64), py - (y0 + res)])
    dist = np.hypot(ddx, ddy)

    dist = np.where(occ, dist, np.inf)
    return np.minimum(dist.min(axis=-1), reach)


def free_space_mask(grid: np.ndarray, res: float, radius: float) -> np.ndarray:
    """Configuration-space free cells for a disc robot of the given radius.

    A cell is C-space free when a disc centred on the cell centre does not touch
    an obstacle.  BARN performs the same inflation before checking whether an
    environment is solvable at all.
    """
    ny, nx = grid.shape
    rows, cols = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    centres = np.stack([(cols + 0.5) * res, (rows + 0.5) * res], axis=-1)
    reach = radius + res
    return (~grid) & (clearance(grid, res, centres, reach) > radius)
