"""Grid search for the hand-set constants, on the training split only.

A comparison between a tuned method and an untuned one measures the tuning, not
the method.  Both hand-designed components here -- the poster's reactive gains
and the safety filter's geometry -- get the same treatment: a grid search on
training maps, with the winner frozen before the held-out split is ever touched.

The search objective is success rate.  Collisions are already failures under it,
so no separate penalty term is needed -- but the objective is only well posed if
the thing being tuned can lose success in *both* directions.  See
:func:`tune_cbf` for where that bites.
"""

from __future__ import annotations

import itertools
import time

from .planners.reactive import ReactiveANN
from .rollout import run_suite
from .safety.cbf import CBFFilter
from .world import NavMap


def _success(records: list[dict]) -> float:
    return sum(r["outcome"] == "success" for r in records) / max(len(records), 1)


def _search(label, grid: dict, make, maps, batch: int, filter_factory=None) -> tuple[dict, float]:
    keys = list(grid)
    best_cfg, best_score = {}, -1.0
    combos = list(itertools.product(*(grid[k] for k in keys)))
    print(f"[{label}] {len(combos)} configurations over {len(maps)} training maps", flush=True)
    for i, combo in enumerate(combos):
        cfg = dict(zip(keys, combo, strict=True))
        planner, filt = make(cfg), (filter_factory(cfg) if filter_factory else None)
        t = time.time()
        score = _success(run_suite(planner, maps, safety_filter=filt, batch=batch))
        flag = ""
        if score > best_score:
            best_cfg, best_score, flag = cfg, score, "  <-- best"
        print(
            f"  [{i + 1:>3}/{len(combos)}] {cfg} success={score:.3f} "
            f"({time.time() - t:.0f}s){flag}",
            flush=True,
        )
    return best_cfg, best_score


def tune_reactive(maps: list[NavMap], batch: int = 30) -> tuple[dict, float]:
    """Grid-search the reactive gains over roughly two decades each.

    The avoidance and braking terms are sums over 36 beams divided by the beam
    count, so their natural scale is not obvious from the wiring and the gain has
    to absorb it.  A first sweep spanning 0.03-0.4 for ``k_avoid`` produced a
    flat 0.117-0.133 across all 144 combinations.  That reads as "the
    architecture is insensitive to its gains", and it would have been a tidy
    finding -- but spot-checking a single value outside the grid beat every point
    inside it, and widening the range took the same controller to 0.450.  **A
    flat tuning surface is evidence about the grid before it is evidence about
    the method.**

    Ranges here are log-spaced over roughly two decades and deliberately run past
    the point where behaviour degrades again, so the winner is interior rather
    than sitting on an edge.
    """
    grid = {
        "k_drive": [0.1, 0.3, 0.6, 1.2],
        "k_steer": [0.7, 2.0, 5.0, 12.0],
        "k_avoid": [0.4, 1.5, 6.0, 15.0],
        "k_brake": [0.05, 0.2, 0.8],
    }
    return _search("reactive", grid, lambda c: ReactiveANN(**c), maps, batch)


def tune_cbf(maps: list[NavMap], batch: int = 30, nominal=None) -> tuple[dict, float]:
    """Tuned against a nominal planner that actually collides.

    The obvious choice is DWA, the strongest classical planner -- and it is the
    wrong one.  DWA is already collision-free on this benchmark, so the filter
    can only ever cost it success, and a search that maximises success rate walks
    straight to the parameters that switch the filter off.  That is exactly what
    the first sweep did: success rose monotonically with the class-K gain
    (0.233 -> 0.283 -> 0.383 -> 0.417 -> 0.467 at alpha = 1.5, 3, 6, 12, 24), and
    as alpha grows the barrier constraint becomes vacuous.  **Tuning a safety
    filter for success alone against a safe planner will always tell you to
    disable it.**

    Tuning against a planner that does collide makes the objective well-posed
    without needing a hand-weighted penalty: too weak a filter leaves the
    nominal's collisions in place, too strong a filter freezes the robot, and
    both show up as lost success. PPO is used when its checkpoint exists, since
    it is the strongest planner here that is still unsafe; otherwise the reactive
    controller stands in.
    """
    grid = {
        "look_ahead": [0.08, 0.12, 0.20],
        "margin": [0.02, 0.06],
        # alpha is the class-K gain: larger lets the barrier fall faster, so the
        # filter intervenes later and less. The first sweep put the optimum on
        # the grid boundary at 6.0, so the range was extended upward -- a winner
        # sitting on an edge means the search, not the method, set the value.
        "alpha": [1.5, 3.0, 6.0, 12.0, 24.0],
    }
    if nominal is None:
        nominal = _unsafe_nominal()
    return _search(
        "cbf", grid, lambda c: nominal, maps, batch, filter_factory=lambda c: CBFFilter(**c)
    )


def _unsafe_nominal():
    """PPO if it has been trained, otherwise the reactive controller."""
    from .config import work_dir

    ckpt = work_dir() / "ppo_nav.zip"
    if ckpt.exists():
        from .planners.learned import PPOPlanner

        return PPOPlanner(checkpoint=ckpt)
    return ReactiveANN()


def tune_all(
    maps: list[NavMap], n_maps: int = 60, batch: int = 30, only: str | None = None
) -> dict:
    """Run both searches on a strided sample of the training split.

    Strided, not the first N: the suite is generated in order of fill
    percentage, so a prefix would be the easiest maps only and would tune every
    constant for open corridors.
    """
    subset = maps[:: max(len(maps) // n_maps, 1)][:n_maps]
    out: dict = {"n_tuning_maps": len(subset)}
    if only in (None, "reactive"):
        cfg, score = tune_reactive(subset, batch=batch)
        out["reactive"], out["reactive_success"] = cfg, score
    if only in (None, "cbf"):
        nominal = _unsafe_nominal()
        cfg, score = tune_cbf(subset, batch=batch, nominal=nominal)
        out["cbf"] = cfg
        out["cbf_nominal"] = nominal.name
        out["cbf_success_with_nominal"] = score
    return out
