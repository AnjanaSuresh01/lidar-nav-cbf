"""Scoring.  Every number reported in RESULTS.md is computed here.

Three families, because they answer different questions:

*   **Outcome rates** - success / collision / timeout, plus the freeze rate,
    which splits timeouts into "ran out of clock while making progress" and
    "stopped moving and never recovered".  The freeze rate is the number that
    a safety filter is most likely to quietly inflate.
*   **SPL** (Anderson et al., 2018) - success weighted by path length, the
    standard efficiency measure for goal-driven navigation.
*   **BARN score** - the ICRA BARN Challenge metric, which rewards speed rather
    than path economy and clips the ratio so that one very slow run cannot
    dominate an average.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .world import NavMap


def spl(records: list[dict], maps: list[NavMap]) -> float:
    """Mean of ``S * l / max(p, l)`` over episodes.

    ``l`` is the A* reference path length, ``p`` the distance actually driven.
    A run that fails scores zero, so SPL is bounded above by the success rate.
    """
    vals = []
    for r in records:
        ref = maps[r["map_index"]].path_length
        driven = max(r["travelled"], 1e-9)
        ok = r["outcome"] == "success"
        vals.append(ok * ref / max(driven, ref))
    return float(np.mean(vals)) if vals else float("nan")


def barn_score(records: list[dict], maps: list[NavMap], max_v: float) -> float:
    """ICRA BARN Challenge score.

    ``s = 1_success * OT / clip(AT, 2*OT, 8*OT)`` with the optimal traversal time
    ``OT`` taken as reference path length at top speed.  The clip means the best
    attainable score is 0.5 (a robot cannot beat twice the optimal time on this
    definition), which is a property of the metric, not of the planners.
    """
    vals = []
    for r in records:
        ot = maps[r["map_index"]].optimal_time(max_v)
        at = r["time"]
        ok = r["outcome"] == "success"
        vals.append(ok * ot / np.clip(at, 2 * ot, 8 * ot))
    return float(np.mean(vals)) if vals else float("nan")


@dataclass
class Summary:
    arm: str
    n: int
    success: float
    collision: float
    timeout: float
    freeze: float
    spl: float
    barn: float
    mean_time_success: float
    min_clearance_p05: float
    infeasible_rate: float = float("nan")  # QP fallbacks, safety-filtered arms only

    def as_row(self) -> dict:
        return {
            "arm": self.arm,
            "n": self.n,
            "success": self.success,
            "collision": self.collision,
            "timeout": self.timeout,
            "freeze": self.freeze,
            "spl": self.spl,
            "barn": self.barn,
            "t_success": self.mean_time_success,
            "clear_p05": self.min_clearance_p05,
            "qp_fallback": self.infeasible_rate,
        }


def summarise(arm: str, records: list[dict], maps: list[NavMap], max_v: float) -> Summary:
    n = len(records)
    outcomes = [r["outcome"] for r in records]
    succ_times = [r["time"] for r in records if r["outcome"] == "success"]
    fallbacks = [r.get("qp_fallback_frac", np.nan) for r in records]
    return Summary(
        arm=arm,
        n=n,
        success=outcomes.count("success") / n,
        collision=outcomes.count("collision") / n,
        timeout=outcomes.count("timeout") / n,
        freeze=sum(r["froze"] for r in records) / n,
        spl=spl(records, maps),
        barn=barn_score(records, maps, max_v),
        mean_time_success=float(np.mean(succ_times)) if succ_times else float("nan"),
        min_clearance_p05=float(np.percentile([r["min_clearance"] for r in records], 5)),
        infeasible_rate=float(np.nanmean(fallbacks))
        if np.any(np.isfinite(fallbacks))
        else float("nan"),
    )


def by_difficulty(
    records: list[dict], maps: list[NavMap], key: str = "closest_obstacle", bins: int = 4
) -> list[dict]:
    """Success rate against a BARN difficulty metric, in equal-count bins.

    This is the plot the poster cannot produce, because it evaluates on one map.
    """
    vals = np.array([maps[r["map_index"]].difficulty[key] for r in records])
    edges = np.quantile(vals, np.linspace(0, 1, bins + 1))
    edges[-1] += 1e-9
    out = []
    for i in range(bins):
        sel = (vals >= edges[i]) & (vals < edges[i + 1])
        chunk = [r for r, s in zip(records, sel, strict=True) if s]
        if not chunk:
            continue
        out.append(
            {
                "bin": i,
                "lo": float(edges[i]),
                "hi": float(edges[i + 1]),
                "n": len(chunk),
                "success": sum(r["outcome"] == "success" for r in chunk) / len(chunk),
                "collision": sum(r["outcome"] == "collision" for r in chunk) / len(chunk),
            }
        )
    return out
