"""Render RESULTS.md from the committed JSON.

Written rather than typed, so the tables cannot drift from the numbers the
harness actually produced.  Prose commentary lives in the module docstrings and
in ``docs/``; this file only formats measurements.
"""

from __future__ import annotations

import json
from pathlib import Path

HEADLINE = [
    ("arm", "arm", "{}"),
    ("success", "success", "{:.3f}"),
    ("collision", "collision", "{:.3f}"),
    ("timeout", "timeout", "{:.3f}"),
    ("freeze", "froze", "{:.3f}"),
    ("spl", "SPL", "{:.3f}"),
    ("barn", "BARN", "{:.3f}"),
    ("t_success", "t_goal (s)", "{:.1f}"),
    ("clear_p05", "clear p05 (m)", "{:.3f}"),
    ("intervened", "filter acts", "{:.3f}"),
    ("relaxed", "relaxed", "{:.3f}"),
    ("qp_fallback", "braked", "{:.3f}"),
]


def _cell(value, fmt: str) -> str:
    if value is None:
        return "--"
    if isinstance(value, float) and value != value:  # NaN
        return "--"
    return fmt.format(value)


def _table(rows: list[dict], columns) -> str:
    head = "| " + " | ".join(label for _, label, _ in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_cell(r.get(key), fmt) for key, _, fmt in columns) + " |" for r in rows
    ]
    return "\n".join([head, rule, *body])


def render(results_dir: Path) -> str:
    summary = json.loads((results_dir / "summary.json").read_text())
    tuning_path = results_dir / "tuning.json"
    ablation_path = results_dir / "ablation_look_ahead.json"

    out = [
        "# Results",
        "",
        f"All numbers below are on the **held-out split: {summary['n_maps']} maps** that no arm",
        "was trained or tuned on, generated to the BARN specification (see",
        "`docs/benchmark.md`). Every map is verified solvable in configuration space",
        "before it enters the suite, so every failure here is the planner's.",
        "",
        "Regenerate with `bahn eval && bahn report`.",
        "",
        "## Headline",
        "",
        _table(summary["table"], HEADLINE),
        "",
        "`froze` counts episodes where the robot moved under 0.15 m in 100 consecutive",
        "control steps; it is a subset of `timeout` and separates 'ran out of clock while",
        "making progress' from 'stopped and never recovered'. `filter acts` is the",
        "fraction of control steps on which the filter changed the command, `relaxed` the",
        "fraction where the barrier set was empty and a slack was needed, `braked` the",
        "fraction where even that failed. `clear p05` is the 5th percentile of per-episode",
        "minimum clearance, which separates 'did not collide' from 'was nowhere near",
        "colliding'. The BARN metric caps at 0.5 by construction.",
        "",
    ]

    key = summary["difficulty_key"].replace("_", " ")
    out += [
        f"## Success against map difficulty ({key})",
        "",
        "The plot a single-map evaluation cannot produce. Bins hold equal numbers of maps;",
        "lower mean clearance is harder.",
        "",
    ]
    curves = summary["curves"]
    bins = max((len(c) for c in curves.values()), default=0)
    if bins:
        any_curve = next(c for c in curves.values() if len(c) == bins)
        header = ["arm"] + [f"{c['lo']:.2f}-{c['hi']:.2f} m (n={c['n']})" for c in any_curve]
        out.append("| " + " | ".join(header) + " |")
        out.append("| " + " | ".join("---" for _ in header) + " |")
        for arm, curve in curves.items():
            cells = [f"{c['success']:.2f}" for c in curve]
            out.append("| " + " | ".join([arm, *cells]) + " |")
        out.append("")

    if ablation_path.exists():
        ab = json.loads(ablation_path.read_text())
        out += [
            "## Ablation: safety-filter look-ahead distance",
            "",
            "The barrier is enforced on a point `l` ahead of the wheel axis, so the safe",
            "radius must be `robot radius + margin + l` for the body to be covered: large",
            "`l` is conservative. But the turn-rate column of the actuation matrix scales",
            "with `l`, so small `l` leaves the filter unable to steer and only able to brake.",
            f"Nominal planner is DWA, {ab['n_maps']} held-out maps.",
            "",
            _table(
                ab["rows"],
                [
                    ("look_ahead", "l (m)", "{:.2f}"),
                    ("safe_radius", "r_s (m)", "{:.2f}"),
                    ("success", "success", "{:.3f}"),
                    ("collision", "collision", "{:.3f}"),
                    ("freeze", "froze", "{:.3f}"),
                    ("spl", "SPL", "{:.3f}"),
                    ("intervened", "filter acts", "{:.3f}"),
                    ("relaxed", "relaxed", "{:.3f}"),
                    ("qp_fallback", "braked", "{:.3f}"),
                ],
            ),
            "",
            "This sweep is a post-hoc analysis on held-out maps; the deployed value was",
            "selected by grid search on the training split before it was run.",
            "",
        ]

    if tuning_path.exists():
        t = json.loads(tuning_path.read_text())
        out += [
            "## Tuning record",
            "",
            f"Grid-searched on {t['n_tuning_maps']} training maps. Comparing a tuned method",
            "against an untuned one measures the tuning, so both hand-designed components",
            "get the same treatment and the winners were frozen before the held-out split",
            "was touched.",
            "",
            "```json",
            json.dumps(
                {k: v for k, v in t.items() if k != "n_tuning_maps"},
                indent=1,
            ),
            "```",
            "",
        ]

    out += [
        "## Files",
        "",
        "| file | contents |",
        "| --- | --- |",
        "| `summary.json` | the tables above, machine-readable |",
        "| `episodes.json` | one record per arm per map: outcome, time, distance, clearance |",
        "| `maps.json` | the 300 generated maps with their five BARN difficulty metrics |",
        "| `tuning.json` | what was searched and what won |",
        "| `ablation_look_ahead.json` | the look-ahead sweep |",
        "",
    ]
    return "\n".join(out)
