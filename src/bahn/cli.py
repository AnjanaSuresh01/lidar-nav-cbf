"""Command line entry point.

    bahn maps      report the generated benchmark suite
    bahn tune      grid-search the hand-tuned baselines on the training split
    bahn bc        behaviour-clone the poster network from a DWA teacher
    bahn ppo       train the PPO local planner
    bahn eval      score every arm on the held-out split
    bahn ablate    sweep the safety filter's look-ahead distance
    bahn report    render RESULTS.md from the committed JSON
    bahn figures   render the trajectory and diagnostic plots

Every command is deterministic given its seed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import ROBOT, work_dir
from .world import barn_suite, split_suite

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def load_maps(res_split: int = 10):
    maps = list(barn_suite())
    train, test = split_suite(maps, n_test=res_split)
    return maps, train, test


def build_arms(ckpt_bc: Path | None, ckpt_ppo: Path | None, cbf_kwargs: dict | None = None):
    """The planner x safety-filter grid that the study reports."""
    from .planners.dwa import DWA
    from .planners.reactive import ReactiveANN
    from .safety.cbf import CBFFilter

    tuning = load_tuning()
    planners = [ReactiveANN(**tuning.get("reactive", {}))]
    if ckpt_bc is not None and Path(ckpt_bc).exists():
        from .planners.learned import ClonedPosterNet

        planners.append(ClonedPosterNet(checkpoint=ckpt_bc))
        # Same architecture and same teacher, raw ranges instead of 1/d: isolates
        # what the poster's inverse encoding costs.
        raw = Path(str(ckpt_bc).replace(".pt", "_raw.pt"))
        if raw.exists():
            planners.append(ClonedPosterNet(checkpoint=raw, encoding="raw", name="reactive-bc-raw"))
    planners.append(DWA())
    if ckpt_ppo is not None and Path(ckpt_ppo).exists():
        from .planners.learned import PPOPlanner

        planners.append(PPOPlanner(checkpoint=ckpt_ppo))

    cbf_kwargs = cbf_kwargs if cbf_kwargs is not None else tuning.get("cbf", {})
    arms = []
    for p in planners:
        arms.append((p.name, p, None))
        arms.append((f"{p.name}+cbf", p, CBFFilter(**cbf_kwargs)))
    return arms


def load_tuning() -> dict:
    path = RESULTS_DIR / "tuning.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


# --------------------------------------------------------------------- maps


def cmd_maps(args):
    maps, train, test = load_maps()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "name": m.name,
            "fill": m.params.fill_pct,
            "smooth": m.params.smooth_iters,
            "index": m.params.index,
            "attempt": m.params.attempt,
            "occupancy": float(m.grid.mean()),
            "path_length": m.path_length,
            **m.difficulty,
        }
        for m in maps
    ]
    (RESULTS_DIR / "maps.json").write_text(json.dumps(rows, indent=1))
    print(f"{len(maps)} maps  ({len(train)} train / {len(test)} test)")
    keys = [
        "closest_obstacle",
        "avg_visibility",
        "dispersion",
        "characteristic_dimension",
        "tortuosity",
    ]
    print(f"{'fill/smooth':>12} {'occ%':>6} {'path':>6} " + " ".join(f"{k[:9]:>9}" for k in keys))
    for fill in sorted({m.params.fill_pct for m in maps}):
        for it in sorted({m.params.smooth_iters for m in maps}):
            sel = [m for m in maps if m.params.fill_pct == fill and m.params.smooth_iters == it]
            stats = " ".join(f"{np.mean([m.difficulty[k] for m in sel]):9.2f}" for k in keys)
            print(
                f"{fill:8.2f}/{it:<3d} {100 * np.mean([m.grid.mean() for m in sel]):6.1f} "
                f"{np.mean([m.path_length for m in sel]):6.2f} {stats}"
            )
    print(f"\nwrote {RESULTS_DIR / 'maps.json'}")


# --------------------------------------------------------------------- train


def cmd_bc(args):
    from .train import train_bc

    _, train, _ = load_maps()
    suffix = "" if args.encoding == "poster" else f"_{args.encoding}"
    train_bc(
        train,
        out_path=work_dir() / f"reactive_bc{suffix}.pt",
        steps=args.steps,
        epochs=args.epochs,
        encoding=args.encoding,
    )


def cmd_ppo(args):
    from .train import train_ppo

    _, train, _ = load_maps()
    train_ppo(train, out_path=work_dir() / "ppo_nav", total_steps=args.steps, n_envs=args.n_envs)


# ---------------------------------------------------------------------- eval


def cmd_eval(args):
    from .metrics import by_difficulty, summarise
    from .rollout import run_suite

    _, train, test = load_maps()
    maps = train if args.split == "train" else test
    arms = build_arms(work_dir() / "reactive_bc.pt", work_dir() / "ppo_nav.zip")
    if args.arm:
        arms = [a for a in arms if a[0] in args.arm]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table, raw, curves = [], {}, {}
    for name, planner, filt in arms:
        records = run_suite(planner, maps, safety_filter=filt, batch=args.batch, seed=args.seed)
        summary = summarise(name, records, maps, ROBOT.max_v)
        row = summary.as_row()
        row["relaxed"] = (
            float(np.nanmean([r.get("relaxed_frac", np.nan) for r in records]))
            if filt
            else float("nan")
        )
        row["intervened"] = (
            float(np.nanmean([r.get("intervened_frac", np.nan) for r in records]))
            if filt
            else float("nan")
        )
        table.append(row)
        raw[name] = records
        curves[name] = by_difficulty(records, maps, key=args.difficulty_key)
        print(_format_row(row), flush=True)

    out = {
        "split": args.split,
        "n_maps": len(maps),
        "seed": args.seed,
        "table": table,
        "difficulty_key": args.difficulty_key,
        "curves": curves,
    }
    tag = "" if args.split == "test" else f"_{args.split}"
    (RESULTS_DIR / f"summary{tag}.json").write_text(json.dumps(out, indent=1))
    (RESULTS_DIR / f"episodes{tag}.json").write_text(json.dumps(raw, indent=1))
    print(f"\nwrote {RESULTS_DIR / f'summary{tag}.json'}")


def _format_row(row: dict) -> str:
    return (
        f"{row['arm']:<18} succ={row['success']:.3f} coll={row['collision']:.3f} "
        f"froze={row['freeze']:.3f} spl={row['spl']:.3f} barn={row['barn']:.3f} "
        f"clr_p05={row['clear_p05']:.3f}"
    )


# -------------------------------------------------------------------- tuning


def cmd_tune(args):
    from .tuning import tune_all

    _, train, _ = load_maps()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    best = tune_all(train, n_maps=args.n_maps, batch=args.batch, only=args.only)
    # Merge, so re-running one search does not discard the other's result.
    merged = {**load_tuning(), **best}
    (RESULTS_DIR / "tuning.json").write_text(json.dumps(merged, indent=1))
    best = merged
    print(json.dumps(best, indent=1))


# ------------------------------------------------------------------ ablation


def cmd_ablate(args):
    """Sweep the safety filter's look-ahead distance.

    The look-ahead is the one parameter with a genuine two-sided cost.  The
    barrier lives on a point ``l`` ahead of the axle, so the safe radius must be
    ``radius + margin + l`` for the *body* to be covered -- a large ``l`` makes
    the filter conservative.  But the omega column of the actuation matrix scales
    with ``l``, so a small ``l`` leaves the filter almost no steering authority
    and it can only brake.  This sweep measures where that trades off.

    Run against the same unsafe nominal the filter was tuned against, so both
    sides of the trade-off are visible: too small an `l` and the filter can only
    brake, so the nominal's collisions become freezes; too large and it refuses
    room the robot does not need.  Sweeping against an already-safe planner would
    only ever show cost.

    Run on the held-out split as a post-hoc analysis; the deployed value was
    chosen on the training split, before this was run.
    """
    from .metrics import summarise
    from .rollout import run_suite
    from .safety.cbf import CBFFilter
    from .tuning import _unsafe_nominal

    _, _, test = load_maps()
    maps = test[:: max(len(test) // args.n_maps, 1)][: args.n_maps]
    nominal = _unsafe_nominal()
    rows = []
    for look_ahead in args.look_ahead:
        cfg = dict(load_tuning().get("cbf", {}))
        cfg["look_ahead"] = look_ahead
        records = run_suite(nominal, maps, safety_filter=CBFFilter(**cfg), batch=args.batch)
        row = summarise(f"{nominal.name}+cbf(l={look_ahead})", records, maps, ROBOT.max_v).as_row()
        row["look_ahead"] = look_ahead
        row["safe_radius"] = CBFFilter(**cfg).safe_radius
        row["relaxed"] = float(np.nanmean([r.get("relaxed_frac", np.nan) for r in records]))
        row["intervened"] = float(np.nanmean([r.get("intervened_frac", np.nan) for r in records]))
        rows.append(row)
        print(
            f"l={look_ahead:.2f} r_s={row['safe_radius']:.2f} succ={row['success']:.3f} "
            f"coll={row['collision']:.3f} froze={row['freeze']:.3f} "
            f"relaxed={row['relaxed']:.3f} intervened={row['intervened']:.3f}",
            flush=True,
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "ablation_look_ahead.json").write_text(
        json.dumps({"n_maps": len(maps), "rows": rows}, indent=1)
    )


# -------------------------------------------------------------------- report


def cmd_report(args):
    from .report import render

    text = render(RESULTS_DIR)
    out = RESULTS_DIR.parent / "RESULTS.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwrote {out}")


# ------------------------------------------------------------------- figures


def cmd_figures(args):
    from .viz import render_all

    _, train, test = load_maps()
    render_all(test, RESULTS_DIR / "figures", build_arms)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="bahn", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("maps").set_defaults(func=cmd_maps)

    b = sub.add_parser("bc")
    b.add_argument("--steps", type=int, default=60_000)
    b.add_argument("--epochs", type=int, default=60)
    b.add_argument("--encoding", choices=["poster", "raw"], default="poster")
    b.set_defaults(func=cmd_bc)

    r = sub.add_parser("ppo")
    r.add_argument("--steps", type=int, default=2_000_000)
    r.add_argument("--n-envs", type=int, default=32)
    r.set_defaults(func=cmd_ppo)

    e = sub.add_parser("eval")
    e.add_argument("--split", choices=["train", "test"], default="test")
    e.add_argument("--batch", type=int, default=30)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--arm", nargs="*", default=None)
    e.add_argument("--difficulty-key", default="closest_obstacle")
    e.set_defaults(func=cmd_eval)

    t = sub.add_parser("tune")
    t.add_argument("--n-maps", type=int, default=60)
    t.add_argument("--batch", type=int, default=30)
    t.add_argument("--only", choices=["reactive", "cbf"], default=None)
    t.set_defaults(func=cmd_tune)

    a = sub.add_parser("ablate")
    a.add_argument("--look-ahead", type=float, nargs="*", default=[0.05, 0.08, 0.12, 0.20, 0.30])
    a.add_argument("--n-maps", type=int, default=60)
    a.add_argument("--batch", type=int, default=30)
    a.set_defaults(func=cmd_ablate)

    sub.add_parser("report").set_defaults(func=cmd_report)
    sub.add_parser("figures").set_defaults(func=cmd_figures)

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
