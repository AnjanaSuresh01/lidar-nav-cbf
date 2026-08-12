"""Figures.

Two of these deliberately mirror the poster -- a trajectory over the occupancy
map, and the wheel-velocity and steering traces -- because the point is to show
the same picture on the same controller and then show what the poster's single
map hides.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import ROBOT
from .rollout import run_trajectory
from .world import NavMap


def _axes(ax, nav: NavMap):
    ny, nx = nav.grid.shape
    ax.imshow(
        nav.grid,
        origin="lower",
        extent=(0, nx * nav.res, 0, ny * nav.res),
        cmap="Greys",
        vmin=0,
        vmax=1.4,
        interpolation="nearest",
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")


def plot_trajectory(ax, nav: NavMap, record: dict, colour: str, label: str):
    _axes(ax, nav)
    ax.plot(nav.path[:, 0], nav.path[:, 1], "--", lw=1.0, color="0.55", label="A* reference")
    poses = record["poses"]
    ax.plot(poses[:, 0], poses[:, 1], color=colour, lw=1.8, label=label)
    ax.plot(*nav.start, "o", mfc="none", mec="tab:green", ms=10, mew=2, label="start")
    ax.plot(*nav.goal, "o", mfc="none", mec="tab:red", ms=10, mew=2, label="goal")
    if record["outcome"] == "collision":
        ax.plot(*poses[-1, :2], "x", color="tab:red", ms=12, mew=3)
    ax.set_title(f"{label}: {record['outcome']} ({record['time']:.1f} s)", fontsize=9)


def figure_poster_replica(nav: NavMap, planner, out: Path):
    """The poster's own four panels, on one map, for the poster's own controller."""
    import matplotlib.pyplot as plt

    rec = run_trajectory(planner, nav)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    plot_trajectory(axes[0, 0], nav, rec, "tab:blue", planner.name)
    axes[0, 0].legend(fontsize=7, loc="upper left")

    scan = rec["scans"][len(rec["scans"]) // 2]
    ang = np.degrees(np.linspace(-np.pi, np.pi, len(scan), endpoint=False))
    axes[0, 1].plot(ang, scan, lw=1.0)
    axes[0, 1].set_xlabel("beam angle (deg)")
    axes[0, 1].set_ylabel("range (m)")
    axes[0, 1].set_title("LiDAR scan (mid-episode)", fontsize=9)

    t = np.arange(len(rec["commands"])) * 0.1
    axes[1, 0].plot(t, rec["steer"], lw=1.0)
    axes[1, 0].axhline(0.0, color="0.7", lw=0.8, ls="--")
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 0].set_ylabel("steering (normalised)")
    axes[1, 0].set_ylim(-0.55, 0.55)
    axes[1, 0].set_title("Steering angle: goal bearing / 360, 0 = on target", fontsize=9)

    v_l, v_r = ROBOT.twist_to_wheels(rec["commands"][:, 0], rec["commands"][:, 1])
    axes[1, 1].plot(t, v_l, lw=1.0, label="left $v_L$")
    axes[1, 1].plot(t, v_r, lw=1.0, color="tab:red", label="right $v_R$")
    axes[1, 1].set_xlabel("time (s)")
    axes[1, 1].set_ylabel("wheel velocity (m/s)")
    axes[1, 1].set_title("Wheel velocities", fontsize=9)
    axes[1, 1].legend(fontsize=7)

    fig.suptitle(f"Poster replication on BARN map {nav.name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def figure_arm_comparison(nav: NavMap, arms, out: Path):
    """The same map under every arm, side by side."""
    import matplotlib.pyplot as plt

    cols = min(len(arms), 4)
    rows = int(np.ceil(len(arms) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.6 * rows), squeeze=False)
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, (name, planner, filt), colour in zip(axes.ravel(), arms, colours * 4, strict=False):
        rec = run_trajectory(planner, nav, safety_filter=filt)
        plot_trajectory(ax, nav, rec, colour, name)
    for ax in axes.ravel()[len(arms) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def figure_difficulty_curve(summary: dict, out: Path):
    """Success rate against BARN difficulty -- the plot one map cannot produce."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for arm, curve in summary["curves"].items():
        if not curve:
            continue
        centres = [0.5 * (c["lo"] + c["hi"]) for c in curve]
        ax.plot(centres, [c["success"] for c in curve], "o-", lw=1.6, ms=5, label=arm)
    ax.set_xlabel(f"BARN difficulty: {summary['difficulty_key'].replace('_', ' ')} (m)")
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Success against map difficulty (held-out split)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def render_all(test_maps: list[NavMap], out_dir: Path, build_arms) -> None:
    import json

    from .planners.reactive import ReactiveANN

    out_dir.mkdir(parents=True, exist_ok=True)
    from .config import work_dir

    arms = build_arms(work_dir() / "reactive_bc.pt", work_dir() / "ppo_nav.zip")

    # A mid-difficulty map, so the picture is representative rather than flattering.
    ranked = sorted(test_maps, key=lambda m: m.difficulty["closest_obstacle"])
    nav = ranked[len(ranked) // 2]

    figure_poster_replica(nav, ReactiveANN(), out_dir / "poster_replica.png")
    figure_arm_comparison(nav, arms, out_dir / "arms_same_map.png")

    summary_path = out_dir.parent / "summary.json"
    if summary_path.exists():
        figure_difficulty_curve(json.loads(summary_path.read_text()), out_dir / "difficulty.png")
    print(f"figures written to {out_dir}")
