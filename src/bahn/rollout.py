"""Episode runner shared by every experiment.

One planner, optionally one safety filter, over a list of maps.  Maps are handed
out sequentially, so two planners evaluated on the same list see exactly the same
environments in the same order and the comparison is paired.
"""

from __future__ import annotations

import numpy as np

from .config import LIDAR, ROBOT, SIM, LidarSpec, RobotSpec, SimSpec
from .sim import BatchSim
from .world import NavMap


def run_suite(
    planner,
    maps: list[NavMap],
    safety_filter=None,
    batch: int = 30,
    robot: RobotSpec = ROBOT,
    lidar: LidarSpec = LIDAR,
    spec: SimSpec = SIM,
    seed: int = 0,
) -> list[dict]:
    """Run every map in ``maps`` exactly once and return one record per map.

    Returns records in map order, so they align with ``maps`` by index.
    """
    n = min(batch, len(maps))
    sim = BatchSim(maps, n_envs=n, robot=robot, lidar=lidar, spec=spec, seed=seed, sequential=True)
    planner.reset(np.ones(n, dtype=bool))
    if safety_filter is not None:
        safety_filter.reset(np.ones(n, dtype=bool))

    done_by_map: dict[int, dict] = {}
    tallies = {k: np.zeros(n) for k in ("fallback", "relaxed", "intervened")}
    step_counts = np.zeros(n)

    while len(done_by_map) < len(maps):
        obs = sim.observe()
        v, w = planner.act(obs)
        if safety_filter is not None:
            v, w, info = safety_filter.filter(obs, v, w)
            for k in tallies:
                tallies[k] += info.get(k, 0.0)
        step_counts += 1
        out = sim.step(v, w)

        finished = np.flatnonzero(out["done"])
        if finished.size:
            for rec, s in zip(sim.episode_records(finished), finished, strict=True):
                steps = max(step_counts[s], 1)
                for k, acc in tallies.items():
                    key = "qp_fallback_frac" if k == "fallback" else f"{k}_frac"
                    rec[key] = float(acc[s] / steps) if safety_filter is not None else float("nan")
                done_by_map.setdefault(rec["map_index"], rec)
            for acc in tallies.values():
                acc[finished] = 0.0
            step_counts[finished] = 0.0
            sim._reset_slots(finished)
            mask = np.zeros(n, dtype=bool)
            mask[finished] = True
            planner.reset(mask)
            if safety_filter is not None:
                safety_filter.reset(mask)

    return [done_by_map[i] for i in range(len(maps))]


def run_trajectory(
    planner,
    nav: NavMap,
    safety_filter=None,
    robot: RobotSpec = ROBOT,
    lidar: LidarSpec = LIDAR,
    spec: SimSpec = SIM,
) -> dict:
    """Single episode with the full state trace, for figures and debugging."""
    sim = BatchSim([nav], n_envs=1, robot=robot, lidar=lidar, spec=spec, sequential=True)
    planner.reset(np.ones(1, dtype=bool))
    if safety_filter is not None:
        safety_filter.reset(np.ones(1, dtype=bool))

    poses, cmds, nominal, clearances, scans, steer = [], [], [], [], [], []
    while sim.outcome[0] == 0:
        obs = sim.observe()
        v, w = planner.act(obs)
        nominal.append([float(v[0]), float(w[0])])
        if safety_filter is not None:
            v, w, _ = safety_filter.filter(obs, v, w)
        poses.append([float(sim.pos[0, 0]), float(sim.pos[0, 1]), float(sim.theta[0])])
        cmds.append([float(v[0]), float(w[0])])
        clearances.append(float(obs.clearance[0]))
        scans.append(obs.ranges[0].copy())
        # The poster's "steering angle (normalised)": goal bearing in degrees
        # over 360.  It is the controller's *input*, not a function of its output.
        steer.append(float(np.degrees(obs.heading_err[0]) / 360.0))
        sim.step(v, w)

    rec = sim.episode_records(np.array([0]))[0]
    rec.update(
        poses=np.array(poses),
        commands=np.array(cmds),
        nominal=np.array(nominal),
        clearances=np.array(clearances),
        scans=np.array(scans),
        steer=np.array(steer),
    )
    return rec
