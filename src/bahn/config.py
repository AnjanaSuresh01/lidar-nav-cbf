"""Shared specifications for the robot, sensor and simulation loop.

Every number here is a modelling choice, so each one carries a note saying where
it came from.  Anything traceable to the BARN benchmark or to the Clearpath
Jackal it is defined on is marked as such; the rest are our own and are stated
as such rather than dressed up as standard.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def work_dir() -> Path:
    """Directory for run artefacts (checkpoints, rollouts, figures).

    Defaults *outside* the repository.  The repository lives in a synced
    OneDrive folder on the development machine, where writing many files starves
    the training process of CPU; keeping artefacts on a plain local path avoids
    that.  Override with ``BAHN_WORK_DIR``.
    """
    root = os.environ.get("BAHN_WORK_DIR")
    if root is None:
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        root = str(Path(base) / "bahn-work")
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class RobotSpec:
    """Differential-drive robot, modelled as a disc.

    ``radius`` and ``wheel_base`` are close to a Clearpath Jackal (0.508 x 0.430 m
    footprint, 0.37 m track width), the platform the BARN benchmark is defined
    on.  We model the footprint as a circumscribed-ish disc rather than a
    rectangle: the robot cannot exploit its own orientation to squeeze through a
    gap, which makes every planner's job slightly harder and identical.
    """

    radius: float = 0.22
    wheel_base: float = 0.40
    max_wheel_speed: float = 1.0  # m/s per wheel
    max_omega: float = 3.0  # rad/s, clamped below the kinematic limit 2*v_w/L
    # Acceleration limits are enforced by the simulator for every planner alike.
    # Without them the robot can stop dead in one control interval, which makes
    # any safety filter trivially satisfiable -- commanding zero is always safe
    # -- and hides the question the filter actually has to answer.
    max_accel_v: float = 2.0  # m/s^2
    max_accel_omega: float = 6.0  # rad/s^2

    @property
    def max_v(self) -> float:
        return self.max_wheel_speed

    def wheels_to_twist(self, v_l: np.ndarray, v_r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Poster eq. 5: v = (v_R + v_L)/2, omega = (v_R - v_L)/L."""
        v = 0.5 * (v_r + v_l)
        omega = (v_r - v_l) / self.wheel_base
        return v, omega

    def twist_to_wheels(self, v: np.ndarray, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        half = 0.5 * self.wheel_base * omega
        return v - half, v + half


@dataclass(frozen=True)
class LidarSpec:
    """2-D planar scanner.

    The poster shows a 360-degree scan.  36 beams (10-degree spacing) is our
    choice: enough angular resolution that a 0.15 m obstacle cell is not missed
    inside the 2.5 m the planners actually react within, and small enough that a
    policy over raw beams trains on a laptop CPU.  ``docs/ablations.md`` reports
    what changing it costs.
    """

    n_beams: int = 36
    fov: float = 2 * np.pi
    # 3.0 m over a 7.5 m map: the goal region is not visible from the start, so
    # every planner here is genuinely a *local* planner reacting to what it sees.
    max_range: float = 3.0
    # Ray marching step; hits are refined by bisection afterwards.
    step: float = 0.05
    n_refine: int = 5

    @property
    def angles(self) -> np.ndarray:
        """Beam angles in the robot frame, ascending, starting at -fov/2."""
        if np.isclose(self.fov, 2 * np.pi):
            return np.linspace(-np.pi, np.pi, self.n_beams, endpoint=False)
        return np.linspace(-self.fov / 2, self.fov / 2, self.n_beams)


@dataclass(frozen=True)
class SimSpec:
    """Episode-level simulation settings."""

    dt: float = 0.1  # 10 Hz control, a realistic rate for a 2-D LiDAR stack
    max_steps: int = 600  # 60 s; BARN clips scores at 8x optimal time anyway
    goal_radius: float = 0.35  # m, counts as arrived (robot radius plus a little)
    # A robot that has moved less than this over `freeze_window` steps is stuck.
    freeze_window: int = 100
    freeze_distance: float = 0.15


ROBOT = RobotSpec()
LIDAR = LidarSpec()
SIM = SimSpec()
