"""The poster's controller: a hand-wired reactive network over inverse LiDAR.

The poster describes an "Artificial Neural Network (ANN)" that maps normalised
inverse LiDAR distances and a normalised steering angle to differential wheel
velocities, with a constant +0.2 added to both wheels to keep the robot rolling.
What it never describes is any way of *obtaining* the weights: there is no
dataset, no loss, no optimiser and no train/test split anywhere on it.  The
architecture is a Braitenberg vehicle, and its weights are a design choice.

So this module implements the canonical Braitenberg wiring the architecture
implies, and :mod:`bahn.planners.learned` implements the same input/output
convention with weights actually fitted to data.  The pair isolates the one
thing the poster leaves unanswered: whether the learning would have mattered.

Wiring, per forward-facing beam ``i`` at robot-frame angle ``a_i``:

*   ``front_i = max(cos a_i, 0)`` gates out beams pointing behind the robot.
*   ``lateral_i = sin a_i`` is positive on the left.  Weighting the inverse
    distance by ``front * lateral`` and adding it to the *left* wheel turns the
    robot away from obstacles on its left, which is the avoider connection.
*   A symmetric ``front`` term subtracted from both wheels brakes for whatever
    is directly ahead.
*   The steering term drives the wheels apart towards the goal bearing.
*   A constant ``k_drive`` on both wheels is the forward drive.  The poster's
    +0.2 wheel bias is described as being there "to maintain forward motion",
    i.e. as a supplement that stops the vehicle stalling -- not as the primary
    drive.  Leaving it to do the driving on its own caps the robot at 0.16 m/s
    in open space, which times out on a 10 m path and would make the baseline
    look bad for a reason of our own construction rather than the poster's.

All four gains are grid-searched on the training split, so the baseline is as
strong as this architecture gets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import ROBOT, RobotSpec
from ..sim import Observation
from .base import poster_encoding

POSTER_BIAS = 0.2  # poster step 7: "Add bias of 0.2 to both wheels"


@dataclass
class ReactiveANN:
    """Hand-wired reactive controller; no parameter here was fitted to data."""

    # Defaults are the grid-search winner on the training split (see
    # results/tuning.json), interior on all four axes.
    k_drive: float = 0.3
    k_steer: float = 5.0
    k_avoid: float = 6.0
    k_brake: float = 0.2
    bias: float = POSTER_BIAS
    robot: RobotSpec = ROBOT
    name: str = "reactive-hand"

    def reset(self, mask: np.ndarray) -> None:  # stateless
        return None

    def act(self, obs: Observation) -> tuple[np.ndarray, np.ndarray]:
        steer, inv = poster_encoding(obs)
        front = np.maximum(np.cos(obs.angles), 0.0)[None, :]
        lateral = np.sin(obs.angles)[None, :]

        avoid = (inv * front * lateral).sum(axis=1) / max(obs.angles.size, 1)
        brake = (inv * front).sum(axis=1) / max(obs.angles.size, 1)

        drive = self.k_drive - self.k_brake * brake
        v_l = drive - self.k_steer * steer + self.k_avoid * avoid
        v_r = drive + self.k_steer * steer - self.k_avoid * avoid

        # The poster applies the bias after the network's own (bounded) output.
        v_l = np.tanh(v_l) * self.robot.max_wheel_speed + self.bias
        v_r = np.tanh(v_r) * self.robot.max_wheel_speed + self.bias
        lim = self.robot.max_wheel_speed
        return self.robot.wheels_to_twist(np.clip(v_l, -lim, lim), np.clip(v_r, -lim, lim))
