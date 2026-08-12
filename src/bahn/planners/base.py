"""Planner interface and the poster's input encoding.

Every planner consumes the same :class:`~bahn.sim.Observation` and returns a
twist ``(v, omega)`` per environment.  Nothing here may read the map, the
reference path or the geodesic field.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..sim import Observation


class Planner(Protocol):
    name: str

    def act(self, obs: Observation) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(v, omega)``, each shape ``(n,)``."""
        ...

    def reset(self, mask: np.ndarray) -> None:
        """Clear any per-episode state for the environments flagged in ``mask``."""


def poster_encoding(obs: Observation) -> tuple[np.ndarray, np.ndarray]:
    """The poster's step 2 and step 4, exactly as written.

    Step 2: "Normalize steering angle to [-0.5, 0.5] (0 = correct direction)" -
    the bearing to the goal in degrees divided by 360.

    Step 4: "Normalize sensor values to [0.1, 1] and take inverse (1/d) (closer
    obstacles -> higher effect)" - so a beam at maximum range contributes 1 and a
    beam against the robot's skin contributes 10.  The floor at 0.1 is what stops
    the inverse blowing up, and it is the reason the encoding saturates: past
    about 0.3 m every beam reports roughly the same very large number, so the
    controller loses the ability to tell "close" from "about to hit".

    Returns:
        ``steer`` of shape ``(n,)`` and ``inv`` of shape ``(n, b)``.
    """
    steer = np.degrees(obs.heading_err) / 360.0
    d_norm = 0.1 + 0.9 * np.clip(obs.ranges / obs.max_range, 0.0, 1.0)
    return steer, 1.0 / d_norm
