"""The two planners whose weights were actually fitted to something.

:class:`ClonedPosterNet` is the poster's own architecture -- its input encoding,
its hidden layer, its ``tanh`` outputs, its +0.2 wheel bias -- with the weights
obtained by behaviour cloning a DWA teacher instead of being hand-chosen.  It
exists to answer the question the poster raises and never tests: how much of the
gap between a reactive controller and a real planner is the architecture, and how
much is the fact that nobody trained it?

:class:`PPOPlanner` wraps a Stable-Baselines3 policy trained on the same maps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import LIDAR, ROBOT, LidarSpec, RobotSpec
from ..sim import Observation
from .base import poster_encoding
from .reactive import POSTER_BIAS

HIDDEN = 32


def build_poster_net(n_beams: int = LIDAR.n_beams, hidden: int = HIDDEN):
    """The network drawn on the poster: inputs -> one tanh hidden layer -> 2 outputs."""
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(n_beams + 1, hidden),
        nn.Tanh(),
        nn.Linear(hidden, 2),
        nn.Tanh(),
    )


def poster_features(obs: Observation) -> np.ndarray:
    """The poster's input vector: normalised inverse ranges, then the steering term."""
    steer, inv = poster_encoding(obs)
    return np.concatenate([inv, steer[:, None]], axis=1).astype(np.float32)


def raw_features(obs: Observation) -> np.ndarray:
    """Identical, except the ranges are passed through instead of inverted.

    The poster's ``1/d`` step is a design decision it never tests.  Inverting a
    distance floored at 0.1 compresses everything beyond about a metre into a
    narrow band near 1 while expanding the last 30 cm across values up to 10, so
    the encoding spends almost all of its dynamic range on obstacles the robot is
    already too close to avoid.  Training the same network on raw ranges from the
    same teacher isolates how much that choice costs.
    """
    steer, _ = poster_encoding(obs)
    scan = np.clip(obs.ranges / obs.max_range, 0.0, 1.0)
    return np.concatenate([scan, steer[:, None]], axis=1).astype(np.float32)


ENCODINGS = {"poster": poster_features, "raw": raw_features}


@dataclass
class ClonedPosterNet:
    """Poster architecture, weights fitted by behaviour cloning."""

    checkpoint: Path | None = None
    robot: RobotSpec = ROBOT
    lidar: LidarSpec = LIDAR
    bias: float = POSTER_BIAS
    encoding: str = "poster"
    name: str = "reactive-bc"
    net: object = field(default=None, repr=False)

    def __post_init__(self):
        import torch

        if self.net is None:
            self.net = build_poster_net(self.lidar.n_beams)
        if self.checkpoint is not None:
            state = torch.load(Path(self.checkpoint), map_location="cpu", weights_only=True)
            self.net.load_state_dict(state)
        self.net.eval()

    def reset(self, mask: np.ndarray) -> None:  # stateless
        return None

    def wheels(self, features: np.ndarray) -> np.ndarray:
        """Forward pass in the poster's output convention, including its bias."""
        import torch

        with torch.no_grad():
            out = self.net(torch.as_tensor(features)).numpy()
        return np.clip(
            out * self.robot.max_wheel_speed + self.bias,
            -self.robot.max_wheel_speed,
            self.robot.max_wheel_speed,
        )

    def act(self, obs: Observation) -> tuple[np.ndarray, np.ndarray]:
        w = self.wheels(ENCODINGS[self.encoding](obs))
        return self.robot.wheels_to_twist(w[:, 0], w[:, 1])


@dataclass
class PPOPlanner:
    """Stable-Baselines3 PPO policy over the local observation."""

    checkpoint: Path
    robot: RobotSpec = ROBOT
    lidar: LidarSpec = LIDAR
    deterministic: bool = True
    name: str = "ppo"
    model: object = field(default=None, repr=False)

    def __post_init__(self):
        from stable_baselines3 import PPO

        if self.model is None:
            self.model = PPO.load(str(self.checkpoint), device="cpu")

    def reset(self, mask: np.ndarray) -> None:  # feed-forward policy, no state
        return None

    def act(self, obs: Observation) -> tuple[np.ndarray, np.ndarray]:
        from ..gym_env import decode_action, encode_obs

        x = encode_obs(obs, self.robot, self.lidar)
        action, _ = self.model.predict(x, deterministic=self.deterministic)
        return decode_action(np.asarray(action), self.robot)
