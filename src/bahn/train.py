"""Training entry points: behaviour cloning the poster net, and PPO.

Both write to :func:`bahn.config.work_dir`, outside the repository.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .config import LIDAR, ROBOT, SIM, work_dir
from .planners.dwa import DWA
from .planners.learned import ENCODINGS, build_poster_net
from .sim import BatchSim
from .world import NavMap


def collect_dwa_demonstrations(
    maps: list[NavMap],
    steps: int = 60_000,
    n_envs: int = 32,
    noise: float = 0.15,
    seed: int = 0,
    encoding: str = "poster",
) -> tuple[np.ndarray, np.ndarray]:
    """Roll DWA out and record (poster input vector, wheel command) pairs.

    Actions are perturbed by Gaussian noise before execution while the *label*
    stays the unperturbed expert command.  Cloning a controller from its own
    on-policy states teaches it nothing about recovering from the states it will
    actually reach once its own errors accumulate; the noise widens the state
    distribution enough to make the clone usable, at a fraction of the cost of a
    full DAgger loop.
    """
    rng = np.random.default_rng(seed)
    teacher = DWA()
    sim = BatchSim(maps, n_envs=n_envs, robot=ROBOT, lidar=LIDAR, spec=SIM, seed=seed)

    xs, ys = [], []
    collected = 0
    while collected < steps:
        obs = sim.observe()
        v, w = teacher.act(obs)
        v_l, v_r = ROBOT.twist_to_wheels(v, w)
        xs.append(ENCODINGS[encoding](obs))
        ys.append(np.stack([v_l, v_r], axis=1).astype(np.float32))
        collected += n_envs

        v_noisy = v + rng.normal(0.0, noise * ROBOT.max_v, size=v.shape)
        w_noisy = w + rng.normal(0.0, noise * ROBOT.max_omega, size=w.shape)
        out = sim.step(v_noisy, w_noisy)
        finished = np.flatnonzero(out["done"])
        if finished.size:
            sim._reset_slots(finished)

    return np.concatenate(xs)[:steps], np.concatenate(ys)[:steps]


def train_bc(
    maps: list[NavMap],
    out_path: Path | None = None,
    steps: int = 60_000,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
    encoding: str = "poster",
) -> Path:
    """Fit the poster architecture to DWA's wheel commands."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from .planners.learned import POSTER_BIAS

    torch.manual_seed(seed)
    out_path = out_path or (work_dir() / "reactive_bc.pt")

    x, y = collect_dwa_demonstrations(maps, steps=steps, seed=seed, encoding=encoding)
    n_val = len(x) // 10
    perm = np.random.default_rng(seed).permutation(len(x))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    net = build_poster_net(LIDAR.n_beams)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(x[train_idx]), torch.as_tensor(y[train_idx])),
        batch_size=batch_size,
        shuffle=True,
    )
    xv = torch.as_tensor(x[val_idx])
    yv = torch.as_tensor(y[val_idx])

    def wheels(raw):
        # Same output convention as the deployed planner, bias included, so the
        # loss is on the command the robot would actually receive.
        return torch.clamp(
            raw * ROBOT.max_wheel_speed + POSTER_BIAS,
            -ROBOT.max_wheel_speed,
            ROBOT.max_wheel_speed,
        )

    best = np.inf
    best_state = None
    for epoch in range(epochs):
        net.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(wheels(net(xb)), yb)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = float(torch.nn.functional.mse_loss(wheels(net(xv)), yv))
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in net.state_dict().items()}
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:3d}  val_mse {val:.5f}  best {best:.5f}", flush=True)

    torch.save(best_state, out_path)
    print(f"behaviour clone [{encoding}] saved to {out_path} (val MSE {best:.5f})")
    return out_path


class OutcomeLogger:
    """Print outcome rates as training proceeds.

    SB3's own table reports ``ep_rew_mean``, which conflates "reached the goal"
    with "collected shaping reward and then crashed".  Success, collision and
    timeout rates are the quantities the study is actually about, so they are
    counted here directly off the simulator.
    """

    def __init__(self, every: int = 20):
        self.every, self.counts, self.rollouts = every, np.zeros(4), 0

    def __call__(self, local_vars: dict, _globals: dict) -> bool:
        # Read the outcome out of `infos`, not off the simulator: by the time the
        # callback runs, the vectorised env has already auto-reset the finished
        # slots and their outcome fields are back to RUNNING.
        done = local_vars.get("dones")
        infos = local_vars.get("infos")
        if done is not None and infos is not None:
            for d, info in zip(np.asarray(done, dtype=bool), infos, strict=False):
                if d:
                    self.counts[int(info.get("outcome", 0))] += 1
        if local_vars["n_steps"] == 0:
            self.rollouts += 1
            if self.rollouts % self.every == 0 and self.counts.sum() > 0:
                n = self.counts.sum()
                print(
                    f"  {local_vars['self'].num_timesteps:>9,} steps | {int(n):5d} episodes | "
                    f"success {self.counts[1] / n:.3f}  collision {self.counts[2] / n:.3f}  "
                    f"timeout {self.counts[3] / n:.3f}",
                    flush=True,
                )
                self.counts[:] = 0
        return True


def train_ppo(
    maps: list[NavMap],
    out_path: Path | None = None,
    total_steps: int = 2_000_000,
    n_envs: int = 32,
    seed: int = 0,
) -> Path:
    """Train a PPO local planner on the training split."""
    import torch
    from stable_baselines3 import PPO

    from .gym_env import make_vec_env_class

    # A 41-input, two-hidden-layer MLP is far too small to amortise thread
    # synchronisation: measured on this machine, one thread runs 40 per cent
    # faster than eight and leaves the rest of the box usable.
    torch.set_num_threads(1)

    out_path = out_path or (work_dir() / "ppo_nav")
    env = make_vec_env_class()(maps, n_envs=n_envs, seed=seed)
    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        device="cpu",  # a 41-input MLP is latency-bound, not FLOP-bound
        n_steps=256,
        batch_size=512,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        policy_kwargs={"net_arch": [128, 128]},
        verbose=0,
    )
    t0 = time.time()
    model.learn(total_timesteps=total_steps, callback=OutcomeLogger(), progress_bar=False)
    model.save(str(out_path))
    print(f"PPO saved to {out_path}.zip after {time.time() - t0:.0f}s")
    return Path(f"{out_path}.zip")
