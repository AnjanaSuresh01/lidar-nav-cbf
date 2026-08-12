# The four planners

All four consume the same observation — 36 LiDAR ranges and the goal bearing —
and emit a twist `(v, omega)`. None sees the map, the reference path or the
geodesic field. They differ only in how the mapping is obtained.

## 1. `reactive-hand` — the poster's controller

The poster describes an "Artificial Neural Network" over normalised inverse
LiDAR distances and a normalised steering angle, emitting differential wheel
velocities, with a constant +0.2 added to both wheels.

Reproduced exactly, including:

* **Step 2**, steering angle normalised to `[-0.5, 0.5]`: the goal bearing in
  degrees divided by 360.
* **Step 4**, ranges normalised to `[0.1, 1]` then inverted: a beam at maximum
  range contributes 1, a beam against the robot's skin contributes 10.
* **Step 7**, `+0.2` on both wheels after the network's bounded output.

What the poster does **not** describe is any way of obtaining the weights. There
is no dataset, no loss, no optimiser, no train/test split anywhere on it. The
architecture is a Braitenberg vehicle and its weights are a design choice, so
this arm implements the canonical wiring the architecture implies:

| Connection | Effect |
| --- | --- |
| `+k_avoid * sum(inv * front * lateral)` on the left wheel | obstacle on the left speeds the left wheel, turning right |
| `-k_brake * sum(inv * front)` on both wheels | brake for whatever is straight ahead |
| `-k_steer * steer` / `+k_steer * steer` | drive the wheels apart towards the goal bearing |

where `front = max(cos a, 0)` gates out rearward beams and `lateral = sin a` is
positive on the left. The three gains are grid-searched on the training split, so
the baseline is as strong as this architecture gets rather than a straw man.

## 2. `reactive-bc` — the same architecture, weights fitted

The network drawn on the poster (inputs → one 32-unit `tanh` hidden layer → two
`tanh` outputs), the poster's input encoding, the poster's `+0.2` bias — with the
weights obtained by behaviour cloning a DWA teacher instead of being hand-chosen.

This arm exists to answer the question the poster raises and never tests: of the
gap between a reactive controller and a real planner, how much is the
architecture and how much is the fact that nobody trained it?

Demonstrations are collected by rolling DWA out with Gaussian noise on the
executed action while the *label* stays the unperturbed expert command. Cloning
a controller purely from its own on-policy states teaches it nothing about
recovering from the states it will reach once its own errors accumulate; the
noise broadens the state distribution at a fraction of the cost of a full DAgger
loop. This is a limitation of the arm, and it means its score is a lower bound on
what the architecture can do — see `RESULTS.md`.

An `encoding="raw"` variant trains the identical network on the same
demonstrations with raw normalised ranges in place of `1/d`. That isolates the
cost of the poster's inverse encoding specifically, separately from the
architecture and from the imitation procedure.

## 3. `dwa` — Dynamic Window Approach

Fox, Burgard & Thrun (1997); the classical baseline and the one BARN's original
paper evaluates. Candidate twists are sampled inside the acceleration window,
rolled out as constant-twist arcs, and scored against obstacle points
reconstructed *from the scan* rather than from the occupancy grid, so it holds no
map advantage over the others.

Two implementation details that turned out to matter:

* **Scoring must reward closing distance, not pointing at the goal.** With a pure
  heading term, a candidate that merely points at the goal outscores one that
  drives towards it, and since standing still maximises the clearance term the
  robot sits and rotates. Measured on a 40-map development subset, that
  configuration scored 0.00 success and froze on 100% of maps; replacing the
  heading term with distance actually closed over the horizon took the same
  subset to 0.72. (That subset is the easy end of the suite — the suite is
  generated in order of fill percentage — so treat those two numbers as a
  before/after on one configuration, not as DWA's score. The held-out numbers are
  in `RESULTS.md`.)
* **Candidates must be executable.** Sampling `v` and `omega` on independent axes
  produces corner combinations outside the wheel-speed diamond; scoring those
  means choosing a trajectory the robot then does not drive, because the
  simulator scales the twist back.

DWA is also the teacher for `reactive-bc`, which is why it is worth having a
properly tuned one.

## 4. `ppo` — a learned local planner

Stable-Baselines3 PPO, `[128, 128]` MLP, trained on the training split only.

* **Observation** (41 floats): 36 normalised ranges, goal distance,
  `sin`/`cos` of the goal bearing, previous `v` and `omega`.
* **Action**: `[-1, 1]^2` mapped onto `v in [-0.3, 1.0]` m/s and
  `omega in [-3, 3]` rad/s.
* **Reward**: geodesic progress (3.0 per metre closed), a small step cost, a
  proximity penalty below 0.45 m clearance, a spin penalty, and `+/-5` terminal.

The geodesic distance field is privileged information, used *only* to score
progress the policy already made, and never entering the observation. Shaping on
straight-line distance instead turns every concave obstacle into a reward wall
the policy learns to press against.

**A bug worth recording.** The geodesic field was initially computed only over
configuration-space free cells. A robot whose disc is collision-free routinely
sits in a cell whose *centre* is not C-space free — that is what hugging an
obstacle looks like on a grid — so 3.1% of lookups returned the unreachable
sentinel. The reward had a standard deviation of 615 with spikes to ±3000, and
PPO learned the only sane response to a signal like that: stand still, scoring 0%
success and freezing on 95% of maps. Nothing in a loss curve showed this. It
showed up in one histogram of per-step reward. The field is now extended to every
cell by nearest-neighbour lookup, and `tests/test_reward.py` asserts the reward
scale directly so it cannot regress silently.
