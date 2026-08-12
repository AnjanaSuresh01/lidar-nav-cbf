# The benchmark

## Why not just use one map

The source material for this repository is a course poster that demonstrates a
neural reactive navigator on a single 4 x 6 m map, with a single run, no
baseline and no metric. A single run cannot distinguish a controller that
navigates from one that got lucky on the geometry it was tuned against, and it
cannot say anything at all about where the controller stops working. Every
design decision below exists to replace that single run with a number that has
error structure.

## Map generation

Maps follow the specification in *Benchmarking Metric Ground Navigation*
(Perille, Truong, Xiao & Stone, 2020, [arXiv:2008.13315]), the dataset the ICRA
BARN Challenge is built on:

| Parameter | Value |
| --- | --- |
| Grid | 30 x 30 cells |
| Initial fill | {0.15, 0.20, 0.25, 0.30} |
| Smoothing iterations | {2, 3, 4} |
| Fill threshold | 5 occupied neighbours |
| Clear threshold | 1 occupied neighbour |
| Neighbourhood | 8-connected |
| Repetitions | 25 per combination |
| Total | **300 environments** |

Generation is a cellular automaton: fill the grid at random, then repeatedly set
a cell occupied if at least 5 of its 8 neighbours are occupied and free if at
most 1 is. Cells outside the grid count as free during smoothing, so the
automaton is not biased towards sealing the border.

Three departures from the paper, all forced by simulating in 2-D rather than in
Gazebo, and all chosen before any planner was run:

1. **Cell size 0.25 m**, so a map spans 7.5 x 7.5 m. With a 0.22 m robot radius,
   a one-cell gap is impassable and a two-cell gap leaves 6 cm either side. That
   places the interesting part of the difficulty range inside the dataset.
2. **Corridor walls** on the top and bottom rows. The Gazebo BARN worlds are
   walled corridors; without walls a planner drives around the obstacle field
   and every map scores identically.
3. **Three free approach columns at each end.** The paper draws start and goal
   from free cells on the left and right edges, but with a walled border no edge
   cell is ever collision-free for a disc of radius 0.22 m. The obstacle field
   itself is untouched at 30 x 30.

## Solvability

Every map is checked in configuration space before it enters the suite:
obstacles are inflated by the robot radius, and a flood fill must connect a free
cell in the left lane to one in the right lane. Unsolvable draws are discarded
and the seed advanced. **Every failure the benchmark reports is therefore the
planner's, not the map's** — a collision-free path always exists, and A* has
found it.

The suite is a deterministic function of its parameters: the seed for each map
is derived from `(fill, iterations, repetition, attempt)`, so regenerating it on
another machine produces byte-identical grids. Nothing is committed as data.

## Difficulty metrics

All five from the paper, averaged over the reference path:

| Metric | Meaning | Direction |
| --- | --- | --- |
| `closest_obstacle` | mean distance to the nearest obstacle along the path | lower is harder |
| `avg_visibility` | mean range over an 8-ray scan | lower is harder |
| `dispersion` | occupied/free alternations around a 16-ray scan | higher is harder |
| `characteristic_dimension` | narrowest width through the cell, over 8 axes | lower is harder |
| `tortuosity` | path arc length over start-goal distance | higher is harder |

Two need an interpretation the paper leaves implicit, stated here so the numbers
are reproducible rather than merely plausible. `dispersion` is evaluated at a
fixed 1.0 m radius, so it counts how many distinct obstacle groups surround the
robot. `characteristic_dimension` sums opposing ray pairs and takes the minimum
over 8 axes at 22.5-degree spacing.

Measured across the generated suite, every metric moves monotonically with fill
percentage — see `results/maps.json` and the table printed by `bahn maps`. That
is the evidence that the suite spans a difficulty range at all; without it,
reporting a mean success rate would be meaningless.

## Splits

Held out by repetition index: the last 10 of the 25 repetitions of *every*
parameter combination form the test split (120 maps), the first 15 form the
training split (180 maps). Splitting this way keeps the difficulty distribution
of the two splits matched while guaranteeing no test map was seen during
training or tuning.

Everything fitted or tuned in this repository — the reactive gains, the safety
filter geometry, the behaviour clone, the PPO policy — touches only the training
split. The held-out split is used once, for the table in `RESULTS.md`.

## Robot and sensor

| | |
| --- | --- |
| Body | disc, radius 0.22 m |
| Drive | differential, wheel base 0.40 m |
| Wheel speed | +/- 1.0 m/s each |
| Turn rate | +/- 3.0 rad/s |
| Acceleration | 2.0 m/s^2, 6.0 rad/s^2 |
| LiDAR | 36 beams over 360 degrees, 3.0 m range |
| Control rate | 10 Hz |
| Episode limit | 600 steps (60 s) |

The dimensions are close to a Clearpath Jackal, the platform BARN is defined on.
The footprint is modelled as a disc rather than a rectangle, so no planner can
exploit its own orientation to squeeze through a gap.

**The acceleration limits matter more than they look.** Without them the robot
stops dead within one control interval, which makes any safety filter trivially
satisfiable — commanding zero velocity is always safe — and hides the question
the filter has to answer. They are enforced by the simulator, identically for
every planner, so no arm can win by requesting the impossible.

**LiDAR range is 3.0 m over a 7.5 m map**, so the goal region is not visible
from the start. Every planner here is a genuinely *local* planner reacting to
what it can see, which is the regime the poster's controller was designed for.

## Metrics

* **Success / collision / timeout** rates.
* **Freeze rate**: the fraction of episodes in which the robot moved less than
  0.15 m over 100 consecutive control steps. This splits timeouts into "ran out
  of clock while making progress" and "stopped moving and never recovered", and
  it is the number a safety filter is most likely to inflate quietly.
* **SPL** (Anderson et al., 2018): `mean(S * l / max(p, l))`, with `l` the A*
  reference length and `p` the distance driven. Bounded above by the success
  rate.
* **BARN score**: `mean(S * OT / clip(AT, 2*OT, 8*OT))`, the ICRA Challenge
  metric. The clip means 0.5 is the maximum attainable value; that is a property
  of the metric, not of the planners.
* **5th-percentile minimum clearance**: how close the worst runs came to
  contact, which separates "did not collide" from "was nowhere near colliding".

[arXiv:2008.13315]: https://arxiv.org/abs/2008.13315
