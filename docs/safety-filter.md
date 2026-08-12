# The safety filter, and what it does not promise

## The construction

A control barrier function filter sits between any planner and the robot. Given
the planner's nominal command it returns the closest command that keeps a safety
function from decreasing too fast.

A unicycle is not control-affine in its own position, so the barrier is written
on a **look-ahead point** `l` metres ahead of the wheel axis:

```
p     = (x + l cos t,  y + l sin t)
p_dot = A(t) u,    A(t) = [[cos t, -l sin t],
                           [sin t,  l cos t]],    u = (v, w)
```

`det A = l`, so for `l > 0` the point `p` is fully actuated and any barrier on it
is a **linear** constraint on `u`. This is the standard near-identity
diffeomorphism used for nonholonomic CBFs.

For each LiDAR return `q_i` reprojected into the world frame:

```
h_i      = ||p - q_i||^2 - r_s^2
h_dot_i + alpha * h_i >= 0
  <=>  2 (p - q_i)^T A(t) u >= -alpha * h_i
```

one linear row per beam. The 8 tightest beams are kept. The command box —
acceleration window, per-wheel speed, turn rate — enters the same problem as ten
more rows, so the filter cannot escape by requesting a command the robot would
have clipped anyway.

The result is a two-variable quadratic program solved exactly by enumerating
active sets (`src/bahn/safety/qp.py`): the optimum of a strictly convex quadratic
over a polyhedron is either the unconstrained point, the projection onto one
constraint, or the intersection of two. Enumerating all three cases has no
tolerance to tune and no iteration limit to hit, and it vectorises across the
whole batch. It is checked against `scipy.optimize.minimize` on random instances
in `tests/test_qp.py`; across 400 random problems the largest relative excess
over scipy's objective was 5e-9, and it never declared a solvable problem
infeasible.

## What the guarantee actually covers

Holding `||p - q_i|| >= r_s` with `r_s = radius + margin + l` puts the *wheel-axis
centre* at least `radius + margin` from every returned point, because the centre
is exactly `l` behind the look-ahead point. So the claim is:

> the robot body stays clear of every obstacle point the scanner reported.

That is narrower than "the robot does not collide", in three specific ways:

1. **Between the beams.** 36 beams over 360 degrees leave 10-degree gaps. At 1 m
   a gap is 17 cm wide — comparable to the robot radius. Obstacle surface that
   falls between two beams is not in the constraint set.
2. **Outside the scan.** Anything beyond 3.0 m does not exist to the filter.
3. **Discretisation.** The barrier is enforced at 10 Hz on a first-order model,
   not in continuous time.

None of these are argued away. The filtered arms are scored on the same
collision counter as everything else, and `RESULTS.md` reports what that counter
says.

## The relaxation, and why it is not optional

In a gap narrower than `2 * r_s` the barrier set is **empty** — no command keeps
every beam at arm's length, because there is not that much room. With
`r_s = 0.22 + margin + l`, a filter with `l = 0.15` and `margin = 0.05` needs
0.84 m of corridor to be strictly satisfiable, and the benchmark is built around
gaps narrower than that.

A filter that brakes whenever the set is empty refuses every passage the
benchmark is about. Measured on a 40-map development subset, that naive version
hard-braked on **17.5%** of control steps and dragged DWA's success rate on that
subset from 0.72 to 0.40.

So the filter instead finds the smallest uniform slack `delta` making the barrier
rows satisfiable and re-solves with `b - delta`. Feasibility is monotone in
`delta`, so bisection is exact to the bracket width. The command box is never
relaxed: those constraints are the physical robot, not a design choice. On the
same subset, hard braking fell to **2.4%** of steps and DWA+CBF recovered to
0.53.

Those two figures are a before/after on one configuration during development, not
a headline result; the held-out numbers are in `RESULTS.md`, which reports the
relaxation and braking rates for every filtered arm. They are reported because a
filter that is relaxed most of the time is not enforcing much, and omitting the
number would make the guarantee sound stronger than it is.

## The look-ahead trade-off, and how a safer-looking filter got less safe

`l` is the one parameter with a genuine two-sided cost:

* `r_s = radius + margin + l` grows with `l`, so a large `l` makes the filter
  conservative — it demands clearance the robot does not physically need.
* The turn-rate column of `A(t)` scales with `l`, so a small `l` leaves the
  filter almost no steering authority: it can slow down but barely turn, and a
  filter that can only brake produces exactly the freezing it should prevent.

Swept against PPO on 60 held-out maps (`bahn ablate`):

| `l` (m) | `r_s` (m) | success | collision | froze | relaxed |
| --- | --- | --- | --- | --- | --- |
| 0.05 | 0.29 | 0.500 | 0.033 | 0.500 | 0.009 |
| 0.08 | 0.32 | 0.517 | 0.017 | 0.483 | 0.021 |
| 0.12 | 0.36 | 0.483 | 0.017 | 0.517 | 0.023 |
| 0.20 | 0.44 | 0.317 | **0.117** | 0.583 | 0.096 |
| 0.30 | 0.54 | 0.183 | **0.183** | 0.633 | **0.220** |

The important column is the last one. Asking for a larger safety radius does not
buy safety here — it *costs* it. As `r_s` grows past what the corridor can
provide, the barrier set is empty more and more often, the slack relaxation has
to fire (0.9% of steps at `l = 0.05`, 22% at `l = 0.30`), and the filter stops
enforcing anything **precisely in the tight passages where it was needed**.
Collisions climb from 0.033 to 0.183, which is most of the way back to
unfiltered PPO's 0.225.

So the guarantee is not monotone in the margin you ask for. A safety filter
configured more conservatively than its environment affords degrades into an
expensive no-op, and the relaxation rate is the diagnostic that says so. That is
the reason it is reported for every filtered arm rather than buried.

The deployed value (`l = 0.12`) was chosen by grid search on the training split
before this sweep was run. On held-out maps `l = 0.08` would have been slightly
better (0.517 vs 0.483) — a one-grid-point discrepancy, reported rather than
retro-fitted.

## Tuning

Both hand-designed components in this repository get the same treatment: a grid
search on training maps, with the winner frozen before the held-out split is
touched. Comparing a tuned method against an untuned one measures the tuning,
not the method. `results/tuning.json` records what was searched and what won.
