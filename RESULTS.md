# Results

All numbers below are on the **held-out split: 120 maps** that no arm
was trained or tuned on, generated to the BARN specification (see
`docs/benchmark.md`). Every map is verified solvable in configuration space
before it enters the suite, so every failure here is the planner's.

Regenerate with `bahn eval && bahn report`.

## Headline

| arm | success | collision | timeout | froze | SPL | BARN | t_goal (s) | clear p05 (m) | filter acts | relaxed | braked |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reactive-hand | 0.400 | 0.600 | 0.000 | 0.000 | 0.400 | 0.174 | 21.8 | 0.216 | -- | -- | -- |
| reactive-hand+cbf | 0.358 | 0.000 | 0.642 | 0.667 | 0.355 | 0.149 | 23.3 | 0.297 | 0.629 | 0.022 | 0.002 |
| reactive-bc | 0.075 | 0.333 | 0.592 | 0.667 | 0.067 | 0.030 | 23.9 | 0.218 | -- | -- | -- |
| reactive-bc+cbf | 0.133 | 0.017 | 0.850 | 0.867 | 0.109 | 0.045 | 30.0 | 0.296 | 0.618 | 0.020 | 0.004 |
| reactive-bc-raw | 0.117 | 0.650 | 0.233 | 0.308 | 0.104 | 0.053 | 18.1 | 0.214 | -- | -- | -- |
| reactive-bc-raw+cbf | 0.217 | 0.017 | 0.767 | 0.767 | 0.182 | 0.086 | 25.8 | 0.240 | 0.623 | 0.027 | 0.004 |
| dwa | 0.400 | 0.000 | 0.600 | 0.600 | 0.400 | 0.200 | 12.3 | 0.239 | -- | -- | -- |
| dwa+cbf | 0.392 | 0.017 | 0.592 | 0.600 | 0.389 | 0.193 | 12.9 | 0.280 | 0.547 | 0.039 | 0.004 |
| ppo | 0.467 | 0.225 | 0.308 | 0.350 | 0.447 | 0.228 | 13.1 | 0.214 | -- | -- | -- |
| ppo+cbf | 0.492 | 0.008 | 0.500 | 0.517 | 0.462 | 0.242 | 13.3 | 0.247 | 0.849 | 0.020 | 0.003 |

`froze` counts episodes where the robot moved under 0.15 m in 100 consecutive
control steps; it is a subset of `timeout` and separates 'ran out of clock while
making progress' from 'stopped and never recovered'. `filter acts` is the
fraction of control steps on which the filter changed the command, `relaxed` the
fraction where the barrier set was empty and a slack was needed, `braked` the
fraction where even that failed. `clear p05` is the 5th percentile of per-episode
minimum clearance, which separates 'did not collide' from 'was nowhere near
colliding'. The BARN metric caps at 0.5 by construction.

## Success against map difficulty (closest obstacle)

The plot a single-map evaluation cannot produce. Bins hold equal numbers of maps;
lower mean clearance is harder.

| arm | 0.39-0.46 m (n=30) | 0.46-0.54 m (n=30) | 0.54-0.70 m (n=30) | 0.70-1.68 m (n=30) |
| --- | --- | --- | --- | --- |
| reactive-hand | 0.07 | 0.07 | 0.57 | 0.90 |
| reactive-hand+cbf | 0.03 | 0.10 | 0.50 | 0.80 |
| reactive-bc | 0.00 | 0.00 | 0.07 | 0.23 |
| reactive-bc+cbf | 0.03 | 0.07 | 0.13 | 0.30 |
| reactive-bc-raw | 0.03 | 0.00 | 0.03 | 0.40 |
| reactive-bc-raw+cbf | 0.07 | 0.07 | 0.20 | 0.53 |
| dwa | 0.07 | 0.20 | 0.57 | 0.77 |
| dwa+cbf | 0.07 | 0.17 | 0.57 | 0.77 |
| ppo | 0.23 | 0.17 | 0.60 | 0.87 |
| ppo+cbf | 0.17 | 0.27 | 0.70 | 0.83 |

## Ablation: safety-filter look-ahead distance

The barrier is enforced on a point `l` ahead of the wheel axis, so the safe
radius must be `robot radius + margin + l` for the body to be covered: large
`l` is conservative. But the turn-rate column of the actuation matrix scales
with `l`, so small `l` leaves the filter unable to steer and only able to brake.
Nominal planner is DWA, 60 held-out maps.

| l (m) | r_s (m) | success | collision | froze | SPL | filter acts | relaxed | braked |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.29 | 0.500 | 0.033 | 0.500 | 0.475 | 0.836 | 0.009 | 0.002 |
| 0.08 | 0.32 | 0.517 | 0.017 | 0.483 | 0.490 | 0.832 | 0.021 | 0.003 |
| 0.12 | 0.36 | 0.483 | 0.017 | 0.517 | 0.458 | 0.838 | 0.023 | 0.004 |
| 0.20 | 0.44 | 0.317 | 0.117 | 0.583 | 0.304 | 0.857 | 0.096 | 0.043 |
| 0.30 | 0.54 | 0.183 | 0.183 | 0.633 | 0.178 | 0.912 | 0.220 | 0.037 |

This sweep is a post-hoc analysis on held-out maps; the deployed value was
selected by grid search on the training split before it was run.

## Tuning record

Grid-searched on 60 training maps. Comparing a tuned method
against an untuned one measures the tuning, so both hand-designed components
get the same treatment and the winners were frozen before the held-out split
was touched.

```json
{
 "reactive": {
  "k_drive": 0.3,
  "k_steer": 5.0,
  "k_avoid": 6.0,
  "k_brake": 0.2
 },
 "reactive_success": 0.5166666666666667,
 "reactive_grid_points": 192,
 "cbf": {
  "look_ahead": 0.12,
  "margin": 0.02,
  "alpha": 12.0
 },
 "cbf_nominal": "ppo",
 "cbf_success_with_nominal": 0.7166666666666667,
 "cbf_grid_points": 30,
 "note_rejected_search": "An earlier CBF sweep used DWA as the nominal planner. DWA is already collision-free here, so success rose monotonically with the class-K gain (0.233/0.283/0.383/0.417/0.467 at alpha=1.5/3/6/12/24) - the search was walking towards disabling the filter. Tuning against a nominal that does collide makes the objective two-sided; the winner above is interior in both look_ahead and alpha."
}
```

## Files

| file | contents |
| --- | --- |
| `summary.json` | the tables above, machine-readable |
| `episodes.json` | one record per arm per map: outcome, time, distance, clearance |
| `maps.json` | the 300 generated maps with their five BARN difficulty metrics |
| `tuning.json` | what was searched and what won |
| `ablation_look_ahead.json` | the look-ahead sweep |
