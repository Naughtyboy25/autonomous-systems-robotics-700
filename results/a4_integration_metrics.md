# Component A4 - Integrated navigation performance

| Metric | Value |
|---|---|
| Goal reached | yes |
| Initial planned path length | 22.92 m |
| Actually travelled distance | 24.11 m |
| Time to goal | 48.5 s |
| Mean cross-track error | 0.026 m |
| Max cross-track error | 0.112 m |
| Online replans (corridor blocked) | 6 |
| Min clearance to obstacles | 0.40 m |

## Critical analysis: perception / planning / control trade-offs

The run starts from a deliberately outdated a-priori map: the initial plan
crosses a region where a real obstacle exists. Only the fused LiDAR+camera
occupancy (autumn overlay in the figure) reveals the blockage, which
triggers online A* replanning around it - perception accuracy is therefore
not a passive quality metric here, it is what makes the plan executable at
all. The cost of that accuracy is sensing latency: fusion lags the truth by
up to one scan period (0.5 s LiDAR / 1.0 s camera), so the planner inflates
obstacles by an extra safety margin (0.30 m) and the controller caps speed
when front clearance drops below 0.8 m. Planning optimality suffers from the
same conservatism: inflation and the optimistic treatment of unknown space
(planned as free until proven occupied) make replans more frequent than a
full-knowledge planner would need, but guarantee the robot never commits to
a corridor it cannot physically fit through. Control responsiveness closes
the loop: the tuned PID (Kp=5, Ki=0.8, Kd=0.4) tracks the (re)planned path
with a mean CTE of 0.026 m, but aggressive tracking of a
freshly replanned path is exactly where an over-tuned controller would
oscillate; moderate gains keep the transient benign at the cost of slightly
slower convergence. In shared human environments every one of these margins
is a direct trade between mission time and safety: slower, wider-margin
behaviour is ethically defensible even when it is sub-optimal on paper.
