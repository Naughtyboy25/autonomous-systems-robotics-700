# Component A3 - PID tuning results (45 deg heading step)

| Gains | Overshoot (%) | Rise time 10-90% (s) | Settling time 2% (s) | SSE (deg) |
|---|---|---|---|---|
| P only (Kp=2) | 0.0 | 0.90 | 1.62 | 0.000 |
| PI (Kp=2, Ki=1) | 15.7 | 0.62 | 5.18 | 0.022 |
| PD aggressive (Kp=8, Kd=0.05) | 8.1 | 0.20 | 0.62 | 0.000 |
| PID tuned (Kp=5, Ki=0.8, Kd=0.4) | 3.9 | 0.42 | 5.58 | 0.430 |

Path tracking on the S-curve: tuned PID mean cross-track error 0.089 m vs P-only 0.147 m.

Interpretation: P-only reacts sluggishly and cannot cancel the
residual droop caused by the actuator lag. Adding I removes the
steady-state error but increases overshoot; D adds damping that
suppresses oscillation. The tuned set (Kp=5, Ki=0.8, Kd=0.4) is the
best compromise between responsiveness and stability and is the set
used in the integrated system (A4).