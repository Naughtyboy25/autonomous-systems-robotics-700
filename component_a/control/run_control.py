"""Component A3 driver: PID tuning study + path-tracking demonstration.

Part 1 - heading step response for four gain sets (P, PI, over-aggressive PD,
tuned PID) with quantitative metrics (overshoot, rise time, settling time,
steady-state error).
Part 2 - closed-loop tracking of a curved reference path with the tuned vs an
under-damped controller, showing the effect of tuning on real tracking.

Run with:  uv run python -m component_a.control.run_control
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from component_a.control.kinematics import DifferentialDriveRobot, wrap_angle
from component_a.control.path_tracker import PIDPathTracker, cross_track_error
from component_a.control.pid import PIDController

PLOTS = Path("plots")
RESULTS = Path("results")
DT = 0.02

GAIN_SETS = {
    "P only (Kp=2)": PIDController(2.0, 0.0, 0.0),
    "PI (Kp=2, Ki=1)": PIDController(2.0, 1.0, 0.0),
    "PD aggressive (Kp=8, Kd=0.05)": PIDController(8.0, 0.0, 0.05),
    "PID tuned (Kp=5, Ki=0.8, Kd=0.4)": PIDController(5.0, 0.8, 0.4),
}


def heading_step_response(pid: PIDController, t_end: float = 6.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a 45-degree heading step; return (t, theta, error)."""
    robot = DifferentialDriveRobot()
    pid.reset()
    t = np.arange(0.0, t_end, DT)
    theta_hist, err_hist = [], []
    for _ in t:
        omega = pid.update(np.deg2rad(45.0), robot.theta, DT)
        robot.step(0.0, omega, DT)
        theta_hist.append(robot.theta)
        err_hist.append(float(wrap_angle(np.deg2rad(45.0) - robot.theta)))
    return t, np.rad2deg(theta_hist), np.rad2deg(err_hist)


def step_metrics(t: np.ndarray, theta: np.ndarray, setpoint: float = 45.0) -> dict[str, float]:
    """Overshoot (%), 10-90% rise time, 2% settling time, steady-state error."""
    peak = float(np.max(theta))
    overshoot = max(0.0, (peak - setpoint) / setpoint * 100.0)
    try:
        t10 = t[np.argmax(theta >= 0.1 * setpoint)]
        t90 = t[np.argmax(theta >= 0.9 * setpoint)]
        rise = float(t90 - t10)
    except ValueError:
        rise = float("nan")
    band = 0.02 * setpoint
    outside = np.where(np.abs(theta - setpoint) > band)[0]
    settling = float(t[outside[-1]] + DT) if len(outside) else 0.0
    sse = float(abs(setpoint - theta[-1]))
    return {"overshoot_pct": overshoot, "rise_time_s": rise,
            "settling_time_s": settling, "sse_deg": sse}


def path_tracking_demo(pid: PIDController) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Track an S-curve reference; return (trajectory, reference, time, mean CTE)."""
    s = np.linspace(0, 10, 120)
    reference = np.column_stack([s, 2.0 * np.sin(s / 1.6) + 0.15 * s])
    robot = DifferentialDriveRobot(x=0.0, y=-1.0, theta=0.0)  # start offset
    tracker = PIDPathTracker(path=reference, heading_pid=pid)
    traj: list[tuple[float, float]] = []
    t = 0.0
    max_t = 40.0
    while t < max_t and not tracker.reached_goal(robot.x, robot.y):
        v, omega = tracker.command(robot, DT)
        robot.step(v, omega, DT)
        traj.append((robot.x, robot.y))
        t += DT
    tr = np.array(traj)
    cte = np.array([cross_track_error(reference, px, py) for px, py in tr[::10]])
    return tr, reference, np.arange(len(tr)) * DT, float(np.mean(cte))


def main() -> None:
    PLOTS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    # ---- part 1: step responses ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Component A3 - PID heading step response (45 deg command)")
    rows = []
    for name, pid in GAIN_SETS.items():
        t, theta, err = heading_step_response(pid, t_end=10.0)
        m = step_metrics(t, theta)
        axes[0].plot(t, theta, label=name)
        axes[1].plot(t, err, label=name)
        rows.append((name, pid, m))
    axes[0].axhline(45, ls="--", c="k", lw=1)
    axes[0].set(xlabel="t (s)", ylabel="heading (deg)", title="Response")
    axes[1].set(xlabel="t (s)", ylabel="heading error (deg)", title="Error")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "a3_pid_step_response.png", dpi=150)
    plt.close(fig)

    # ---- part 2: path tracking ----------------------------------------
    tuned = PIDController(5.0, 0.8, 0.4)
    sloppy = PIDController(0.8, 0.0, 0.0)
    tr_tuned, ref, tt, cte_tuned = path_tracking_demo(tuned)
    tr_sloppy, _, ts, cte_sloppy = path_tracking_demo(sloppy)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Component A3 - Path tracking with tuned vs under-tuned PID")
    axes[0].plot(ref[:, 0], ref[:, 1], "k--", label="reference")
    axes[0].plot(tr_tuned[:, 0], tr_tuned[:, 1], "g-", label=f"tuned PID (mean CTE {cte_tuned:.3f} m)")
    axes[0].plot(tr_sloppy[:, 0], tr_sloppy[:, 1], "r-", label=f"P only (mean CTE {cte_sloppy:.3f} m)")
    axes[0].set(xlabel="x (m)", ylabel="y (m)", title="Trajectories")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].plot(tt, [cross_track_error(ref, x, y) for x, y in tr_tuned], "g-", label="tuned PID")
    axes[1].plot(ts, [cross_track_error(ref, x, y) for x, y in tr_sloppy], "r-", label="P only")
    axes[1].set(xlabel="t (s)", ylabel="cross-track error (m)", title="Tracking error over time")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "a3_pid_path_tracking.png", dpi=150)
    plt.close(fig)

    # ---- metrics table -------------------------------------------------
    lines = [
        "# Component A3 - PID tuning results (45 deg heading step)",
        "",
        "| Gains | Overshoot (%) | Rise time 10-90% (s) | Settling time 2% (s) | SSE (deg) |",
        "|---|---|---|---|---|",
    ]
    for name, _pid, m in rows:
        lines.append(
            f"| {name} | {m['overshoot_pct']:.1f} | {m['rise_time_s']:.2f} "
            f"| {m['settling_time_s']:.2f} | {m['sse_deg']:.3f} |"
        )
    lines += [
        "",
        f"Path tracking on the S-curve: tuned PID mean cross-track error "
        f"{cte_tuned:.3f} m vs P-only {cte_sloppy:.3f} m.",
        "",
        "Interpretation: P-only reacts sluggishly and cannot cancel the",
        "residual droop caused by the actuator lag. Adding I removes the",
        "steady-state error but increases overshoot; D adds damping that",
        "suppresses oscillation. The tuned set (Kp=5, Ki=0.8, Kd=0.4) is the",
        "best compromise between responsiveness and stability and is the set",
        "used in the integrated system (A4).",
    ]
    table = "\n".join(lines)
    (RESULTS / "a3_pid_tuning_table.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
