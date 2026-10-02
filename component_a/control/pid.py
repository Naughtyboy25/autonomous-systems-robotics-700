"""PID controller with anti-windup (Component A3).

Design decisions
----------------
* **Anti-windup by integral clamping**: the integral term is only allowed to
  accumulate within ``integral_limit``; without this, actuator saturation
  (heading commands are clamped to ``output_limit``) causes the classic
  windup overshoot.
* **Derivative on measurement** (option via ``derivative_on_measurement``):
  avoids the derivative kick when the setpoint changes step-wise, which is
  exactly what happens at every waypoint switch during path tracking.
* The controller is deliberately framework-free (pure float math) so it can
  be transplanted verbatim into the Webots controller in ``arena.wbt``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PIDController:
    kp: float
    ki: float
    kd: float
    output_limit: float = 5.0          # rad/s heading command clamp
    integral_limit: float = 1.0        # anti-windup bound on the integrator
    derivative_on_measurement: bool = True
    _integral: float = field(default=0.0, init=False)
    _prev_error: float = field(default=0.0, init=False)
    _prev_measurement: float = field(default=0.0, init=False)
    _initialized: bool = field(default=False, init=False)

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_measurement = 0.0
        self._initialized = False

    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        error = setpoint - measurement

        p = self.kp * error

        self._integral += error * dt
        self._integral = max(-self.integral_limit, min(self.integral_limit, self._integral))
        i = self.ki * self._integral

        if self.derivative_on_measurement and self._initialized:
            d = -self.kd * (measurement - self._prev_measurement) / dt
        elif self._initialized:
            d = self.kd * (error - self._prev_error) / dt
        else:
            d = 0.0
        self._prev_error = error
        self._prev_measurement = measurement
        self._initialized = True

        output = p + i + d
        return max(-self.output_limit, min(self.output_limit, output))
