"""Conventional controllers retained alongside the learned policy."""


class PIDController:
    """Small scalar PID controller."""

    def __init__(self, kp: float, ki: float, kd: float, dt: float) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = 0.0
        self.previous_error = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = 0.0

    def update(self, error: float) -> float:
        self.integral += error * self.dt
        derivative = (error - self.previous_error) / self.dt
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative
