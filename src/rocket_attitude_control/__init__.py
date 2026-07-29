"""Rocket attitude-control simulation and reinforcement-learning environment."""

from .env import RocketEnv
from .simulation import RocketSimulation

__all__ = ["RocketEnv", "RocketSimulation"]
__version__ = "0.1.0"
