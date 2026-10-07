"""
nominal_ds.py

Nominal joint-space dynamical system q_dot = f(q) (Eq. 1 / Eq. 36 of
Koptev, Figueroa, Billard, "Reactive collision-free motion generation in
joint space via dynamical systems and sampling-based MPC", IJRR 2024):

    f(q) = - speed * (q - q*) / max(||q - q*||, slow_radius)

i.e. the normalized linear DS of Eq. (36) (constant speed far from the
attractor) that turns into a plain linear DS -speed/slow_radius * (q - q*)
inside the ball ||q - q*|| < slow_radius, which keeps it G.A.S. and smooth
at the attractor (the "no normalization close to the attractor" remark of
Sec. 8.1).

CyclicJointDS switches between several attractors once the current one is
reached, like DSMotionGenerator in mj_sim/fr3_spawn.py but in joint space
(the paper's real-robot experiment alternates between two attractors).
"""

from typing import Sequence

import numpy as np


class LinearJointDS:
    """Normalized linear DS towards q_goal.

    Args:
        q_goal:      (d,) attractor q*.
        speed:       joint-space speed far from the attractor [rad/s].
        slow_radius: radius of the linear (non-normalized) region [rad].
    """

    def __init__(self, q_goal: Sequence[float], speed: float = 0.6, slow_radius: float = 0.15):
        self.q_goal = np.asarray(q_goal, dtype=float)
        self.speed = speed
        self.slow_radius = slow_radius

    def __call__(self, q: np.ndarray) -> np.ndarray:
        e = np.asarray(q, dtype=float) - self.q_goal
        return -self.speed * e / max(np.linalg.norm(e), self.slow_radius)

    def distance(self, q: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(q) - self.q_goal))


class CyclicJointDS:
    """Cycles through a list of joint attractors.

    Args:
        goals:      list of (d,) attractors, visited in order and repeated.
        speed, slow_radius: see LinearJointDS.
        switch_tol: switch to the next attractor once ||q - q*|| < switch_tol [rad].
    """

    def __init__(self, goals: Sequence[Sequence[float]], speed: float = 0.6,
                 slow_radius: float = 0.15, switch_tol: float = 0.02):
        self.goals = [np.asarray(g, dtype=float) for g in goals]
        self.speed = speed
        self.slow_radius = slow_radius
        self.switch_tol = switch_tol
        self.index = 0
        self.reached = 0                  # number of attractors reached so far
        self.ds = LinearJointDS(self.goals[0], speed, slow_radius)

    @property
    def q_goal(self) -> np.ndarray:
        return self.ds.q_goal

    def __call__(self, q: np.ndarray) -> np.ndarray:
        if self.ds.distance(q) < self.switch_tol:
            self.reached += 1
            self.index = (self.index + 1) % len(self.goals)
            self.ds = LinearJointDS(self.goals[self.index], self.speed, self.slow_radius)
        return self.ds(q)
