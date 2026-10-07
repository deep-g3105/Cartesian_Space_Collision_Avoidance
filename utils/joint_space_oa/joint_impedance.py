"""
joint_impedance.py

Joint-space impedance (PD + gravity/Coriolis compensation) controller that
tracks the reference produced by the modulated DS (Koptev et al. IJRR 2024,
Fig. 7: the modulated DS streams (q, q_dot) to a low-level impedance
controller):

    tau = K (q_ref - q) + B (q_dot_ref - q_dot) + c(q, q_dot)

with c the bias forces (MuJoCo qfrc_bias), clipped to the torque limits.
The tracking error is kept small, which matters because the collision
check is done on q_ref, not on the measured q.
"""

from typing import Optional, Sequence

import numpy as np

# FR3 defaults: stiff shoulder/elbow, softer wrist (Nm/rad), torque limits (Nm)
FR3_STIFFNESS = np.array([150.0, 150.0, 150.0, 150.0, 100.0, 75.0, 20.0])
FR3_TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])


class JointImpedanceController:
    """
    Args:
        stiffness:     (d,) K [Nm/rad].
        damping:       (d,) B [Nms/rad]; default critical-ish 2 * sqrt(K).
        torque_limits: (d,) |tau| limits [Nm].
    """

    def __init__(self, stiffness: Sequence[float] = FR3_STIFFNESS,
                 damping: Optional[Sequence[float]] = None,
                 torque_limits: Sequence[float] = FR3_TORQUE_LIMITS):
        self.K = np.asarray(stiffness, dtype=float)
        self.B = 2.0 * np.sqrt(self.K) if damping is None else np.asarray(damping, dtype=float)
        self.tau_max = np.asarray(torque_limits, dtype=float)

    def torque(self, q: np.ndarray, qdot: np.ndarray, q_ref: np.ndarray,
               qdot_ref: np.ndarray, bias: np.ndarray) -> np.ndarray:
        tau = self.K * (q_ref - q) + self.B * (qdot_ref - qdot) + bias
        return np.clip(tau, -self.tau_max, self.tau_max)
