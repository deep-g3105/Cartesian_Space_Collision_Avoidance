"""
modulation.py

Joint-space DS modulation for obstacle avoidance with a learned distance
Gamma(q) (Koptev, Figueroa, Billard, IJRR 2024, Sec. 4.1, 5.1, 7.1, 7.4):

    M(q)   = E(q) D(q) E(q)^T                                     (5)
    E(q)   = [n(q) e_1 ... e_{d-1}],  n = grad Gamma / ||grad Gamma||   (6)
    D(q)   = diag(lambda_n, lambda_tau, ..., lambda_tau)          (7)

    sigma(Gamma; d1, d2, l1, l2, k) = l1 + (l2 - l1) / (1 + exp(-k (Gamma - (d1 + d2) / 2)))   (12)
    lambda_n   = sigma(.; 1 cm, 10 cm, 0, 1, k)                   (13)
    lambda_tau = sigma(.; 1 cm, 10 cm, 2, 1, k)

    tail effect (29)-(30): lambda_v = 1 / (1 + exp(-100 <n, f/||f||>))
                           lambda_n' = lambda_v + (1 - lambda_v) lambda_n
    safety repulsion (33): f_rep = (eps - Gamma) / dt * n   if Gamma < eps, else 0

    discrete modulated DS (34)-(35), MPC navigation kernels g(q) off (alpha = 0):
        q_dot   = M(q) (f(q) + alpha(q) M_perp(q) g(q)) + f_rep(q)
        q_{t+1} = q_t + q_dot * dt

Because E is orthonormal and all tangential eigenvalues are equal, (5)
reduces to the closed form
    M = lambda_tau I + (lambda_n' - lambda_tau) n n^T,
so no QR / Gram-Schmidt of the tangent basis is needed (the paper notes QR
is its main CPU cost), and M_perp of Eq. (32) is I - n n^T.

Units: the paper writes d1, d2 in cm with k = 2 [1/cm]; here everything is
in metres, so the default k = 200 [1/m].
"""

from typing import Dict, Optional, Tuple

import numpy as np


def sigmoid_gain(gamma: float, d1: float, d2: float, l1: float, l2: float, k: float) -> float:
    """Eq. (12): smooth step from l1 (Gamma << (d1+d2)/2) to l2 (Gamma >> (d1+d2)/2)."""
    z = np.clip(-k * (gamma - 0.5 * (d1 + d2)), -60.0, 60.0)
    return l1 + (l2 - l1) / (1.0 + np.exp(z))


class JointSpaceModulator:
    """Modulated joint-space DS (Eqs. 5-7, 12-13, 29-30, 33-35), no MPC.

    Args:
        d1, d2:          distances [m] where the modulation saturates / vanishes (1 cm, 10 cm).
        k:               sigmoid steepness [1/m] (paper: 2 / cm).
        lambda_tau_max:  tangential gain at contact (paper: 2).
        tail_effect:     use the tail-effect compensation (29)-(30).
        tail_k:          steepness of lambda_v (paper: 100).
        safety_eps:      safety threshold eps [m] for the repulsion (33) (paper: 1 cm).
        dt:              integration step [s] (paper: 0.002).
        max_qdot:        (d,) or scalar joint speed limit [rad/s]; the whole q_dot is scaled down.
        q_min, q_max:    (d,) joint limits; q_{t+1} is clamped inside [q_min + margin, q_max - margin].
        limit_margin:    margin to the joint limits [rad].
    """

    def __init__(self, d1: float = 0.01, d2: float = 0.10, k: float = 200.0,
                 lambda_tau_max: float = 2.0, tail_effect: bool = True, tail_k: float = 100.0,
                 safety_eps: float = 0.01, dt: float = 0.002,
                 max_qdot=None, q_min: Optional[np.ndarray] = None,
                 q_max: Optional[np.ndarray] = None, limit_margin: float = 0.05):
        self.d1, self.d2, self.k = d1, d2, k
        self.lambda_tau_max = lambda_tau_max
        self.tail_effect = tail_effect
        self.tail_k = tail_k
        self.eps = safety_eps
        self.dt = dt
        self.max_qdot = None if max_qdot is None else np.asarray(max_qdot, dtype=float)
        self.q_min = None if q_min is None else np.asarray(q_min, dtype=float) + limit_margin
        self.q_max = None if q_max is None else np.asarray(q_max, dtype=float) - limit_margin

    # ------------------------------------------------------------------ #
    # gains, Eq. (12)-(13), (29)-(30)
    # ------------------------------------------------------------------ #
    def lambda_n(self, gamma: float) -> float:
        return sigmoid_gain(gamma, self.d1, self.d2, 0.0, 1.0, self.k)

    def lambda_tau(self, gamma: float) -> float:
        return sigmoid_gain(gamma, self.d1, self.d2, self.lambda_tau_max, 1.0, self.k)

    def lambda_v(self, n: np.ndarray, f: np.ndarray) -> float:
        """~1 if the nominal DS points away from the obstacle, ~0 if towards it."""
        nf = np.linalg.norm(f)
        if nf < 1e-9:
            return 1.0
        z = np.clip(-self.tail_k * float(n @ f) / nf, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(z))

    # ------------------------------------------------------------------ #
    # modulation
    # ------------------------------------------------------------------ #
    def modulation_matrix(self, n: np.ndarray, f: np.ndarray, gamma: float
                          ) -> Tuple[np.ndarray, Dict[str, float]]:
        """M(q) = E D E^T in closed form, with tail-effect compensation."""
        lam_n = self.lambda_n(gamma)
        lam_t = self.lambda_tau(gamma)
        lam_v = self.lambda_v(n, f) if self.tail_effect else 0.0
        lam_n_eff = lam_v + (1.0 - lam_v) * lam_n
        M = lam_t * np.eye(len(n)) + (lam_n_eff - lam_t) * np.outer(n, n)
        return M, {"lambda_n": lam_n_eff, "lambda_tau": lam_t, "lambda_v": lam_v}

    def velocity(self, f: np.ndarray, gamma: float, grad: np.ndarray,
                 g: Optional[np.ndarray] = None, alpha: float = 0.0
                 ) -> Tuple[np.ndarray, Dict]:
        """Modulated joint velocity, Eq. (34).

        Args:
            f:      (d,) nominal DS velocity f(q).
            gamma:  Gamma(q) [m].
            grad:   (d,) grad_q Gamma(q).
            g, alpha: optional tangential deflection (navigation kernels, MPC) - unused for now.
        """
        f = np.asarray(f, dtype=float)
        gn = np.linalg.norm(grad)
        if not np.isfinite(gamma) or gn < 1e-9:
            info = {"lambda_n": 1.0, "lambda_tau": 1.0, "lambda_v": 1.0, "repulsion": 0.0}
            return self._limit(f), info

        n = grad / gn
        M, info = self.modulation_matrix(n, f, gamma)
        u = f
        if g is not None and alpha > 0.0:
            u = f + alpha * (g - n * float(n @ g))           # M_perp g, Eq. (32)
        qdot = M @ u

        rep = 0.0
        if gamma < self.eps:                                  # Eq. (33)
            rep = (self.eps - gamma) / self.dt
            qdot = qdot + rep * n
        info["repulsion"] = rep
        return self._limit(qdot), info

    def step(self, q: np.ndarray, f: np.ndarray, gamma: float, grad: np.ndarray,
             g: Optional[np.ndarray] = None, alpha: float = 0.0
             ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """One Euler step of the modulated DS, Eq. (35). Returns q_{t+1}, q_dot, info."""
        qdot, info = self.velocity(f, gamma, grad, g, alpha)
        q_next = np.asarray(q, dtype=float) + qdot * self.dt
        if self.q_min is not None:
            q_next = np.clip(q_next, self.q_min, self.q_max)
            qdot = (q_next - q) / self.dt
        return q_next, qdot, info

    def _limit(self, qdot: np.ndarray) -> np.ndarray:
        if self.max_qdot is None:
            return qdot
        ratio = np.max(np.abs(qdot) / self.max_qdot)
        return qdot / ratio if ratio > 1.0 else qdot
