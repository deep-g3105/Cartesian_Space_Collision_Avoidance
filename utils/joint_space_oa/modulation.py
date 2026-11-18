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

    discrete modulated DS (34)-(35):
        q_dot   = M(q) (f(q) + alpha(q) M_perp(q) g(q)) + f_rep(q)
        q_{t+1} = q_t + q_dot * dt

    kernel activation (15), (31):
        alpha(q) = alpha_Gamma * alpha_n * alpha_f   if ||q - q*|| > r, else 0
        alpha_Gamma = 1 - lambda_n,  alpha_n = 1 - lambda_v,  alpha_f = min(1, ||q - q*||)
    (the paper prints alpha_f = max(1, ...), which is not in [0, 1]; the text
    "alpha_f goes from 1 to 0 as q approaches q*" means min.)

The tangential field g(q) (navigation kernels) and its MPPI optimisation
live in mpc.py; with g = None / alpha = 0 this is the plain modulation.

Because E is orthonormal and all tangential eigenvalues are equal, (5)
reduces to the closed form
    M = lambda_tau I + (lambda_n' - lambda_tau) n n^T,
so no QR / Gram-Schmidt of the tangent basis is needed (the paper notes QR
is its main CPU cost), and M_perp of Eq. (32) is I - n n^T. Everything is
vectorised over a batch of states (velocity_batch) for the MPPI roll-outs.

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
    """Modulated joint-space DS (Eqs. 5-7, 12-13, 15, 29-35).

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
        max_deflection:  ||alpha M_perp g|| is capped at max_deflection * ||f|| (None: no cap).
                         Not in the paper: MPPI-sampled directions can be several rad/s, and a
                         reference that moves much faster than the nominal DS is not tracked by
                         the impedance controller (the robot lags the collision-checked q_ref).
        lead_slow, lead_stop: reference governor of step() [rad] (None: off).
    """

    def __init__(self, d1: float = 0.01, d2: float = 0.10, k: float = 200.0,
                 lambda_tau_max: float = 2.0, tail_effect: bool = True, tail_k: float = 100.0,
                 safety_eps: float = 0.01, dt: float = 0.002,
                 max_qdot=None, q_min: Optional[np.ndarray] = None,
                 q_max: Optional[np.ndarray] = None, limit_margin: float = 0.05,
                 max_deflection: Optional[float] = 1.0,
                 lead_slow: Optional[float] = None, lead_stop: Optional[float] = None):
        self.max_deflection = max_deflection
        self.lead_slow, self.lead_stop = lead_slow, lead_stop
        if lead_stop is not None:
            assert lead_slow is not None and 0.0 <= lead_slow < lead_stop
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

    def lambda_v(self, n: np.ndarray, f: np.ndarray):
        """~1 if the nominal DS points away from the obstacle, ~0 if towards it.
        n, f: (d,) -> float, or (N, d) -> (N,)."""
        n, f = np.asarray(n, dtype=float), np.asarray(f, dtype=float)
        nf = np.linalg.norm(f, axis=-1)
        cos = (n * f).sum(-1) / np.maximum(nf, 1e-9)
        z = np.clip(-self.tail_k * cos, -60.0, 60.0)
        lam = np.where(nf < 1e-9, 1.0, 1.0 / (1.0 + np.exp(z)))
        return float(lam) if lam.ndim == 0 else lam

    def activation(self, gamma, n, f, goal_dist, goal_radius: float = 0.1):
        """Navigation-kernel activation alpha(q) in [0, 1], Eqs. (15), (31).

        Args:
            gamma:      Gamma(q) [m], scalar or (N,).
            n:          (d,) or (N, d) unit obstacle normal.
            f:          (d,) or (N, d) nominal DS velocity.
            goal_dist:  ||q - q*|| [rad], scalar or (N,).
            goal_radius: r of the ball B_r around q* where alpha = 0 (keeps q* L.A.S., Prop. 2).
        """
        gamma = np.asarray(gamma, dtype=float)
        a_gamma = 1.0 - self.lambda_n(np.where(np.isfinite(gamma), gamma, 1e3))
        a_n = 1.0 - self.lambda_v(n, f)
        goal_dist = np.asarray(goal_dist, dtype=float)
        a_f = np.minimum(1.0, goal_dist)
        a = np.where(goal_dist > goal_radius, a_gamma * a_n * a_f, 0.0)
        return float(a) if a.ndim == 0 else a

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
            g:      (d,) navigation-kernel field g(q) (mpc.NavigationKernels), or None.
            alpha:  kernel activation alpha(q) (activation()).
        """
        G = None if g is None else np.asarray(g, dtype=float)[None]
        qdot, info = self.velocity_batch(np.asarray(f, dtype=float)[None], np.array([gamma], dtype=float),
                                         np.asarray(grad, dtype=float)[None], G, np.array([alpha], dtype=float))
        return qdot[0], {k: float(v[0]) for k, v in info.items()}

    def velocity_batch(self, F: np.ndarray, gamma: np.ndarray, grad: np.ndarray,
                       G: Optional[np.ndarray] = None, alpha: Optional[np.ndarray] = None,
                       dt: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Eq. (34) for N states at once: F, grad, G (N, d); gamma, alpha (N,).

        dt overrides the step used in the repulsion (33) (the MPC roll-outs
        integrate with a coarser step than the controller).
        """
        F = np.asarray(F, dtype=float)
        gamma = np.asarray(gamma, dtype=float)
        gn = np.linalg.norm(grad, axis=1)
        valid = np.isfinite(gamma) & (gn > 1e-9)
        n = np.where(valid[:, None], grad / np.maximum(gn, 1e-9)[:, None], 0.0)
        g_safe = np.where(valid, gamma, 1e3)

        lam_n = np.where(valid, self.lambda_n(g_safe), 1.0)
        lam_t = np.where(valid, self.lambda_tau(g_safe), 1.0)
        lam_v = np.where(valid, self.lambda_v(n, F), 1.0) if self.tail_effect else np.where(valid, 0.0, 1.0)
        lam_ne = lam_v + (1.0 - lam_v) * lam_n                             # Eq. (29)

        U = F
        if G is not None and alpha is not None:
            Gt = G - n * (n * G).sum(1, keepdims=True)                     # M_perp g, Eq. (32)
            Gt = np.where(valid, np.asarray(alpha, dtype=float), 0.0)[:, None] * Gt
            if self.max_deflection is not None:
                cap = self.max_deflection * np.linalg.norm(F, axis=1)
                Gt = Gt * np.minimum(1.0, cap / np.maximum(np.linalg.norm(Gt, axis=1), 1e-12))[:, None]
            U = F + Gt
        # M U with M = lambda_tau I + (lambda_n' - lambda_tau) n n^T
        qdot = lam_t[:, None] * U + ((lam_ne - lam_t) * (n * U).sum(1))[:, None] * n

        rep = np.where(valid & (gamma < self.eps), (self.eps - g_safe) / (dt or self.dt), 0.0)   # Eq. (33)
        qdot = qdot + rep[:, None] * n
        info = {"lambda_n": lam_ne, "lambda_tau": lam_t, "lambda_v": lam_v, "repulsion": rep}
        return self._limit(qdot), info

    def step(self, q: np.ndarray, f: np.ndarray, gamma: float, grad: np.ndarray,
             g: Optional[np.ndarray] = None, alpha: float = 0.0,
             q_measured: Optional[np.ndarray] = None
             ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """One Euler step of the modulated DS, Eq. (35). Returns q_{t+1}, q_dot, info.

        q_measured: the robot's measured q. If given and lead_slow / lead_stop are set,
        the reference governor (not in the paper) scales q_dot by 1 while
        ||q - q_measured|| < lead_slow, linearly down to 0 at lead_stop, and if the lead
        still exceeds lead_stop it restarts the reference at q_measured. Gamma is
        evaluated on the reference q, so the reference must not run away from a
        robot that lags behind (fast deflections, contact, low gains).
        """
        qdot, info = self.velocity(f, gamma, grad, g, alpha)
        info["governor"] = 1.0
        q = np.asarray(q, dtype=float)
        if q_measured is not None and self.lead_stop is not None:
            lead = float(np.linalg.norm(q - q_measured))
            if lead >= 0.95 * self.lead_stop:
                # the robot does not follow (lag / contact): restart the reference from the robot, so the
                # next step evaluates Gamma on the REAL configuration (and repels it if it is too close)
                info["governor"] = -1.0
                return np.asarray(q_measured, dtype=float).copy(), np.zeros_like(q), info
            scale = float(np.clip((self.lead_stop - lead) / (self.lead_stop - self.lead_slow), 0.0, 1.0))
            qdot = qdot * scale
            info["governor"] = scale
        q_next = q + qdot * self.dt
        if self.q_min is not None:
            q_next = np.clip(q_next, self.q_min, self.q_max)
            qdot = (q_next - q) / self.dt
        return q_next, qdot, info

    def _limit(self, qdot: np.ndarray) -> np.ndarray:
        """Scale each q_dot (row) down uniformly to respect max_qdot (keeps its direction)."""
        if self.max_qdot is None:
            return qdot
        ratio = np.max(np.abs(qdot) / self.max_qdot, axis=-1, keepdims=True)
        return qdot / np.maximum(ratio, 1.0)
