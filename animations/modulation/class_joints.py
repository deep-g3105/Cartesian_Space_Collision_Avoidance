from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Helper: Gram-Schmidt tangent basis
# ---------------------------------------------------------------------------

def _gram_schmidt_tangents(normal: np.ndarray, d: int) -> List[np.ndarray]:
   
    tangents: List[np.ndarray] = []
    for i in range(d):
        ei = np.zeros(d)
        ei[i] = 1.0
        vi = ei - np.dot(ei, normal) * normal
        for t in tangents:
            vi = vi - np.dot(vi, t) * t
        norm_vi = np.linalg.norm(vi)
        if norm_vi > 1e-8:
            tangents.append(vi / norm_vi)
        if len(tangents) >= d - 1:
            break

    # Fallback: random QR completion (should never be needed in practice)
    if len(tangents) < d - 1:
        Q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((d, d)))
        extra = [Q[:, i] for i in range(d) if abs(np.dot(Q[:, i], normal)) < 1 - 1e-3]
        tangents = (tangents + extra)[: d - 1]

    return tangents[: d - 1]


# ---------------------------------------------------------------------------
# Per-obstacle specification
# ---------------------------------------------------------------------------

@dataclass
class JointSpaceObstacle:
    
    gamma_fn:      Callable[[np.ndarray], float]
    grad_gamma_fn: Callable[[np.ndarray], np.ndarray]
    name:          str = "obstacle"


# ---------------------------------------------------------------------------
# KB2024 Sigmoid eigenvalue functions  (Eqs. 12–13)
# ---------------------------------------------------------------------------

def _sigmoid(
    gamma_val: float,
    d1: float,
    d2: float,
    lam1: float,
    lam2: float,
    k: float,
) -> float:
    
    mid = 0.5 * (d1 + d2)
    return lam1 + (lam2 - lam1) / (1.0 + np.exp(-k * (gamma_val - mid)))


def _lambda_n(gamma_val: float, d1: float = 0.01, d2: float = 0.1, k: float = 2.0) -> float:
    
    return _sigmoid(gamma_val, d1, d2, lam1=0.0, lam2=1.0, k=k)


def _lambda_tau(gamma_val: float, d1: float = 0.01, d2: float = 0.1, k: float = 2.0) -> float:
    
    return _sigmoid(gamma_val, d1, d2, lam1=2.0, lam2=1.0, k=k)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class JointSpaceAvoider:
    

    def __init__(
        self,
        obstacles: Sequence[JointSpaceObstacle],
        dim: int,
        d1: float = 1.0,
        d2: float = 10.0,
        k: float = 2.0,
        gamma_norm_offset: float = 1.0,
        enforce_impenetrability: bool = True,
    ):
        if not obstacles:
            raise ValueError("At least one obstacle must be provided.")
        self.obstacles = list(obstacles)
        self.dim = int(dim)
        self.d1 = float(d1)
        self.d2 = float(d2)
        self.k = float(k)
        self.gamma_norm_offset = float(gamma_norm_offset)
        self.enforce_impenetrability = enforce_impenetrability

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def modulate(self, q: np.ndarray, f: np.ndarray) -> np.ndarray:
        
        q = np.asarray(q, float)
        f = np.asarray(f, float)
        assert q.shape == (self.dim,), f"Expected q shape ({self.dim},), got {q.shape}"
        assert f.shape == (self.dim,), f"Expected f shape ({self.dim},), got {f.shape}"

        # Per-obstacle raw Γ values (physical cm) and normalised Γ_norm
        gammas_raw  = np.array([obs.gamma_fn(q) for obs in self.obstacles])  # shape (N,)
        gammas_norm = gammas_raw + self.gamma_norm_offset                     # Γ_norm ≥ 1

        # Per-obstacle modulation matrices
        M_list = [self._M_single(q, obs, g_raw) for obs, g_raw in zip(self.obstacles, gammas_raw)]

        # ω-weighted combination  [KB2012 §6]
        M_combined = self._combine(M_list, gammas_norm)

        q_dot = M_combined @ f

        # Optional hard impenetrability enforcement
        if self.enforce_impenetrability:
            q_dot = self._enforce_impenetrability(q, q_dot, gammas_raw)

        return q_dot

    def M_combined(self, q: np.ndarray) -> np.ndarray:
        
        q = np.asarray(q, float)
        gammas_raw  = np.array([obs.gamma_fn(q) for obs in self.obstacles])
        gammas_norm = gammas_raw + self.gamma_norm_offset
        M_list = [self._M_single(q, obs, g_raw) for obs, g_raw in zip(self.obstacles, gammas_raw)]
        return self._combine(M_list, gammas_norm)

    def eigenvalues(self, q: np.ndarray) -> List[Tuple[float, float]]:
        
        q = np.asarray(q, float)
        out = []
        for obs in self.obstacles:
            g = float(obs.gamma_fn(q))
            ln  = _lambda_n(g,   self.d1, self.d2, self.k)
            lt  = _lambda_tau(g, self.d1, self.d2, self.k)
            out.append((ln, lt))
        return out

    # ------------------------------------------------------------------
    # Per-obstacle modulation matrix  M_i(q)  [KB2024 Eqs. 5–7]
    # ------------------------------------------------------------------

    def _M_single(
        self,
        q: np.ndarray,
        obs: JointSpaceObstacle,
        gamma_raw: float,
    ) -> np.ndarray:
        
        E = self._E(q, obs)                          # (d, d) orthonormal
        D = self._D(gamma_raw)                        # (d, d) diagonal
        return E @ D @ E.T

    def _E(self, q: np.ndarray, obs: JointSpaceObstacle) -> np.ndarray:
        
        d = self.dim
        grad = np.asarray(obs.grad_gamma_fn(q), float)
        assert grad.shape == (d,), \
            f"grad_gamma_fn for '{obs.name}' returned shape {grad.shape}, expected ({d},)"

        # Unit normal in joint space  [KB2024 below Eq. 6]
        n_unit = grad / (np.linalg.norm(grad) + _EPS)

        # d-1 tangent vectors via Gram-Schmidt  [KB2024 Eq. 6]
        tangents = _gram_schmidt_tangents(n_unit, d)

        return np.column_stack([n_unit] + tangents)  # (d, d) orthonormal

    def _D(self, gamma_raw: float) -> np.ndarray:
        
        d  = self.dim
        ln = _lambda_n(gamma_raw,   self.d1, self.d2, self.k)
        lt = _lambda_tau(gamma_raw, self.d1, self.d2, self.k)
        return np.diag([ln] + [lt] * (d - 1))

    # ------------------------------------------------------------------
    # Multi-obstacle ω-weighted combination  [KB2012 §6]
    # ------------------------------------------------------------------

    @staticmethod
    def _omega_weights(gammas_norm: np.ndarray) -> np.ndarray:
        
        N = len(gammas_norm)
        if N == 1:
            return np.array([1.0])

        log_G = np.log(np.clip(gammas_norm, _EPS, None))  # log-space for stability
        total_log = np.sum(log_G)

        # ∏_{j≠i} Γ_j = exp(total_log - log_G[i])
        numerators = np.exp(total_log - log_G)
        denom = np.sum(numerators)
        return numerators / (denom + _EPS)

    def _combine(
        self,
        M_list: List[np.ndarray],
        gammas_norm: np.ndarray,
    ) -> np.ndarray:
        
        weights = self._omega_weights(gammas_norm)
        d = self.dim
        M = np.zeros((d, d))
        for w, Mi in zip(weights, M_list):
            M += w * Mi
        return M

    # ------------------------------------------------------------------
    # Hard impenetrability post-processing
    # ------------------------------------------------------------------

    def _enforce_impenetrability(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        gammas_raw: np.ndarray,
    ) -> np.ndarray:
        
        for obs, g_raw in zip(self.obstacles, gammas_raw):
            if g_raw > self.d1:
                continue  # far from boundary — no correction needed
            grad = np.asarray(obs.grad_gamma_fn(q), float)
            n_unit = grad / (np.linalg.norm(grad) + _EPS)
            normal_vel = float(n_unit @ q_dot)
            if normal_vel < 0:
                q_dot = q_dot - normal_vel * n_unit
        return q_dot


# ---------------------------------------------------------------------------
# Convenience factory: analytical sphere obstacle in joint space
# ---------------------------------------------------------------------------

def make_sphere_obstacle(
    gamma_fn_cartesian: Callable[[np.ndarray], float],
    fk_fn: Callable[[np.ndarray], np.ndarray],
    jac_fn: Callable[[np.ndarray], np.ndarray],
    name: str = "sphere",
) -> JointSpaceObstacle:
    
    def gamma_joint(q: np.ndarray) -> float:
        x = fk_fn(q)
        return float(gamma_fn_cartesian(x))

    def grad_gamma_joint(q: np.ndarray) -> np.ndarray:
        x   = fk_fn(q)
        J   = jac_fn(q)                        # (3, d)
        # Numerical gradient of Γ w.r.t. x
        eps  = 1e-5
        n_x  = x.shape[0]
        grad_x = np.zeros(n_x)
        for i in range(n_x):
            x_plus  = x.copy(); x_plus[i]  += eps
            x_minus = x.copy(); x_minus[i] -= eps
            grad_x[i] = (gamma_fn_cartesian(x_plus) - gamma_fn_cartesian(x_minus)) / (2 * eps)
        return J.T @ grad_x                    # (d,)

    return JointSpaceObstacle(gamma_joint, grad_gamma_joint, name=name)


