from typing import Callable, Optional
import numpy as np

_EPS = 1e-9


class ModulationAvoider:

    def __init__(
        self,
        gamma: Callable[[np.ndarray], float],
        grad_gamma: Callable[[np.ndarray], np.ndarray],
        dim: Optional[int] = None,
        rho: float = 1.0,
        safety_eta: float = 1.0,
        tail_effect: bool = True,
        lambda_min: float = -10.0,
        lambda_max_factor: float = 100.0,
    ):
        assert rho > 0, "rho must be > 0"
        assert 0 < safety_eta <= 1.0, "safety_eta must be in (0,1]"

        self.gamma = gamma
        self.grad_gamma = grad_gamma
        self.rho = float(rho)
        self.safety_eta = float(safety_eta)
        self.tail_effect = bool(tail_effect)
        self.lambda_min = lambda_min
        self.lambda_max_factor = lambda_max_factor
        self.dim = dim

    # ---------------------- Public API ----------------------

    def modulate(self, x: np.ndarray, x_tilde: np.ndarray, f: np.ndarray, v_obs: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute modulated velocity dx_dt = M(x_tilde) f(x)."""

        x = np.asarray(x, float)
        x_tilde = np.asarray(x_tilde, float)
        f = np.asarray(f, float)
        v_o = np.zeros_like(f) if v_obs is None else np.asarray(v_obs, float)

        d = x.shape[0]
        if self.dim is None:
            self.dim = d
        assert d == self.dim, "x dimension mismatch"

        # Safety-inflated modulation anchor
        # x_tilde_safe = x_tilde / self.safety_eta
        f_rel = f - v_o
        M = self.M(x_tilde, f=f_rel)
        dx_dt = M.dot(f_rel) + v_o

        # # Enforce impenetrability
        # x_tilde_safe = x_tilde * self.safety_eta
        # n = self._get_normal(x_tilde_safe)
        # n_unit = n / (np.linalg.norm(n) + _EPS)

        # gamma_val = self.gamma(x_tilde)
        # effective_boundary = 1/self.safety_eta

        # if gamma_val <= effective_boundary:
        #     normal_vel = float(n_unit.dot(dx_dt))
        #     if normal_vel < 0:
        #         dx_dt = dx_dt - normal_vel * n_unit

        return dx_dt

    def M(self, x_tilde: np.ndarray, f: Optional[np.ndarray] = None) -> np.ndarray:
        x_tilde = np.asarray(x_tilde, float)
        x_tilde_safe = x_tilde * self.safety_eta
        E = self.E(x_tilde_safe)
        D = self.D(x_tilde, f=f)

        try:
            E_inv = np.linalg.inv(E)
        except np.linalg.LinAlgError:
            E_inv = np.linalg.pinv(E)

        return E.dot(D).dot(E_inv)

    # def E(self, x_tilde: np.ndarray) -> np.ndarray:
    #     n = self._get_normal(x_tilde)
    #     d = n.shape[0]
    #     n_unit = n / (np.linalg.norm(n) + _EPS)

    #     _, _, Vt = np.linalg.svd(n_unit.reshape(1, -1), full_matrices=True)
    #     tangents = Vt[1:, :]
    #     tangents = tangents.T

    #     # tangents = []
    #     # for i in range(d):
    #     #     ei = np.zeros(d)
    #     #     ei[i] = 1.0
    #     #     vi = ei - np.dot(ei, n_unit) * n_unit

    #     #     for t in tangents:
    #     #         vi = vi - np.dot(vi, t) * t

    #     #     norm_vi = np.linalg.norm(vi)
    #     #     if norm_vi > 1e-8:
    #     #         tangents.append(vi / norm_vi)
    #     #     if len(tangents) >= d - 1:
    #     #         break

    #     # if len(tangents) < d - 1:
    #     #     rand_mat = np.random.randn(d, d)
    #     #     Q, _ = np.linalg.qr(rand_mat)
    #     #     tangents = [Q[:, i] for i in range(d) if abs(np.dot(Q[:, i], n_unit)) < 1 - 1e-3]
    #     #     tangents = tangents[: max(0, d - 1)]

    #     E = np.column_stack([n_unit] + tangents[: d - 1])
    #     return E

    #alternate 
    def E(self, x_tilde: np.ndarray) -> np.ndarray:


        x_tilde = np.asarray(x_tilde, dtype=float)

        # --------------------------------------------------
        # 1. Get obstacle normal
        # --------------------------------------------------
        n = self._get_normal(x_tilde)

        d = n.shape[0]

        norm_n = np.linalg.norm(n)

        if norm_n < _EPS:
            raise ValueError(
                "Obstacle gradient is too small to define a normal."
            )

        n_unit = n / norm_n

        # --------------------------------------------------
        # 2. Construct an orthonormal basis using QR
        # --------------------------------------------------
        #
        # Start with the normal as the first column.
        # Add arbitrary independent vectors and orthogonalize.
        #
        A = np.eye(d)

        # Put normal as first column
        A[:, 0] = n_unit

        # Gram-Schmidt / QR decomposition
        Q, _ = np.linalg.qr(A)

        # Make sure first column has the correct normal direction
        if np.dot(Q[:, 0], n_unit) < 0:
            Q[:, 0] *= -1

        return Q

    def D(self, x_tilde: np.ndarray, f: Optional[np.ndarray] = None) -> np.ndarray:
        # gamma_raw = abs(float(self.gamma(x_tilde))) + _EPS
        
        #Inflate obstacle boundary
        x_tilde_safe = x_tilde * self.safety_eta
        gamma_val = abs(float(self.gamma(x_tilde_safe))) + _EPS
        d = x_tilde.shape[0]

        denom = gamma_val ** (1.0 / self.rho)
        lam_n = 1.0 - 1.0 / denom
        lam_tau = 1.0 + 1.0 / denom

        # Tail-effect
        if self.tail_effect and f is not None:
            x_tilde_safe = x_tilde * self.safety_eta
            n = self._get_normal(x_tilde_safe)
            n_unit = n / (np.linalg.norm(n) + _EPS)
            if float(np.dot(n_unit, f)) >= 0:
                lam_n = 1.0

        lam_n = max(lam_n, self.lambda_min)
        lam_tau = min(lam_tau, 1.0 + self.lambda_max_factor)

        return np.diag([lam_n] + [lam_tau] * (d - 1))

    # ---------------------- Helpers ----------------------

    def _get_normal(self, x_tilde: np.ndarray) -> np.ndarray:
        n = np.asarray(self.grad_gamma(x_tilde), float)
        if n.ndim != 1 or n.shape[0] != x_tilde.shape[0]:
            raise ValueError("grad_gamma must return a 1D vector of same dimension")
        return n

    # ---------------------- Static Utility Methods ----------------------

    @staticmethod
    def gamma_sphere(x: np.ndarray, center: np.ndarray, radius: float) -> float:
        return float(np.linalg.norm(x - center) / (radius + _EPS))

    @staticmethod
    def grad_gamma_sphere(x: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
        v = x - center
        return (v / (np.linalg.norm(v) + _EPS)) / (radius + _EPS)

    @staticmethod
    def gamma_ellipsoid(x: np.ndarray, center: np.ndarray, A: np.ndarray) -> float:
        v = x - center
        return float(np.sqrt(np.dot(v, A.dot(v)) + _EPS))
    
    @staticmethod
    def grad_gamma_ellipsoid(x: np.ndarray, center: np.ndarray, A: np.ndarray) -> np.ndarray:
        v = x - center
        return A.dot(v) / (np.sqrt(np.dot(v, A.dot(v)) + _EPS) + _EPS)
    
    @staticmethod
    def gamma_superellipsoid(x, center, radii, p):
        x = np.asarray(x)
        center = np.asarray(center)
        radii = np.asarray(radii)

        v = (x - center) / radii
        S = np.sum(np.abs(v) ** p)

        return (S + 1e-9) ** (1.0 / p)

    @staticmethod
    def grad_gamma_superellipsoid(x, center, radii, p):
        x = np.asarray(x)
        center = np.asarray(center)
        radii = np.asarray(radii)

        v = (x - center) / radii
        abs_v = np.abs(v)

        S = np.sum(abs_v ** p) + 1e-9
        coeff = S ** (1.0 / p - 1.0)

        term = (abs_v ** (p - 2)) * v
        grad = coeff * (term / radii)

        return grad
    


