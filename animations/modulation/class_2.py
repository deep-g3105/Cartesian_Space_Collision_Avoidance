from typing import Callable, Optional, List, Union
import numpy as np

_EPS = 1e-9


class ModulationAvoider:

    def __init__(
        self,
        gamma: Union[Callable[[np.ndarray], float], List[Callable[[np.ndarray], float]]],
        grad_gamma: Union[Callable[[np.ndarray], np.ndarray], List[Callable[[np.ndarray], np.ndarray]]],
        dim: Optional[int] = None,
        rho: float = 1.0,
        safety_eta: float = 1.0,
        tail_effect: bool = False,
        lambda_min: float = -10.0,
        lambda_max_factor: float = 100.0,
    ):
        assert rho > 0, "rho must be > 0"
        assert 0 < safety_eta <= 1.0, "safety_eta must be in (0,1]"

        self.gamma_list = gamma if isinstance(gamma, (list, tuple)) else [gamma]
        self.grad_gamma_list = (
            grad_gamma if isinstance(grad_gamma, (list, tuple)) else [grad_gamma]
        )

        assert len(self.gamma_list) == len(self.grad_gamma_list)

        self.K = len(self.gamma_list)
        self.rho = float(rho)
        self.safety_eta = float(safety_eta)
        self.tail_effect = bool(tail_effect)
        self.lambda_min = lambda_min
        self.lambda_max_factor = lambda_max_factor
        self.dim = dim

    # ---------------------- Public API ----------------------
    def modulate(self, x: np.ndarray, x_tilde: np.ndarray, f: np.ndarray) -> np.ndarray:
        """Compute modulated velocity dx_dt = M_bar(x_tilde) f(x)."""

        x = np.asarray(x, float)
        x_tilde = np.asarray(x_tilde, float)
        f = np.asarray(f, float)

        d = x.shape[0]
        if self.dim is None:
            self.dim = d
        assert d == self.dim, "x dimension mismatch"

        # Safety-inflated anchor
        x_tilde_safe = x_tilde / self.safety_eta

        M_bar = self.M_bar(x_tilde_safe)
        dx_dt = M_bar.dot(f)

        # Impenetrability (checked per obstacle)
        for k in range(self.K):
            n = self._get_normal_k(x_tilde_safe, k)
            n_unit = n / (np.linalg.norm(n) + _EPS)
            gamma_val = self.gamma_list[k](x_tilde_safe)

            if gamma_val <= 1.0:
                normal_vel = float(n_unit.dot(dx_dt))
                if normal_vel < 0:
                    dx_dt = dx_dt - normal_vel * n_unit

        return dx_dt

    # ---------------------- Modulation ----------------------
    def M_bar(self, x_tilde: np.ndarray) -> np.ndarray:
        """Product modulation matrix."""
        d = x_tilde.shape[0]
        M_bar = np.eye(d)

        for k in range(self.K):
            M_bar = M_bar @ self.M_k(x_tilde, k)

        return M_bar

    def M_k(self, x_tilde: np.ndarray, k: int) -> np.ndarray:
        E = self.E_k(x_tilde, k)
        gammas = np.array([g(x_tilde) for g in self.gamma_list])
        D = self.D_k(x_tilde, k, gammas)

        try:
            E_inv = np.linalg.inv(E)
        except np.linalg.LinAlgError:
            E_inv = np.linalg.pinv(E)

        return E @ D @ E_inv

    # ---------------------- Basis ----------------------
    def E_k(self, x_tilde: np.ndarray, k: int) -> np.ndarray:
        n = self._get_normal_k(x_tilde, k)
        d = n.shape[0]
        n_unit = n / (np.linalg.norm(n) + _EPS)

        tangents = []
        for i in range(d):
            ei = np.zeros(d)
            ei[i] = 1.0
            vi = ei - np.dot(ei, n_unit) * n_unit

            for t in tangents:
                vi = vi - np.dot(vi, t) * t

            norm_vi = np.linalg.norm(vi)
            if norm_vi > 1e-8:
                tangents.append(vi / norm_vi)
            if len(tangents) >= d - 1:
                break

        if len(tangents) < d - 1:
            Q, _ = np.linalg.qr(np.random.randn(d, d))
            tangents = [Q[:, i] for i in range(d)
                        if abs(np.dot(Q[:, i], n_unit)) < 1 - 1e-3]
            tangents = tangents[: d - 1]

        return np.column_stack([n_unit] + tangents[: d - 1])

    # ---------------------- Eigenvalues ----------------------
    def D_k(self, x_tilde: np.ndarray, k: int, gammas: np.ndarray) -> np.ndarray:
        d = x_tilde.shape[0]

        Gamma_k = gammas[k] + _EPS
        omega_k = self._omega_k(gammas, k)

        denom = Gamma_k ** (1.0 / self.rho)

        lam_n = 1.0 - omega_k / denom
        lam_tau = 1.0 + omega_k / denom

        lam_n = max(lam_n, self.lambda_min)
        lam_tau = min(lam_tau, 1.0 + self.lambda_max_factor)

        return np.diag([lam_n] + [lam_tau] * (d - 1))

    # ---------------------- Obstacle Weight ----------------------
    def _omega_k(self, gammas: np.ndarray, k: int) -> float:
        omega = 1.0
        for i in range(self.K):
            if i == k:
                continue
            omega *= (gammas[i] - 1.0) / (
                (gammas[k] - 1.0) + (gammas[i] - 1.0) + _EPS
            )
        return omega

    # ---------------------- Helpers ----------------------
    def _get_normal_k(self, x_tilde: np.ndarray, k: int) -> np.ndarray:
        n = np.asarray(self.grad_gamma_list[k](x_tilde), float)
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
