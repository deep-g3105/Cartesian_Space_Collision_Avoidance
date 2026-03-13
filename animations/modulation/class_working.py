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
        lambda_min: float = 0.0,
        lambda_max_factor: float = 100.0,
        reference_point: Optional[np.ndarray] = None,
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
        self.reference_point = (np.asarray(reference_point, float) if reference_point is not None else None)

    # ---------------------- Public API ----------------------

    def modulate(self, x: np.ndarray, x_tilde: np.ndarray, f: np.ndarray) -> np.ndarray:
        """Compute modulated velocity dx_dt = M(x_tilde) f(x)."""

        x = np.asarray(x, float)
        x_tilde = np.asarray(x_tilde, float)
        f = np.asarray(f, float)

        d = x.shape[0]
        if self.dim is None:
            self.dim = d
        assert d == self.dim, "x dimension mismatch"

        # Safety-inflated modulation anchor
        # x_tilde_safe = x_tilde / self.safety_eta
        M = self.M(x_tilde, f=f)
        dx_dt = M.dot(f)

        # Enforce impenetrability
        x_tilde_safe = x_tilde * self.safety_eta
        n = self._get_normal(x_tilde_safe)
        n_unit = n / (np.linalg.norm(n) + _EPS)

        gamma_val = self.gamma(x_tilde)
        effective_boundary = 1/self.safety_eta

        if gamma_val <= effective_boundary:
            normal_vel = float(n_unit.dot(dx_dt))
            if normal_vel < 0:
                dx_dt = dx_dt - normal_vel * n_unit

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

    #     tangents = []
    #     for i in range(d):
    #         ei = np.zeros(d)
    #         ei[i] = 1.0
    #         vi = ei - np.dot(ei, n_unit) * n_unit

    #         for t in tangents:
    #             vi = vi - np.dot(vi, t) * t

    #         norm_vi = np.linalg.norm(vi)
    #         if norm_vi > 1e-8:
    #             tangents.append(vi / norm_vi)
    #         if len(tangents) >= d - 1:
    #             break

    #     if len(tangents) < d - 1:
    #         rand_mat = np.random.randn(d, d)
    #         Q, _ = np.linalg.qr(rand_mat)
    #         tangents = [Q[:, i] for i in range(d) if abs(np.dot(Q[:, i], n_unit)) < 1 - 1e-3]
    #         tangents = tangents[: max(0, d - 1)]

    #     E = np.column_stack([n_unit] + tangents[: d - 1])
    #     return E

    def E(self, x_tilde: np.ndarray) -> np.ndarray:
        x_tilde = np.asarray(x_tilde, float)
        d       = x_tilde.shape[0]

        if self.reference_point is not None:
            # [Huber2019 Eq.5] Reference-direction mode
            ref_vec  = x_tilde - self.reference_point
            norm_ref = np.linalg.norm(ref_vec)
            first_col = ref_vec / norm_ref if norm_ref > _EPS else np.eye(d)[:, 0]

            # Tangents ⊥ ∇Γ (required for impenetrability, Huber2019 App.A)
            grad    = self._get_normal(x_tilde)
            n_unit  = grad / (np.linalg.norm(grad) + _EPS)
            tangents = _gram_schmidt_tangents(n_unit, d)
        else:
            # [KB2012 Eqs.12-14] Gradient-normal mode
            n        = self._get_normal(x_tilde)
            n_unit   = n / (np.linalg.norm(n) + _EPS)
            first_col = n_unit
            tangents  = _gram_schmidt_tangents(n_unit, d)

        return np.column_stack([first_col] + tangents[: d - 1])

    def D(self, x_tilde: np.ndarray, f: Optional[np.ndarray] = None) -> np.ndarray:
        
        
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
    # ── [Huber2019 Eq.3] ──────────────────────────────────────────────────────────
    @staticmethod
    def gamma_star_shaped(
        x: np.ndarray,
        reference_point: np.ndarray,
        radius_func: Callable[[np.ndarray], float],
    ) -> float:
        """
        Γ(x̃) = ‖x̃ - ξ^r‖ / R(θ)   [Huber2019 Eq.3]
        Γ = 1 on boundary, > 1 outside, < 1 inside.
        radius_func returns R(θ) given x̃ (obstacle-relative position).
        """
        x   = np.asarray(x,               float)
        ref = np.asarray(reference_point, float)
        dist = np.linalg.norm(x - ref) + _EPS
        R    = float(radius_func(x)) + _EPS
        return dist / R

    @staticmethod
    def grad_gamma_star_shaped(
        x: np.ndarray,
        reference_point: np.ndarray,
        R_base: float,
        amp: float,
        n: int,
    ) -> np.ndarray:
        """
        ∇Γ for polar star boundary R(θ) = R_base · (1 + amp · cos(n · θ))
        [Huber2019 Eq.3 gradient]

        Derivation:
        Γ = r / R(θ),   r = ‖x̃‖,   θ = atan2(ỹ, x̃)
        ∂Γ/∂x̃ = x̃/(r·R)  +  (dR/dθ / R²·r) · [ỹ, -x̃]
        The second term corrects for the non-circular boundary curvature.
        """
        p     = np.asarray(x, float) - np.asarray(reference_point, float)
        r     = np.linalg.norm(p) + _EPS
        theta = float(np.arctan2(p[1], p[0]))
        R     = R_base * (1.0 + amp * np.cos(n * theta))
        dRdt  = -R_base * amp * n * np.sin(n * theta)
        perp  = np.array([p[1], -p[0]])   # x̃ rotated 90° clockwise
        return p / (r * R) + (dRdt / (R**2 * r)) * perp
    
def _gram_schmidt_tangents(normal: np.ndarray, d: int):
        tangents = []
        for i in range(d):
            ei = np.zeros(d); ei[i] = 1.0
            vi = ei - np.dot(ei, normal) * normal
            for t in tangents:
                vi = vi - np.dot(vi, t) * t
            norm_vi = np.linalg.norm(vi)
            if norm_vi > 1e-8:
                tangents.append(vi / norm_vi)
            if len(tangents) >= d - 1:
                break
        if len(tangents) < d - 1:
            Q, _ = np.linalg.qr(np.random.randn(d, d))
            extra = [Q[:, i] for i in range(d)
                    if abs(np.dot(Q[:, i], normal)) < 1 - 1e-3]
            tangents = (tangents + extra)[: d - 1]
        return tangents[: d - 1]    