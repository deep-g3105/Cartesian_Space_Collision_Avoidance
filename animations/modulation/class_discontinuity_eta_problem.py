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
        tail_effect: bool = False,
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
        x_tilde_safe = x_tilde / self.safety_eta 
        
        # Calculate Modulated Velocity
        M = self.M(x_tilde_safe, f=f)
        dx_dt = M.dot(f)

        # --- Impenetrability Part (Hard Safety Constraint) ---
        # While modulation theoretically handles this, we add a hard clamp
        # to handle numerical precision issues near the boundary.
        gamma_val = self.gamma(x_tilde_safe)
        
        # Check if we are at (or inside) the boundary
        if gamma_val <= 1.0 + 1e-2:
            n = self._get_normal(x_tilde_safe)
            n_unit = n / (np.linalg.norm(n) + _EPS)
            
            # Project velocity onto the normal
            normal_vel = float(n_unit.dot(dx_dt))
            
            # If velocity is pointing INTO the obstacle, remove that component
            if normal_vel < 0:
                dx_dt = dx_dt - normal_vel * n_unit

        return dx_dt

    def M(self, x_tilde: np.ndarray, f: Optional[np.ndarray] = None) -> np.ndarray:
        x_tilde = np.asarray(x_tilde, float)
        E = self.E(x_tilde)
        D = self.D(x_tilde, f=f)

        # For non-orthonormal basis, we must use the inverse, not transpose.
        try:
            E_inv = np.linalg.inv(E)
        except np.linalg.LinAlgError:
            E_inv = np.linalg.pinv(E)

        return E.dot(D).dot(E_inv)

    def E(self, x_tilde: np.ndarray) -> np.ndarray:
        """
        Constructs the Non-Orthonormal Basis Matrix E.
        Column 0: Reference vector r(x) (prevents spurious attractor)
        Columns 1..d: Tangent vectors e(x) (orthogonal to normal n(x))
        """
        d = x_tilde.shape[0]
        
        # --- 1. Reference Vector r(x) ---
        # Vector from obstacle center to robot (Slide 25)
        r_vec = x_tilde / (np.linalg.norm(x_tilde) + _EPS)

        # --- 2. Tangent Vectors e(x) ---
        # These must be orthogonal to the NORMAL n(x), not r(x).
        n = self._get_normal(x_tilde)
        n_unit = n / (np.linalg.norm(n) + _EPS)

        # Stable Tangent Generation (Fixes "weird potential fields" / jitter)
        if d == 2:
            # Analytic 2D tangent [-y, x] is smooth and continuous
            tangents = [np.array([-n_unit[1], n_unit[0]])]
        else:
            # Stable 3D tangents using cross-product with a fixed axis
            k = np.array([0.0, 0.0, 1.0])
            # If normal is parallel to k, swap k to avoid singularity
            if abs(np.dot(n_unit, k)) > 0.99:
                k = np.array([0.0, 1.0, 0.0])
            
            t1 = np.cross(n_unit, k)
            t1 /= np.linalg.norm(t1) + _EPS
            t2 = np.cross(n_unit, t1)
            tangents = [t1, t2]

        # Stack columns: [r(x), e1(x), e2(x)...]
        E = np.column_stack([r_vec] + tangents)
        return E

    def D(self, x_tilde: np.ndarray, f: Optional[np.ndarray] = None) -> np.ndarray:
        gamma_val = abs(float(self.gamma(x_tilde))) + _EPS
        d = x_tilde.shape[0]

        denom = gamma_val ** (1.0 / self.rho)
        lam_n = 1.0 - 1.0 / denom
        lam_tau = 1.0 + 1.0 / denom

        # Tail-effect logic (Preserved)
        if self.tail_effect and f is not None:
            n = self._get_normal(x_tilde)
            n_unit = n / (np.linalg.norm(n) + _EPS)
            # If velocity is moving away from the obstacle, disable repulsion
            if float(np.dot(n_unit, f)) >= 0:
                lam_n = 1.0
                # Ideally lam_tau should also be 1.0 here to fully disable modulation,
                # but following your snippet we keep lam_tau logic or ensure minimal impact.
                lam_tau = 1.0 

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