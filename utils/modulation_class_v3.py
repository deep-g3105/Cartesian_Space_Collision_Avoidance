"""
modulation_class_v3.py

Implements the modulation approach of:
    L. Huber, A. Billard, J.-J. Slotine,
    "Avoidance of Convex and Concave Obstacles With Convergence Ensured
    Through Contraction", IEEE RA-L, 2019.

Key departure from v2 (Khansari & Billard, 2012):
    v2's E(x) uses the surface normal n(x) = grad(gamma)/|grad(gamma)| as
    its FIRST column, and builds the tangent directions as the orthogonal
    complement of that same normal -> E is fully orthonormal, but this
    restricts the obstacle to be convex (star-shaped only if the reference
    point is the centroid).

    Huber decouples the two roles:
      - the tangent directions e_1..e_{d-1} are still the orthogonal
        complement of the gradient (the normal direction), so the boundary
        is still respected locally,
      - but the FIRST column of E is r(x) = (x - x_r)/||x - x_r||, the
        direction from a chosen REFERENCE POINT x_r inside the obstacle,
        not the local normal.

    Because r(x) and the local normal need not be parallel, E(x) is in
    general non-orthogonal but still invertible (as long as x_r is chosen
    so every ray from x_r crosses the boundary exactly once, i.e. the
    obstacle is *star-shaped* w.r.t. x_r). This one change is what makes
    the method valid for non-convex, concave, star-shaped obstacles, and
    (via the multi-obstacle interpolation below) for concave regions
    formed by intersecting/unioned convex obstacles that share one common
    reference point (Sec. IV-B / Theorem 3 of the paper).

Contents:
    HuberModulationAvoider   - single-obstacle modulation (Eqs. 4-8, 17-18)
    MultiObstacleHuberAvoider- weighted magnitude + kappa-space direction
                               interpolation for several obstacles (Eqs.
                               11-16), required whenever a concave region
                               is built from several convex pieces
    Helper / utility functions:
        numerical_gradient            - generic central-difference gradient
        gamma_star_from_radius        \
        grad_gamma_star_from_radius    } arbitrary star-shaped obstacle
        radius_function_polygon_2d    / defined by a radius-vs-direction
                                         function (supports concave shapes,
                                         e.g. star polygons in 2D)
        is_valid_reference_point      - sampling-based check that x_r sees
                                         the boundary exactly once per ray
        gamma_sphere / grad_gamma_sphere
        gamma_ellipsoid / grad_gamma_ellipsoid
        gamma_superellipsoid / grad_gamma_superellipsoid
            (same convex primitives as v2, reused unchanged)

What I still need from you to finish wiring this to your setup:
    1. How you represent obstacle geometry for the concave objects you
       care about (mesh? polygon vertices? analytic radius function?) -
       radius_function_polygon_2d only covers 2D polygons right now.
    2. Whether obstacles move (HEAL/Franka links, human arms) - if so I
       need their linear/angular velocity source so modulate_moving() can
       be hooked into your control loop the same way v2's v_obs is today.
    3. Confirm the reference-point selection strategy per obstacle (fixed
       centroid vs. something you compute per obstacle instance) so
       is_valid_reference_point can be wired into your obstacle setup code
       as a sanity check.
"""

from typing import Callable, List, Optional, Sequence, Union
import numpy as np

_EPS = 1e-9

RefPoint = Union[np.ndarray, Callable[[], np.ndarray]]


def _resolve_ref_point(ref: RefPoint) -> np.ndarray:
    return np.asarray(ref(), float) if callable(ref) else np.asarray(ref, float)


def _orthonormal_complement(unit_vec: np.ndarray) -> np.ndarray:
    """Columns 1..d-1 of a QR-completed basis whose first column is unit_vec.
    Returns only the d-1 complement columns (shape d x (d-1))."""
    d = unit_vec.shape[0]
    A = np.eye(d)
    A[:, 0] = unit_vec
    Q, _ = np.linalg.qr(A)
    if np.dot(Q[:, 0], unit_vec) < 0:
        Q[:, 0] *= -1
    return Q[:, 1:]


def numerical_gradient(func: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Generic central-difference gradient. Use this for any gamma(x) whose
    analytic gradient is inconvenient (e.g. ray-cast/polygon-based gammas)."""
    x = np.asarray(x, float)
    grad = np.zeros_like(x)
    for i in range(x.shape[0]):
        dx = np.zeros_like(x)
        dx[i] = h
        grad[i] = (func(x + dx) - func(x - dx)) / (2 * h)
    return grad


# ---------------------- Single obstacle ----------------------

class HuberModulationAvoider:
    """Single-obstacle modulation per Huber et al. 2019.

    gamma / grad_gamma follow the same contract as in v2: gamma(x) -> float,
    grad_gamma(x) -> (d,) vector, with gamma(x) = 1 on the boundary and
    increasing away from the obstacle.

    reference_point is the key new input: a point x_r strictly inside the
    obstacle (or a zero-arg callable returning one, for a moving obstacle)
    such that every ray from x_r crosses the boundary exactly once (star-
    shaped condition, Sec. III-C / III-B). Use is_valid_reference_point to
    check a candidate.
    """

    def __init__(
        self,
        gamma: Callable[[np.ndarray], float],
        grad_gamma: Callable[[np.ndarray], np.ndarray],
        reference_point: RefPoint,
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
        self.reference_point = reference_point
        self.rho = float(rho)
        self.safety_eta = float(safety_eta)
        self.tail_effect = bool(tail_effect)
        self.lambda_min = lambda_min
        self.lambda_max_factor = lambda_max_factor
        self.dim = dim

    # ---------------------- Public API ----------------------

    def modulate(self, x: np.ndarray, f: np.ndarray, v_obs: Optional[np.ndarray] = None) -> np.ndarray:
        """Static-obstacle modulation: dx_dt = M(x)(f(x) - v_obs) + v_obs."""
        x = np.asarray(x, float)
        f = np.asarray(f, float)
        v_o = np.zeros_like(f) if v_obs is None else np.asarray(v_obs, float)

        d = x.shape[0]
        if self.dim is None:
            self.dim = d
        assert d == self.dim, "x dimension mismatch"

        f_rel = f - v_o
        M = self.M(x, f=f_rel)
        return M.dot(f_rel) + v_o

    def modulate_moving(
        self,
        x: np.ndarray,
        f: np.ndarray,
        obstacle_center: np.ndarray,
        obstacle_lin_vel: np.ndarray,
        obstacle_ang_vel: Optional[np.ndarray] = None,
        clip_outward_only: bool = True,
        tangential_damping: float = 1.0,
    ) -> np.ndarray:
        """Moving-obstacle modulation, Eq. (17): modulation is done in the
        obstacle frame, i.e. relative to the obstacle's instantaneous rigid
        velocity, then the obstacle velocity is added back.

        obstacle_ang_vel: for d==2 a scalar (or length-1 array) angular rate,
        for d==3 an (3,) angular-velocity vector; omit for pure translation.
        clip_outward_only implements Sec. V-A: the obstacle's velocity
        component along the outward normal is only carried over when it is
        moving away from x_r (>=0), avoiding the robot getting dragged along
        with a passing obstacle.
        """
        x = np.asarray(x, float)
        f = np.asarray(f, float)
        obstacle_center = np.asarray(obstacle_center, float)
        obstacle_lin_vel = np.asarray(obstacle_lin_vel, float)
        d = x.shape[0]

        x_rel = x - obstacle_center
        if obstacle_ang_vel is None:
            v_transport = obstacle_lin_vel.copy()
        elif d == 2:
            omega = float(np.asarray(obstacle_ang_vel).reshape(-1)[0])
            # 2D "cross product" of a scalar omega with a planar vector
            rot = omega * np.array([-x_rel[1], x_rel[0]])
            v_transport = obstacle_lin_vel + rot
        elif d == 3:
            v_transport = obstacle_lin_vel + np.cross(np.asarray(obstacle_ang_vel, float), x_rel)
        else:
            raise ValueError("obstacle_ang_vel only supported for d in {2, 3}")

        if clip_outward_only:
            n = self.grad_gamma(x)
            n_unit = n / (np.linalg.norm(n) + _EPS)
            v_n = float(np.dot(v_transport, n_unit))
            if v_n < 0:
                v_transport = v_transport - v_n * n_unit  # remove inward component

        if tangential_damping < 1.0:
            n = self.grad_gamma(x)
            n_unit = n / (np.linalg.norm(n) + _EPS)
            v_n_component = float(np.dot(v_transport, n_unit)) * n_unit
            v_tangential = v_transport - v_n_component
            v_transport = v_n_component + tangential_damping * v_tangential

        return self.modulate(x, f, v_obs=v_transport)

    def _apply_safety(self, x: np.ndarray) -> np.ndarray:
        x_r = _resolve_ref_point(self.reference_point)
        return x_r + (x - x_r) * self.safety_eta

    def M(self, x: np.ndarray, f: Optional[np.ndarray] = None) -> np.ndarray:
        x = np.asarray(x, float)
        x_safe = self._apply_safety(x)
        E = self.E(x_safe)
        D = self.D(x, f=f)
        try:
            E_inv = np.linalg.inv(E)
        except np.linalg.LinAlgError:
            E_inv = np.linalg.pinv(E)
        return E.dot(D).dot(E_inv)

    def E(self, x: np.ndarray) -> np.ndarray:
        """Eq. (5): E(x) = [r(x), e_1(x), .., e_{d-1}(x)].

        r(x) is the reference direction from x_r (NOT the surface normal);
        the tangents e_i still span the orthogonal complement of the
        gradient of gamma. This is the sole structural change from v2's E().
        """
        x_r = _resolve_ref_point(self.reference_point)
        rel = x - x_r
        rel_norm = np.linalg.norm(rel)
        if rel_norm < _EPS:
            raise ValueError("x coincides with the reference point; r(x) is undefined there.")
        r_unit = rel / rel_norm

        n = self._get_normal(x)
        n_norm = np.linalg.norm(n)
        if n_norm < _EPS:
            raise ValueError("Obstacle gradient is too small to define a normal.")
        n_unit = n / n_norm

        tangents = _orthonormal_complement(n_unit)
        return np.column_stack([r_unit, tangents])

    def D(self, x: np.ndarray, f: Optional[np.ndarray] = None) -> np.ndarray:
        """Eq. (6), (8): D(x) = diag(lambda_r, lambda_e, .., lambda_e),
        generalized with the same rho exponent used in v2 so gamma(x)**(1/rho)
        replaces the plain gamma(x) of the original paper (rho=1 recovers it)."""
        x_safe = self._apply_safety(x)
        gamma_val = abs(float(self.gamma(x_safe))) + _EPS
        d = x.shape[0]

        denom = gamma_val ** (1.0 / self.rho)
        lam_r = 1.0 - 1.0 / denom
        lam_tau = 1.0 + 1.0 / denom

        if self.tail_effect and f is not None:
            n = self._get_normal(x_safe)
            n_unit = n / (np.linalg.norm(n) + _EPS)
            if float(np.dot(n_unit, f)) >= 0:
                lam_r = 1.0

        lam_r = max(lam_r, self.lambda_min)
        lam_tau = min(lam_tau, 1.0 + self.lambda_max_factor)

        return np.diag([lam_r] + [lam_tau] * (d - 1))

    # ---------------------- Helpers ----------------------

    def _get_normal(self, x: np.ndarray) -> np.ndarray:
        n = np.asarray(self.grad_gamma(x), float)
        if n.ndim != 1 or n.shape[0] != x.shape[0]:
            raise ValueError("grad_gamma must return a 1D vector of same dimension")
        return n

    # ---------------------- Static convex primitives (same as v2) ----------------------

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
        x, center, radii = np.asarray(x), np.asarray(center), np.asarray(radii)
        v = (x - center) / radii
        S = np.sum(np.abs(v) ** p)
        return (S + 1e-9) ** (1.0 / p)

    @staticmethod
    def grad_gamma_superellipsoid(x, center, radii, p):
        x, center, radii = np.asarray(x), np.asarray(center), np.asarray(radii)
        v = (x - center) / radii
        abs_v = np.abs(v)
        S = np.sum(abs_v ** p) + 1e-9
        coeff = S ** (1.0 / p - 1.0)
        term = (abs_v ** (p - 2)) * v
        return coeff * (term / radii)

    # ---------------------- Star-shaped / concave obstacles ----------------------

    @staticmethod
    def gamma_star_from_radius(
        x: np.ndarray, reference_point: np.ndarray, radius_func: Callable[[np.ndarray], float], p: int = 1
    ) -> float:
        """Eq. (3) generalized to an arbitrary radius function R(direction):
        Gamma(x) = (||x - x_r|| / R(d))^(2p), d = unit direction from x_r to x.

        radius_func(unit_direction) -> distance from x_r to the boundary in
        that direction. Because radius_func can be anything (a lookup table,
        a ray-cast against a mesh/polygon, a Fourier series in angle, ...),
        this single helper covers arbitrary star-shaped obstacles, convex or
        concave, as long as x_r sees the whole boundary (one crossing/ray).
        """
        x = np.asarray(x, float)
        reference_point = np.asarray(reference_point, float)
        rel = x - reference_point
        dist = np.linalg.norm(rel)
        if dist < _EPS:
            return 0.0
        d_unit = rel / dist
        R = float(radius_func(d_unit))
        return (dist / (R + _EPS)) ** (2 * p)

    @staticmethod
    def grad_gamma_star_from_radius(
        x: np.ndarray,
        reference_point: np.ndarray,
        radius_func: Callable[[np.ndarray], float],
        p: int = 1,
        h: float = 1e-6,
    ) -> np.ndarray:
        """Finite-difference gradient of gamma_star_from_radius. Used by
        default because radius_func is often only piecewise differentiable
        (e.g. a polygon ray-cast) so no closed form is assumed."""
        f = lambda xx: HuberModulationAvoider.gamma_star_from_radius(xx, reference_point, radius_func, p)
        return numerical_gradient(f, np.asarray(x, float), h=h)

    @staticmethod
    def radius_function_polygon_2d(unit_direction: np.ndarray, reference_point: np.ndarray, vertices: np.ndarray) -> float:
        """Ray-cast from reference_point along unit_direction against a
        closed 2D polygon (vertices in order, shape (N, 2), CW or CCW).
        Returns the distance to the nearest edge intersection.

        Works directly for concave / star-shaped polygons: pass this as
        radius_func into gamma_star_from_radius, together with the same
        reference_point, to get gamma/grad_gamma for that concave shape.
        (reference_point must be a point from which every polygon edge is
        visible along a single ray - check with is_valid_reference_point.)
        """
        p0 = np.asarray(reference_point, float)
        d = np.asarray(unit_direction, float)
        d = d / (np.linalg.norm(d) + _EPS)
        n = vertices.shape[0]
        best_t = np.inf
        for i in range(n):
            a = vertices[i]
            b = vertices[(i + 1) % n]
            edge = b - a
            # Solve p0 + t*d = a + s*edge, t >= 0, s in [0,1]
            denom = d[0] * edge[1] - d[1] * edge[0]
            if abs(denom) < _EPS:
                continue
            ap = a - p0
            t = (ap[0] * edge[1] - ap[1] * edge[0]) / denom
            s = (ap[0] * d[1] - ap[1] * d[0]) / denom
            if t >= 0 and -1e-9 <= s <= 1 + 1e-9:
                best_t = min(best_t, t)
        if not np.isfinite(best_t):
            raise ValueError("Ray from reference_point does not hit the polygon boundary; "
                              "reference_point is likely outside the polygon or not star-shaped in this direction.")
        return float(best_t)

    @staticmethod
    def is_valid_reference_point(
        gamma: Callable[[np.ndarray], float],
        reference_point: np.ndarray,
        dim: int,
        n_samples: int = 200,
        t_max: float = 100.0,
        n_steps: int = 2000,
    ) -> bool:
        """Sampling-based sanity check for the star-shaped condition
        (Sec. III-B/C): march outward from reference_point along many random
        directions and count sign changes of (gamma - 1). A valid reference
        point should give exactly one crossing per ray. Returns True iff
        every sampled ray has exactly one crossing.

        This is a heuristic check (finite samples, finite step size), not a
        proof - use it to catch obviously bad reference points, e.g. ones
        outside the obstacle or ones for which the obstacle is not actually
        star-shaped.
        """
        reference_point = np.asarray(reference_point, float)
        if gamma(reference_point) >= 1.0:
            return False  # must start inside the obstacle

        rng = np.random.default_rng(0)
        ts = np.linspace(t_max / n_steps, t_max, n_steps)
        for _ in range(n_samples):
            d = rng.normal(size=dim)
            d = d / (np.linalg.norm(d) + _EPS)
            prev = gamma(reference_point) - 1.0
            crossings = 0
            for t in ts:
                cur = gamma(reference_point + t * d) - 1.0
                if prev < 0 <= cur or prev >= 0 > cur:
                    crossings += 1
                prev = cur
            if crossings != 1:
                return False
        return True


# ---------------------- Multiple obstacles ----------------------

class MultiObstacleHuberAvoider:
    """Weighted combination of several obstacles' modulated velocities,
    Eqs. (11)-(16). Required whenever a concave region is represented as
    several convex (or star-shaped) HuberModulationAvoider pieces that share
    ONE common reference_point (Theorem 3 / Sec. IV-B) - e.g. two ellipsoids
    modeling intersecting human forearms, or a robot approximated by two
    convex hulls with a shared reference point (Fig. 3b in the paper).

    All avoiders passed in should therefore be built with the *same*
    reference_point (or reference points that all coincide at the moment of
    contact) for the convergence guarantee (Theorem 3) to apply.
    """

    def __init__(self, avoiders: Sequence[HuberModulationAvoider]):
        assert len(avoiders) >= 1
        self.avoiders = list(avoiders)

    # ---------------------- Public API ----------------------

    def modulate(self, x: np.ndarray, f: np.ndarray, v_obs: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
        x = np.asarray(x, float)
        f = np.asarray(f, float)
        n_obs = len(self.avoiders)
        v_obs_list = [None] * n_obs if v_obs is None else list(v_obs)

        gammas = np.array([abs(float(a.gamma(x))) + _EPS for a in self.avoiders])
        weights = self._weights(gammas)

        xdots = np.stack([
            a.modulate(x, f, v_obs=v_obs_list[i]) for i, a in enumerate(self.avoiders)
        ])  # (n_obs, d)

        magnitude = float(np.sum(weights * np.linalg.norm(xdots, axis=1)))  # Eq. (13)
        n_bar = self._direction_interpolation(f, xdots, weights)  # Eqs. (14)-(16)
        return n_bar * magnitude

    # ---------------------- Internals ----------------------

    @staticmethod
    def _weights(gammas_minus_zero: np.ndarray) -> np.ndarray:
        """Eq. (11)-(12): w_o = prod_{i != o}(Gamma_i - 1) / sum_k prod_{i != k}(Gamma_i - 1),
        with the boundary special case w_o = 1 on obstacle o's own boundary."""
        d = gammas_minus_zero - 1.0  # (Gamma_i - 1), >= 0 outside, 0 on boundary
        n = d.shape[0]

        near_boundary = d < 1e-8
        if np.any(near_boundary):
            w = np.zeros(n)
            # if several are ~0 simultaneously, split evenly between them
            w[near_boundary] = 1.0 / np.sum(near_boundary)
            return w

        prods = np.array([np.prod(np.delete(d, o)) for o in range(n)])
        total = np.sum(prods)
        if total < _EPS:
            return np.ones(n) / n
        return prods / total

    @staticmethod
    def _direction_interpolation(f: np.ndarray, xdots: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Eqs. (14)-(16): interpolate directions on the log-map (kappa) of
        the (d-1)-sphere centered on f(x)'s direction, then map back."""
        d = f.shape[0]
        f_norm = np.linalg.norm(f)
        if f_norm < _EPS:
            # No preferred axis; fall back to a plain weighted vector mean.
            mean = np.sum(weights[:, None] * xdots, axis=0)
            mnorm = np.linalg.norm(mean)
            return mean / mnorm if mnorm > _EPS else np.zeros(d)

        f_unit = f / f_norm
        Rf = np.column_stack([f_unit, _orthonormal_complement(f_unit)])  # (d, d)

        kappas = []
        for xdot in xdots:
            xdot_norm = np.linalg.norm(xdot)
            if xdot_norm < _EPS:
                kappas.append(np.zeros(d - 1))
                continue
            n_hat = xdot / xdot_norm
            local = Rf.T.dot(n_hat)  # local[0] = cos(theta), local[1:] has norm sin(theta)
            cos_t = np.clip(local[0], -1.0, 1.0)
            theta = np.arccos(cos_t)
            sin_t = np.linalg.norm(local[1:])
            if sin_t < 1e-8:
                kappas.append(np.zeros(d - 1))
            else:
                kappas.append(theta / sin_t * local[1:])
        kappas = np.stack(kappas)  # (n_obs, d-1)

        kappa_bar = np.sum(weights[:, None] * kappas, axis=0)  # Eq. (14), linear mean in tangent space
        theta_bar = np.linalg.norm(kappa_bar)
        if theta_bar < 1e-8:
            local_bar = np.zeros(d)
            local_bar[0] = 1.0
        else:
            local_bar = np.concatenate([[np.cos(theta_bar)], np.sin(theta_bar) / theta_bar * kappa_bar])
        return Rf.dot(local_bar)  # Eq. (15), n_bar