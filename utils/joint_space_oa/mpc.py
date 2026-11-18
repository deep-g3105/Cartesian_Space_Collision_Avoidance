"""
mpc.py

Navigation kernels and their sampling-based MPC (MPPI) optimisation, the
part of Koptev, Figueroa, Billard, "Reactive collision-free motion
generation in joint space via dynamical systems and sampling-based MPC",
IJRR 2024 (Secs. 5.2-6.4, 7.2-7.5) that escapes the local minima of the
plain modulation (f anti-parallel to the obstacle normal n).

Deflected modulated DS (11), (34):
    q_dot = M(q) (f(q) + alpha(q) M_perp(q) g(q)) + f_rep(q)

Navigation kernels (14):
    g(q) = sum_k g_k exp(-gamma_k ||q - q_k||)        centre q_k, direction g_k, width gamma_k = gamma_hat
activated by alpha(q) (15), (31) (JointSpaceModulator.activation).

MPPI over the kernel directions (16)-(19): every iteration
    1. sample N direction sets  g_{i,k} ~ N(mu_k, Sigma_k),  i = 1..N, k = 1..K
       (sample 0 is the current mean = the policy the controller runs)
    2. roll out (34)-(35) H steps for each set from the current state
    3. cost of each roll-out (20)-(27):
         c_goal = ||q* - q_{i,H}||                         (20)
         c_coll = sum_h [Gamma(q_{i,h}) <= 0]              (21)-(22)
         c_lim  = sum_h [q_{i,h} outside the joint limits] (23)-(24)
         c_stay = c_goal / max(eps_stay, ||q_{i,1} - q_{i,H}||)   (25)
         c_DS   = sum_h Gamma(q_{i,h}) ||q_dot_{i,h} - f(q_{i,h})||  (26)
         c_i    = sum_t w_t c_t                            (27)
       (no self-collision term: there is no Gamma_self here)
    4. w_i = exp(-(c_i - min c) / beta)                    (17) (shifted by min c for numerics)
       mu_k    <- (1 - a_mu) mu_k + a_mu sum_i w_i g_{i,k} / sum_i w_i              (18)
       Sigma_k <- (1 - a_S) Sigma_k + a_S sum_i w_i (g_{i,k}-mu_k)(g_{i,k}-mu_k)^T / sum_i w_i   (19)
       (Sigma_k is kept >= sigma_min^2 I so the search never collapses)
    5. kernel placement (28): a roll-out state q_{i,h} becomes a new kernel centre if
         ||q* - q_{i,h}|| > delta_*          far from the attractor
         Gamma(q_{i,h})   < delta_Gamma      close to an obstacle
         <n, f/||f||>     < -1 + delta_n     possible local minimum
         min_k ||q_{i,h} - q_k|| > delta_k   far from existing kernels
       its direction distribution is copied from the closest kernel (warm start),
       or N(0, sigma_init^2 I) for the first one.

Sample count (Sec. 7.5): a single roll-out (the current policy) is enough
to predict the motion and detect a local minimum; N samples are used only
while the predicted path passes through an active kernel.

Units follow the rest of the package: q in rad, Gamma in m, g in rad/s.
Algorithm 1 / Table 4 (appendix) are not in our copy of the paper; the
default hyper-parameters below are our own choices and are documented.
"""

import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .modulation import JointSpaceModulator

# Gamma(Q (N, d), centers, radii) -> gamma (N,), grad (N, d)
BatchDistance = Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]

DEFAULT_WEIGHTS = {"goal": 1.0, "coll": 10.0, "lim": 10.0, "stay": 1.0, "ds": 0.05}


class NavigationKernels:
    """g(q) = sum_k g_k exp(-width ||q - q_k||), Eq. (14).

    Args:
        dim:   joint-space dimension d.
        width: kernel width gamma_hat [1/rad] (same for all kernels, Sec. 6.2).
    """

    def __init__(self, dim: int, width: float = 3.0):
        self.dim, self.width = dim, width
        self.centers = np.zeros((0, dim))
        self.directions = np.zeros((0, dim))

    def __len__(self) -> int:
        return len(self.centers)

    def rbf(self, Q: np.ndarray) -> np.ndarray:
        """(N, d) -> (N, K) radial-basis activations."""
        return np.exp(-self.width * np.linalg.norm(Q[:, None] - self.centers[None], axis=-1))

    def __call__(self, q: np.ndarray, directions: Optional[np.ndarray] = None) -> np.ndarray:
        """g(q) for q (d,) -> (d,) or Q (N, d) -> (N, d).
        directions: (K, d) shared, or (N, K, d) per state (sampled sets); default the policy."""
        Q = np.atleast_2d(np.asarray(q, dtype=float))
        if len(self) == 0:
            out = np.zeros_like(Q)
        else:
            D = self.directions if directions is None else directions
            W = self.rbf(Q)
            out = W @ D if D.ndim == 2 else np.einsum("nk,nkd->nd", W, D)
        return out[0] if np.ndim(q) == 1 else out

    def add(self, center: np.ndarray, direction: np.ndarray):
        self.centers = np.vstack([self.centers, center])
        self.directions = np.vstack([self.directions, direction])

    def remove(self, k: int):
        self.centers = np.delete(self.centers, k, 0)
        self.directions = np.delete(self.directions, k, 0)

    def clear(self):
        self.centers = np.zeros((0, self.dim))
        self.directions = np.zeros((0, self.dim))


class KernelMPPI:
    """Places navigation kernels and optimises their directions with MPPI (Secs. 6.1-6.4).

    Call update() at the MPC rate (the paper: ~20 Hz, asynchronous) and
    policy() at the control rate (the paper: 500 Hz) - policy() uses the
    latest kernels and directions.

    Args:
        modulator:      the JointSpaceModulator the controller uses (same gains in the roll-outs).
        distance_batch: Gamma for a batch, e.g. JSDFCollisionDistance.batch / PlanarArmDistance.batch.
        dim:            joint-space dimension d.
        n_samples:      N roll-outs while a kernel is active (otherwise 1).
        horizon:        H roll-out steps.
        dt:             roll-out integration step [s] (coarser than the control step).
        temperature:    beta of Eq. (17).
        lr_mean, lr_cov: alpha_mu, alpha_Sigma of Eqs. (18)-(19).
        sigma_init:     initial std of a first kernel's direction [rad/s].
        sigma_min:      lower bound of the direction std [rad/s].
        select:         "mean" (mu_k, default) or "best" (lowest-cost sample) as the policy.
        kernel_width:   gamma_hat [1/rad].
        goal_radius:    r of the ball around q* without deflection [rad] (Eq. 15).
        place_goal, place_gamma, place_normal, place_spacing: delta_*, delta_Gamma [m], delta_n,
                        delta_k [rad] of Eq. (28b).
        weights:        cost weights w_t (Eq. 27), keys goal, coll, lim, stay, ds.
        stay_eps:       eps of Eq. (25) [rad].
        gamma_cap:      Gamma is capped at this value [m] in c_DS (Gamma = inf without obstacles).
        active_threshold: N samples are used while alpha * RBF > this along the predicted path.
        max_kernels:    the oldest kernel is dropped beyond this number.
        q_min, q_max:   (d,) joint limits for c_lim (roll-outs are clamped to them).
    """

    def __init__(self, modulator: JointSpaceModulator, distance_batch: BatchDistance, dim: int,
                 n_samples: int = 32, horizon: int = 40, dt: float = 0.04,
                 temperature: float = 0.1, lr_mean: float = 1.0, lr_cov: float = 0.3,
                 sigma_init: float = 1.0, sigma_min: float = 0.15, select: str = "mean",
                 kernel_width: float = 3.0, goal_radius: float = 0.1,
                 place_goal: float = 0.3, place_gamma: float = 0.05, place_normal: float = 0.5,
                 place_spacing: float = 0.5, weights: Optional[Dict[str, float]] = None,
                 stay_eps: float = 0.1, gamma_cap: float = 0.5, active_threshold: float = 0.05,
                 max_kernels: int = 30, q_min=None, q_max=None, seed: int = 0):
        self.mod, self.distance, self.dim = modulator, distance_batch, dim
        self.n_samples, self.horizon, self.dt = n_samples, horizon, dt
        self.temperature, self.lr_mean, self.lr_cov = temperature, lr_mean, lr_cov
        self.sigma_init, self.sigma_min = sigma_init, sigma_min
        assert select in ("mean", "best")
        self.select = select
        self.goal_radius = goal_radius
        self.place_goal, self.place_gamma = place_goal, place_gamma
        self.place_normal, self.place_spacing = place_normal, place_spacing
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.stay_eps, self.gamma_cap = stay_eps, gamma_cap
        self.active_threshold = active_threshold
        self.max_kernels = max_kernels
        self.q_min = None if q_min is None else np.asarray(q_min, dtype=float)
        self.q_max = None if q_max is None else np.asarray(q_max, dtype=float)
        self.rng = np.random.default_rng(seed)
        self.kernels = NavigationKernels(dim, kernel_width)
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self):
        """Forget all kernels (e.g. when the attractor changes)."""
        self.kernels.clear()
        self.mu = np.zeros((0, self.dim))
        self.Sigma = np.zeros((0, self.dim, self.dim))
        self.active = False
        self.iterations = 0
        self.last: Dict = {}

    @property
    def num_kernels(self) -> int:
        return len(self.kernels)

    # ------------------------------------------------------------------ #
    # control side: Eq. (34) inputs from the latest policy
    # ------------------------------------------------------------------ #
    def policy(self, q: np.ndarray, f: np.ndarray, gamma: float, grad: np.ndarray,
               q_goal: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """g(q) and alpha(q) for JointSpaceModulator.velocity / step."""
        gn = np.linalg.norm(grad)
        if self.num_kernels == 0 or not np.isfinite(gamma) or gn < 1e-9:
            return None, 0.0
        alpha = self.mod.activation(gamma, grad / gn, f, np.linalg.norm(q - q_goal), self.goal_radius)
        return self.kernels(q), alpha

    def policy_batch(self, Q: np.ndarray, F: np.ndarray, gamma: np.ndarray, grad: np.ndarray,
                     q_goal: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """policy() for N states (e.g. to draw the deflected vector field)."""
        if self.num_kernels == 0:
            return None, None
        gn = np.linalg.norm(grad, axis=1)
        valid = np.isfinite(gamma) & (gn > 1e-9)
        n = np.where(valid[:, None], grad / np.maximum(gn, 1e-9)[:, None], 0.0)
        alpha = self.mod.activation(gamma, n, F, np.linalg.norm(Q - q_goal, axis=1), self.goal_radius)
        return self.kernels(Q), np.where(valid, alpha, 0.0)

    # ------------------------------------------------------------------ #
    # MPC side: one MPPI iteration
    # ------------------------------------------------------------------ #
    def update(self, q0: np.ndarray, ds, centers: np.ndarray, radii: np.ndarray) -> Dict:
        """One MPPI iteration from the current state q0.

        Args:
            q0:      (d,) current (reference) configuration.
            ds:      nominal DS with .q_goal and .batch(Q) (LinearJointDS / CyclicJointDS).
            centers, radii: obstacle spheres, passed to distance_batch.
        Returns info: rollouts (H+1, N, d), cost (N,), best, n_samples, placed, ms.
        """
        t0 = time.perf_counter()
        q0 = np.asarray(q0, dtype=float)
        q_goal = ds.q_goal
        K, d, H = self.num_kernels, self.dim, self.horizon
        N = self.n_samples if (K > 0 and self.active) else 1

        # 1. sample direction sets, sample 0 = the mean (current policy)
        if K > 0:
            L = np.linalg.cholesky(self.Sigma)
            Z = self.rng.standard_normal((N, K, d))
            Z[0] = 0.0
            Gs = self.mu[None] + np.einsum("kij,nkj->nki", L, Z)
        else:
            Gs = np.zeros((N, 0, d))

        # 2. roll-outs of (34)-(35)
        Q = np.repeat(q0[None], N, 0)
        traj = np.empty((H + 1, N, d))
        traj[0] = Q
        c_coll, c_lim, c_ds = np.zeros(N), np.zeros(N), np.zeros(N)
        act0 = 0.0
        candidate = None
        for h in range(H):
            F = ds.batch(Q)
            gam, grad = self.distance(Q, centers, radii)
            gn = np.linalg.norm(grad, axis=1)
            valid = np.isfinite(gam) & (gn > 1e-9)
            n = np.where(valid[:, None], grad / np.maximum(gn, 1e-9)[:, None], 0.0)
            gd = np.linalg.norm(Q - q_goal, axis=1)
            G = alpha = None
            if K > 0:
                rbf = self.kernels.rbf(Q)
                G = np.einsum("nk,nkd->nd", rbf, Gs)
                alpha = np.where(valid, self.mod.activation(gam, n, F, gd, self.goal_radius), 0.0)
                act0 = max(act0, float(alpha[0] * rbf[0].max()))
            Qd, _ = self.mod.velocity_batch(F, gam, grad, G, alpha, dt=self.dt)

            gcap = np.where(np.isfinite(gam), np.clip(gam, 0.0, self.gamma_cap), self.gamma_cap)
            c_ds += gcap * np.linalg.norm(Qd - F, axis=1)                          # (26)
            c_coll += gam <= 0.0                                                   # (21)-(22)

            # 5. kernel placement candidates (28b), earliest state along any roll-out
            if candidate is None:
                fn = np.linalg.norm(F, axis=1)
                cos = (n * F).sum(1) / np.maximum(fn, 1e-9)
                ok = valid & (gd > self.place_goal) & (gam < self.place_gamma) & (cos < -1.0 + self.place_normal)
                if K > 0:
                    ok &= np.linalg.norm(Q[:, None] - self.kernels.centers[None], axis=-1).min(1) > self.place_spacing
                if ok.any():
                    candidate = Q[int(np.argmax(ok))].copy()

            Qn = Q + Qd * self.dt
            if self.q_min is not None:
                c_lim += np.any((Qn < self.q_min) | (Qn > self.q_max), axis=1)    # (23)-(24)
                Qn = np.clip(Qn, self.q_min, self.q_max)
            Q = Qn
            traj[h + 1] = Q

        # 3. costs (20), (25), (27)
        c_goal = np.linalg.norm(q_goal - Q, axis=1)
        c_stay = c_goal / np.maximum(self.stay_eps, np.linalg.norm(traj[1] - traj[-1], axis=1))
        w = self.weights
        cost = (w["goal"] * c_goal + w["coll"] * c_coll + w["lim"] * c_lim
                + w["stay"] * c_stay + w["ds"] * c_ds)
        best = int(np.argmin(cost))

        # 4. MPPI update of the direction distributions (17)-(19)
        if K > 0 and N > 1:
            wi = np.exp(-(cost - cost[best]) / self.temperature)
            wi /= wi.sum()
            mean = np.einsum("n,nkd->kd", wi, Gs)
            self.mu = (1.0 - self.lr_mean) * self.mu + self.lr_mean * mean
            dev = Gs - self.mu[None]
            cov = np.einsum("n,nki,nkj->kij", wi, dev, dev)
            self.Sigma = (1.0 - self.lr_cov) * self.Sigma + self.lr_cov * cov
            lo = np.linalg.eigvalsh(self.Sigma)[:, 0]
            self.Sigma += np.maximum(0.0, self.sigma_min ** 2 - lo)[:, None, None] * np.eye(d)
            self.kernels.directions = Gs[best].copy() if self.select == "best" else self.mu.copy()
        self.active = act0 > self.active_threshold

        # 5. add the new kernel, warm-started from the closest one
        placed = candidate is not None
        if placed:
            if K > 0:
                k_star = int(np.argmin(np.linalg.norm(self.kernels.centers - candidate, axis=1)))
                mu_new, S_new = self.mu[k_star].copy(), self.Sigma[k_star].copy()
            else:
                mu_new, S_new = np.zeros(d), self.sigma_init ** 2 * np.eye(d)
            self.kernels.add(candidate, mu_new)
            self.mu = np.vstack([self.mu, mu_new])
            self.Sigma = np.concatenate([self.Sigma, S_new[None]])
            if self.num_kernels > self.max_kernels:
                self.kernels.remove(0)
                self.mu, self.Sigma = self.mu[1:], self.Sigma[1:]
            self.active = True                   # explore the new kernel's direction next iteration

        self.iterations += 1
        self.last = {"rollouts": traj, "cost": cost, "best": best, "n_samples": N, "placed": placed,
                     "ms": 1e3 * (time.perf_counter() - t0), "costs": {
                         "goal": c_goal[best], "coll": c_coll[best], "lim": c_lim[best],
                         "stay": c_stay[best], "ds": c_ds[best]}}
        return self.last
