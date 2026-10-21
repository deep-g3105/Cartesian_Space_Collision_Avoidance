"""
planar_arm.py

Analytic joint-space distance Gamma(q) for a planar 2-link arm among circular
obstacles - the 2D counterpart of distance.JSDFCollisionDistance (same call
signature), used by the 2D animations to illustrate the joint-space DS
modulation of Koptev, Figueroa, Billard, IJRR 2024 (Figs. 2-5).

Links are capsules (segment + radius w). For link k (segment P_k(q, s)) and
circle s (centre c_s, radius r_s):

    Gamma_ks(q)  = min_s' ||P_k(q, s') - c_s|| - r_s - w
    Gamma(q)     = min_{k, s} Gamma_ks(q)                                 (IJRR Eq. 3)
    grad Gamma   = J_p(q)^T (p - c) / ||p - c||                            (envelope theorem)

with p = P_k(q, s*) the closest point on the closest link and J_p its
position Jacobian. So the learned network of the 7-DoF case is replaced here
by exact geometry; everything downstream (JointSpaceModulator) is identical.
"""

from typing import Dict, Tuple

import numpy as np


class PlanarArm:
    """Planar 2R arm: base at the origin, q = [theta1, theta2] (theta2 relative)."""

    def __init__(self, l1: float = 0.5, l2: float = 0.4, link_radius: float = 0.03):
        self.l1, self.l2, self.w = l1, l2, link_radius

    def fk(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Base, elbow and end-effector positions."""
        t1, t12 = q[0], q[0] + q[1]
        elbow = self.l1 * np.array([np.cos(t1), np.sin(t1)])
        ee = elbow + self.l2 * np.array([np.cos(t12), np.sin(t12)])
        return np.zeros(2), elbow, ee

    def ik(self, x: np.ndarray, elbow_up: bool = True) -> np.ndarray:
        """Joint angles reaching x (clamped to the reachable annulus)."""
        d2 = float(x @ x)
        c2 = np.clip((d2 - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2), -1.0, 1.0)
        t2 = np.arccos(c2) * (-1.0 if elbow_up else 1.0)
        t1 = np.arctan2(x[1], x[0]) - np.arctan2(self.l2 * np.sin(t2), self.l1 + self.l2 * np.cos(t2))
        return (np.array([t1, t2]) + np.pi) % (2 * np.pi) - np.pi

    def point_jacobian(self, q: np.ndarray, link: int, s: float) -> np.ndarray:
        """2x2 Jacobian of the point at arc length s along link 0 or 1."""
        t1, t12 = q[0], q[0] + q[1]
        J = np.zeros((2, 2))
        if link == 0:
            J[:, 0] = s * np.array([-np.sin(t1), np.cos(t1)])
        else:
            J[:, 0] = self.l1 * np.array([-np.sin(t1), np.cos(t1)]) + s * np.array([-np.sin(t12), np.cos(t12)])
            J[:, 1] = s * np.array([-np.sin(t12), np.cos(t12)])
        return J


def _closest_on_segments(a, b, c):
    """Closest points of segments a->b (..., 2) to points c (..., 2); returns point and t in [0, 1]."""
    ab = b - a
    t = np.clip(((c - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0.0, 1.0)
    return a + t[..., None] * ab, t


class PlanarArmDistance:
    """Gamma(q), grad Gamma(q) for a PlanarArm and circles (centres (S, 2), radii (S,))."""

    def __init__(self, arm: PlanarArm):
        self.arm = arm

    def __call__(self, q: np.ndarray, centers: np.ndarray, radii: np.ndarray
                 ) -> Tuple[float, np.ndarray, Dict]:
        q = np.asarray(q, dtype=float)
        if len(centers) == 0:
            return float("inf"), np.zeros(2), {"link": None, "sphere": None}
        base, elbow, ee = self.arm.fk(q)
        best = (np.inf, None)
        for k, (a, b, L) in enumerate(((base, elbow, self.arm.l1), (elbow, ee, self.arm.l2))):
            p, t = _closest_on_segments(a[None], b[None], centers)
            d = np.linalg.norm(p - centers, axis=-1) - radii - self.arm.w
            s = int(np.argmin(d))
            if d[s] < best[0]:
                best = (d[s], (k, s, p[s], t[s] * L))
        gamma, (k, s, p, arc) = best
        diff = p - centers[s]
        u = diff / max(np.linalg.norm(diff), 1e-9)
        grad = self.arm.point_jacobian(q, k, arc).T @ u
        return float(gamma), grad, {"link": f"link{k + 1}", "sphere": s, "point": p}

    def gamma_grid(self, Q: np.ndarray, centers: np.ndarray, radii: np.ndarray) -> np.ndarray:
        """Vectorized Gamma for many configurations Q (N, 2) -> (N,) (for joint-space maps)."""
        if len(centers) == 0:
            return np.full(len(Q), np.inf)
        t1, t12 = Q[:, 0], Q[:, 0] + Q[:, 1]
        elbow = self.arm.l1 * np.stack([np.cos(t1), np.sin(t1)], -1)
        ee = elbow + self.arm.l2 * np.stack([np.cos(t12), np.sin(t12)], -1)
        out = np.full(len(Q), np.inf)
        c = centers[None]                                            # (1, S, 2)
        for a, b in ((np.zeros_like(elbow), elbow), (elbow, ee)):
            p, _ = _closest_on_segments(a[:, None], b[:, None], c)    # (N, S, 2)
            d = (np.linalg.norm(p - c, axis=-1) - radii[None] - self.arm.w).min(1)
            out = np.minimum(out, d)
        return out
