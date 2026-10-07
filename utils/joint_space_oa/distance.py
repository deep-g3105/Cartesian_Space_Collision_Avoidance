"""
distance.py

Joint-space distance-to-collision Gamma(q) and its gradient from the learned
neural joint-space SDF (utils/sdf, Koptev et al. RA-L 2023), as used by the
modulation of Koptev et al. IJRR 2024, Eq. (3):

    Gamma(q)      = min_{s} min_{k} ( Gamma_k(q, y_s) - r_s )
    grad Gamma(q) = d Gamma(q) / d q            (repulsive direction, R^d)

for obstacles given as S spheres (centres y_s, radii r_s; a point cloud is
r_s = 0). All S x K distances come from one batched network pass, the
gradient from one backward pass through the minimum.

The hard minimum (paper) has a gradient that jumps when the closest
link/sphere pair changes. Optionally a soft minimum
    Gamma_beta(q) = -1/beta * log sum_{k,s} exp(-beta * Gamma_ks)
gives a smooth gradient; it is always <= the hard minimum (conservative)
and converges to it as beta -> inf.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from utils.sdf import JSDFNet


class JSDFCollisionDistance:
    """Gamma(q) and grad Gamma(q) for a set of spherical obstacles.

    Args:
        checkpoint:   path to a trained JSDFNet checkpoint (neural_sdf/runs/*/best.pt).
        device:       torch device ("cuda" recommended for many spheres).
        softmin_beta: None -> hard minimum (paper); else soft minimum sharpness [1/m].
        ignore_links: link names excluded from the minimum (e.g. ["link0"]).
        link_margins: {link name: extra margin [m]} subtracted from that link's
                      distance, e.g. {"hand": 0.02} for the gripper fingers that
                      are not part of the learned SDF.
    """

    def __init__(self, checkpoint: str, device: str = "cpu",
                 softmin_beta: Optional[float] = None,
                 ignore_links: Sequence[str] = (), link_margins: Optional[Dict[str, float]] = None):
        self.device = torch.device(device)
        self.net, ckpt = JSDFNet.load(checkpoint, map_location=self.device)
        self.net.to(self.device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.link_names = list(ckpt.get("link_names", [f"link{k}" for k in range(self.net.num_links)]))
        self.q_min = np.asarray(ckpt["q_min"].cpu()) if "q_min" in ckpt else None
        self.q_max = np.asarray(ckpt["q_max"].cpu()) if "q_max" in ckpt else None
        self.softmin_beta = softmin_beta
        mask = torch.ones(self.net.num_links, dtype=torch.bool)
        for name in ignore_links:
            mask[self.link_names.index(name)] = False
        self.link_mask = mask.to(self.device)
        margins = torch.zeros(self.net.num_links)
        for name, m in (link_margins or {}).items():
            margins[self.link_names.index(name)] = m
        self.link_margins = margins.to(self.device)

    def __call__(self, q: np.ndarray, centers: np.ndarray, radii: np.ndarray
                 ) -> Tuple[float, np.ndarray, Dict]:
        """
        Args:
            q:       (d,) joint configuration.
            centers: (S, 3) obstacle sphere centres in the robot base frame.
            radii:   (S,) sphere radii.
        Returns:
            gamma:   Gamma(q) [m]
            grad:    (d,) grad_q Gamma(q)
            info:    {"link": closest link name, "sphere": closest sphere index,
                      "per_link": (K,) min distance of every link [m]}
        """
        d = len(q)
        if len(centers) == 0:
            return float("inf"), np.zeros(d), {"link": None, "sphere": None,
                                               "per_link": np.full(self.net.num_links, np.inf)}

        qt = torch.as_tensor(np.asarray(q), dtype=torch.float32, device=self.device).requires_grad_(True)
        y = torch.as_tensor(np.asarray(centers), dtype=torch.float32, device=self.device)
        r = torch.as_tensor(np.asarray(radii), dtype=torch.float32, device=self.device)

        with torch.enable_grad():
            G = self.net(qt.expand(y.shape[0], -1), y) - r[:, None] - self.link_margins   # (S, K)
            G = torch.where(self.link_mask, G, torch.full_like(G, float("inf")))
            if self.softmin_beta is None:
                gamma = G.min()
            else:
                Gv = G[:, self.link_mask]
                gamma = -torch.logsumexp(-self.softmin_beta * Gv.reshape(-1), dim=0) / self.softmin_beta
            gamma.backward()

        G = G.detach()
        flat = int(torch.argmin(G))
        s, k = divmod(flat, G.shape[1])
        info = {"link": self.link_names[k], "sphere": s, "per_link": G.min(0).values.cpu().numpy()}
        return float(gamma.detach()), qt.grad.cpu().numpy().astype(float), info
