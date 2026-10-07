"""
jsdf_network.py

Neural joint-space implicit signed distance function Gamma(q, y)
(Sec. IV-A / IV-C of Koptev et al., RA-L 2023):

    Gamma : R^m x R^3 -> R^K,   Gamma_k(q, y) ~ d_k(q, y)

Architecture (paper's best, Table II):
    input  x = [q, cos(q), sin(q), y]          (positional encoding, p.e.)
    MLP    D hidden layers of N units, ReLU    (D = 5, N = 256)
    output K link distances [m]
    optional skip-connection (s.c.) that re-concatenates x at layer `skip_layer`.

The joint-space repulsive field of Eq. (3)-(4) is the Jacobian
    J_q[k, i] = dGamma_k / dq_i
computed with torch.func (vmap + jacrev). The task-space gradient
dGamma_k / dy (outward normal of link k at y) is returned as well, which is
handy for the Cartesian-space modulation code in utils/.

Note: ReLU makes Gamma only piecewise linear, so its gradients are
piecewise constant - fine for QP-IK as in the paper. Use activation="softplus"
or "silu" if you need a smoother C1 field.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.func import jacrev, vmap

_ACTIVATIONS = {"relu": nn.ReLU, "softplus": lambda: nn.Softplus(beta=100), "silu": nn.SiLU}


class JSDFNet(nn.Module):
    """MLP approximating the K per-link signed distances.

    Args:
        num_joints: m, size of q.
        num_links:  K, number of outputs.
        hidden:     N, units per hidden layer.
        depth:      D, number of hidden layers.
        pos_enc:    use [q, cos q, sin q] instead of raw q.
        skip_layer: hidden layer index (1-based) that also receives the input
                    features, or None to disable the skip-connection.
        activation: "relu" (paper), "softplus" or "silu".
    """

    def __init__(self, num_joints: int = 7, num_links: int = 9, hidden: int = 256,
                 depth: int = 5, pos_enc: bool = True, skip_layer: Optional[int] = None,
                 activation: str = "relu"):
        super().__init__()
        self.hparams = dict(num_joints=num_joints, num_links=num_links, hidden=hidden,
                            depth=depth, pos_enc=pos_enc, skip_layer=skip_layer,
                            activation=activation)
        self.num_joints = num_joints
        self.num_links = num_links
        self.pos_enc = pos_enc
        self.skip_layer = skip_layer

        in_dim = (3 * num_joints if pos_enc else num_joints) + 3
        act = _ACTIVATIONS[activation]
        layers = []
        d_prev = in_dim
        for i in range(1, depth + 1):
            d_in = d_prev + (in_dim if skip_layer == i else 0)
            layers.append(nn.Linear(d_in, hidden))
            d_prev = hidden
        self.hidden_layers = nn.ModuleList(layers)
        self.act = act()
        self.out = nn.Linear(hidden, num_links)

    # ------------------------------------------------------------------ #
    def features(self, q: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.pos_enc:
            return torch.cat([q, torch.cos(q), torch.sin(q), y], dim=-1)
        return torch.cat([q, y], dim=-1)

    def forward(self, q: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """q (..., m), y (..., 3) -> Gamma (..., K) in metres."""
        x = self.features(q, y)
        h = x
        for i, layer in enumerate(self.hidden_layers, start=1):
            if self.skip_layer == i:
                h = torch.cat([h, x], dim=-1)
            h = self.act(layer(h))
        return self.out(h)

    # ------------------------------------------------------------------ #
    # gradients, Eq. (3)-(4)
    # ------------------------------------------------------------------ #
    def value_and_jacobians(self, q: torch.Tensor, y: torch.Tensor
                            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gamma and its Jacobians for a batch of (q, y) pairs.

        Args:
            q: (B, m), y: (B, 3)
        Returns:
            gamma: (B, K)
            J_q:   (B, K, m)  row k = grad_q Gamma_k (joint-space repulsion)
            J_y:   (B, K, 3)  row k = grad_y Gamma_k (task-space normal of link k)
        """
        def single(qi, yi):
            g = self.forward(qi, yi)
            return g, g

        (jq, jy), gamma = vmap(jacrev(single, argnums=(0, 1), has_aux=True))(q, y)
        return gamma, jq, jy

    def min_distance(self, q: torch.Tensor, y: torch.Tensor, radius: torch.Tensor = 0.0
                     ) -> torch.Tensor:
        """Distance between the robot at q (m,) and spheres (S, 3) of radius r:
        min_k,s Gamma_k(q, y_s) - r_s  (Sec. V)."""
        qb = q.expand(y.shape[0], -1)
        return (self.forward(qb, y) - torch.as_tensor(radius, device=y.device).reshape(-1, 1)).min()

    # ------------------------------------------------------------------ #
    # checkpoint helpers
    # ------------------------------------------------------------------ #
    def save(self, path: str, extra: Optional[Dict] = None):
        torch.save({"hparams": self.hparams, "state_dict": self.state_dict(), **(extra or {})}, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> Tuple["JSDFNet", Dict]:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        net = cls(**ckpt["hparams"])
        net.load_state_dict(ckpt["state_dict"])
        return net, ckpt
