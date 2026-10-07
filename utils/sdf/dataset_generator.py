"""
dataset_generator.py

Synthetic training data for the neural joint-space SDF (Sec. IV-B of
Koptev et al., RA-L 2023).

Each sample is (q, y, d) with
    q in R^m   - joint configuration, uniform within joint limits
                 (expanded by `joint_margin` x range for the training set),
    y in R^3   - workspace point,
    d in R^K   - exact signed distance from y to each of the K links.

Per configuration, `points_per_config` points are drawn:
    near half  - exists k: d_k <= near_threshold (1 cm). Points are put on
                 the surface of a link (links used in equal proportion) and
                 pushed along the outward normal by t ~ U(-inside_depth,
                 near_threshold), so |d_k| <= |t| for that link.
    far half   - for all k: d_k > near_threshold. A surface point is pushed
                 by r ~ U(near_threshold, far_max) along a random direction
                 in the outward hemisphere; candidates whose true minimum
                 distance is <= near_threshold are rejected and redrawn.
"""

from typing import Dict, Optional

import numpy as np
import torch

from .robot_mesh_model import RobotMeshModel


class JSDFDatasetGenerator:
    """Generates (q, y, d) tuples with exact labels from a RobotMeshModel.

    Args:
        robot:              geometric robot model (holds the device).
        points_per_config:  workspace points per configuration (paper: 1000).
        near_fraction:      share of near / colliding points (paper: 0.5).
        near_threshold:     near/far boundary in metres (paper: 0.01).
        far_max:            max distance of far points in metres (paper: 1.0).
        inside_depth:       max penetration depth of near points in metres.
        seed:               RNG seed (numpy for q, torch for points).
    """

    def __init__(self, robot: RobotMeshModel,
                 points_per_config: int = 1000,
                 near_fraction: float = 0.5,
                 near_threshold: float = 0.01,
                 far_max: float = 1.0,
                 inside_depth: float = 0.01,
                 seed: int = 0):
        self.robot = robot
        self.device = robot.device
        self.P = int(points_per_config)
        self.n_near = int(round(self.P * near_fraction))
        self.n_far = self.P - self.n_near
        self.near_threshold = near_threshold
        self.far_max = far_max
        self.inside_depth = inside_depth
        self.rng = np.random.default_rng(seed)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)

    # ------------------------------------------------------------------ #
    def _rand(self, *shape):
        return torch.rand(*shape, device=self.device, generator=self.gen)

    def _near_points(self, R, t):
        """(B, n_near, 3) points within near_threshold of some link."""
        K = self.robot.num_links
        counts = np.full(K, self.n_near // K)
        counts[self.rng.choice(K, self.n_near % K, replace=False)] += 1
        pts = []
        for k in range(K):
            if counts[k] == 0:
                continue
            p, n = self.robot.sample_link_surface(R, t, k, int(counts[k]), generator=self.gen)
            off = -self.inside_depth + (self.inside_depth + self.near_threshold) * self._rand(*p.shape[:2], 1)
            pts.append(p + off * n)
        return torch.cat(pts, dim=1)

    def _far_candidates(self, R, t, n):
        """(B, n, 3) candidate points pushed away from a random link."""
        B, K = R.shape[0], self.robot.num_links
        per_link = np.full(K, n // K)
        per_link[: n % K] += 1
        pts = []
        for k in range(K):
            p, nrm = self.robot.sample_link_surface(R, t, k, int(per_link[k]), generator=self.gen)
            v = torch.randn(p.shape, device=self.device, generator=self.gen)
            v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            v = torch.where((v * nrm).sum(-1, keepdim=True) < 0, -v, v)  # outward hemisphere
            r = self.near_threshold + (self.far_max - self.near_threshold) * self._rand(*p.shape[:2], 1)
            pts.append(p + r * v)
        return torch.cat(pts, dim=1)

    @torch.no_grad()
    def _batch(self, q: np.ndarray, max_tries: int = 10):
        R, t = self.robot.geom_poses(q)
        B = q.shape[0]

        y_near = self._near_points(R, t)
        d_near = self.robot.link_distances(R, t, y_near)

        # far points: rejection sampling until every config has n_far valid ones
        y_far = torch.empty(B, self.n_far, 3, device=self.device)
        d_far = torch.empty(B, self.n_far, self.robot.num_links, device=self.device)
        filled = torch.zeros(B, dtype=torch.long, device=self.device)
        for _ in range(max_tries):
            todo = (filled < self.n_far).nonzero().squeeze(-1)
            if todo.numel() == 0:
                break
            cand = self._far_candidates(R[todo], t[todo], 2 * self.n_far)
            dc = self.robot.link_distances(R[todo], t[todo], cand)
            ok = dc.min(-1).values > self.near_threshold
            for j, b in enumerate(todo.tolist()):
                sel = ok[j].nonzero().squeeze(-1)[: self.n_far - filled[b]]
                s, e = filled[b].item(), filled[b].item() + sel.numel()
                y_far[b, s:e] = cand[j, sel]
                d_far[b, s:e] = dc[j, sel]
                filled[b] = e
        if (filled < self.n_far).any():
            raise RuntimeError("far-point rejection sampling did not converge; increase far_max")

        y = torch.cat([y_near, y_far], dim=1)
        d = torch.cat([d_near, d_far], dim=1)
        qt = torch.as_tensor(q, dtype=torch.float32, device=self.device)[:, None, :].expand(B, self.P, -1)
        return qt.reshape(-1, q.shape[1]), y.reshape(-1, 3), d.reshape(-1, self.robot.num_links)

    def generate(self, n_configs: int, joint_margin: float = 0.05,
                 config_batch: int = 64, verbose: bool = True) -> Dict[str, torch.Tensor]:
        """Generate n_configs x points_per_config samples (returned on CPU).

        Returns dict with q (N, m), y (N, 3), d (N, K) float32 tensors plus
        meta-data (link names, joint names/limits) needed at inference time.
        """
        qs = self.robot.sample_configurations(n_configs, margin=joint_margin, rng=self.rng)
        out_q, out_y, out_d = [], [], []
        for s in range(0, n_configs, config_batch):
            q, y, d = self._batch(qs[s:s + config_batch])
            out_q.append(q.cpu()); out_y.append(y.cpu()); out_d.append(d.cpu())
            if verbose:
                done = min(s + config_batch, n_configs)
                print(f"\r  configs {done}/{n_configs}", end="", flush=True)
        if verbose:
            print()
        return {
            "q": torch.cat(out_q), "y": torch.cat(out_y), "d": torch.cat(out_d),
            "link_names": self.robot.link_names,
            "joint_names": self.robot.joint_names,
            "q_min": torch.as_tensor(self.robot.q_min, dtype=torch.float32),
            "q_max": torch.as_tensor(self.robot.q_max, dtype=torch.float32),
            "joint_margin": joint_margin,
        }
