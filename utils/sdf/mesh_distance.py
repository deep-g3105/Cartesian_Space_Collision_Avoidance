"""
mesh_distance.py

Exact signed distance between query points and a closed triangle mesh,
batched on the GPU with plain torch (no FCL / trimesh / open3d needed).

Used to generate the ground-truth labels d_k(q, y) of Eq. (1) in:
    M. Koptev, N. Figueroa, A. Billard,
    "Neural Joint Space Implicit Signed Distance Functions for Reactive
    Robot Manipulator Control", IEEE RA-L, 2023.

Method:
    unsigned distance - brute force over every triangle: the closest point
        is either the orthogonal projection onto the triangle plane (if it
        falls inside the triangle) or the closest point on one of its three
        edges.
    sign              - generalized winding number (Jacobson et al. 2013):
        w(p) = sum_f Omega_f(p) / (4 pi), with Omega_f the solid angle of
        triangle f seen from p (Van Oosterom & Strackee formula). p is
        inside the mesh when |w(p)| > 0.5. Robust to small holes, but the
        mesh should be closed for clean labels.

The FR3 collision meshes are 200-300 triangles each, so brute force over
all faces is cheap and exact. Work is chunked over points so memory stays
bounded (max_pairs = points x faces evaluated at once).
"""

import math

import numpy as np
import torch


def _cross(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """torch.cross with broadcasting over leading dims."""
    u, v = torch.broadcast_tensors(u, v)
    return torch.cross(u, v, dim=-1)


class TriangleMeshSDF:
    """Signed distance to one rigid triangle mesh, expressed in its own frame.

    Args:
        vertices: (V, 3) mesh vertices in the mesh (geom) frame.
        faces:    (F, 3) integer vertex indices.
        device:   torch device used for the queries.
        max_pairs: max number of point-triangle pairs evaluated per chunk.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray,
                 device: torch.device = torch.device("cpu"),
                 max_pairs: int = 4_000_000):
        self.device = torch.device(device)
        self.max_pairs = int(max_pairs)

        v = torch.as_tensor(np.asarray(vertices), dtype=torch.float32, device=self.device)
        f = torch.as_tensor(np.asarray(faces), dtype=torch.long, device=self.device)
        tris = v[f]                                        # (F, 3, 3)
        self.a, self.b, self.c = tris[:, 0], tris[:, 1], tris[:, 2]

        # per-face constants for the closest-point query
        self.ab = self.b - self.a
        self.bc = self.c - self.b
        self.ca = self.a - self.c
        n = torch.cross(self.ab, self.c - self.a, dim=-1)
        self.normal = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        self.num_faces = tris.shape[0]

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def signed_distance(self, points: torch.Tensor) -> torch.Tensor:
        """Signed distance (negative inside) for (N, 3) points in mesh frame."""
        points = points.to(self.device, torch.float32)
        out = torch.empty(points.shape[0], device=self.device)
        chunk = max(1, self.max_pairs // self.num_faces)
        for s in range(0, points.shape[0], chunk):
            p = points[s:s + chunk]
            d = self._unsigned_distance(p)
            inside = self._winding_number(p).abs() > 0.5   # abs: robust to face orientation
            out[s:s + chunk] = torch.where(inside, -d, d)
        return out

    @torch.no_grad()
    def unsigned_distance(self, points: torch.Tensor) -> torch.Tensor:
        points = points.to(self.device, torch.float32)
        out = torch.empty(points.shape[0], device=self.device)
        chunk = max(1, self.max_pairs // self.num_faces)
        for s in range(0, points.shape[0], chunk):
            out[s:s + chunk] = self._unsigned_distance(points[s:s + chunk])
        return out

    # ------------------------------------------------------------------ #
    # internals, p is (M, 3) and everything broadcasts to (M, F)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _segment_dist2(p, a, ab):
        """Squared distance from points p (M,1,3) to segments a + t*ab, t in [0,1]."""
        ap = p - a
        t = (ap * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-12)
        t = t.clamp(0.0, 1.0).unsqueeze(-1)
        diff = ap - t * ab
        return (diff * diff).sum(-1)

    def _unsigned_distance(self, p: torch.Tensor) -> torch.Tensor:
        p = p[:, None, :]                                   # (M, 1, 3)

        # projection onto the triangle plane and inside test via edge normals
        h = ((p - self.a) * self.normal).sum(-1)            # (M, F) signed plane height
        proj = p - h.unsqueeze(-1) * self.normal
        s1 = (_cross(self.ab, proj - self.a) * self.normal).sum(-1)
        s2 = (_cross(self.bc, proj - self.b) * self.normal).sum(-1)
        s3 = (_cross(self.ca, proj - self.c) * self.normal).sum(-1)
        inside = (s1 >= 0) & (s2 >= 0) & (s3 >= 0)

        edge_d2 = torch.minimum(
            torch.minimum(self._segment_dist2(p, self.a, self.ab),
                          self._segment_dist2(p, self.b, self.bc)),
            self._segment_dist2(p, self.c, self.ca))

        d2 = torch.where(inside, h * h, edge_d2)            # (M, F)
        return d2.min(dim=-1).values.clamp_min(0.0).sqrt()

    def _winding_number(self, p: torch.Tensor) -> torch.Tensor:
        p = p[:, None, :]
        a = self.a - p
        b = self.b - p
        c = self.c - p
        la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
        det = (a * _cross(b, c)).sum(-1)
        den = (la * lb * lc + (a * b).sum(-1) * lc
               + (b * c).sum(-1) * la + (c * a).sum(-1) * lb)
        solid_angle = 2.0 * torch.atan2(det, den)
        return solid_angle.sum(-1) / (4.0 * math.pi)
