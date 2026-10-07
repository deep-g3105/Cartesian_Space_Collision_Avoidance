"""
isosurface.py

Extracts iso-surfaces of the learned joint-space SDF for a fixed robot
configuration q (cf. Fig. 1 of Koptev et al., RA-L 2023):

    S_k(level)   = { y : Gamma_k(q, y) = level }       per link
    S_min(level) = { y : min_k Gamma_k(q, y) = level } whole robot

Gamma is evaluated on a regular grid around the robot (batched on the GPU)
and meshed with marching cubes (scikit-image). The exact signed distance of
the resulting vertices tells how far the learned surface is from the true
collision mesh.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from skimage.measure import marching_cubes

from .jsdf_network import JSDFNet
from .robot_mesh_model import RobotMeshModel

Mesh = Tuple[np.ndarray, np.ndarray]   # vertices (V, 3), faces (F, 3)


class JSDFIsosurface:
    """Grid evaluation + marching cubes of Gamma(q, .) for one configuration.

    Args:
        net:        trained JSDFNet.
        robot:      RobotMeshModel (used for the bounding box and exact errors).
        q:          (m,) joint configuration.
        resolution: grid spacing in metres.
        padding:    grid padding around the robot bounding box in metres.
        batch:      points per network forward pass.
    """

    def __init__(self, net: JSDFNet, robot: RobotMeshModel, q: np.ndarray,
                 resolution: float = 0.005, padding: float = 0.15, batch: int = 1 << 20):
        self.net = net.eval()
        self.robot = robot
        self.device = next(net.parameters()).device
        self.q = np.asarray(q, dtype=np.float64)
        self.res = resolution

        verts = torch.cat([t.reshape(-1, 3) for t in robot.world_triangles(self.q)]).cpu().numpy()
        self.lo = verts.min(0) - padding
        hi = verts.max(0) + padding
        self.shape = tuple(np.ceil((hi - self.lo) / resolution).astype(int) + 1)
        self.gamma = self._evaluate_grid(batch)                      # (nx, ny, nz, K)

    @torch.no_grad()
    def _evaluate_grid(self, batch: int) -> np.ndarray:
        axes = [torch.arange(n, device=self.device) * self.res + float(l)
                for n, l in zip(self.shape, self.lo)]
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3).float()
        q = torch.as_tensor(self.q, dtype=torch.float32, device=self.device)
        out = torch.cat([self.net(q.expand(grid[s:s + batch].shape[0], -1), grid[s:s + batch])
                         for s in range(0, grid.shape[0], batch)])
        return out.reshape(*self.shape, -1).cpu().numpy()

    def _mesh(self, field: np.ndarray, level: float) -> Optional[Mesh]:
        if not (field.min() < level < field.max()):
            return None
        v, f, _, _ = marching_cubes(field, level=level, spacing=(self.res,) * 3)
        # marching_cubes orients faces towards increasing field -> flip for outward normals
        return (v + self.lo).astype(np.float32), f[:, ::-1].astype(np.int32).copy()

    def link_surface(self, k: int, level: float = 0.0) -> Optional[Mesh]:
        return self._mesh(self.gamma[..., k], level)

    def robot_surface(self, level: float = 0.0) -> Optional[Mesh]:
        return self._mesh(self.gamma.min(-1), level)

    def surface_error(self, mesh: Mesh, k: Optional[int] = None) -> Dict[str, float]:
        """Exact signed distance of learned surface vertices to link k (or whole robot).

        For a perfect Gamma the zero level set has error 0 everywhere.
        """
        y = torch.as_tensor(mesh[0], device=self.robot.device)[None]
        d = self.robot.distances(self.q[None], y)[0]
        d = d[:, k] if k is not None else d.min(-1).values
        d = d.abs()
        return {"mean": d.mean().item(), "p95": d.quantile(0.95).item(), "max": d.max().item()}
