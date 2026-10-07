"""
robot_mesh_model.py

Exact geometric model of the robot body B = U_k B_k (Sec. III-A of Koptev
et al., RA-L 2023), built from the MuJoCo collision meshes of the FR3 +
Franka hand used in mj_sim/.

Provides:
    - the K link meshes (one TriangleMeshSDF per collision geom),
    - batched forward kinematics f(q, .) via MuJoCo (pose of every geom),
    - the exact per-link signed distance d_k(q, y) of Eq. (1),
    - area-weighted surface sampling with outward normals (used by the
      dataset generator to place points near the robot surface).

Default links (K = 9, gripper fingers excluded as in the paper - their
meshes are also not closed, so the winding-number sign would be wrong):
    fr3_link0 ... fr3_link7, fr3_hand_hand
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import mujoco as mj
import numpy as np
import torch

from .mesh_distance import TriangleMeshSDF

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ARM_XML = os.path.join(_PROJECT_ROOT, "mj_sim/robot_files/franka_fr3/fr3.xml")
DEFAULT_HAND_XML = os.path.join(_PROJECT_ROOT, "mj_sim/robot_files/fr3_hand/fr3_hand.xml")

# link name -> list of MuJoCo bodies whose collision geoms form that link
DEFAULT_LINKS: Dict[str, List[str]] = {
    "link0": ["fr3_link0"],
    "link1": ["fr3_link1"],
    "link2": ["fr3_link2"],
    "link3": ["fr3_link3"],
    "link4": ["fr3_link4"],
    "link5": ["fr3_link5"],
    "link6": ["fr3_link6"],
    "link7": ["fr3_link7"],
    "hand":  ["fr3_hand_hand"],
}

DEFAULT_ARM_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]


class RobotMeshModel:
    """Per-link triangle meshes + forward kinematics of the FR3 (+ hand).

    Args:
        arm_xml / hand_xml: MJCF files, attached the same way as mj_sim/fr3_spawn.FR3.
        links:        ordered {link_name: [body names]} -> defines the K outputs.
        arm_joints:   names of the m controlled joints (order of q).
        collision_group: MuJoCo geom group holding the collision meshes.
        device:       torch device for distance queries.
    """

    def __init__(self,
                 arm_xml: str = DEFAULT_ARM_XML,
                 hand_xml: Optional[str] = DEFAULT_HAND_XML,
                 links: Optional[Dict[str, List[str]]] = None,
                 arm_joints: Optional[Sequence[str]] = None,
                 collision_group: int = 3,
                 device: torch.device = torch.device("cpu")):
        self.device = torch.device(device)
        self.links = dict(links or DEFAULT_LINKS)
        self.link_names = list(self.links.keys())
        self.num_links = len(self.link_names)

        spec = mj.MjSpec.from_file(arm_xml)
        if hand_xml is not None:
            spec.attach(mj.MjSpec.from_file(hand_xml), site="attachment_site", prefix="fr3_hand_")
        self.model = spec.compile()
        self.data = mj.MjData(self.model)

        # joints -> qpos addresses and limits
        joint_names = list(arm_joints or DEFAULT_ARM_JOINTS)
        jids = [self._id(mj.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
        self.joint_names = joint_names
        self.num_joints = len(jids)
        self.qpos_adr = np.array([self.model.jnt_qposadr[j] for j in jids])
        self.q_min = self.model.jnt_range[jids, 0].copy()
        self.q_max = self.model.jnt_range[jids, 1].copy()

        # collision geoms per link
        self.geom_ids: List[int] = []         # flat list of all used geoms
        self.geom_link: List[int] = []        # link index of each geom
        self.geom_sdf: List[TriangleMeshSDF] = []
        self._geom_surface: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for k, (link, bodies) in enumerate(self.links.items()):
            bids = {self._id(mj.mjtObj.mjOBJ_BODY, b) for b in bodies}
            found = False
            for g in range(self.model.ngeom):
                if self.model.geom_bodyid[g] not in bids or self.model.geom_group[g] != collision_group:
                    continue
                if self.model.geom_type[g] != mj.mjtGeom.mjGEOM_MESH:
                    raise ValueError(f"link '{link}': only mesh collision geoms are supported (geom {g})")
                verts, faces = self._mesh_of_geom(g)
                self.geom_ids.append(g)
                self.geom_link.append(k)
                self.geom_sdf.append(TriangleMeshSDF(verts, faces, device=self.device))
                self._geom_surface.append(self._surface_tables(verts, faces))
                found = True
            if not found:
                raise ValueError(f"link '{link}' has no collision mesh in group {collision_group}")
        self.geom_link_t = torch.as_tensor(self.geom_link, device=self.device)

        # surface area per link, used to sample uniformly over a whole link
        self.link_geoms = [[i for i, kk in enumerate(self.geom_link) if kk == k] for k in range(self.num_links)]

    # ------------------------------------------------------------------ #
    # MuJoCo helpers
    # ------------------------------------------------------------------ #
    def _id(self, obj, name):
        i = mj.mj_name2id(self.model, obj, name)
        if i < 0:
            raise ValueError(f"'{name}' not found in the MuJoCo model")
        return i

    def _mesh_of_geom(self, g):
        mid = self.model.geom_dataid[g]
        va, vn = self.model.mesh_vertadr[mid], self.model.mesh_vertnum[mid]
        fa, fn = self.model.mesh_faceadr[mid], self.model.mesh_facenum[mid]
        # mesh_vert is already expressed in the geom frame after compilation
        return self.model.mesh_vert[va:va + vn].copy(), self.model.mesh_face[fa:fa + fn].copy()

    def _surface_tables(self, verts, faces):
        v = torch.as_tensor(verts, dtype=torch.float32, device=self.device)
        tri = v[torch.as_tensor(faces, dtype=torch.long, device=self.device)]   # (F,3,3)
        n = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
        area = 0.5 * n.norm(dim=-1)
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        # make normals point outwards (mesh winding may be either way)
        centroid = v.mean(0)
        if ((tri.mean(1) - centroid) * n).sum(-1).mean() < 0:
            n = -n
        return tri, n, area

    # ------------------------------------------------------------------ #
    # kinematics
    # ------------------------------------------------------------------ #
    def sample_configurations(self, n: int, margin: float = 0.0,
                              rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Uniform q within joint limits expanded by `margin` x range on each side."""
        rng = rng or np.random.default_rng()
        span = self.q_max - self.q_min
        lo, hi = self.q_min - margin * span, self.q_max + margin * span
        return rng.uniform(lo, hi, size=(n, self.num_joints))

    def geom_poses(self, q: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """World pose of every used collision geom for a batch of configs.

        Returns R (B, G, 3, 3) and t (B, G, 3) such that x_world = R @ x_geom + t.
        """
        q = np.atleast_2d(q)
        B, G = q.shape[0], len(self.geom_ids)
        R = np.empty((B, G, 3, 3))
        t = np.empty((B, G, 3))
        for b in range(B):
            self.data.qpos[:] = 0.0
            self.data.qpos[self.qpos_adr] = q[b]
            mj.mj_kinematics(self.model, self.data)
            R[b] = self.data.geom_xmat[self.geom_ids].reshape(G, 3, 3)
            t[b] = self.data.geom_xpos[self.geom_ids]
        return (torch.as_tensor(R, dtype=torch.float32, device=self.device),
                torch.as_tensor(t, dtype=torch.float32, device=self.device))

    @torch.no_grad()
    def world_triangles(self, q: np.ndarray) -> List[torch.Tensor]:
        """Collision triangles of every geom in world frame at a single config q: [(F, 3, 3)]."""
        R, t = self.geom_poses(np.asarray(q)[None])
        return [torch.einsum("ij,fvj->fvi", R[0, gi], self._geom_surface[gi][0]) + t[0, gi]
                for gi in range(len(self.geom_ids))]

    # ------------------------------------------------------------------ #
    # exact distances, Eq. (1)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def link_distances(self, R: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Exact signed distances d_k(q, y) for every link.

        Args:
            R, t: geom poses from geom_poses(), (B, G, 3, 3) and (B, G, 3).
            y:    world points, (B, P, 3) - P points per configuration.
        Returns:
            d: (B, P, K) signed distance of each point to each link [m].
        """
        y = y.to(self.device, torch.float32)
        B, P, _ = y.shape
        d = torch.full((B, P, self.num_links), float("inf"), device=self.device)
        for gi, sdf in enumerate(self.geom_sdf):
            # x_geom = R^T (y - t)  ->  row-vector form (y - t) @ R
            local = torch.einsum("bpi,bij->bpj", y - t[:, gi, None, :], R[:, gi])
            dg = sdf.signed_distance(local.reshape(-1, 3)).reshape(B, P)
            k = self.geom_link[gi]
            d[..., k] = torch.minimum(d[..., k], dg)
        return d

    def distances(self, q: np.ndarray, y: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper: q (B, m), y (B, P, 3) -> d (B, P, K)."""
        R, t = self.geom_poses(q)
        return self.link_distances(R, t, y)

    # ------------------------------------------------------------------ #
    # surface sampling
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample_link_surface(self, R: torch.Tensor, t: torch.Tensor, link: int, n: int,
                            generator: Optional[torch.Generator] = None
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Area-uniform points on link `link` for every config in the batch.

        Returns points (B, n, 3) and outward unit normals (B, n, 3) in world frame.
        """
        B = R.shape[0]
        geoms = self.link_geoms[link]
        tri = torch.cat([self._geom_surface[g][0] for g in geoms])
        nrm = torch.cat([self._geom_surface[g][1] for g in geoms])
        area = torch.cat([self._geom_surface[g][2] for g in geoms])
        owner = torch.cat([torch.full((len(self._geom_surface[g][2]),), g, device=self.device)
                           for g in geoms])

        f = torch.multinomial(area, B * n, replacement=True, generator=generator).view(B, n)
        u = torch.rand(B, n, 2, device=self.device, generator=generator)
        flip = u.sum(-1) > 1.0
        u[flip] = 1.0 - u[flip]
        a, b, c = tri[f, 0], tri[f, 1], tri[f, 2]
        p_local = a + u[..., :1] * (b - a) + u[..., 1:] * (c - a)
        n_local = nrm[f]

        gidx = owner[f]                                                   # (B, n)
        Rg = torch.gather(R, 1, gidx[..., None, None].expand(B, n, 3, 3))
        tg = torch.gather(t, 1, gidx[..., None].expand(B, n, 3))
        p = torch.einsum("bnij,bnj->bni", Rg, p_local) + tg
        nw = torch.einsum("bnij,bnj->bni", Rg, n_local)
        return p, nw
