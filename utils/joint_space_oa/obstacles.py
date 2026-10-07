"""
obstacles.py

Obstacle shapes for joint-space collision avoidance with the learned SDF.

The neural joint-space SDF Gamma_k(q, y) measures distances to *points*, so
every obstacle is approximated by a set of spheres (y_s, r_s) and
Gamma_ks = Gamma_k(q, y_s) - r_s (Koptev et al. RA-L 2023 Sec. V, IJRR 2024
Sec. 3.1). Each shape here provides both
    geoms()          - MuJoCo-style geom dicts (rendering / physics), local frame
    local_spheres()  - (centres (S, 3), radii (S,)) in the obstacle frame

Sphere approximation (radius rho = sphere_radius, spacing h = 0.75 rho):
(ObstacleSet then pads every sphere of an obstacle by the largest gap
between its true surface and the sphere union - box corners / edges are
rounded off by the spheres - so the sphere set always CONTAINS the obstacle
and Gamma never over-estimates the clearance; padding ~0.7 rho for boxes.)
    Sphere   - exact (one sphere).
    Capsule  - spheres of the capsule radius along its axis (near exact).
    Box      - spheres centred on the surface of the box shrunk by rho, so
               their union matches the box up to dips of ~0.15 rho between
               spheres. Dimensions thinner than 2 rho are inflated to 2 rho
               (conservative), which keeps plates/walls at a sane sphere count.
    Cylinder - same idea on the shrunk cylinder (side rings + cap disks).
    Star     - spheres centred at (R(u) - rho) u for directions u, for any
               star-shaped radius function R(u) (e.g. the lobed "flower"
               obstacle of mj_sim/main.py).
    Compound - union of child shapes (concave obstacles: cross, shelf, ...).

surface_points(h) samples the TRUE obstacle surface (spacing ~h); it is
used to monitor the real clearance independently of the sphere approximation.

Geom dict convention (MuJoCo sizes): sphere [r], box [hx, hy, hz],
cylinder [r, half_height], capsule [r, half_length], mesh: vertices/faces.
Quaternions are (w, x, y, z).
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_RGBA = (0.85, 0.15, 0.10, 0.65)


# ---------------------------------------------------------------------- #
# small rotation helpers (w, x, y, z)
# ---------------------------------------------------------------------- #
def quat_to_mat(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_mul(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def axis_angle_quat(axis: Sequence[float], angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


def fibonacci_directions(n: int) -> np.ndarray:
    """n roughly uniform unit vectors on the sphere."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=1)


def _n_on_sphere(r: float, h: float) -> int:
    return max(20, int(np.ceil(4 * np.pi * r * r / (h * h))))


def _ring(r: float, z: float, h: float) -> np.ndarray:
    if r <= 0:
        return np.array([[0.0, 0.0, z]])
    n = max(6, int(np.ceil(2 * np.pi * r / h)))
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t), np.full(n, z)], 1)


def _cylinder_side(r: float, half_height: float, h: float) -> np.ndarray:
    zs = np.linspace(-half_height, half_height, max(2, int(np.ceil(2 * half_height / h)) + 1))
    return np.concatenate([_ring(r, z, h) for z in zs])


# ---------------------------------------------------------------------- #
# shapes
# ---------------------------------------------------------------------- #
class Obstacle:
    """Base class: a rigid obstacle with a pose (pos, quat) in the world/base frame."""

    def __init__(self, name: str, pos: Sequence[float], quat: Sequence[float] = (1, 0, 0, 0),
                 rgba: Sequence[float] = DEFAULT_RGBA, sphere_radius: float = 0.015,
                 spacing: float = 0.75):
        self.name = name
        self.pos = np.asarray(pos, dtype=float)
        self.quat = np.asarray(quat, dtype=float)
        self.rgba = list(rgba)
        self.rho = sphere_radius
        self.h = spacing * sphere_radius

    def geoms(self) -> List[Dict]:
        raise NotImplementedError

    def local_spheres(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def surface_points(self, h: float = 0.006) -> np.ndarray:
        """(P, 3) points on the true surface, local frame, spacing ~h."""
        raise NotImplementedError

    def _geom(self, gtype: str, size: Sequence[float], **kw) -> Dict:
        g = {"type": gtype, "size": list(size), "pos": [0.0, 0.0, 0.0],
             "quat": [1.0, 0.0, 0.0, 0.0], "rgba": self.rgba}
        g.update(kw)
        return g


class SphereObstacle(Obstacle):
    def __init__(self, name, pos, radius: float, **kw):
        super().__init__(name, pos, **kw)
        self.radius = radius

    def geoms(self):
        return [self._geom("sphere", [self.radius])]

    def local_spheres(self):
        return np.zeros((1, 3)), np.array([self.radius])

    def surface_points(self, h=0.006):
        return self.radius * fibonacci_directions(_n_on_sphere(self.radius, h))


class CapsuleObstacle(Obstacle):
    """Capsule along the local z axis."""

    def __init__(self, name, pos, radius: float, half_length: float, **kw):
        super().__init__(name, pos, **kw)
        self.radius, self.half_length = radius, half_length

    def geoms(self):
        return [self._geom("capsule", [self.radius, self.half_length])]

    def local_spheres(self):
        n = max(2, int(np.ceil(2 * self.half_length / (0.5 * self.radius))) + 1)
        z = np.linspace(-self.half_length, self.half_length, n)
        return np.stack([np.zeros(n), np.zeros(n), z], 1), np.full(n, self.radius)

    def surface_points(self, h=0.006):
        side = _cylinder_side(self.radius, self.half_length, h)
        u = fibonacci_directions(_n_on_sphere(self.radius, h))
        caps = [self.radius * u[u[:, 2] * s >= 0] + [0, 0, s * self.half_length] for s in (-1, 1)]
        return np.concatenate([side] + caps)


class BoxObstacle(Obstacle):
    def __init__(self, name, pos, half_size: Sequence[float], **kw):
        super().__init__(name, pos, **kw)
        self.half_size = np.asarray(half_size, dtype=float)

    def geoms(self):
        return [self._geom("box", self.half_size)]

    def local_spheres(self):
        inner = np.maximum(self.half_size - self.rho, 0.0)
        axes = [np.linspace(-a, a, max(1, int(np.ceil(2 * a / self.h)) + 1)) if a > 0 else np.zeros(1)
                for a in inner]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        on_shell = np.any(np.isclose(np.abs(grid), inner) & (inner > 0), axis=1) | np.any(inner == 0)
        c = grid[on_shell]
        return c, np.full(len(c), self.rho)

    def surface_points(self, h=0.006):
        pts = []
        for ax in range(3):
            o = [a for a in range(3) if a != ax]
            u = np.linspace(-self.half_size[o[0]], self.half_size[o[0]], max(2, int(np.ceil(2 * self.half_size[o[0]] / h)) + 1))
            v = np.linspace(-self.half_size[o[1]], self.half_size[o[1]], max(2, int(np.ceil(2 * self.half_size[o[1]] / h)) + 1))
            U, V = np.meshgrid(u, v, indexing="ij")
            for s in (-1, 1):
                p = np.zeros((U.size, 3))
                p[:, o[0]], p[:, o[1]], p[:, ax] = U.ravel(), V.ravel(), s * self.half_size[ax]
                pts.append(p)
        return np.concatenate(pts)


class CylinderObstacle(Obstacle):
    """Cylinder along the local z axis."""

    def __init__(self, name, pos, radius: float, half_height: float, **kw):
        super().__init__(name, pos, **kw)
        self.radius, self.half_height = radius, half_height

    def geoms(self):
        return [self._geom("cylinder", [self.radius, self.half_height])]

    def local_spheres(self):
        R = max(self.radius - self.rho, 0.0)
        H = max(self.half_height - self.rho, 0.0)
        nz = max(1, int(np.ceil(2 * H / self.h)) + 1)
        zs = np.linspace(-H, H, nz) if H > 0 else np.zeros(1)
        pts = [self._ring(R, z) for z in zs]                         # side wall
        for z in ({-H, H} if H > 0 else {0.0}):                      # caps
            for r in np.linspace(0, R, max(1, int(np.ceil(R / self.h)) + 1))[:-1]:
                pts.append(self._ring(r, z))
        c = np.concatenate(pts)
        return c, np.full(len(c), self.rho)

    def surface_points(self, h=0.006):
        caps = [np.concatenate([_ring(r, s * self.half_height, h)
                                for r in np.linspace(0, self.radius, max(2, int(np.ceil(self.radius / h)) + 1))])
                for s in (-1, 1)]
        return np.concatenate([_cylinder_side(self.radius, self.half_height, h)] + caps)

    def _ring(self, r, z):
        if r <= 0:
            return np.array([[0.0, 0.0, z]])
        n = max(3, int(np.ceil(2 * np.pi * r / self.h)))
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.stack([r * np.cos(t), r * np.sin(t), np.full(n, z)], 1)


class StarObstacle(Obstacle):
    """Star-shaped obstacle with boundary r = R(u) for unit directions u."""

    def __init__(self, name, pos, radius_func: Callable[[np.ndarray], float], **kw):
        super().__init__(name, pos, **kw)
        self.radius_func = radius_func

    def geoms(self):
        v, f = self._mesh()
        return [self._geom("mesh", [], vertices=v, faces=f)]

    def local_spheres(self):
        r_max = max(self.radius_func(u) for u in fibonacci_directions(200))
        n = int(np.ceil(4 * np.pi * max(r_max - self.rho, self.h) ** 2 / self.h ** 2)) + 1
        u = fibonacci_directions(n)
        R = np.array([self.radius_func(ui) for ui in u])
        c = np.maximum(R - self.rho, 0.0)[:, None] * u
        return c, np.full(len(c), self.rho)

    def surface_points(self, h=0.006):
        r_max = max(self.radius_func(u) for u in fibonacci_directions(200))
        u = fibonacci_directions(_n_on_sphere(r_max, h))
        return np.array([self.radius_func(ui) for ui in u])[:, None] * u

    def _mesh(self, n_theta: int = 40, n_phi: int = 20):
        """UV-sphere mesh with per-direction radius (same topology as fr3_spawn.build_star_mesh)."""
        R = self.radius_func
        thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        verts = [np.array([0, 0, R(np.array([0.0, 0, 1]))])]
        for phi in np.linspace(0, np.pi, n_phi + 1)[1:-1]:
            for th in thetas:
                u = np.array([np.sin(phi) * np.cos(th), np.sin(phi) * np.sin(th), np.cos(phi)])
                verts.append(R(u) * u)
        verts.append(np.array([0, 0, -R(np.array([0.0, 0, -1]))]))
        faces, rings, south = [], n_phi - 1, len(verts) - 1
        start = lambda r: 1 + r * n_theta
        for j in range(n_theta):
            faces.append([0, start(0) + j, start(0) + (j + 1) % n_theta])
        for r in range(rings - 1):
            a, b = start(r), start(r + 1)
            for j in range(n_theta):
                j2 = (j + 1) % n_theta
                faces += [[a + j, b + j, b + j2], [a + j, b + j2, a + j2]]
        for j in range(n_theta):
            faces.append([south, start(rings - 1) + (j + 1) % n_theta, start(rings - 1) + j])
        return np.array(verts), np.array(faces)


class CompoundObstacle(Obstacle):
    """Union of child obstacles; child pos/quat are relative to this obstacle's frame."""

    def __init__(self, name, pos, parts: Sequence[Obstacle], **kw):
        super().__init__(name, pos, **kw)
        self.parts = list(parts)

    def geoms(self):
        out = []
        for p in self.parts:
            R = quat_to_mat(p.quat)
            for g in p.geoms():
                g = dict(g)
                g["pos"] = list(p.pos + R @ np.asarray(g["pos"]))
                g["quat"] = list(quat_mul(p.quat, g["quat"]))
                out.append(g)
        return out

    def local_spheres(self):
        cs, rs = [], []
        for p in self.parts:
            c, r = p.local_spheres()
            cs.append(p.pos + c @ quat_to_mat(p.quat).T)
            rs.append(r)
        return np.concatenate(cs), np.concatenate(rs)

    def surface_points(self, h=0.006):
        return np.concatenate([p.pos + p.surface_points(h) @ quat_to_mat(p.quat).T for p in self.parts])


# ---------------------------------------------------------------------- #
# obstacle set -> world spheres for the distance function
# ---------------------------------------------------------------------- #
class ObstacleSet:
    """Holds the local sphere sets of several obstacles and maps them to the world.

    The obstacle poses come from the simulator each step (mocap pos/quat),
    so moving obstacles need no special handling here.
    """

    def __init__(self, obstacles: Sequence[Obstacle], pad: bool = True, h: float = 0.004):
        self.obstacles = list(obstacles)
        self._local, self.padding = [], []
        for o in self.obstacles:
            c, r = o.local_spheres()
            gap = 0.0
            if pad:     # largest distance from the true surface to the sphere union
                p = o.surface_points(h)
                gap = max(0.0, float(max((np.linalg.norm(p[i:i + 2048, None] - c[None], axis=-1) - r[None]).min(1).max()
                                         for i in range(0, len(p), 2048))))
            self._local.append((c, r + gap))
            self.padding.append(gap)
        self.num_spheres = int(sum(len(r) for _, r in self._local))

    def world_spheres(self, poses: Sequence[Tuple[np.ndarray, np.ndarray]]
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """poses: [(pos (3,), quat (4,)) per obstacle] -> centres (S, 3), radii (S,)."""
        if not self.obstacles:
            return np.zeros((0, 3)), np.zeros(0)
        cs = [p + c @ quat_to_mat(q).T for (c, _), (p, q) in zip(self._local, poses)]
        return np.concatenate(cs), np.concatenate([r for _, r in self._local])

    def world_surface_points(self, poses: Sequence[Tuple[np.ndarray, np.ndarray]],
                             h: float = 0.006) -> np.ndarray:
        """(P, 3) points on the true obstacle surfaces in the world (for monitoring)."""
        if not hasattr(self, "_surf") or self._surf_h != h:
            self._surf, self._surf_h = [o.surface_points(h) for o in self.obstacles], h
        if not self.obstacles:
            return np.zeros((0, 3))
        return np.concatenate([p + s @ quat_to_mat(q).T for s, (p, q) in zip(self._surf, poses)])

    def sphere_geoms(self, index: int, group: int = 5,
                     rgba: Sequence[float] = (1.0, 0.8, 0.1, 0.35)) -> List[Dict]:
        """Visual-only geoms showing the sphere approximation of obstacle `index`."""
        c, r = self._local[index]
        return [{"type": "sphere", "size": [ri], "pos": list(ci), "quat": [1, 0, 0, 0],
                 "rgba": list(rgba), "contype": 0, "conaffinity": 0, "group": group}
                for ci, ri in zip(c, r)]
