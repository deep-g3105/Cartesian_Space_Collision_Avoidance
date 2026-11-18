"""
check_scene.py

Pre-flight check of hw_config.py before running on the robot (no ROS needed):
  - Gamma (learned SDF, with the safety margins) and the TRUE clearance (robot
    collision meshes vs. obstacle surface) at Q_A, Q_B and the current/home pose,
  - the same along the straight joint-space path Q_A -> Q_B (where the plain
    nominal DS would go): it should come close to / hit the obstacle,
    otherwise the experiment shows no avoidance,
  - optionally (--viz) a MuJoCo viewer cycling the robot through Q_A, Q_B
    and the nominal path, with the obstacle drawn where hw_config puts it.

    python3 hardware/joint_space/check_scene.py [--viz] [--q "q1 ... q7"]
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "mj_sim"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

import hw_config as cfg  # noqa: E402
from utils.joint_space_oa import JSDFCollisionDistance, ObstacleSet  # noqa: E402
from utils.sdf import RobotMeshModel  # noqa: E402

READY = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--q", default=None, help="also check this configuration (7 numbers, e.g. the robot's current q)")
    args = ap.parse_args()

    obstacles = cfg.make_obstacles()
    oset = ObstacleSet(obstacles)
    poses = [(np.asarray(o.pos, float), np.asarray(o.quat, float)) for o in obstacles]
    centers, radii = oset.world_spheres(poses)
    margins = {n: cfg.SAFETY_MARGIN + (cfg.HAND_MARGIN if n == "hand" else 0.0)
               for n in ("link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand")}
    dist = JSDFCollisionDistance(cfg.CHECKPOINT, device="cpu", link_margins=margins)
    exact = RobotMeshModel(device="cpu")
    pts = torch.as_tensor(oset.world_surface_points(poses, 0.008), dtype=torch.float32)

    def true_clearance(q):
        d = exact.distances(np.asarray(q)[None], pts[None])[0].min(0).values.numpy()
        k = int(np.argmin(d))
        return float(d[k]), exact.link_names[k]

    print("obstacle(s): " + ", ".join(f"{o.name} {type(o).__name__} at {np.round(o.pos, 3)}" for o in obstacles)
          + f" -> {len(centers)} spheres | margins {100 * cfg.SAFETY_MARGIN:.0f} cm (+{100 * cfg.HAND_MARGIN:.0f} cm hand)")
    checks = [("Q_A", cfg.Q_A), ("Q_B", cfg.Q_B), ("ready pose", READY)]
    if args.q:
        checks.append(("--q", np.array([float(v) for v in args.q.replace(",", " ").split()])))
    ok = True
    for name, q in checks:
        g, _, info = dist(q, centers, radii)
        tc, link = true_clearance(q)
        flag = "OK" if g > 0.03 else "TOO CLOSE"
        ok &= g > 0.03 or name == "--q"
        print(f"  {name:10s} Gamma {100 * g:6.1f} cm ({info['link']:5s}) | true clearance {100 * tc:6.1f} cm ({link:5s})"
              f" | {flag}")

    s = np.linspace(0, 1, 41)
    path = cfg.Q_A[None] + s[:, None] * (cfg.Q_B - cfg.Q_A)[None]
    gam = np.array([dist(q, centers, radii)[0] for q in path])
    tcs = np.array([true_clearance(q)[0] for q in path])
    i = int(np.argmin(tcs))
    print(f"  nominal Q_A->Q_B path: min Gamma {100 * gam.min():.1f} cm, min true clearance {100 * tcs[i]:.1f} cm "
          f"at s = {s[i]:.2f} -> " + ("the nominal motion hits the obstacle (avoidance needed)" if tcs[i] < 0.02
                                       else "the nominal motion already clears it (no avoidance will be visible)"))
    print("scene OK for the attractors" if ok else "MOVE THE OBSTACLE OR THE ATTRACTORS: an attractor is within 3 cm")

    if args.viz:
        import mujoco as mj
        import mujoco.viewer
        from fr3_spawn import FR3
        spawn = [{"name": o.name, "pos": o.pos, "quat": o.quat, "geoms": o.geoms()} for o in obstacles]
        robot = FR3(obstacles=spawn)
        m, d = robot.get_model(), robot.get_data()
        seq = [cfg.Q_A] * 30 + list(path) + [cfg.Q_B] * 30 + list(path[::-1])
        with mujoco.viewer.launch_passive(m, d) as v:
            v.cam.lookat[:] = [0.35, 0.0, 0.3]
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 2.0, 150, -25
            k = 0
            while v.is_running():
                d.qpos[:7] = seq[k % len(seq)]
                mj.mj_forward(m, d)
                v.sync()
                k += 1
                time.sleep(0.04)


if __name__ == "__main__":
    main()
