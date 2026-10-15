"""
fr3_joint_space_oa.py

Joint-space obstacle avoidance on the FR3 in MuJoCo with the learned neural
joint-space SDF and DS modulation (Koptev, Figueroa, Billard, IJRR 2024),
nominal modulation only (no MPC / navigation kernels yet):

    nominal DS      q_dot = f(q), normalized linear DS cycling between Q_A and Q_B   (Eq. 36)
    distance        Gamma(q), grad Gamma(q) from the learned SDF over obstacle spheres (Eq. 3)
    modulation      q_dot = M(q) f(q) + f_rep(q),  M = E D E^T                       (Eqs. 5-7, 12-13, 29-30, 33-34)
    integration     q_ref <- q_ref + q_dot dt                                         (Eq. 35)
    control         joint impedance tau = K (q_ref - q) + B (q_dot_ref - q_dot) + bias

Obstacles are presets in OBSTACLES below: flip "enabled" or pick them on
the command line. They are placed along the arc swept by the hand between
Q_A and Q_B so that the nominal motion runs into them. Obstacles at or
below hand height are passed (the arm goes over them); the presets marked
LOCAL MINIMUM (pole, cross, box_facing) stop the arm safely in front of the
obstacle - escaping those is what the MPC navigation kernels of the paper
are for.

Safety: Gamma is reduced by --margin (1 cm, learned-SDF error) on all
links and --hand-margin (2 cm) on the hand (fingers are not in the SDF),
and obstacle spheres are padded to fully contain the true shapes.

The terminal / HUD shows the learned Gamma(q_ref) next to the TRUE
clearance between the robot collision meshes and the obstacle geoms
(MuJoCo mj_geomDistance), so the network is checked independently.

Usage (from the project root or mj_sim/):
    python3 mj_sim/fr3_joint_space_oa.py                               # default obstacles
    python3 mj_sim/fr3_joint_space_oa.py --obstacles pole star moving_sphere
    python3 mj_sim/fr3_joint_space_oa.py --obstacles cross --show-spheres
    python3 mj_sim/fr3_joint_space_oa.py --no-avoidance                # nominal DS only (collides)
    python3 mj_sim/fr3_joint_space_oa.py --list                        # list obstacle presets
    python3 mj_sim/fr3_joint_space_oa.py --headless --duration 20      # no viewer, print summary
"""

import argparse
import os
import sys
import time

import mujoco as mj
import mujoco.viewer
import numpy as np
import torch

# sort the project root directory and add it to the system path for module imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJ_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (PROJECT_ROOT, MJ_SIM_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from fr3_spawn import FR3, FrankaGripper, SinusoidalObstacle
from utils.sdf import RobotMeshModel
from utils.joint_space_oa import (CyclicJointDS, JSDFCollisionDistance, JointSpaceModulator,
                                  JointImpedanceController, ObstacleSet, SphereObstacle, BoxObstacle,
                                  CylinderObstacle, CapsuleObstacle, StarObstacle, CompoundObstacle,
                                  axis_angle_quat)

CHECKPOINT = os.path.join(PROJECT_ROOT, "neural_sdf", "runs", "big_weighted", "best.pt")

# ---------------------------------------------------------------------- #
# reaching task: the hand sweeps an arc of radius ~0.6 m at z ~ 0.31 m
# ---------------------------------------------------------------------- #
Q_A = np.array([-1.0, 0.3, 0.0, -2.0, 0.0, 2.3, 0.785])
Q_B = np.array([1.0, 0.3, 0.0, -2.0, 0.0, 2.3, 0.785])
DS_SPEED = 1.0            # rad/s far from the attractor
SAFETY_MARGIN = 0.01      # m, subtracted from every link's Gamma: covers the learned SDF error (~1 cm)
HAND_MARGIN = 0.02        # m, extra on the hand: the gripper fingers are not in the learned SDF
MAX_QDOT = np.array([2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5])   # rad/s, below FR3 limits


def on_arc(deg, z, r=0.6):
    """Point on the hand's sweep arc at azimuth `deg` (0 = straight ahead)."""
    a = np.deg2rad(deg)
    return np.array([r * np.cos(a), r * np.sin(a), z])


def lobed_star(u, base_radius=0.07, lobe_amplitude=0.03, n_lobes=5):
    """Non-convex 'flower' (same family as radius_function_lobed_3d in main.py)."""
    return base_radius + lobe_amplitude * np.cos(n_lobes * np.arctan2(u[1], u[0]))


def yaw(deg):
    return axis_angle_quat([0, 0, 1], np.deg2rad(deg))


# ---------------------------------------------------------------------- #
# obstacle presets: set "enabled" to choose the default scene
# ---------------------------------------------------------------------- #
OBSTACLES = {
    # single convex primitives
    "sphere": dict(enabled=False, info="sphere r=7 cm in the hand path",
                   make=lambda: SphereObstacle("sphere", on_arc(-30, 0.26), radius=0.07)),
    "box": dict(enabled=False, info="12 cm cube, faces oblique to the path",
                make=lambda: BoxObstacle("box", on_arc(30, 0.21), [0.06, 0.06, 0.06], quat=yaw(0))),
    "box_facing": dict(enabled=False, info="12 cm cube with a face perpendicular to the path "
                                            "(f anti-parallel to n: LOCAL MINIMUM without MPC)",
                       make=lambda: BoxObstacle("box_facing", on_arc(30, 0.30), [0.06, 0.06, 0.06], quat=yaw(30))),
    "pole": dict(enabled=False, info="vertical cylinder r=3 cm, floor to z=0.45 m (LOCAL MINIMUM without MPC)",
                 make=lambda: CylinderObstacle("pole", on_arc(0, 0.225, r=0.62), radius=0.03, half_height=0.225)),
    "bar": dict(enabled=False, info="horizontal capsule r=2.5 cm across the path",
                make=lambda: CapsuleObstacle("bar", on_arc(-10, 0.24), radius=0.025, half_length=0.15,
                                             quat=axis_angle_quat([0, 1, 0], np.pi / 2))),
    "plate": dict(enabled=False, info="thin 30x25 cm plate (wall) facing the robot",
                  make=lambda: BoxObstacle("plate", on_arc(15, 0.35, r=0.66), [0.005, 0.15, 0.125],
                                           quat=yaw(15))),
    # non-convex shapes
    "star": dict(enabled=True, info="lobed star-shaped (non-convex) obstacle",
                 make=lambda: StarObstacle("star", on_arc(-15, 0.30), lobed_star, sphere_radius=0.2)),
    "cross": dict(enabled=False, info="cross of 9 spheres (IJRR Fig. 9), concave (LOCAL MINIMUM without MPC)",
                  make=lambda: CompoundObstacle("cross", on_arc(0, 0.40, r=0.62), [
                      SphereObstacle("c", [0, dy, dz], radius=0.045)
                      for dy, dz in [(0, 0)] + [(s * i * 0.08, 0) for s in (-1, 1) for i in (1, 2)]
                      + [(0, s * i * 0.08) for s in (-1, 1) for i in (1, 2)]], quat=yaw(0))),
    "shelf": dict(enabled=False, info="U-shaped shelf (concave), passed from this side",
                  make=lambda: CompoundObstacle("shelf", on_arc(0, 0.30, r=0.72), [
                      BoxObstacle("top", [0, 0, 0.12], [0.12, 0.20, 0.01], sphere_radius=0.02),
                      BoxObstacle("bottom", [0, 0, -0.12], [0.12, 0.20, 0.01], sphere_radius=0.02),
                      BoxObstacle("back", [0.11, 0, 0], [0.01, 0.20, 0.12], sphere_radius=0.02)],
                      quat=yaw(0), rgba=(0.55, 0.35, 0.2, 0.7))),
    # dynamic
    "moving_sphere": dict(enabled=False, info="sphere r=6 cm oscillating up/down through the path",
                          make=lambda: SphereObstacle("moving_sphere", on_arc(10, 0.31), radius=0.06,
                                                      rgba=(0.9, 0.5, 0.1, 0.7)),
                          motion=dict(amplitude=[0.18, 0.0, 0.0], frequency=0.15, phase=0.0)),
}


def parse_args():
    p = argparse.ArgumentParser(description="FR3 joint-space DS modulation with the learned SDF.")
    p.add_argument("--obstacles", nargs="*", default=None, help="obstacle presets to enable (overrides 'enabled')")
    p.add_argument("--list", action="store_true", help="list obstacle presets and exit")
    p.add_argument("--no-avoidance", action="store_true", help="run the nominal DS without modulation")
    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--speed", type=float, default=DS_SPEED, help="nominal DS speed [rad/s]")
    p.add_argument("--softmin", type=float, default=None, metavar="BETA",
                   help="soft-min sharpness [1/m] for a smooth Gamma (default: hard min, as in the paper)")
    p.add_argument("--margin", type=float, default=SAFETY_MARGIN,
                   help="safety margin on all links for the learned SDF error [m]")
    p.add_argument("--hand-margin", type=float, default=HAND_MARGIN,
                   help="extra margin on the hand link for the fingers [m]")
    p.add_argument("--show-spheres", action="store_true", help="show the sphere approximation of the obstacles")
    p.add_argument("--headless", action="store_true", help="no viewer; print a summary at the end")
    p.add_argument("--duration", type=float, default=30.0, help="sim time for --headless [s]")
    return p.parse_args()


class ClearanceMonitor:
    """TRUE clearance, independent of the learned SDF and of the sphere approximation:
    exact signed distance of the robot collision meshes (utils/sdf RobotMeshModel,
    links 0-7 + hand) to dense points on the true obstacle surfaces, plus MuJoCo
    contacts (which also cover the gripper fingers).

    (mj_geomDistance is not used: with native CCD it returns spurious zeros for
    mesh-box pairs that are far apart, and legacy CCD is off by up to ~2 cm.)
    """

    def __init__(self, model, data, obstacle_set, obstacle_names, device, h=0.008):
        self.model, self.data, self.obstacle_set, self.h = model, data, obstacle_set, h
        self.exact = RobotMeshModel(device=device)
        self.obs_bodies = {mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, n) for n in obstacle_names}
        self.min_seen = np.inf
        self.closest = None
        self.contact_steps = 0

    def contact(self):
        """Deepest physical robot-obstacle contact this step (MuJoCo), or None."""
        deepest = None
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if (b1 in self.obs_bodies) != (b2 in self.obs_bodies) and c.dist < 0:
                rb = b2 if b1 in self.obs_bodies else b1
                name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, rb)
                if deepest is None or c.dist < deepest[0]:
                    deepest = (c.dist, name.replace("fr3_hand_", "").replace("fr3_", ""))
        if deepest is not None:
            self.contact_steps += 1
            self.min_seen = min(self.min_seen, deepest[0])
        return deepest

    def __call__(self, q, poses):
        pts = self.obstacle_set.world_surface_points(poses, self.h)
        d = self.exact.distances(np.asarray(q)[None], torch.as_tensor(pts, dtype=torch.float32)[None])[0]
        per_link = d.min(0).values.cpu().numpy()
        k = int(np.argmin(per_link))
        self.closest = self.exact.link_names[k]
        self.min_seen = min(self.min_seen, per_link[k])
        return float(per_link[k])


def main():
    args = parse_args()
    if args.list:
        for name, cfg in OBSTACLES.items():
            print(f"  {name:14s} {'[on] ' if cfg['enabled'] else '[off]'} {cfg['info']}")
        return

    names = args.obstacles if args.obstacles is not None else [n for n, c in OBSTACLES.items() if c["enabled"]]
    unknown = [n for n in names if n not in OBSTACLES]
    if unknown:
        raise SystemExit(f"unknown obstacles {unknown}; choose from {list(OBSTACLES)}")
    obstacles = [OBSTACLES[n]["make"]() for n in names]
    obstacle_set = ObstacleSet(obstacles)

    # spawn the FR3 with the obstacles as mocap bodies
    spawn = []
    for i, o in enumerate(obstacles):
        geoms = o.geoms() + (obstacle_set.sphere_geoms(i) if args.show_spheres else [])
        spawn.append({"name": o.name, "pos": o.pos, "quat": o.quat, "geoms": geoms})
    robot = FR3(obstacles=spawn)
    model, data = robot.get_model(), robot.get_data()
    dt = model.opt.timestep
    gripper = FrankaGripper(model, data)
    gripper.open()

    movers = [SinusoidalObstacle(model, data, n, center=o.pos, **OBSTACLES[n]["motion"])
              for n, o in zip(names, obstacles) if "motion" in OBSTACLES[n]]
    mocap_ids = [model.body_mocapid[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, o.name)] for o in obstacles]

    # start at Q_A, cycle Q_B <-> Q_A
    data.qpos[:7] = Q_A
    mj.mj_forward(model, data)
    q_ref = Q_A.copy()
    ds = CyclicJointDS([Q_B, Q_A], speed=args.speed)

    distance = JSDFCollisionDistance(args.checkpoint, device=args.device, softmin_beta=args.softmin,
                                     link_margins={name: args.margin + (args.hand_margin if name == "hand" else 0.0)
                                                   for name in ("link0", "link1", "link2", "link3", "link4",
                                                                "link5", "link6", "link7", "hand")})
    q_lo, q_hi = model.jnt_range[:7, 0], model.jnt_range[:7, 1]
    modulator = JointSpaceModulator(dt=dt, max_qdot=MAX_QDOT, q_min=q_lo, q_max=q_hi)
    controller = JointImpedanceController()
    monitor = ClearanceMonitor(model, data, obstacle_set, [o.name for o in obstacles], args.device)

    print("[oa] sphere approximation: " + ", ".join(
        f"{o.name} {len(obstacle_set._local[i][1])} spheres (+{100 * obstacle_set.padding[i]:.1f} cm pad)"
        for i, o in enumerate(obstacles)))
    print(f"[oa] obstacles: {names or 'none'} ({obstacle_set.num_spheres} spheres for Gamma), "
          f"device {args.device}, avoidance {'OFF' if args.no_avoidance else 'ON'}")

    state = {"gamma": np.inf, "clear": np.inf, "info": {}, "link": None, "compute": [],
             "contact": None, "step": 0}

    def control_step():
        t0 = time.perf_counter()
        for m in movers:
            m.step(dt)
        poses = [(data.mocap_pos[i].copy(), data.mocap_quat[i].copy()) for i in mocap_ids]
        centers, radii = obstacle_set.world_spheres(poses)

        f = ds(q_ref)
        gamma, grad, dinfo = distance(q_ref, centers, radii)
        if args.no_avoidance:
            q_next, qdot = q_ref + f * dt, f
            minfo = {}
        else:
            q_next, qdot, minfo = modulator.step(q_ref, f, gamma, grad)
        q_ref[:] = q_next

        q, qd = data.qpos[:7], data.qvel[:7]
        data.ctrl[:7] = controller.torque(q, qd, q_ref, qdot, data.qfrc_bias[:7])
        gripper.open()
        state["compute"].append(time.perf_counter() - t0)
        mj.mj_step(model, data)

        state.update(gamma=gamma, info=minfo, link=dinfo["link"])
        state["step"] += 1
        if obstacles:
            state["contact"] = monitor.contact() or state["contact"]
            if state["step"] % 10 == 0:                       # true clearance at 50 Hz
                state["clear"] = monitor(data.qpos[:7], poses)

    def status():
        lam = state["info"].get("lambda_n", 1.0)
        hit = state["contact"]
        state["contact"] = None
        contact = f" | CONTACT {hit[1]} {100 * hit[0]:.1f} cm" if hit else ""
        return (f"t {data.time:6.2f}s | goal {'B' if ds.index == 0 else 'A'} dist {ds.ds.distance(q_ref):.2f} rad"
                f" | Gamma {100 * state['gamma']:6.1f} cm ({state['link']}) | true clearance "
                f"{100 * state['clear']:6.1f} cm ({monitor.closest}) | lambda_n {lam:.2f} | track err "
                f"{np.linalg.norm(q_ref - data.qpos[:7]):.3f} rad{contact}")

    if args.headless:
        n_steps = int(args.duration / dt)
        for i in range(n_steps):
            control_step()
            if i % int(1.0 / dt) == 0:
                print(status())
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            with viewer.lock():
                viewer.opt.geomgroup[5] = int(args.show_spheres)
                viewer.cam.lookat[:] = [0.35, 0.0, 0.35]
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.0, 150, -25
            last_print = 0.0
            while viewer.is_running():
                t_wall = time.perf_counter()
                control_step()
                if data.time - last_print > 0.5:
                    last_print = data.time
                    print(status())
                    if hasattr(viewer, "set_texts"):
                        viewer.set_texts((None, mj.mjtGridPos.mjGRID_TOPLEFT,
                                          "Gamma (learned)\ntrue clearance\ngoals reached",
                                          f"{100 * state['gamma']:.1f} cm\n{100 * state['clear']:.1f} cm\n{ds.reached}"))
                viewer.sync()
                time.sleep(max(0.0, dt - (time.perf_counter() - t_wall)))

    comp = 1e3 * np.array(state["compute"])
    print(f"\n[oa] summary: sim {data.time:.1f} s | goals reached {ds.reached} | "
          f"min true clearance {100 * monitor.min_seen:.1f} cm | steps in contact {monitor.contact_steps} | "
          f"control step {comp.mean():.2f} ms mean / {np.percentile(comp, 99):.2f} ms p99")


if __name__ == "__main__":
    main()
