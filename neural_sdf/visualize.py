"""
neural_sdf/visualize.py

MuJoCo viewer of the robot collision geometry used by the neural joint-space
SDF, for one configuration q:

    geom group 2 - visual meshes                      (off by default)
    geom group 3 - exact collision meshes, coloured per link (training labels)
    geom group 4 - learned surfaces Gamma_k(q, y) = 0, same colour per link
    geom group 5 - learned safety shell min_k Gamma_k(q, y) = --margin
                   (transparent, as in Fig. 1 of Koptev et al., RA-L 2023)

Toggle the groups in the viewer's left panel (Rendering -> Geom groups) or
with the keys 2-5. Gripper fingers are drawn in grey: they are not part of
the SDF (as in the paper).

Usage:
    python3 neural_sdf/visualize.py                         # home pose, default run
    python3 neural_sdf/visualize.py --random 3              # random config (seed 3)
    python3 neural_sdf/visualize.py --q 0 0 0 -1.5 0 1.5 0.8
    python3 neural_sdf/visualize.py --exact-only            # no trained model needed
    python3 neural_sdf/visualize.py --checkpoint neural_sdf/runs/long/best.pt --res 0.003
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
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.sdf import RobotMeshModel, JSDFNet
from utils.sdf.robot_mesh_model import DEFAULT_HAND_XML
from utils.sdf.isosurface import JSDFIsosurface

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(PROJECT_ROOT, "mj_sim/robot_files/franka_fr3/scene.xml")   # fr3 + floor + lights
HOME_Q = np.array([0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398])

# one colour per link (link0 ... link7, hand)
PALETTE = np.array([
    [0.40, 0.40, 0.45], [0.89, 0.29, 0.20], [0.95, 0.60, 0.15], [0.95, 0.85, 0.20],
    [0.40, 0.75, 0.30], [0.20, 0.65, 0.65], [0.25, 0.45, 0.85], [0.50, 0.35, 0.80],
    [0.85, 0.35, 0.65]])
UNUSED_RGBA = [0.6, 0.6, 0.6, 1.0]
SHELL_RGBA = [0.3, 0.6, 1.0, 0.18]


def parse_args():
    p = argparse.ArgumentParser(description="Visualize exact and learned robot collision geometry.")
    p.add_argument("--checkpoint", default=os.path.join(HERE, "runs", "default", "best.pt"))
    p.add_argument("--exact-only", action="store_true", help="only show the exact collision meshes")
    p.add_argument("--q", type=float, nargs=7, default=None, help="joint configuration [rad]")
    p.add_argument("--random", type=int, default=None, metavar="SEED",
                   help="random configuration within joint limits")
    p.add_argument("--margin", type=float, default=0.10, help="level of the safety shell [m]")
    p.add_argument("--res", type=float, default=0.005, help="marching-cubes grid spacing [m]")
    p.add_argument("--show-visual", action="store_true", help="also show the visual meshes")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def add_mesh_geom(spec, name, mesh, group, rgba):
    v, f = mesh
    m = spec.add_mesh()
    m.name = name
    m.uservert = v.reshape(-1).tolist()
    m.userface = f.reshape(-1).tolist()
    m.inertia = mj.mjtMeshInertia.mjMESH_INERTIA_SHELL   # open / thin surfaces are fine
    spec.worldbody.add_geom(name=name, type=mj.mjtGeom.mjGEOM_MESH, meshname=name, group=group,
                            contype=0, conaffinity=0, rgba=rgba)


def learned_surfaces(args, robot, q, spec):
    net, ckpt = JSDFNet.load(args.checkpoint, map_location=args.device)
    net.to(args.device)
    print(f"[viz] checkpoint {args.checkpoint} (epoch {ckpt.get('epoch')})")
    t0 = time.time()
    iso = JSDFIsosurface(net, robot, q, resolution=args.res)
    print(f"[viz] Gamma evaluated on a {iso.shape} grid in {time.time() - t0:.1f} s")

    print("[viz] learned surface Gamma_k = 0 vs exact mesh  |  |d_exact| at learned vertices [mm]")
    for k, name in enumerate(robot.link_names):
        mesh = iso.link_surface(k, 0.0)
        if mesh is None:
            print(f"   {name:6s} no zero crossing in the grid")
            continue
        err = iso.surface_error(mesh, k)
        print(f"   {name:6s} mean {1e3 * err['mean']:5.1f} | p95 {1e3 * err['p95']:5.1f} | max {1e3 * err['max']:5.1f}")
        add_mesh_geom(spec, f"learned_{name}", mesh, 4, [*PALETTE[k], 1.0])

    shell = iso.robot_surface(args.margin)
    if shell is not None:
        add_mesh_geom(spec, "learned_shell", shell, 5, SHELL_RGBA)


def main():
    args = parse_args()
    robot = RobotMeshModel(device=args.device)
    if args.q is not None:
        q = np.array(args.q)
    elif args.random is not None:
        q = robot.sample_configurations(1, rng=np.random.default_rng(args.random))[0]
    else:
        q = HOME_Q
    print(f"[viz] q = {np.round(q, 3).tolist()}")

    # same robot model as RobotMeshModel / mj_sim.fr3_spawn.FR3
    spec = mj.MjSpec.from_file(SCENE_XML)
    spec.attach(mj.MjSpec.from_file(DEFAULT_HAND_XML), site="attachment_site", prefix="fr3_hand_")

    show_learned = not args.exact_only and os.path.exists(args.checkpoint)
    if not args.exact_only and not show_learned:
        print(f"[viz] {args.checkpoint} not found -> showing exact collision meshes only")
    if show_learned:
        learned_surfaces(args, robot, q, spec)

    model = spec.compile()
    data = mj.MjData(model)

    # colour the exact collision meshes by link, unused ones (fingers) grey
    body_link = {mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, b): k
                 for k, bodies in enumerate(robot.links.values()) for b in bodies}
    for g in range(model.ngeom):
        if model.geom_group[g] == 3:
            k = body_link.get(model.geom_bodyid[g])
            model.geom_rgba[g] = [*PALETTE[k], 1.0] if k is not None else UNUSED_RGBA

    data.qpos[robot.qpos_adr] = q
    mj.mj_kinematics(model, data)

    print("[viz] geom groups: 2 visual | 3 exact collision | 4 learned Gamma=0 | "
          f"5 learned shell Gamma={args.margin:.2f} m  (toggle with keys 2-5)")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.opt.geomgroup[:] = 0
            viewer.opt.geomgroup[0] = 1                      # floor
            viewer.opt.geomgroup[2] = int(args.show_visual)
            viewer.opt.geomgroup[3] = 1
            viewer.opt.geomgroup[5] = int(show_learned)
            viewer.cam.lookat[:] = [0.2, 0.0, 0.45]
            viewer.cam.distance = 2.0
            viewer.cam.azimuth, viewer.cam.elevation = 135, -20
        # no physics stepping: the robot is frozen at q, only the camera moves
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
