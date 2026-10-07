"""
neural_sdf/evaluate.py

Compares trained Neural-JSDF checkpoints on the same data:

    1. test-set metrics (Table IV of Koptev et al., RA-L 2023) on one shared
       test set: overall RMSE, per-link close RMSE (d <= 10 cm), a_coll;
    2. learned-surface error: for N random configurations, the zero level set
       Gamma_k(q, .) = 0 of each link is meshed (marching cubes) and the exact
       distance of its vertices to the true link mesh is averaged [mm].
       A link whose zero level set is missing in a configuration is counted
       in the "missing" column (the network never predicts contact there).

Usage:
    python3 neural_sdf/evaluate.py neural_sdf/runs/default/best.pt neural_sdf/runs/big_weighted/best.pt
    python3 neural_sdf/evaluate.py RUN.pt --test-data neural_sdf/data/test.pt --surface-configs 50
"""

import argparse
import os
import sys

import numpy as np
import torch

# sort the project root directory and add it to the system path for module imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.sdf import RobotMeshModel, JSDFNet, evaluate
from utils.sdf.isosurface import JSDFIsosurface

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="Compare Neural-JSDF checkpoints.")
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--test-data", default=os.path.join(HERE, "data", "test.pt"))
    p.add_argument("--surface-configs", type=int, default=20, help="random configs for the surface error")
    p.add_argument("--res", type=float, default=0.006, help="marching-cubes grid spacing [m]")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def surface_errors(net, robot, qs, res):
    """(K,) mean surface error [m] over configs and (K,) count of missing surfaces."""
    K = robot.num_links
    err, missing = [[] for _ in range(K)], np.zeros(K, dtype=int)
    for q in qs:
        iso = JSDFIsosurface(net, robot, q, resolution=res)
        for k in range(K):
            mesh = iso.link_surface(k, 0.0)
            if mesh is None:
                missing[k] += 1
            else:
                err[k].append(iso.surface_error(mesh, k)["mean"])
    return np.array([np.mean(e) if e else np.nan for e in err]), missing


def main():
    args = parse_args()
    robot = RobotMeshModel(device=args.device)
    test = torch.load(args.test_data, weights_only=False)
    test = {k: test[k].to(args.device) for k in ("q", "y", "d")}
    qs = robot.sample_configurations(args.surface_configs, rng=np.random.default_rng(args.seed))
    print(f"test set: {args.test_data} ({test['q'].shape[0]} samples), "
          f"surface error over {args.surface_configs} random configs\n")

    for path in args.checkpoints:
        net, ckpt = JSDFNet.load(path, map_location=args.device)
        net.to(args.device)
        m = evaluate(net, test)
        surf, miss = surface_errors(net, robot, qs, args.res)
        print(f"== {path}")
        print(f"   test RMSE {100 * m['rmse']:.3f} cm | close RMSE {100 * m['rmse_close_mean']:.3f} cm"
              f" | a_coll {m['acc_coll_mean']:.3f} | mean surface error {1e3 * np.nanmean(surf):.1f} mm")
        print(f"   {'link':6s} {'close cm':>8s} {'a_coll':>7s} {'surf mm':>8s} {'missing':>8s}")
        for k, name in enumerate(robot.link_names):
            print(f"   {name:6s} {100 * m['rmse_close'][k]:8.2f} {m['acc_coll'][k]:7.3f}"
                  f" {1e3 * surf[k]:8.1f} {miss[k]:5d}/{len(qs)}")
        print()


if __name__ == "__main__":
    main()
