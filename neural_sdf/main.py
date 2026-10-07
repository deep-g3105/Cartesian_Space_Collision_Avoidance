"""
neural_sdf/main.py

Entry point for training the neural joint-space SDF Gamma(q, y) of
Koptev, Figueroa, Billard, "Neural Joint Space Implicit Signed Distance
Functions for Reactive Robot Manipulator Control", RA-L 2023, on the
FR3 + Franka hand model in mj_sim/robot_files.

All the math lives in utils/sdf/. This script only wires it together:
    1. generate  - exact (q, y, d) datasets from the collision meshes
                   train: joint limits +5 %, test: nominal limits (paper)
                   cached in neural_sdf/data/{train,test}.pt
    2. train     - MLP (D=5, N=256, positional encoding) with RMSE loss
                   checkpoints + log in neural_sdf/runs/<run_name>/

Usage (from the project root or anywhere):
    python3 neural_sdf/main.py                      # generate (if missing) + train
    python3 neural_sdf/main.py --stage generate     # only build the datasets
    python3 neural_sdf/main.py --stage train --epochs 300 --run-name long
    python3 neural_sdf/main.py --stage train --resume neural_sdf/runs/default/last.pt
    python3 neural_sdf/main.py --quick              # small smoke test (~1 min)
"""

import argparse
import json
import os
import sys
import time

import torch

# sort the project root directory and add it to the system path for module imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.sdf import RobotMeshModel, JSDFDatasetGenerator, JSDFNet, JSDFTrainer

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="Train the neural joint-space SDF (Neural-JSDF).")
    p.add_argument("--stage", choices=["all", "generate", "train"], default="all")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="tiny dataset / few epochs smoke test")

    g = p.add_argument_group("dataset (paper: 5000 x 1000 train, 1000 x 1000 test)")
    g.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    g.add_argument("--regenerate", action="store_true", help="overwrite cached datasets")
    g.add_argument("--train-configs", type=int, default=5000)
    g.add_argument("--test-configs", type=int, default=1000)
    g.add_argument("--points-per-config", type=int, default=1000)
    g.add_argument("--joint-margin", type=float, default=0.05, help="train joint-limit expansion")
    g.add_argument("--near-threshold", type=float, default=0.01)
    g.add_argument("--far-max", type=float, default=1.0)
    g.add_argument("--inside-depth", type=float, default=0.01)

    m = p.add_argument_group("network (paper: D=5, N=256, p.e., no skip)")
    m.add_argument("--hidden", type=int, default=256)
    m.add_argument("--depth", type=int, default=5)
    m.add_argument("--no-pos-enc", action="store_true")
    m.add_argument("--skip-layer", type=int, default=None)
    m.add_argument("--activation", choices=["relu", "softplus", "silu"], default="relu")

    t = p.add_argument_group("training")
    t.add_argument("--run-name", default="default")
    t.add_argument("--epochs", type=int, default=200)
    t.add_argument("--batch-size", type=int, default=8192)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--lr-min", type=float, default=1e-5)
    t.add_argument("--eval-every", type=int, default=5)
    t.add_argument("--compile", action="store_true", help="torch.compile the model")
    t.add_argument("--near-weight", type=float, default=0.0,
                   help="extra loss weight at the surface, w = 1 + a*exp(-|d|/sigma) (0 = paper RMSE)")
    t.add_argument("--near-sigma", type=float, default=0.02, help="decay length of --near-weight [m]")
    t.add_argument("--resume", default=None, help="checkpoint to resume training from")
    args = p.parse_args()

    if args.quick:
        args.train_configs, args.test_configs = 200, 50
        args.epochs, args.eval_every = 5, 1
        args.data_dir = os.path.join(HERE, "data_quick")
        args.run_name = "quick"
    return args


def generate_datasets(args, robot):
    os.makedirs(args.data_dir, exist_ok=True)
    paths = {s: os.path.join(args.data_dir, f"{s}.pt") for s in ("train", "test")}
    plan = {"train": (args.train_configs, args.joint_margin, args.seed),
            "test": (args.test_configs, 0.0, args.seed + 1)}
    for split, (n_cfg, margin, seed) in plan.items():
        if os.path.exists(paths[split]) and not args.regenerate:
            print(f"[data] {paths[split]} exists, skipping (use --regenerate to rebuild)")
            continue
        gen = JSDFDatasetGenerator(robot, points_per_config=args.points_per_config,
                                   near_threshold=args.near_threshold, far_max=args.far_max,
                                   inside_depth=args.inside_depth, seed=seed)
        print(f"[data] generating {split}: {n_cfg} configs x {args.points_per_config} points")
        t0 = time.time()
        data = gen.generate(n_cfg, joint_margin=margin)
        torch.save(data, paths[split])
        n = data["q"].shape[0]
        near = (data["d"].min(-1).values <= args.near_threshold).float().mean().item()
        print(f"[data] {split}: {n} samples, {100 * near:.1f}% near/colliding, "
              f"{time.time() - t0:.1f} s -> {paths[split]}")
    return paths


def train(args, paths):
    train_data = torch.load(paths["train"], weights_only=False)
    test_data = torch.load(paths["test"], weights_only=False)
    num_joints, num_links = train_data["q"].shape[1], train_data["d"].shape[1]
    print(f"[train] {train_data['q'].shape[0]} train / {test_data['q'].shape[0]} test samples, "
          f"m={num_joints}, K={num_links}")

    model = JSDFNet(num_joints=num_joints, num_links=num_links, hidden=args.hidden,
                    depth=args.depth, pos_enc=not args.no_pos_enc,
                    skip_layer=args.skip_layer, activation=args.activation)
    print(f"[train] JSDFNet {model.hparams}, {sum(p.numel() for p in model.parameters())} params")

    out_dir = os.path.join(HERE, "runs", args.run_name)
    meta = {k: train_data[k] for k in ("link_names", "joint_names", "q_min", "q_max")}
    trainer = JSDFTrainer(model, train_data, test_data, out_dir, device=args.device,
                          epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                          lr_min=args.lr_min, eval_every=args.eval_every,
                          compile=args.compile, near_weight=args.near_weight,
                          near_sigma=args.near_sigma, meta=meta)
    if args.resume:
        trainer.resume(args.resume)
    with open(os.path.join(out_dir, "args.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)

    final = trainer.fit()
    print("[train] final test metrics (cm):")
    for k, name in enumerate(meta["link_names"]):
        print(f"   {name:6s} a_coll {final['acc_coll'][k]:.3f} | close {100 * final['rmse_close'][k]:.2f}"
              f" | far {100 * final['rmse_far'][k]:.2f}")
    print(f"   overall RMSE {100 * final['rmse']:.3f} cm, best {100 * trainer.best_rmse:.3f} cm"
          f" -> {os.path.join(out_dir, 'best.pt')}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    print(f"[setup] device: {args.device}"
          + (f" ({torch.cuda.get_device_name()})" if args.device.startswith("cuda") else ""))

    paths = {s: os.path.join(args.data_dir, f"{s}.pt") for s in ("train", "test")}
    if args.stage in ("all", "generate"):
        robot = RobotMeshModel(device=args.device)
        print(f"[setup] links: {robot.link_names}, joints: {robot.num_joints}")
        paths = generate_datasets(args, robot)
    if args.stage in ("all", "train"):
        train(args, paths)


if __name__ == "__main__":
    main()
