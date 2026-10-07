"""
trainer.py

Training loop and evaluation metrics for JSDFNet (Sec. IV-C / IV-D of
Koptev et al., RA-L 2023).

    loss    - RMSE over all K link distances (paper), optionally weighted
              towards the surface:  w = 1 + near_weight * exp(-|d| / near_sigma),
              loss = sqrt( sum w (Gamma - d)^2 / sum w ).
              The zero level set (the learned robot surface) is defined by the
              near-surface samples, so up-weighting them sharpens it.
    metrics - overall RMSE, and per link (Table IV):
                close RMSE  (samples with d_k <= 10 cm),
                far RMSE    (samples with d_k >  10 cm),
                a_coll      sign accuracy of Gamma_k for samples with d_k < 3 cm.

The whole dataset lives on the GPU (5 M samples x 19 floats ~ 0.4 GB), so
mini-batches are just random index gathers - no DataLoader overhead.
"""

import csv
import json
import math
import os
import time
from typing import Dict, Optional

import torch

from .jsdf_network import JSDFNet


@torch.no_grad()
def evaluate(model: JSDFNet, data: Dict[str, torch.Tensor], batch_size: int = 65536,
             close_threshold: float = 0.10, coll_threshold: float = 0.03) -> Dict:
    """Paper metrics on a dataset dict {q, y, d} (tensors already on the model's device)."""
    model.eval()
    preds = [model(data["q"][s:s + batch_size], data["y"][s:s + batch_size])
             for s in range(0, data["q"].shape[0], batch_size)]
    pred = torch.cat(preds)
    d = data["d"]
    err2 = (pred - d) ** 2

    close = d <= close_threshold
    coll = d < coll_threshold
    sign_ok = (torch.sign(pred) == torch.sign(d)).float()

    def masked_rmse(mask):
        n = mask.sum(0).clamp_min(1)
        return ((err2 * mask).sum(0) / n).sqrt()

    res = {
        "rmse": err2.mean().sqrt().item(),
        "rmse_close": masked_rmse(close).tolist(),
        "rmse_far": masked_rmse(~close).tolist(),
        "acc_coll": ((sign_ok * coll).sum(0) / coll.sum(0).clamp_min(1)).tolist(),
    }
    res["rmse_close_mean"] = sum(res["rmse_close"]) / len(res["rmse_close"])
    res["acc_coll_mean"] = sum(res["acc_coll"]) / len(res["acc_coll"])
    model.train()
    return res


class JSDFTrainer:
    """Mini-batch Adam training of JSDFNet with cosine LR decay.

    Args:
        model:       JSDFNet (moved to `device`).
        train_data / test_data: dicts with q (N, m), y (N, 3), d (N, K) tensors.
        out_dir:     where checkpoints / log.csv are written.
        epochs:      passes over the training set.
        batch_size:  samples per step.
        lr, lr_min:  cosine schedule from lr to lr_min over all steps.
        eval_every:  evaluate + checkpoint every n epochs.
        compile:     wrap the model in torch.compile for the training forward.
        near_weight: extra loss weight at the surface (0 = plain RMSE, paper).
        near_sigma:  decay length of the extra weight in metres.
    """

    def __init__(self, model: JSDFNet, train_data: Dict, test_data: Dict, out_dir: str,
                 device: torch.device, epochs: int = 200, batch_size: int = 8192,
                 lr: float = 1e-3, lr_min: float = 1e-5, weight_decay: float = 0.0,
                 eval_every: int = 5, compile: bool = False, near_weight: float = 0.0,
                 near_sigma: float = 0.02, meta: Optional[Dict] = None):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.fwd = torch.compile(self.model) if compile else self.model
        self.train = {k: train_data[k].to(self.device) for k in ("q", "y", "d")}
        self.test = {k: test_data[k].to(self.device) for k in ("q", "y", "d")}
        self.out_dir = out_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.eval_every = eval_every
        self.meta = meta or {}
        self.near_weight = near_weight
        self.near_sigma = near_sigma

        self.n_train = self.train["q"].shape[0]
        self.steps_per_epoch = math.ceil(self.n_train / batch_size)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=epochs * self.steps_per_epoch, eta_min=lr_min)
        self.start_epoch = 0
        self.best_rmse = float("inf")
        os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    def _checkpoint(self, name: str, epoch: int, metrics: Optional[Dict] = None):
        self.model.save(os.path.join(self.out_dir, name), extra={
            "epoch": epoch, "optimizer": self.opt.state_dict(), "scheduler": self.sched.state_dict(),
            "best_rmse": self.best_rmse, "metrics": metrics, **self.meta})

    def resume(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.opt.load_state_dict(ckpt["optimizer"])
        self.sched.load_state_dict(ckpt["scheduler"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_rmse = ckpt.get("best_rmse", float("inf"))
        print(f"resumed from {path} at epoch {self.start_epoch}")

    def fit(self):
        log_path = os.path.join(self.out_dir, "log.csv")
        new_log = not os.path.exists(log_path) or self.start_epoch == 0
        with open(log_path, "w" if new_log else "a", newline="") as fh:
            log = csv.writer(fh)
            if new_log:
                log.writerow(["epoch", "train_rmse_cm", "test_rmse_cm", "test_close_rmse_cm",
                              "test_acc_coll", "lr", "epoch_time_s"])
            self.model.train()
            for epoch in range(self.start_epoch, self.epochs):
                t0 = time.time()
                perm = torch.randperm(self.n_train, device=self.device)
                run_loss = torch.zeros((), device=self.device)
                for s in range(self.steps_per_epoch):
                    idx = perm[s * self.batch_size:(s + 1) * self.batch_size]
                    pred = self.fwd(self.train["q"][idx], self.train["y"][idx])
                    d = self.train["d"][idx]
                    err2 = (pred - d) ** 2
                    if self.near_weight > 0:
                        w = 1.0 + self.near_weight * torch.exp(-d.abs() / self.near_sigma)
                        loss = torch.sqrt((w * err2).sum() / w.sum())
                    else:
                        loss = torch.sqrt(err2.mean())
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    self.opt.step()
                    self.sched.step()
                    run_loss += err2.detach().mean().sqrt()     # unweighted RMSE for logging
                train_rmse = (run_loss / self.steps_per_epoch).item()
                dt = time.time() - t0

                if (epoch + 1) % self.eval_every == 0 or epoch == self.epochs - 1:
                    m = evaluate(self.model, self.test)
                    lr = self.sched.get_last_lr()[0]
                    log.writerow([epoch, 100 * train_rmse, 100 * m["rmse"], 100 * m["rmse_close_mean"],
                                  m["acc_coll_mean"], lr, dt])
                    fh.flush()
                    print(f"epoch {epoch + 1:4d}/{self.epochs} | train {100 * train_rmse:.3f} cm | "
                          f"test {100 * m['rmse']:.3f} cm | close {100 * m['rmse_close_mean']:.3f} cm | "
                          f"a_coll {m['acc_coll_mean']:.3f} | lr {lr:.2e} | {dt:.1f} s/epoch")
                    if m["rmse"] < self.best_rmse:
                        self.best_rmse = m["rmse"]
                        self._checkpoint("best.pt", epoch, m)
                    self._checkpoint("last.pt", epoch, m)
                else:
                    print(f"epoch {epoch + 1:4d}/{self.epochs} | train {100 * train_rmse:.3f} cm | {dt:.1f} s/epoch")

        final = evaluate(self.model, self.test)
        with open(os.path.join(self.out_dir, "final_metrics.json"), "w") as fh:
            json.dump({"link_names": self.meta.get("link_names"), **final}, fh, indent=2)
        return final
