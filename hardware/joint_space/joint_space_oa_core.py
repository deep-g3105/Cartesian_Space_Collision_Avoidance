"""
joint_space_oa_core.py

Hardware-independent core of the joint-space obstacle avoidance
(Koptev, Figueroa, Billard, IJRR 2024, Fig. 7), shared by the ROS 2 node
(real FR3) and the MuJoCo preview backend:

    control thread (CONTROL_RATE, ~250 Hz)
        q_dot_ref = M(q_ref) (f(q_ref) + alpha M_perp g(q_ref)) + f_rep    Eq. (34)
        q_ref    <- q_ref + q_dot_ref dt                                   Eq. (35)
        -> (q_ref, q_dot_ref) streamed to the joint impedance controller (1 kHz)
        Gamma on the CPU (one configuration, ~1.5 ms)
    MPC thread (as fast as possible, <= 1 / MPC_PERIOD)
        one MPPI iteration (KernelMPPI.update) from the latest q_ref, then the
        new kernels / directions are published to the control thread
        Gamma on the GPU (N x H batched roll-outs)

The nominal DS cycles Q_A -> Q_B -> Q_A ... (CyclicJointDS); the kernels are
cleared when the attractor switches.
"""

import threading
import time
from typing import Dict, Optional

import numpy as np

from utils.joint_space_oa import (CyclicJointDS, JSDFCollisionDistance, JointSpaceModulator, KernelMPPI,
                                  LinearJointDS, NavigationKernels, ObstacleSet)


class JointSpaceOARunner:
    """
    Args:
        cfg:        the hw_config module.
        use_mpc:    False -> plain modulation (no navigation kernels).
        mpc_device: torch device of the MPC roll-outs ("cuda" recommended).
        ctrl_device: torch device of the per-step Gamma ("cpu" is faster for one configuration).
    """

    def __init__(self, cfg, use_mpc: bool = True, mpc_device: str = "cuda", ctrl_device: str = "cpu",
                 speed: Optional[float] = None):
        self.cfg = cfg
        self.dt = 1.0 / cfg.CONTROL_RATE
        self.use_mpc = use_mpc
        margins = {name: cfg.SAFETY_MARGIN + (cfg.HAND_MARGIN if name == "hand" else 0.0)
                   for name in ("link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand")}
        self.dist_ctrl = JSDFCollisionDistance(cfg.CHECKPOINT, device=ctrl_device, link_margins=margins)
        self.dist_mpc = (JSDFCollisionDistance(cfg.CHECKPOINT, device=mpc_device, link_margins=margins)
                         if use_mpc else None)

        self.obstacles = cfg.make_obstacles()
        self.obstacle_set = ObstacleSet(self.obstacles)
        poses = [(np.asarray(o.pos, dtype=float), np.asarray(o.quat, dtype=float)) for o in self.obstacles]
        self.centers, self.radii = self.obstacle_set.world_spheres(poses)

        self.speed = cfg.DS_SPEED if speed is None else speed
        self.ds = CyclicJointDS([cfg.Q_A, cfg.Q_B], speed=self.speed, switch_tol=cfg.SWITCH_TOL)
        self.modulator = JointSpaceModulator(dt=self.dt, max_qdot=cfg.MAX_QDOT, q_min=cfg.Q_MIN, q_max=cfg.Q_MAX,
                                             lead_slow=cfg.LEAD_SLOW, lead_stop=cfg.LEAD_STOP)
        self.planner = (KernelMPPI(self.modulator, self.dist_mpc.batch, 7, n_samples=cfg.MPC_SAMPLES,
                                   horizon=cfg.MPC_HORIZON, dt=cfg.MPC_DT,
                                   q_min=cfg.Q_MIN + 0.05, q_max=cfg.Q_MAX - 0.05)
                        if use_mpc else None)

        # policy used by the control thread: a copy of the planner's kernels (thread-safe swap)
        self.policy = NavigationKernels(7, self.planner.kernels.width if use_mpc else 3.0)
        self.lock = threading.Lock()
        self.q_ref: Optional[np.ndarray] = None
        self.qdot_ref = np.zeros(7)
        self.goal_epoch = 0                        # incremented at every attractor switch
        self.info: Dict = {"gamma": np.inf, "alpha": 0.0, "lambda_n": 1.0, "link": None}
        self.mpc_stats = {"iterations": 0, "ms": [], "full": 0, "placed": 0}
        self._running = False
        self._mpc_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    def start(self, q_measured: np.ndarray):
        """Start from the measured configuration (no jump in the reference)."""
        self.q_ref = np.asarray(q_measured, dtype=float).copy()
        self._running = True
        if self.use_mpc:
            self._mpc_thread = threading.Thread(target=self._mpc_loop, daemon=True)
            self._mpc_thread.start()

    def stop(self):
        self._running = False
        if self._mpc_thread is not None:
            self._mpc_thread.join(timeout=2.0)

    @property
    def goal_index(self) -> int:
        """0 -> heading to Q_A, 1 -> heading to Q_B."""
        return self.ds.index

    # ------------------------------------------------------------------ #
    # control thread
    # ------------------------------------------------------------------ #
    def step(self, q_measured: Optional[np.ndarray] = None):
        """One control step; returns (q_ref, q_dot_ref). q_measured feeds the reference governor."""
        q = self.q_ref
        index_before = self.ds.index
        f = self.ds(q)
        if self.ds.index != index_before:          # attractor switched: kernels belong to the old one
            with self.lock:
                self.goal_epoch += 1
                self.policy.clear()
        gamma, grad, dinfo = self.dist_ctrl(q, self.centers, self.radii)

        g, alpha = None, 0.0
        with self.lock:
            if len(self.policy) > 0 and np.isfinite(gamma) and np.linalg.norm(grad) > 1e-9:
                n = grad / np.linalg.norm(grad)
                alpha = self.modulator.activation(gamma, n, f, self.ds.distance(q), self.planner.goal_radius)
                g = self.policy(q)
        q_next, qdot, minfo = self.modulator.step(q, f, gamma, grad, g, alpha, q_measured=q_measured)
        with self.lock:
            self.q_ref = q_next
            self.qdot_ref = qdot
        self.info = {"gamma": gamma, "alpha": alpha, "lambda_n": minfo.get("lambda_n", 1.0),
                     "link": dinfo.get("link"), "governor": minfo.get("governor", 1.0)}
        return q_next, qdot

    def measured_gamma(self, q: np.ndarray) -> float:
        return self.dist_ctrl(q, self.centers, self.radii)[0]

    # ------------------------------------------------------------------ #
    # MPC thread
    # ------------------------------------------------------------------ #
    def _mpc_loop(self):
        epoch = -1
        while self._running:
            t0 = time.perf_counter()
            with self.lock:
                q = self.q_ref.copy()
                cur_epoch = self.goal_epoch
                goal = self.ds.q_goal.copy()
            if cur_epoch != epoch:
                self.planner.reset()
                epoch = cur_epoch
            ds = LinearJointDS(goal, speed=self.speed, slow_radius=self.ds.slow_radius)
            info = self.planner.update(q, ds, self.centers, self.radii)
            with self.lock:
                if self.goal_epoch == epoch:           # discard if the attractor switched meanwhile
                    self.policy.centers = self.planner.kernels.centers.copy()
                    self.policy.directions = self.planner.kernels.directions.copy()
            s = self.mpc_stats
            s["iterations"] += 1
            s["ms"].append(info["ms"])
            s["full"] += info["n_samples"] > 1
            s["placed"] += info["placed"]
            time.sleep(max(0.0, self.cfg.MPC_PERIOD - (time.perf_counter() - t0)))

    # ------------------------------------------------------------------ #
    def status(self) -> str:
        last = self.planner.last if self.use_mpc else {}
        mpc = (f" | kernels {len(self.policy)} alpha {self.info['alpha']:.2f} MPPI N={last.get('n_samples', 0)} "
               f"{last.get('ms', 0.0):.0f} ms" if self.use_mpc else "")
        return (f"goal {'A' if self.goal_index == 0 else 'B'} dist {self.ds.distance(self.q_ref):.2f} rad"
                f" | reached {self.ds.reached} | Gamma(q_ref) {100 * self.info['gamma']:5.1f} cm ({self.info['link']})"
                f" | lambda_n {self.info['lambda_n']:.2f} | governor {self.info.get('governor', 1.0):.2f}{mpc}")

    def summary(self) -> str:
        out = f"goals reached {self.ds.reached}"
        s = self.mpc_stats
        if self.use_mpc and s["ms"]:
            ms = np.array(s["ms"])
            out += (f" | MPPI {s['iterations']} iterations ({s['full']} with N={self.cfg.MPC_SAMPLES}), "
                    f"{ms.mean():.0f} ms mean / {ms.max():.0f} ms max, kernels placed {s['placed']}")
        return out
