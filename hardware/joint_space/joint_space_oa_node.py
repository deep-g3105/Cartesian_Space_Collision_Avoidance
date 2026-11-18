"""
joint_space_oa_node.py

Joint-space obstacle avoidance on the real FR3 (Koptev, Figueroa, Billard,
IJRR 2024): modulated DS + MPPI navigation kernels, streaming (q_ref, q_dot_ref)
to the IITGN joint impedance controller (iitgn_controllers/ImpedanceController,
direct-tracking mode). Same task as mj_sim/fr3_joint_space_oa.py: two
attractors Q_A, Q_B (switching when reached) and one hard-coded obstacle, all
set in hw_config.py.

Backends
    --backend ros      real robot (default). Needs franka_iitgn_bringup running with
                       config/joint_space_oa_controllers.yaml (see README.md).
    --backend mujoco   the same code against a MuJoCo FR3 tracked by the same impedance
                       gains: dry run of the whole pipeline before touching the robot.

Options
    --dry-run          ros backend: read the robot, compute everything, publish NOTHING.
    --viz              MuJoCo viewer mirroring the robot (measured q), obstacle, predicted
                       hand path (blue) and navigation kernels (green).
    --no-mpc           plain modulation only (expect a local minimum in front of the box).
    --speed            nominal DS speed [rad/s] (default hw_config.DS_SPEED).
    --duration         stop after this many seconds (0 = until Ctrl+C).

Safety (hardware): the node stops (sends "stop" -> the controller holds the
current position) on Ctrl+C, stale joint states, tracking error above
MAX_TRACK_ERROR, Gamma(measured q) below MIN_MEASURED_GAMMA, or control-loop
exceptions. Keep a hand on the user stop button.

Run with the system python of ROS 2 Humble (not a conda env), from anywhere:
    python3 hardware/joint_space/joint_space_oa_node.py --backend mujoco --viz
    python3 hardware/joint_space/joint_space_oa_node.py --dry-run --viz
    python3 hardware/joint_space/joint_space_oa_node.py --viz
"""

import argparse
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "mj_sim"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import hw_config as cfg  # noqa: E402
from joint_space_oa_core import JointSpaceOARunner  # noqa: E402


# ---------------------------------------------------------------------- #
# robot interfaces
# ---------------------------------------------------------------------- #
class RosIO:
    """Joint states in, JointState direct-tracking commands out (iitgn ImpedanceController)."""

    def __init__(self, dry_run: bool):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String

        self.rclpy, self.JointState, self.String = rclpy, JointState, String
        self.dry_run = dry_run
        rclpy.init()
        self.node = rclpy.create_node("joint_space_oa")
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.node.create_subscription(JointState, cfg.JOINT_STATE_TOPIC, self._on_state, qos)
        self.cmd_pub = self.node.create_publisher(JointState, cfg.COMMAND_TOPIC, qos)
        self.cfg_pub = self.node.create_publisher(String, cfg.CONFIG_TOPIC, 10)
        self._q, self._stamp = None, 0.0
        self._lock = threading.Lock()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self._spin = threading.Thread(target=self.executor.spin, daemon=True)
        self._spin.start()

    def _on_state(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        if not all(n in idx for n in cfg.JOINT_NAMES):
            return
        q = np.array([msg.position[idx[n]] for n in cfg.JOINT_NAMES])
        with self._lock:
            self._q, self._stamp = q, time.monotonic()

    def get_q(self):
        with self._lock:
            return (None, np.inf) if self._q is None else (self._q.copy(), time.monotonic() - self._stamp)

    def send(self, q_ref, qdot_ref):
        if self.dry_run:
            return
        msg = self.JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = ["direct_track"]                  # direct tracking of position + velocity
        msg.position = [float(v) for v in q_ref]
        msg.velocity = [float(v) for v in qdot_ref]
        self.cmd_pub.publish(msg)

    def hold(self):
        """Controller holds the current position (position_impedance on q measured)."""
        if self.dry_run:
            return
        for _ in range(3):
            self.cfg_pub.publish(self.String(data="stop"))
            time.sleep(0.02)

    def close(self):
        self.executor.shutdown()
        self.node.destroy_node()
        self.rclpy.shutdown()


class MujocoIO:
    """MuJoCo FR3 (with the obstacle) tracked at 1 kHz by the same joint impedance law
    as the IITGN controller: tau = K (q_d - q) + D (dq_d - dq) (+ gravity, done by the
    real robot internally)."""

    def __init__(self, obstacles):
        import mujoco as mj
        from fr3_spawn import FR3, FrankaGripper 
        self.mj = mj
        spawn = [{"name": o.name, "pos": o.pos, "quat": o.quat, "geoms": o.geoms()} for o in obstacles]
        robot = FR3(obstacles=spawn)
        self.model, self.data = robot.get_model(), robot.get_data()
        self.model.opt.timestep = 0.001
        self.gripper = FrankaGripper(self.model, self.data)
        self.data.qpos[:7] = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])   # franka "ready" pose
        mj.mj_forward(self.model, self.data)
        self.q_d = self.data.qpos[:7].copy()
        self.dq_d = np.zeros(7)
        self.obs_bodies = {mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, o.name) for o in obstacles}
        self.lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        dt = self.model.opt.timestep
        while self._running:
            t0 = time.perf_counter()
            with self.lock:
                q, dq = self.data.qpos[:7], self.data.qvel[:7]
                tau = cfg.K_GAINS * (self.q_d - q) + cfg.D_GAINS * (self.dq_d - dq) + self.data.qfrc_bias[:7]
                self.data.ctrl[:7] = np.clip(tau, -cfg.TORQUE_LIMITS, cfg.TORQUE_LIMITS)
                self.gripper.open()
                self.mj.mj_step(self.model, self.data)
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))

    def get_q(self):
        with self.lock:
            return self.data.qpos[:7].copy(), 0.0

    def send(self, q_ref, qdot_ref):
        with self.lock:
            self.q_d, self.dq_d = np.asarray(q_ref).copy(), np.asarray(qdot_ref).copy()

    def hold(self):
        with self.lock:
            self.q_d, self.dq_d = self.data.qpos[:7].copy(), np.zeros(7)

    def contacts(self):
        """Robot-obstacle penetrations (sim only)."""
        with self.lock:
            out = []
            for c in self.data.contact[:self.data.ncon]:
                b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
                other = b2 if b1 in self.obs_bodies else b1
                if c.dist < 0 and ((b1 in self.obs_bodies) != (b2 in self.obs_bodies)) and other != 0:
                    out.append(c.dist)
            return out

    def close(self):
        self._running = False
        self._thread.join(timeout=1.0)


# ---------------------------------------------------------------------- #
# viewer (kinematic mirror of the measured robot)
# ---------------------------------------------------------------------- #
class Viewer:
    def __init__(self, obstacles, runner, io):
        import mujoco as mj
        import mujoco.viewer
        from fr3_spawn import FR3
        from fr3_joint_space_oa import MPCOverlay
        self.mj = mj
        self.io, self.runner = io, runner
        if isinstance(io, MujocoIO):                  # show the simulated robot itself
            self.model, self.data, self.mirror = io.model, io.data, False
        else:
            spawn = [{"name": o.name, "pos": o.pos, "quat": o.quat, "geoms": o.geoms()} for o in obstacles]
            robot = FR3(obstacles=spawn)
            self.model, self.data, self.mirror = robot.get_model(), robot.get_data(), True
        self.overlay = MPCOverlay(self.model) if runner.use_mpc else None
        self._drawn_iteration = -1
        if isinstance(io, MujocoIO):                  # the viewer copies mjData at launch: not during mj_step
            with io.lock:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        with self.viewer.lock():
            self.viewer.cam.lookat[:] = [0.35, 0.0, 0.3]
            self.viewer.cam.distance, self.viewer.cam.azimuth, self.viewer.cam.elevation = 2.0, 150, -25

    def running(self):
        return self.viewer.is_running()

    def update(self):
        if self.mirror:
            q, _ = self.io.get_q()
            if q is not None:
                self.data.qpos[:7] = q
                self.mj.mj_forward(self.model, self.data)
        planner = self.runner.planner
        if (self.overlay is not None and "rollouts" in planner.last
                and planner.iterations != self._drawn_iteration):     # only redraw on a new MPPI iteration
            self._drawn_iteration = planner.iterations
            self.overlay.update(planner)
            with self.viewer.lock():
                self.overlay.draw(self.viewer.user_scn)
        if isinstance(self.io, MujocoIO):
            with self.io.lock:
                self.viewer.sync()
        else:
            self.viewer.sync()

    def close(self):
        self.viewer.close()


# ---------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="FR3 joint-space obstacle avoidance (modulated DS + MPPI)")
    ap.add_argument("--backend", choices=["ros", "mujoco"], default="ros")
    ap.add_argument("--dry-run", action="store_true", help="ros backend: do not publish any command")
    ap.add_argument("--viz", action="store_true", help="MuJoCo viewer with obstacle, predicted path and kernels")
    ap.add_argument("--no-mpc", action="store_true", help="plain modulation, no navigation kernels")
    ap.add_argument("--speed", type=float, default=None, help=f"DS speed [rad/s] (default {cfg.DS_SPEED})")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after this many seconds (0: Ctrl+C)")
    ap.add_argument("--mpc-device", default="cuda")
    args = ap.parse_args()

    import torch
    if args.mpc_device == "cuda" and not torch.cuda.is_available():
        print("[oa] WARNING: CUDA not available, MPC on CPU (slow: expect a few Hz)")
        args.mpc_device = "cpu"
    torch.set_num_threads(4)

    runner = JointSpaceOARunner(cfg, use_mpc=not args.no_mpc, mpc_device=args.mpc_device,
                                ctrl_device="cpu", speed=args.speed)
    print(f"[oa] obstacle: {len(runner.centers)} spheres | Q_A {cfg.Q_A} | Q_B {cfg.Q_B} | speed {runner.speed} rad/s"
          f" | control {cfg.CONTROL_RATE:.0f} Hz | MPC {'OFF' if args.no_mpc else 'ON (' + args.mpc_device + ')'}"
          f" | backend {args.backend}{' DRY RUN' if args.dry_run else ''}")

    io = RosIO(args.dry_run) if args.backend == "ros" else MujocoIO(runner.obstacles)

    # wait for the first joint state
    t_wait = time.monotonic()
    while True:
        q, age = io.get_q()
        if q is not None and age < cfg.MAX_STATE_AGE:
            break
        if time.monotonic() - t_wait > 10.0:
            print(f"[oa] no joint states on {cfg.JOINT_STATE_TOPIC} after 10 s - is the robot bringup running?")
            io.close()
            return
        time.sleep(0.01)

    gamma0 = runner.measured_gamma(q)
    print(f"[oa] start q = {np.round(q, 3)} | Gamma(q) = {100 * gamma0:.1f} cm")
    if gamma0 < 0.0:
        print("[oa] the robot is already inside the obstacle margin - check the obstacle pose in hw_config.py")
        io.close()
        return

    viewer = Viewer(runner.obstacles, runner, io) if args.viz else None
    runner.start(q)
    stop_reason = None
    stop_flag = threading.Event()
    stats = {"overruns": 0, "steps": 0, "max_err": 0.0, "min_gamma_meas": np.inf, "contacts": 0}

    def control_loop():
        nonlocal stop_reason
        dt = runner.dt
        t_next = time.perf_counter()
        t_start = time.perf_counter()
        last_print = 0.0
        try:
            while not stop_flag.is_set():
                q, age = io.get_q()
                if age > cfg.MAX_STATE_AGE:
                    stop_reason = f"joint states are {1e3 * age:.0f} ms old"
                    break
                q_ref, qdot_ref = runner.step(None if args.dry_run else q)
                io.send(q_ref, qdot_ref)

                err = float(np.linalg.norm(q_ref - q))
                stats["max_err"] = max(stats["max_err"], err)
                if err > cfg.MAX_TRACK_ERROR and not args.dry_run:
                    stop_reason = f"tracking error {err:.3f} rad > {cfg.MAX_TRACK_ERROR}"
                    break
                if stats["steps"] % 10 == 0:          # Gamma of the MEASURED robot at ~25 Hz
                    gm = runner.measured_gamma(q)
                    stats["min_gamma_meas"] = min(stats["min_gamma_meas"], gm)
                    if gm < cfg.MIN_MEASURED_GAMMA and not args.dry_run:
                        stop_reason = f"Gamma(measured q) = {100 * gm:.1f} cm"
                        break
                if isinstance(io, MujocoIO) and io.contacts():
                    stats["contacts"] += 1
                stats["steps"] += 1

                t = time.perf_counter()
                if t - last_print > 1.0:
                    last_print = t
                    print(f"[oa] t {t - t_start:6.1f}s | {runner.status()} | track err {err:.3f} rad")
                if args.duration > 0 and t - t_start > args.duration:
                    stop_reason = "duration reached"
                    break
                t_next += dt
                if t_next < time.perf_counter():
                    stats["overruns"] += 1
                    t_next = time.perf_counter()
                time.sleep(max(0.0, t_next - time.perf_counter()))
        except Exception as e:   # noqa: BLE001 - anything -> hold the robot
            stop_reason = f"exception in control loop: {e!r}"
        finally:
            io.hold()
            stop_flag.set()

    ctrl = threading.Thread(target=control_loop, daemon=True)
    ctrl.start()
    try:
        while not stop_flag.is_set():
            if viewer is not None:
                if not viewer.running():
                    stop_reason = "viewer closed"
                    break
                viewer.update()
            time.sleep(1 / 20)
    except KeyboardInterrupt:
        stop_reason = "Ctrl+C"
    stop_flag.set()
    ctrl.join(timeout=2.0)
    runner.stop()
    io.hold()
    print(f"\n[oa] stopped: {stop_reason} -> controller holds the current position")
    print(f"[oa] {runner.summary()} | control steps {stats['steps']} ({stats['overruns']} overruns) | "
          f"max track err {stats['max_err']:.3f} rad | min Gamma(measured) {100 * stats['min_gamma_meas']:.1f} cm"
          + (f" | sim contact steps {stats['contacts']}" if isinstance(io, MujocoIO) else ""))
    io.close()                      # stop the sim thread / ROS before tearing down the viewer
    if viewer is not None:
        viewer.close()
        time.sleep(0.2)


if __name__ == "__main__":
    main()
