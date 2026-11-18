# Joint-space obstacle avoidance on the FR3 (hardware)

Modulated DS + MPPI navigation kernels (Koptev, Figueroa, Billard, IJRR 2024) on the real
FR3, the same task as `mj_sim/fr3_joint_space_oa.py`: the robot goes to `Q_A`, then cycles
`Q_B -> Q_A -> ...` (switching when an attractor is reached) around one hard-coded obstacle.

The robot is driven by the existing **IITGN joint impedance controller**
(`iitgn_controllers/ImpedanceController` in `franka_ros2_ws/src/franka_ws`). No new
controller is added; this folder only provides a parameter file for it and a Python node
that streams `(q_ref, q_dot_ref)` to it in direct-tracking mode.

```
 joint_space_oa_node.py (python, this PC)                    franka_iitgn_bringup (ros2_control, 1 kHz)
 ├─ control thread 250 Hz: modulated DS, Eq. (34)-(35)  ──►  /impedance_command_joint  (JointState, name[0]="direct_track")
 │    Gamma(q) on CPU (learned SDF)                          ImpedanceController: tau = K (q_ref - q) + D (dq_ref - dq)
 ├─ MPC thread ~10-20 Hz: MPPI kernels, GPU             ◄──  /NS_1/franka/joint_states
 └─ safety: stop -> /impedance_command_config "stop" (controller holds the current position)
```

## Files

| file | what |
|---|---|
| `hw_config.py` | **edit this**: `Q_A`, `Q_B`, obstacle pose/size (robot base frame), speed, margins, MPC, topics |
| `config/joint_space_oa_controllers.yaml` | ros2_control parameters for the bringup: impedance controller with stiff gains, holds position on start/stop |
| `joint_space_oa_node.py` | the node (`--backend ros` real robot, `--backend mujoco` preview, `--dry-run`, `--viz`, `--no-mpc`) |
| `joint_space_oa_core.py` | control loop + MPC thread (shared by both backends) |
| `check_scene.py` | pre-flight check of the scene (clearance at `Q_A`, `Q_B`, along the nominal path) |

The math is in `utils/joint_space_oa/` (`modulation.py`, `mpc.py`, `distance.py`, ...).

## Requirements

- Ubuntu 22.04 + ROS 2 Humble, `franka_ros2_ws` built (`iitgn_controllers`, `franka_iitgn_bringup`).
- **System python** `/usr/bin/python3` (3.10, the one ROS uses), **not a conda env** (deactivate conda:
  `conda deactivate` until no env is shown). It needs `numpy`, `torch` (CUDA build), `mujoco` (only for
  `--viz`, `--backend mujoco`, `check_scene.py`). Check:
  `python3 -c "import rclpy, torch, mujoco; print(torch.cuda.is_available())"` -> `True`.
- **GPU**: recommended for the MPC thread (NVIDIA + CUDA torch). Without it the node falls back to CPU
  and the MPC runs only at a few Hz (the modulated DS itself runs on the CPU either way).
- The trained network `neural_sdf/runs/big_weighted/best.pt` (already in the repo).
- Robot: FCI enabled in Desk, brakes unlocked, robot in execution mode (blue light), the PC on the
  robot network (`robot_ip` in `franka_ws/franka_iitgn_bringup/config/franka_iitgn.config.yaml`,
  currently `192.168.1.11`), and a real-time kernel for libfranka (the usual Franka setup; on a
  non-RT kernel libfranka reports communication violations).
- Obstacle: a box 10 x 10 x 40 cm (default in `hw_config.py`) placed **exactly** where the config says,
  measured in the robot base frame (`fr3_link0`: x forward, y left, z up, origin at the base on the table).

## How to run (one terminal each)

Paths below: `REPO=~/deepak_ws/"Obstacle Avoidance"/Cartesian_Space_Collision_Avoidance`,
`WS=~/deepak_ws/franka_ros2_ws`.

### Before the robot (no ROS, no robot needed)

```bash
cd $REPO
python3 hardware/joint_space/check_scene.py --viz          # obstacle where you expect it? attractors clear?
python3 hardware/joint_space/joint_space_oa_node.py --backend mujoco --viz   # full pipeline in MuJoCo, Ctrl+C to stop
```
`check_scene.py` must print `scene OK` and "the nominal motion hits the obstacle". The MuJoCo run should
show the arm going around the box (blue = predicted hand path, green = navigation kernels) with
`sim contact steps 0` in the final summary.

### Terminal 1: robot bringup + impedance controller

First, with Desk (guiding / pilot buttons), put the arm in a pose with free space to `Q_A`,
e.g. the Franka "ready" pose, and keep the obstacle out of the way of that first motion.
After the launch the controller holds the arm stiff, so hand guiding is no longer possible.

```bash
source /opt/ros/humble/setup.bash
source $WS/install/setup.bash
ros2 launch franka_iitgn_bringup example_iitgn.launch.py controller_name:=impedance_controller \
    controllers_file:=$REPO/hardware/joint_space/config/joint_space_oa_controllers.yaml
```
Wait for `IITGN Impedance Controller activated`. The robot now holds its current position.

Check (optional, any terminal with ROS sourced):
```bash
ros2 control list_controllers -c /NS_1/controller_manager       # impedance_controller active
ros2 topic hz /NS_1/franka/joint_states                          # ~1000 Hz
```

### Terminal 2: dry run on the real robot (nothing is sent)

```bash
conda deactivate                                  # system python
source /opt/ros/humble/setup.bash
source $WS/install/setup.bash
cd $REPO
python3 hardware/joint_space/joint_space_oa_node.py --dry-run --viz
```
The viewer mirrors the real arm next to the obstacle as configured; compare with the real scene.
The terminal prints `Gamma(q)` of the real start pose. Ctrl+C to stop.

### Terminal 2: run

```bash
python3 hardware/joint_space/joint_space_oa_node.py --viz        # (same sourced terminal)
```
Options: `--no-mpc` (plain modulation: the arm stops in front of the box, the local minimum
the MPC removes), `--speed 0.3` (rad/s, default 0.4), `--duration 60`.
**Ctrl+C** (or closing the viewer) sends `stop`: the controller holds the current position.
Keep the external stop button in your hand.

The node stops by itself (and the controller holds) when: joint states are older than 50 ms,
`||q_ref - q|| > 0.2 rad`, `Gamma(measured q) < 0` (inside the safety margins), or on any error.
Before that, a reference governor slows `q_ref` down when it leads the measured `q` by more
than 0.05 rad (stops it at 0.12 rad), so the collision-checked reference stays with the robot.

## Notes

- The table / floor and self-collisions are **not** in `Gamma` (same as the simulation); `Q_A`,
  `Q_B` keep the hand ~31 cm above the table.
- Safety margins (hw_config): 2 cm on every link + 2 cm on the hand + the sphere padding of the
  obstacle (2.2 cm for the default box) on top of the learned SDF (~1 cm error).
- Gains: `config/joint_space_oa_controllers.yaml` (K 600/250/150/50, D 30/10/5, as the Franka joint
  impedance examples). If you change them, also change `K_GAINS`, `D_GAINS` in `hw_config.py` so
  the MuJoCo preview tracks like the robot.
- The bringup's default `controllers.yaml` is **not** modified; without `controllers_file:=` the
  impedance controller starts compliant (`k = 0`) and must not be used with this node.
