"""
hw_config.py

Everything you may want to change for the FR3 hardware run of the joint-space
obstacle avoidance (Koptev, Figueroa, Billard, IJRR 2024): attractors,
obstacle pose/size, speeds, safety margins, ROS topics, MPC settings.

All positions are in the ROBOT BASE frame (fr3_link0): x forward, y left,
z up, origin at the centre of the base flange on the table. The table is
NOT an obstacle in Gamma (same as in the MuJoCo simulation).
"""

import os

import numpy as np

from utils.joint_space_oa import BoxObstacle, CylinderObstacle, SphereObstacle, axis_angle_quat  # noqa: F401

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT = os.path.join(PROJECT_ROOT, "neural_sdf", "runs", "big_weighted", "best.pt")

# ---------------------------------------------------------------------- #
# task: same two attractors as the MuJoCo simulation (mj_sim/fr3_joint_space_oa.py).
# The robot first goes to Q_A, then cycles Q_B -> Q_A -> Q_B ...
# The hand sweeps an arc of radius ~0.6 m at z ~0.31 m in front of the robot.
# ---------------------------------------------------------------------- #
Q_A = np.array([-1.0, 0.3, 0.0, -2.0, 0.0, 2.3, 0.785])
Q_B = np.array([1.0, 0.3, 0.0, -2.0, 0.0, 2.3, 0.785])
SWITCH_TOL = 0.03          # rad, attractor reached -> switch to the other one

# ---------------------------------------------------------------------- #
# obstacle (hard-coded). Measure it on the real setup in the base frame.
# Default: an upright box 10 x 10 x 40 cm standing on the table, 0.62 m
# straight in front of the robot, in the middle of the hand's sweep. The
# plain modulation stops in front of it (local minimum); the MPC navigation
# kernels take the arm over / around it.
# Other examples (uncomment one):
#   CylinderObstacle("obstacle", [0.62, 0.0, 0.225], radius=0.03, half_height=0.225)   # pole, as the sim preset
#   SphereObstacle("obstacle", [0.52, -0.30, 0.26], radius=0.07)
# quat is (w, x, y, z); axis_angle_quat([0, 0, 1], yaw_rad) rotates about z.
# ---------------------------------------------------------------------- #
BOX_SIZE = np.array([0.10, 0.10, 0.40])                 # full size x, y, z [m]
BOX_CENTER = np.array([0.62, 0.0, BOX_SIZE[2] / 2])     # box standing on the table (z = half height)
BOX_YAW_DEG = 0.0
# Gamma sees the obstacle as spheres of this radius filling its shell: larger -> fewer spheres
# (faster MPC) but a bigger conservative padding (0.03 m: 138 spheres, +2.2 cm padding for this box;
# 0.015 m default: 1024 spheres, +1.1 cm). check_scene.py prints both.
SPHERE_RADIUS = 0.03


def make_obstacles():
    return [
        BoxObstacle("obstacle", BOX_CENTER, BOX_SIZE / 2, quat=axis_angle_quat([0, 0, 1], np.deg2rad(BOX_YAW_DEG)),
                    sphere_radius=SPHERE_RADIUS),
    ]


# ---------------------------------------------------------------------- #
# motion + safety
# ---------------------------------------------------------------------- #
DS_SPEED = 0.4             # rad/s far from the attractor (sim uses 1.0; start slow on hardware)
MAX_QDOT = np.array([0.8, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0])   # rad/s cap on the reference (FR3 limits ~2.6)
CONTROL_RATE = 250.0       # Hz, modulated DS + streaming to the impedance controller
SAFETY_MARGIN = 0.02       # m subtracted from every link's Gamma (learned SDF error ~1 cm + pose error)
HAND_MARGIN = 0.02         # m extra on the hand (gripper fingers are not in the learned SDF)

LEAD_SLOW = 0.05           # rad, reference governor: q_ref slows down when it leads the measured q by more,
LEAD_STOP = 0.12           # rad, ... and stops at this lead (Gamma is checked on q_ref, the robot must follow it)
MAX_TRACK_ERROR = 0.20     # rad, ||q_ref - q|| above this -> stop (robot is not following the reference)
MAX_STATE_AGE = 0.05       # s, joint states older than this -> stop
MIN_MEASURED_GAMMA = 0.0   # m, Gamma(q measured) below this -> stop

# ---------------------------------------------------------------------- #
# MPC (MPPI navigation kernels), runs in its own thread on the GPU
# ---------------------------------------------------------------------- #
MPC_SAMPLES = 32
MPC_HORIZON = 30
MPC_DT = 0.06              # s per roll-out step (H * dt = 1.8 s look-ahead at the slower hardware speed)
MPC_PERIOD = 0.05          # s between MPPI iterations (20 Hz target; it runs as fast as the GPU allows)

# ---------------------------------------------------------------------- #
# ROS 2 (franka_iitgn_bringup, namespace NS_1, iitgn_controllers/ImpedanceController)
# ---------------------------------------------------------------------- #
JOINT_STATE_TOPIC = "/NS_1/franka/joint_states"
COMMAND_TOPIC = "/impedance_command_joint"      # sensor_msgs/JointState, name[0] = "direct_track"
CONFIG_TOPIC = "/impedance_command_config"      # std_msgs/String, "stop" -> hold current position
JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]

# gains of the impedance controller (must match config/joint_space_oa_controllers.yaml;
# used by the MuJoCo preview backend so that it tracks like the real robot)
K_GAINS = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0])
D_GAINS = np.array([30.0, 30.0, 30.0, 30.0, 10.0, 10.0, 5.0])
TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

# FR3 joint limits (same as the IITGN controller)
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
