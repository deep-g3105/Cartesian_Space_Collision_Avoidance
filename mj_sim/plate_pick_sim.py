import os 
import sys
import numpy as np
import mujoco as mj
import time

#sort the project root directory and add it to the system path for module imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

#import all the necessary modules and classes from the project here
from fr3_spawn import FR3, DSMotionGenerator, FrankaGripper, OscillatingObstacle, SinusoidalObstacle, build_star_mesh
from cartesian_impedance import CartesianImpedanceController
from utils.modulation_class_v2 import ModulationAvoider
from utils.modulation_class_v3 import MultiObstacleHuberAvoider, HuberModulationAvoider

#define the mass, stiffness, and damping matrices for the Cartesian impedance controller
M = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # Mass matrix
K = np.diag([100.0, 100.0, 100.0, 50.0, 50.0, 50.0])  # Stiffness matrix
B = np.diag([50.0, 50.0, 50.0, 50.0, 50.0, 50.0])  # Damping matrix

#define 2 points A and B
pt_A = np.array([0.4, -0.4, 0.4])
pt_B = np.array([0.4,  0.4, 0.4])
A_gain = np.diag([2.0, 2.0, 2.0])

#call the functions to setup the robot, Impedance controller, gripper and motion generator
robot = FR3()
cartesian_controller = CartesianImpedanceController(M, K, B)
gripper = FrankaGripper(robot.model, robot.data)
ds = DSMotionGenerator(pt_A, pt_B, A_gain, switch_threshold=0.025)

#keep the gripper at the beginning
gripper.open()

with  mj.viewer.launch_passive(robot.model, robot.data) as viewer:
    while viewer.is_running():
        position, quaternion, velocity = robot.get_cartesian_state()
        desired_velocity = ds.step(position)
        control_input = cartesian_controller.compute_wrench(position, quaternion, velocity, desired_velocity)
        robot.apply_control(control_input)
        robot.step()
        viewer.sync()