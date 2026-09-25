import os
import sys
import numpy as np
import mujoco as mj
import time

from fr3_spawn import FR3, DSMotionGenerator, FrankaGripper, OscillatingObstacle, SinusoidalObstacle
from cartesian_impedance import CartesianImpedanceController

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from utils.modulation_class_v2 import ModulationAvoider
from utils.modulation_class_v3 import MultiObstacleHuberAvoider, HuberModulationAvoider


M = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # Mass matrix
K = np.diag([100.0, 100.0, 100.0, 50.0, 50.0, 50.0])  # Stiffness matrix
B = np.diag([50.0, 50.0, 50.0, 50.0, 50.0, 50.0])  # Damping matrix

point_A = np.array([0.6, -0.4, 0.4])
point_B = np.array([0.6,  0.4, 0.4])
A_gain = np.diag([2.0, 2.0, 2.0]) 

obstacle_A = np.array([0.5, 0.4, 0.35])
obstacle_B = np.array([0.5, -0.4, 0.35])
obstacle_radius = 0.05

robot = FR3(obstacles =[{"name":"obstacle1", "pos": obstacle_A, "radius": obstacle_radius}])

# robot.print_site_info("attachment_site")

home_position, home_quaternion, home_velocity = robot.get_cartesian_state()
x_ref = home_position.copy()

# print("Home Position:", home_position)
# print("Home Quaternion:", home_quaternion)

# print("simulation time:", robot.model.opt.timestep )

controller = CartesianImpedanceController(M, K, B)

ds = DSMotionGenerator(point_A, point_B, A_gain, switch_threshold=0.025)

gripper = FrankaGripper(robot.model, robot.data)

# obstacle = OscillatingObstacle(
#     robot.model, robot.data,
#     body_name="obstacle1",     # must match the name passed to add_sphere_obstacle
#     point_start=obstacle_A,
#     point_end=obstacle_B,
#     period=6.0,                  # seconds for a full up-down-up cycle
# )
obstacle = SinusoidalObstacle(robot.model, robot.data, "obstacle1",
                         center=[0.6, 0.0, 0.4],
                         amplitude=[0.0, 0.2, 0.05],
                         frequency=[0, 2.0, 2.0],
                         phase=[0, 0, np.pi / 2])

obstacle.set_params(frequency=0.5, amplitude=[0.1, 0.0, 0.1])

def obstacle_gamma(x):
    return HuberModulationAvoider.gamma_sphere(x, obstacle.get_position(), obstacle_radius)


def obstacle_grad_gamma(x):
    return HuberModulationAvoider.grad_gamma_sphere(x, obstacle.get_position(), obstacle_radius)


avoider = HuberModulationAvoider(gamma = obstacle_gamma,
                            grad_gamma= obstacle_grad_gamma,
                            reference_point=obstacle.get_position,
                            dim=3,
                            rho = 1.0,
                            safety_eta = 0.5)

# controller.set_target_pose(position=[0.5, 0.2, 0.5], quaternion=home_quaternion)  # Target position and orientation

# pose_error = controller.get_pose_error([0.5, 0.2, 0.5], home_quaternion)

dt = robot.model.opt.timestep
# p_obs_prev = obstacle.get_position().copy()
gripper.open()   # optional: call once here instead of every tick

with mj.viewer.launch_passive(robot.get_model(), robot.get_data()) as viewer:
    # viewer.opt.sitegroup[4] = 1
    # viewer.opt.label = mj.mjtLabel.mjLABEL_SITE
    while viewer.is_running():

        position, quaternion, V = robot.get_cartesian_state()

        x_dot, active_target = ds.step(x_ref)
        p_obs = obstacle.get_position().copy()
        v_obs = obstacle.get_velocity()
        p_obs_prev = p_obs

        # x_tilde = x_ref - p_obs
        # v_obs = np.zeros(3)
        x_dot = avoider.modulate_moving(x_ref, x_dot, obstacle_center=p_obs, obstacle_lin_vel=v_obs, tangential_damping=0.3)
        x_ref = x_ref + x_dot * dt
        gripper.open()  # Open the gripper at the start of the motion

        controller.set_target_pose(position=x_ref, quaternion=home_quaternion)  # Keep orientation constant
        controller.V0[:3] = x_dot  # Set desired linear velocity

        wrench = controller.compute_wrench(position, quaternion, V)

        J = robot.get_jacobian()
        Lambda = robot.get_operational_space_mass(J)
        tau = J.T @ (Lambda @ wrench) + robot.data.qfrc_bias[:7]  # Add Coriolis and gravity compensation

        robot.data.ctrl[:7] = tau
        # gripper.close()
        obstacle.step(robot.model.opt.timestep)

        mj.mj_step(robot.model, robot.data)
        time.sleep(0.002)


        viewer.sync()




# robot.print_sites()