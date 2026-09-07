import numpy as np
import os
import mujoco as mj
import time

from fr3_spawn import FR3, DSMotionGenerator
from cartesian_impedance import CartesianImpedanceController


robot = FR3()

# robot.print_site_info("attachment_site")

home_position, home_quaternion, home_velocity = robot.get_cartesian_state()



# print("Home Position:", home_position)
# print("Home Quaternion:", home_quaternion)

# print("simulation time:", robot.model.opt.timestep )

M = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # Mass matrix
K = np.diag([100.0, 100.0, 100.0, 50.0, 50.0, 50.0])  # Stiffness matrix
B = np.diag([50.0, 50.0, 50.0, 50.0, 50.0, 50.0])  # Damping matrix


controller = CartesianImpedanceController(M, K, B)

point_A = np.array([0.5, -0.4, 0.5])
point_B = np.array([0.5,  0.4, 0.5])
A_gain = np.diag([0.5, 0.5, 0.5]) 

ds = DSMotionGenerator(point_A, point_B, A_gain, switch_threshold=0.03)

# controller.set_target_pose(position=[0.5, 0.2, 0.5], quaternion=home_quaternion)  # Target position and orientation

# pose_error = controller.get_pose_error([0.5, 0.2, 0.5], home_quaternion)


with mj.viewer.launch_passive(robot.get_model(), robot.get_data()) as viewer:
    while viewer.is_running():

        position, quaternion, V = robot.get_cartesian_state()

        x_dot, active_target = ds.step(position)
        controller.set_target_pose(position=active_target, quaternion=home_quaternion)  # Keep orientation constant
        controller.V0[:3] = x_dot  # Set desired linear velocity

        wrench = controller.compute_wrench(position, quaternion, V)

        J = robot.get_jacobian()
        Lambda = robot.get_operational_space_mass(J)
        tau = J.T @ (Lambda @ wrench) + robot.data.qfrc_bias[:7]  # Add Coriolis and gravity compensation

        robot.data.ctrl[:7] = tau

        mj.mj_step(robot.model, robot.data)
        time.sleep(0.002)

        viewer.sync()




# robot.print_sites()