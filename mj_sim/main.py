import numpy as np
import os
import mujoco as mj

from fr3_spawn import FR3
from cartesian_impedance import CartesianImpedanceController

robot = FR3()

M = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # Mass matrix
K = np.diag([100.0, 100.0, 100.0, 50.0, 50.0, 50.0])  # Stiffness matrix
B = np.diag([20.0, 20.0, 20.0, 5.0, 5.0, 5.0])  # Damping matrix

controller = CartesianImpedanceController(M, K, B)

controller.set_target_pose(position=[0.5, 0.5, 0.2], quaternion=[1.0, 0.0, 0.0, 0.0])

with mj.viewer.launch_passive(robot.get_model(), robot.get_data()) as viewer:
    while viewer.is_running():

        position, quaternion, V = robot.get_cartesian_state()
        wrench = controller.compute_wrench(position, quaternion, V)

        J = robot.get_jacobian()
        tau = J.T @ wrench

        robot.data.ctrl[:7] = tau

        mj.mj_step(robot.model, robot.data)

        viewer.sync()




# robot.print_sites()