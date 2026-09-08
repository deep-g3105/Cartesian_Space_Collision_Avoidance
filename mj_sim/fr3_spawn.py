from __future__ import annotations

import os 
import time 
import numpy as np
import mujoco as mj
import mujoco.viewer as viewer

# Original code

# fr3_spec = mj.MjSpec.from_file(os.path.join(os.path.dirname(__file__), "robot_files/franka_fr3/scene.xml"))
# fr3_hand_spec = mj.MjSpec.from_file(os.path.join(os.path.dirname(__file__), "robot_files/fr3_hand/fr3_hand.xml"))

# fr3_spec.attach(fr3_hand_spec, site = "attachment_site", prefix = "fr3_hand_") 

# model = fr3_spec.compile()
# data = mj.MjData(model)

# home = np.array([0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398])
# # data.qpos[:7] = home
# # mj.mj_forward(model, data)
# key_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "home")
# mj.mj_resetDataKeyframe(model, data, key_id)

# mj.viewer.launch(model,data)



# Converting above code to class file

class FR3:
        def __init__(self, obstacles=None):
                base_dir = os.path.dirname(__file__)

                fr3_path = os.path.join(base_dir, "robot_files/franka_fr3/scene.xml")
                hand_path = os.path.join(base_dir, "robot_files/fr3_hand/fr3_hand.xml")

                fr3_spec = mj.MjSpec.from_file(fr3_path)
                hand_spec = mj.MjSpec.from_file(hand_path)

                fr3_spec.attach(hand_spec, site="attachment_site", prefix="fr3_hand_")
                for obs in (obstacles or []):
                       
                        self.add_sphere_obstacle(fr3_spec, obs["name"], np.asarray(obs["pos"]), radius = obs.get("radius", 0.05))

                self.model = fr3_spec.compile()
                self.data = mj.MjData(self.model)

                self.home = np.array([0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398])

                key_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_KEY, "home")
                mj.mj_resetDataKeyframe(self.model, self.data, key_id)
                mj.mj_forward(self.model, self.data)

        def launch(self):
                mj.viewer.launch(self.model, self.data)

        def get_model(self):
                return self.model

        def get_data(self):
                return self.data

        def print_sites(self):

                for i in range(self.model.nsite):
                        name = mj.mj_id2name(
                        self.model,
                        mj.mjtObj.mjOBJ_SITE,
                        i
                        )

                        print(i, name)


        def print_site_info(self, site_name):

                site_id = mj.mj_name2id(
                        self.model,
                        mj.mjtObj.mjOBJ_SITE,
                        site_name
                )

                if site_id == -1:
                        raise ValueError(f"Site '{site_name}' not found.")

                print("Site ID:", site_id)
                print("Site name:", site_name)
                print("Site position:", self.data.site_xpos[site_id])
                print("Site orientation:")
                print(self.data.site_xmat[site_id].reshape(3, 3))

        def get_jacobian(self, site_name = "attachment_site"):
                site_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, site_name)

                if site_id == -1:
                        raise ValueError(f"Site '{site_name}' not found.")

                jacp = np.zeros((3, self.model.nv))
                jacr = np.zeros((3, self.model.nv))

                mj.mj_jacSite(self.model, self.data, jacp, jacr, site_id)

                J = np.vstack((jacp, jacr))

                J_arm = J[:, :7]  # Extract the first 7 columns for the arm joints

                return J_arm

        def get_position(self, site_name = "attachment_site"):
                site_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, site_name)

                if site_id == -1:
                        raise ValueError(f"Site '{site_name}' not found.")

                position = self.data.site_xpos[site_id].copy()
                return position

        def get_orientation(self, site_name = "attachment_site"):
                site_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, site_name)

                if site_id == -1:
                        raise ValueError(f"Site '{site_name}' not found.")

                orientation_matrix = self.data.site_xmat[site_id].reshape(3, 3).copy()
                quaternion = np.zeros(4)
                mj.mju_mat2Quat(quaternion, orientation_matrix.flatten())

                return quaternion

        def get_cartesian_velocity(self, site_name="attachment_site"):

                J = self.get_jacobian(site_name)

                q_dot = self.data.qvel[:7]

                V = J @ q_dot

                return V

        def get_cartesian_state(self, site_name="attachment_site"):
                position = self.get_position(site_name)
                quaternion = self.get_orientation(site_name)
                V = self.get_cartesian_velocity(site_name)

                return position, quaternion, V

        # def apply_cartesian_impedance_torque(self, wrench, site_name="attachment_site"):
        #         J = self.get_jacobian(site_name)
        #         wrench = np.asarray(wrench, dtype=float)

        #         if wrench.shape != (6,):
        #                 raise ValueError("Wrench must be a 6D vector.")

        #         tau = J.T @ wrench + self.data.qfrc_bias[:7]  # Add Coriolis and gravity compensation

        #         self.data.ctrl[:7] = tau

        #         return tau

        # def step(self):
        #         mj.mj_step(self.model, self.data)

        def get_operational_space_mass(self, J):
                """
                Compute the operational-space (Cartesian) mass matrix:
                        Lambda = (J @ M^-1 @ J^T)^-1
                where M is the robot's joint-space mass matrix at the current qpos.
                """
                nv = self.model.nv
                M_full = np.zeros((nv, nv))
                mj.mj_fullM(self.model, M_full, self.data.qM)
                M_joint = M_full[:7, :7]

                M_inv = np.linalg.inv(M_joint)
                Lambda_inv = J @ M_inv @ J.T
                Lambda = np.linalg.inv(Lambda_inv)
                return Lambda

        def add_sphere_obstacle(self, spec, name, pos, radius=0.05, rgba=(1.0, 0.0, 0.0, 0.6)):
                body = spec.worldbody.add_body(name=name, mocap=True, pos=pos)
                body.add_geom(
                        type=mj.mjtGeom.mjGEOM_SPHERE,
                        size=[radius, 0, 0],
                        rgba=rgba,
                        contype=1,      # participates in collision detection
                        conaffinity=1,
                )
                return body


class DSMotionGenerator:
    

    def __init__(self, point_a, point_b, A, switch_threshold=0.02):
        self.point_a = np.asarray(point_a, dtype=float)
        self.point_b = np.asarray(point_b, dtype=float)
        self.A = np.asarray(A, dtype=float)          # 3x3 gain matrix
        self.threshold = switch_threshold
        self.active_target = self.point_b             # start heading toward B

    def step(self, x_current):
        x_current = np.asarray(x_current, dtype=float)

        # Switch attractor once close enough to the current one
        if np.linalg.norm(x_current - self.active_target) < self.threshold:
            self.active_target = (
                self.point_a if np.array_equal(self.active_target, self.point_b)
                else self.point_b
            )

        x_dot = -self.A @ (x_current - self.active_target)
        return x_dot, self.active_target

class FrankaGripper:
        ACTUATOR_NAME = "fr3_hand_actuator8"

        OPEN_CTRL = 255.0
        CLOSE_CTRL = 0.0

        def __init__(self, model, data) -> None:
                self.model = model
                self.data = data

                self.actuator_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, self.ACTUATOR_NAME)

                if self.actuator_id == -1:
                        available = [mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)]
                        raise ValueError(f"Actuator '{self.ACTUATOR_NAME}' not found. Available actuators: {available}")

        def open(self) -> None:
                self.data.ctrl[self.actuator_id] = self.OPEN_CTRL

        def close(self) -> None:
                self.data.ctrl[self.actuator_id] = self.CLOSE_CTRL

        def set(self, value) -> None:
               value = max(self.CLOSE_CTRL, min(self.OPEN_CTRL, value))  # Clamp value
               self.data.ctrl[self.actuator_id] = value

        def get_state(self) -> str:
               value = self.data.ctrl[self.actuator_id]
               if value >= self.OPEN_CTRL - 1:
                      return "open"    
               if value <= self.CLOSE_CTRL + 1:
                      return "closed"
               return "intermediate"

class OscillatingObstacle:
        def __init__(self, model, data, body_name, point_start, point_end, period=2.0):
                self.model = model
                self.data = data
                self.point_start = np.asarray(point_start, dtype=float)
                self.point_end = np.asarray(point_end, dtype=float)
                self.period = period
                
                self.body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, body_name)
                if self.body_id == -1:
                        raise ValueError(f"Body '{body_name}' not found in the model.")
                
                self.mocap_id = model.body_mocapid[self.body_id]
                if self.mocap_id == -1:
                        raise ValueError(f"Body '{body_name}' does not have an associated mocap body.")
                
                self._t = 0.0

        def step(self, dt):
               self._t += dt
               phase = 0.5 * (1 - np.cos(2*np.pi * self._t / self.period))
               pos = self.point_start + phase*(self.point_end - self.point_start)
               self.data.mocap_pos[self.mocap_id] = pos

        def get_position(self):
               return self.data.mocap_pos[self.mocap_id].copy()