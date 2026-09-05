import numpy as np

class CartesianImpedanceController:

    def __init__(self, M, K, B):

        self.M = np.asarray(M, dtype=float) 
        self.K = np.asarray(K, dtype=float)
        self.B = np.asarray(B, dtype=float)

        self.position_target = np.zeros(3)  # Desired position 
        self.quaternion_target = np.array([1.0, 0.0, 0.0, 0.0])  # Desired orientation (quaternion)
        self.V0 = np.zeros(6)  # Desired velocity

    def set_target_pose(self, position, quaternion):
        position = np.asarray(position, dtype=float)
        quaternion = np.asarray(quaternion, dtype=float)

        if position.shape != (3,):
            raise ValueError("Position must be a 3D vector.")
        if quaternion.shape != (4,):
            raise ValueError("Quaternion must be a 4D vector.")

        quaternion /= np.linalg.norm(quaternion)  # Normalize quaternion
        self.position_target = position
        self.quaternion_target = quaternion

    @staticmethod
    def quaternion_multiply(q1, q2):

        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    @staticmethod
    def quaternion_inverse(q):
        q = np.asarray(q, dtype=float)

        return np.array([q[0], -q[1], -q[2], -q[3]]) 

    @staticmethod
    def quaternion_to_rotation_vector(q):
        q = np.asarray(q, dtype=float)
        q = q/np.linalg.norm(q)  # Normalize quaternion

        if q[0] < 0:
            q = -q

        w = q[0]
        v = q[1:]

        v_norm = np.linalg.norm(v)

        if v_norm < 1e-8:
            return np.zeros(3)

        theta = 2.0 * np.arctan2(v_norm, w)

        axis = v / v_norm
        return axis * theta

    def get_pose_error(self, position_current, quaternion_current):
        position_current = np.asarray(position_current, dtype=float)
        quaternion_current = np.asarray(quaternion_current, dtype=float)

        # Position error
        position_error = self.position_target - position_current

        # Orientation error
        q_current_inv = self.quaternion_inverse(quaternion_current)
        q_error = self.quaternion_multiply(self.quaternion_target, q_current_inv)
        orientation_error = self.quaternion_to_rotation_vector(q_error)

        return np.concatenate((position_error, orientation_error))

    def  get_velocity_error(self, velocity_current):

        velocity_current = np.asarray(velocity_current, dtype=float)

        if velocity_current.shape != (6,):
            raise ValueError("Velocity must be a 6D vector.")

        return self.V0 - velocity_current

    def compute_acceleration(self, position_current, quaternion_current, velocity_current, F_ext=None):

        pose_error = self.get_pose_error(position_current, quaternion_current)

        velocity_error = self.get_velocity_error(velocity_current)

        spring_term = self.K @ pose_error
        damping_term = self.B @ velocity_error

        if F_ext is None:
            F_ext = np.zeros(6)

        F_ext = np.asarray(F_ext, dtype=float)

        if F_ext.shape != (6,):
            raise ValueError("External force must be a 6D vector.")

        acceleration = np.linalg.inv(self.M) @ (spring_term + damping_term + F_ext)

        return acceleration
    
    def compute_wrench(self, position_current, quaternion_current, velocity_current, F_ext=None):

        pose_error = self.get_pose_error(position_current, quaternion_current)

        velocity_error = self.get_velocity_error(velocity_current)

        spring_term = self.K @ pose_error
        damping_term = self.B @ velocity_error

        if F_ext is None:
            F_ext = np.zeros(6)

        F_ext = np.asarray(F_ext, dtype=float)

        if F_ext.shape != (6,):
            raise ValueError("External force must be a 6D vector.")

        wrench = spring_term + damping_term + F_ext

        return wrench

