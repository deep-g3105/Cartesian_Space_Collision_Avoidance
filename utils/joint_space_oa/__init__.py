# utils/joint_space_oa/__init__.py
# Joint-space DS modulation with the learned SDF (Koptev, Figueroa, Billard, IJRR 2024), no MPC
from .nominal_ds import LinearJointDS, CyclicJointDS
from .modulation import JointSpaceModulator, sigmoid_gain
from .obstacles import (Obstacle, SphereObstacle, BoxObstacle, CylinderObstacle, CapsuleObstacle,
                        StarObstacle, CompoundObstacle, ObstacleSet, axis_angle_quat)
from .joint_impedance import JointImpedanceController
from .planar_arm import PlanarArm, PlanarArmDistance

try:    # needs torch + the trained network; optional for the numpy-only 2D animations
    from .distance import JSDFCollisionDistance
except ImportError:
    JSDFCollisionDistance = None

__all__ = ["LinearJointDS", "CyclicJointDS", "JSDFCollisionDistance", "JointSpaceModulator",
           "sigmoid_gain", "Obstacle", "SphereObstacle", "BoxObstacle", "CylinderObstacle",
           "CapsuleObstacle", "StarObstacle", "CompoundObstacle", "ObstacleSet", "axis_angle_quat",
           "JointImpedanceController", "PlanarArm", "PlanarArmDistance"]
