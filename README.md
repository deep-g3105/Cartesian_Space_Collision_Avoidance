## Mathematical Foundations

This repository implements collision avoidance in Cartesian space using modulation-based approaches. The core idea is to modulate the robot's velocity or trajectory in the presence of obstacles, ensuring smooth and safe navigation.

### Core Concepts

- **Obstacle Modulation:** The robot's desired velocity $\dot{x}_d$ is modulated by a matrix $M(x)$ that depends on the robot's position relative to obstacles:

	$$
	\dot{x}_{mod} = M(x) \cdot \dot{x}_d
	$$

- **Modulation Matrix:** $M(x)$ is constructed such that it slows down or redirects the robot as it approaches obstacles, based on the obstacle's geometry (circle, ellipse, star-shaped, etc.).

- **Dynamical Systems:** The framework supports both kinematic and dynamic systems, and can be extended to multiple obstacles and higher dimensions.

### References

- [Dynamical Systems Based Obstacle Avoidance](https://ieeexplore.ieee.org/document/6630914)
- [Nonlinear Modulation for Obstacle Avoidance](https://ieeexplore.ieee.org/document/6630914)

### Visual Examples

Below are example animations (GIFs) demonstrating the algorithms. Add your GIFs/images to the `media/` folder and update the links as needed.

<div style="display: flex; gap: 24px;">
	<div style="border: 2px solid #888; border-radius: 8px; padding: 8px; text-align: center; width: 48%;">
		<strong>2D Circle Obstacle Avoidance</strong><br>
		<img src="media/2d_circle.gif" alt="2D Circle Obstacle" style="max-width: 100%; height: auto;">
	</div>
	<div style="border: 2px solid #888; border-radius: 8px; padding: 8px; text-align: center; width: 48%;">
		<strong>2D Star-Shaped Obstacle Avoidance</strong><br>
		<img src="media/2d_star.gif" alt="2D Star Obstacle" style="max-width: 100%; height: auto;">
	</div>
</div>


<br/>

<div style="display: flex; gap: 24px;">
	<div style="border: 2px solid #888; border-radius: 8px; padding: 8px; text-align: center; width: 48%;">
		<strong>3D Spherical Obstacle Avoidance</strong><br>
		<img src="media/3d_sphere.gif" alt="3D Spherical Obstacle" style="max-width: 100%; height: auto;">
	</div>
	<div style="border: 2px solid #888; border-radius: 8px; padding: 8px; text-align: center; width: 48%;">
		<strong>3D Star-Shaped Obstacle Avoidance</strong><br>
		<img src="media/3d_star.gif" alt="3D Star Obstacle" style="max-width: 100%; height: auto;">
	</div>
</div>


# Cartesian Space Collision Avoidance

## Table of Contents

- [Overview](#overview)
- [Pipeline Summary](#pipeline-summary)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)


## Mathematical Foundations

This repository implements collision avoidance in Cartesian space using modulation-based approaches. The core idea is to modulate the robot's velocity or trajectory in the presence of obstacles, ensuring smooth and safe navigation.

### Core Concepts

- **Obstacle Modulation:** The robot's desired velocity $\dot{x}_d$ is modulated by a matrix $M(x)$ that depends on the robot's position relative to obstacles:

	$$
	\dot{x}_{\text{mod}} = M(x) \, \dot{x}_d
	$$

- **Modulation Matrix:** $M(x)$ is constructed such that it slows down or redirects the robot as it approaches obstacles, based on the obstacle's geometry (circle, ellipse, star-shaped, etc.).

- **Dynamical Systems:** The framework supports both kinematic and dynamic systems, and can be extended to multiple obstacles and higher dimensions.

#### Example: Modulation Matrix for a Circular Obstacle

For a circular obstacle centered at $x_o$ with radius $r$, the modulation matrix can be defined as:

$$
M(x) = E(x) D(x) E(x)^{-1}
$$

where $E(x)$ contains the orthonormal basis vectors (normal and tangential to the obstacle), and $D(x)$ is a diagonal matrix that scales the velocity along these directions.

#### References

- [Dynamical Systems Based Obstacle Avoidance](https://ieeexplore.ieee.org/document/6630914)
- [Nonlinear Modulation for Obstacle Avoidance](https://ieeexplore.ieee.org/document/6630914)

### Visual Examples

Below are example animations (GIFs) demonstrating the algorithms. Add your GIFs/images to the `media/` folder and update the links as needed.

#### 2D Circle Obstacle Avoidance
![2D Circle Obstacle](media/2d_circle.gif)

#### 2D Star-Shaped Obstacle Avoidance
![2D Star Obstacle](media/2d_star.gif)

#### 3D Spherical Obstacle Avoidance
![3D Spherical Obstacle](media/3d_sphere.gif)

---

## Overview

This repository provides a collection of algorithms and scripts for collision avoidance in Cartesian space for robotic manipulators. It includes 2D and 3D simulation environments, hardware integration scripts, and utilities for motion planning and obstacle avoidance. The codebase is modular, supporting different robot platforms and obstacle types, and is designed for both research and educational use.

---

## Pipeline Summary

1. **Simulation**: Run 2D/3D animations to visualize and test collision avoidance algorithms.
2. **Hardware Integration**: Deploy algorithms on real robots using provided scripts.
3. **Utilities**: Use planners, velocity commanders, and gripper control modules for advanced tasks.

---

## Prerequisites

- Ubuntu Linux (tested on 20.04/22.04)
- Python 3.10+ (see `env.yaml` and `env-hardware.yaml` for details)
- [pygame](https://www.pygame.org/), [matplotlib](https://matplotlib.org/), [numpy](https://numpy.org/), [scipy](https://scipy.org/)
- For hardware: [pyserial](https://pyserial.readthedocs.io/en/latest/)

---

## Installation

1. Clone the repository:
	```bash
	git clone <this-repo-url>
	cd Cartesian_Space_Collision_Avoidance
	```
2. Create and activate a conda environment:
	```bash
	conda env create -f env.yaml
	conda activate obstacle-avoidance
	# For hardware:
	conda env create -f env-hardware.yaml
	conda activate obstacle-hw
	```
3. Install additional Python dependencies if needed:
	```bash
	pip install -r requirements.txt  # If such a file exists
	```

---

## Directory Structure

```
Cartesian_Space_Collision_Avoidance/
├── animations/
│   ├── 2d_circle_1obs.py
│   ├── ...
│   ├── modulation/
│   └── trash_animations/
├── hardware/
│   ├── pick_place_ik.py
│   ├── ...
├── homing_codes/
│   ├── dual_going_home.py
│   ├── ...
├── utils/
│   ├── modulation_class.py
│   ├── modulation_class_v2.py
│   ├── ...
│   └── gripper_commands/
├── env.yaml
├── env-hardware.yaml
└── README.md
```

---

## Usage

### 2D/3D Simulations

Run any script in the `animations/` folder to visualize obstacle avoidance:

```bash
python animations/2d_circle_1obs.py
python animations/3d_spherical_1obs.py
```

### Hardware Control

Scripts in `hardware/` allow you to run pick-and-place and obstacle avoidance on real robots (e.g., Franka, Heal):

```bash
python hardware/pick_place_ik.py
python hardware/pick_place_ik_ds_oa.py
```

### Utilities

Use planners and gripper commands from the `utils/` directory as needed in your projects.

---

## Troubleshooting

- Ensure all dependencies are installed and the correct Python environment is activated.
- For hardware, check serial connections and permissions.
- For visualization, ensure your system supports `pygame` and `matplotlib`.

---

## Acknowledgements

This repository was developed at IIT Gandhinagar Robotics Lab. It draws inspiration from various open-source motion planning and robotics projects.
