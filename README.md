

> **Cartesian Space Collision Avoidance**  
> Designed for robotic manipulators to navigate safely around obstacles in Cartesian space using modulation-based dynamical systems.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

---

<!-- ============================================================
     BANNER IMAGE PLACEHOLDER
================================================================ -->
<div align="center">
  <img src="media/oa_banner.gif" alt="Cartesian Space Collision Avoidance Banner" width="600" style="border-radius: 8px;">
</div>




---

## Table of Contents
- [Overview](#overview)
- [Pipeline Summary](#pipeline-summary)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Mathematical Foundations](#mathematical-foundations)
  - [Real-Time Obstacle Avoidance](#real-time-obstacle-avoidance)
  - [Reactive Pipeline and Parameters](#reactive-pipeline-and-parameters)
  - [Multiple Obstacles](#multiple-obstacles)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)

---

## Overview
This repository provides a collection of algorithms and scripts for collision avoidance in Cartesian space for robotic manipulators. It includes 2D and 3D simulation environments, hardware integration scripts, and utilities for motion planning and obstacle avoidance. The codebase is modular, supporting different robot platforms and obstacle types, and is designed for both research and educational use.

---
<!-- ============================================================
     VISUAL EXAMPLES
================================================================ -->
<div align="center">

| 2D Obstacle Avoidance | 3D Obstacle Avoidance |
|:---:|:---:|
| <img src="media/rho=1.0,%20eta%20=1.0.gif" alt="2D Simulation" width="300"> | <img src="media/sphere.gif" alt="3D Simulation" width="300"> |

</div>

---

## Pipeline Summary
| Stage | Tool/Category | Description |
|---|---|---|
| Simulation | 2D/3D Animations | Run animations to visualize and test collision avoidance algorithms |
| Hardware Integration | Hardware Scripts | Deploy algorithms on real robots |
| Utilities | Modules | Planners, velocity commanders, and gripper control |

---

## Prerequisites
Before installation, ensure the following are available on your system:

| Requirement | Version | Notes |
|---|---|---|
| Ubuntu | 22.04 | Tested environment |
| Python | 3.10+ | Required for conda paths |
| Visualization | - | pygame, matplotlib |
| Scientific Math | - | numpy, scipy |
| Hardware | - | pyserial |

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

---

## Mathematical Foundations

This repository draws its obstacle avoidance theory and implementation from modulation-based approaches. 

### Real-Time Obstacle Avoidance

For an obstacle-centered state $\xi = x - x_o$, the modulated dynamics are:

$$ \dot{\xi} = M(\xi)f(\xi) $$
$$ M(\xi) = E(\xi)D(\xi)E(\xi)^{-1} $$

with $\Gamma(\xi) = 1$ on the boundary and $\Gamma(\xi) > 1$ outside. The eigenvalues of the modulation matrix are formulated as:

$$ \lambda_1 = 1 - \frac{1}{|\Gamma|^{1/\rho}}, \quad \lambda_{2..d} = 1 + \frac{1}{|\Gamma|^{1/\rho}} $$

so the normal velocity is suppressed at the boundary and the flow is redirected tangentially.

<div align="center">

| Free Flow | Circular Obstacle | Polygonal Obstacle |
|:---:|:---:|:---:|
| <img src="media/free_flow.png" alt="Free Flow" width="250" height="190" style="width: 250px; height: 190px; object-fit: contain; border: 1px solid #888; border-radius: 8px; padding: 5px; background: #fff;"> | <img src="media/circle.png" alt="Circular Obstacle" width="250" height="190" style="width: 250px; height: 190px; object-fit: contain; border: 1px solid #888; border-radius: 8px; padding: 5px; background: #fff;"> | <img src="media/poly.png" alt="Polygonal Obstacle" width="250" height="190" style="width: 250px; height: 190px; object-fit: contain; border: 1px solid #888; border-radius: 8px; padding: 5px; background: #fff;"> |

<em>Modulation Flow Dynamics: Free-flow dynamics and obstacle-aware modulation</em>
</div>

### Reactive Pipeline and Parameters

Online execution follows: state input $\rightarrow$ distance evaluation $\Gamma(\tilde{\xi}) \rightarrow$ basis computation $E(\tilde{\xi}) \rightarrow$ modulation $M(\tilde{\xi})f(\xi)$.

```mermaid
graph TD
    classDef default fill:#ffffff,stroke:#3333ff,stroke-width:2px,color:#000000,font-family:sans-serif;
    A["State Input ξ"] --> B["Distance Evaluation Γ(ξ̃)"]
    B --> C["Compute Basis E(ξ̃)"]
    C --> D["Parameter Modulation (ρ, η)"]
    D --> E["Modulate ξ̇ = M(ξ̃)f(ξ)"]
    E --> F["State Update ξ_t+1"]
```

Safety inflation and reactivity are tuned with $\eta$ and $\rho$:
- A **larger $\rho$** gives earlier deflection.
- A **smaller $\eta$** increases clearance.

### Multiple Obstacles

For $K$ convex obstacles, the method is extended by weighting the influence of each obstacle:

$$ \lambda_1^k = 1 - \frac{\omega_k}{|\Gamma_k(\xi_k)|^{1/\rho}}, \quad \lambda_{2..d}^k = 1 + \frac{\omega_k}{|\Gamma_k(\xi_k)|^{1/\rho}} $$

where the weights $\omega_k$ are given by:

$$ \omega_k = \prod_{i \neq k} \frac{\Gamma_i - 1}{(\Gamma_i - 1) + (\Gamma_k - 1)} $$

The overall modulation matrix is the product of individual matrices:

$$ \bar{M}(\xi) = \prod_{k=1}^K M_k(\xi) $$
$$ \dot{\xi} = \bar{M}(\xi)f(\xi) $$

<div align="center">
  <img src="media/elliptical-multi.gif" alt="Multiple Obstacles" width="500"><br>
  <em>Effect of obstacle inflation $\eta$ and reactivity parameter $\rho$ on modulated trajectories</em>
</div>

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
Scripts in `hardware/` allow you to run pick-and-place and obstacle avoidance on real robots:

```bash
python hardware/pick_place_ik.py
python hardware/pick_place_ik_ds_oa.py
```

---

## Troubleshooting
| Problem | Likely Cause | Fix |
|---|---|---|
| Dependencies missing | Environment not activated | Run `conda activate obstacle-avoidance` |
| Hardware connection fails | Serial permissions / connection | Check serial port and run `sudo chmod a+rw /dev/ttyUSB0` |
| Visualization errors | System missing libraries | Ensure `pygame` and `matplotlib` are installed correctly |

---

## Limitations and Future Goals

### Limitations
- **Local Minima and Saddle Points**: The current collision avoidance algorithm can get trapped in local minima or saddle points when dealing with non-convex obstacles.

### Future Goals
- **Non-Convex Obstacles**: Implement strategies to overcome local minima and saddle points for complex, non-convex obstacles.
- **Joint Space Collision Avoidance**: Extend the framework from Cartesian space to joint space to ensure the entire robot body safely avoids collisions.

<div align="center">
  <img src="media/star.gif" alt="Local Minima with Non-Convex Obstacles" width="500" style="border-radius: 8px;">
  <br>
  <em>Visualization of local minima or saddle points with non-convex obstacles</em>
</div>

---

## Acknowledgements

This repository was developed at IIT Gandhinagar Robotics Lab. The theory and implementation of the Cartesian obstacle avoidance algorithm leverages concepts developed for dynamic robotic tasks.


