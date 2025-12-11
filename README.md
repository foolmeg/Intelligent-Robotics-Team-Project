# Intelligent-Robotics-Team-Project
This repository contains our full implementation for the Intelligent Robotics Team Project - Autonomous Navigation System 2025.

Our system performs autonomous navigation in Webots using LiDAR-based mapping, global path planning with D* Lite, and dynamic obstacle avoidance using the Dynamic Window Approach (DWA).

The goal of this project was not to use off-the-shelf SLAM or navigation packages, but to design, implement, and integrate these components ourselves using only the low-level sensor APIs provided by Webots.
# System Overview
Our navigation pipeline is composed of four core modules:

1. Robot Pose Estimation

2. LiDAR-based Occupancy Grid Mapping

3. Global Path Planning using D* Lite

4. Local Collision Avoidance using DWA

A Webots Supervisor-based Pedestrian Controller is also included to create dynamic obstacles.
# What We Implemented Ourselves

## Robot Pose Estimation (Odometry Implementation + Supervisor Ground Truth)

We implemented a full wheel-encoder–based odometry module, including:
- Reading encoder values
- Computing wheel displacements
- Estimating the robot’s forward motion and rotation
- Updating the robot’s pose (x, y, theta)

However, for the final mapping and navigation module, the system uses the **Supervisor’s ground-truth pose** instead of the odometry estimate.  
This choice was made because the standard Robot controller cannot access accurate global pose information, and the project requires stable pose data for mapping and planning.

The odometry implementation remains part of the project (and works), but the Supervisor pose is used during actual execution.

---

## LiDAR-to-World Coordinate Transformation

The transformation logic from LiDAR range values to world coordinates was implemented manually.

When converting LiDAR points to the world frame, the system uses the **Supervisor-provided robot pose** to maintain accuracy in mapping.  
The transformation steps (local LiDAR → world coordinates → grid map) were implemented by us.

---

## Occupancy Grid Mapping
We implemented mapping logic that turns LiDAR data into a real-time occupancy grid:

- Converting world coordinates to grid pixel coordinates  
- Clearing free space using a custom Bresenham line algorithm  
- Marking detected obstacles on the grid  
- Rendering the grid dynamically on the Webots Display  

This entire mapping system (free space clearing, obstacle marking, visualisation) was implemented manually.

---

## Dynamic Window Approach (DWA) Local Planner

We implemented a complete DWA controller used for real-time obstacle avoidance and motion generation.

Our DWA system includes:

Velocity Sampling
- Compute admissible dynamic window based on robot acceleration limits
- Sample feasible (v, ω) pairs

Trajectory Simulation
- Predict future robot poses over a time horizon
- Evaluate trajectory safety using obstacle distances

Scoring Function
Weighted objective combining:
- Heading alignment
- Clearance from obstacles
- Forward velocity
- Path-following bias using D* Lite path waypoints

Collision Checking
- Reject trajectories with predicted collisions
- Penalise low-clearance movements

Detour Commitment Logic
- When the goal direction is blocked, the robot “commits” to a chosen side to reduce oscillations

Fallback Behaviors
- Escape rotation when no valid trajectory exists

This module produces real wheel velocities for the Pioneer robot. All logic, sampling, simulation, and scoring were implemented manually, whilst taking inspiration from the logic from the GitHub https://github.com/AtsushiSakai/PythonRobotics/tree/master/PathPlanning/DynamicWindowApproach

---

## Pedestrian & Dynamic Obstacle Controller

To evaluate dynamic obstacle performance, we created a custom Supervisor-based pedestrian:

- Moves with randomised or bounded waypoint patterns
- Maintains orientation and smooth motion
- Ensures the pedestrian stays within arena boundaries
- Adjustable walking speed & wait times

This module was implemented by us to test unpredictable obstacle interactions

## What Pre-Programmed Packages Were Used

### Webots API
Used only for hardware-level access:

- `Robot`, `Motor`, `PositionSensor`, `Lidar`  
- Raw encoder values  
- Raw LiDAR range readings  
- Display device for drawing the map  
- Built-in Pioneer 3-DX PROTO model  

### Python Standard Libraries
- `math`  
- `numpy`  

Used only for basic math operations.

### References
- https://github.com/AtsushiSakai/PythonRobotics/tree/master/PathPlanning/DynamicWindowApproach
- https://ieeexplore.ieee.org/document/580977

