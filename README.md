# Intelligent-Robotics-Team-Project
Repository for Intelligent Robotics Team Project 2025 
# Robotics Project – Mapping & Odometry Module

## What I Implemented Myself

## 1. Robot Pose Estimation (Odometry Implementation + Supervisor Ground Truth)

I implemented a full wheel-encoder–based odometry module, including:
- Reading encoder values
- Computing wheel displacements
- Estimating the robot’s forward motion and rotation
- Updating the robot’s pose (x, y, theta)

However, for the final mapping and navigation module, the system uses the **Supervisor’s ground-truth pose** instead of the odometry estimate.  
This choice was made because the standard Robot controller cannot access accurate global pose information, and the project requires stable pose data for mapping and planning.

The odometry implementation remains part of the project (and works), but the Supervisor pose is used during actual execution.

---

## 2. LiDAR-to-World Coordinate Transformation

The transformation logic from LiDAR range values to world coordinates was implemented manually.

When converting LiDAR points to the world frame, the system uses the **Supervisor-provided robot pose** to maintain accuracy in mapping.  
The transformation steps (local LiDAR → world coordinates → grid map) were implemented by me.

---

### 3. Occupancy Grid Mapping
I implemented the mapping logic that turns LiDAR data into a real-time occupancy grid:

- Converting world coordinates to grid pixel coordinates  
- Clearing free space using a custom Bresenham line algorithm  
- Marking detected obstacles on the grid  
- Rendering the grid dynamically on the Webots Display  

This entire mapping system (free space clearing, obstacle marking, visualization) was implemented manually.

---

## What Pre-Programmed Packages Were Used

### Webots API
Used only for hardware-level access:

- `Robot`, `Motor`, `PositionSensor`, `Lidar`  
- Raw encoder values  
- Raw LiDAR range readings  
- Display device for drawing the map  
- Built-in Pioneer 3-DX PROTO model  

These components provide sensor data—the processing and mapping logic was implemented by me.

### Python Standard Libraries
- `math`  
- `numpy`  

Used only for basic math operations.
