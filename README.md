# Intelligent-Robotics-Team-Project
Repository for Intelligent Robotics Team Project 2025 
# Robotics Project – Mapping & Odometry Module

## What I Implemented Myself

### 1. Odometry Computation (Wheel Encoders → Robot Pose)
I implemented the entire odometry module manually. This includes:

- Reading raw wheel encoder values every timestep  
- Computing wheel movement between updates  
- Estimating the robot’s forward motion and rotation  
- Updating the robot’s global pose (x, y, theta)  
- Handling missing or invalid sensor values safely  

Webots does not provide the robot’s pose directly.  
Everything related to computing the pose from encoder values was implemented by me.

---

### 2. LiDAR-to-World Coordinate Transformation
I implemented the full conversion pipeline from raw LiDAR range data to world-coordinate points:

- Converting each LiDAR measurement into a 2D point  
- Applying the robot’s current pose (from odometry) to transform those points into world space  
- Filtering out invalid distance readings  
- Preparing transformed points so they can be mapped onto a global occupancy grid  

Webots only provides raw distance measurements; the transformation logic is fully implemented by me.

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

---

## Project Structure (Example)
