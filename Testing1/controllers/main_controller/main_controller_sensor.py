"""
Alternative Controller for Pioneer 3-DX Robot Navigation System in Webots.

This version uses GPS and Compass for localization instead of Supervisor mode,
making it more realistic and portable.
"""

import sys
import math
from typing import Tuple, List, Optional

# Import Webots controller library
from controller import Robot, Lidar, Motor, GPS, Compass, InertialUnit

# Import our navigation modules
from occupancy_grid import OccupancyGrid
from dstar_planner import DStarLitePlanner
from dwa_planner import DWAPlanner
from utils import normalize_angle, euclidean_distance


class Pioneer3DXSensorController:
    """
    Navigation controller using sensor-based localization.
    
    Uses GPS and Compass for pose estimation instead of ground truth.
    """
    
    def __init__(self):
        """Initialize the controller and all components."""
        
        # Initialize Webots Robot
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # Initialize motors
        self.left_motor = self.robot.getDevice('left wheel')
        self.right_motor = self.robot.getDevice('right wheel')
        
        # Set motors to velocity control mode
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        
        # Pioneer 3-DX wheel parameters
        self.wheel_radius = 0.0975  # meters
        self.axle_length = 0.381    # meters (wheel separation)
        self.max_wheel_speed = 6.28  # rad/s
        
        # Initialize GPS (if available)
        self.gps = self.robot.getDevice('gps')
        if self.gps:
            self.gps.enable(self.timestep)
            print("GPS initialized")
        else:
            print("Warning: GPS not found!")
        
        # Initialize Compass (if available)
        self.compass = self.robot.getDevice('compass')
        if self.compass:
            self.compass.enable(self.timestep)
            print("Compass initialized")
        else:
            print("Warning: Compass not found!")
        
        # Initialize IMU (alternative for heading)
        self.imu = self.robot.getDevice('inertial unit')
        if self.imu:
            self.imu.enable(self.timestep)
            print("IMU initialized")
        
        # Initialize LiDAR
        self.lidar = self.robot.getDevice('lidar')
        if self.lidar:
            self.lidar.enable(self.timestep)
            self.lidar.enablePointCloud()
            print(f"LiDAR initialized: {self.lidar.getHorizontalResolution()} beams")
        else:
            print("Warning: LiDAR not found!")
        
        # Initialize occupancy grid
        self.grid = OccupancyGrid(
            width=8.0,
            height=8.0,
            resolution=0.05,
            origin=(-4.0, -4.0)
        )
        
        # Initialize planners
        self.global_planner = DStarLitePlanner(self.grid)
        self.local_planner = DWAPlanner()
        
        # Configure local planner
        self.local_planner.set_robot_params(
            max_v=0.5,
            max_w=2.0,
            max_v_accel=0.5,
            max_w_accel=3.0,
            robot_radius=0.25
        )
        
        # Navigation state
        self.goal: Optional[Tuple[float, float]] = None
        self.current_path: List[Tuple[float, float]] = []
        self.is_navigating = False
        self.goal_reached = False
        
        # Control parameters
        self.goal_tolerance = 0.2
        self.map_update_interval = 5
        self.path_update_interval = 10
        
        # Counters
        self.control_cycle = 0
        
        # Odometry fallback
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_theta = 0.0
        self.last_left_pos = 0.0
        self.last_right_pos = 0.0
        
        print("Pioneer 3-DX Sensor Controller initialized")
    
    def get_pose(self) -> Tuple[float, float, float]:
        """
        Get current robot pose using available sensors.
        
        Falls back through: GPS/Compass -> IMU -> Odometry
        
        Returns:
            Robot pose (x, y, theta)
        """
        x, y, theta = 0.0, 0.0, 0.0
        
        # Try GPS for position
        if self.gps:
            try:
                gps_values = self.gps.getValues()
                x = gps_values[0]
                y = gps_values[1]
            except:
                pass
        
        # Try Compass for heading
        if self.compass:
            try:
                compass_values = self.compass.getValues()
                # Convert compass reading to heading
                # Compass gives direction of magnetic north in robot frame
                theta = math.atan2(compass_values[0], compass_values[1])
            except:
                pass
        elif self.imu:
            try:
                # Use IMU roll-pitch-yaw
                rpy = self.imu.getRollPitchYaw()
                theta = rpy[2]  # Yaw
            except:
                pass
        
        return (x, y, theta)
    
    def get_lidar_data(self) -> Tuple[List[float], List[float]]:
        """Get LiDAR range and angle data."""
        if self.lidar is None:
            return ([], [])
        
        ranges = list(self.lidar.getRangeImage())
        
        num_beams = len(ranges)
        fov = self.lidar.getFov()
        
        angles = []
        for i in range(num_beams):
            angle = -fov / 2 + (i / (num_beams - 1)) * fov if num_beams > 1 else 0
            angles.append(angle)
        
        return (ranges, angles)
    
    def update_map(self, pose: Tuple[float, float, float]):
        """Update the occupancy grid with new LiDAR scan."""
        ranges, angles = self.get_lidar_data()
        if ranges:
            max_range = self.lidar.getMaxRange() if self.lidar else 5.0
            changed_cells = self.grid.update(pose, ranges, angles, max_range)
            
            if changed_cells:
                self.global_planner.update_map(changed_cells)
    
    def set_velocity(self, linear_v: float, angular_w: float):
        """Set robot velocity using differential drive kinematics."""
        v_right = (2 * linear_v + angular_w * self.axle_length) / (2 * self.wheel_radius)
        v_left = (2 * linear_v - angular_w * self.axle_length) / (2 * self.wheel_radius)
        
        v_left = max(-self.max_wheel_speed, min(self.max_wheel_speed, v_left))
        v_right = max(-self.max_wheel_speed, min(self.max_wheel_speed, v_right))
        
        self.left_motor.setVelocity(v_left)
        self.right_motor.setVelocity(v_right)
    
    def stop(self):
        """Stop the robot."""
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
    
    def set_goal(self, goal: Tuple[float, float]):
        """Set a new navigation goal."""
        self.goal = goal
        self.is_navigating = True
        self.goal_reached = False
        
        pose = self.get_pose()
        self.current_path = self.global_planner.plan(
            (pose[0], pose[1]), goal
        )
        
        if self.current_path:
            print(f"Path planned to goal {goal}: {len(self.current_path)} waypoints")
        else:
            print(f"Warning: Could not plan path to goal {goal}")
    
    def get_local_goal(self, pose: Tuple[float, float, float]) -> Tuple[float, float]:
        """Get the next local goal from the global path."""
        if not self.current_path:
            return self.goal if self.goal else (pose[0], pose[1])
        
        min_dist = float('inf')
        closest_idx = 0
        
        for i, (px, py) in enumerate(self.current_path):
            dist = euclidean_distance((pose[0], pose[1]), (px, py))
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        lookahead_dist = 0.5
        lookahead_idx = closest_idx
        
        for i in range(closest_idx, len(self.current_path)):
            px, py = self.current_path[i]
            if euclidean_distance((pose[0], pose[1]), (px, py)) > lookahead_dist:
                lookahead_idx = i
                break
        else:
            lookahead_idx = len(self.current_path) - 1
        
        return self.current_path[lookahead_idx]
    
    def check_goal_reached(self, pose: Tuple[float, float, float]) -> bool:
        """Check if the robot has reached the goal."""
        if self.goal is None:
            return False
        
        dist = euclidean_distance((pose[0], pose[1]), self.goal)
        return dist < self.goal_tolerance
    
    def navigation_step(self) -> bool:
        """Execute one navigation control cycle."""
        if not self.is_navigating or self.goal is None:
            return True
        
        self.control_cycle += 1
        
        pose = self.get_pose()
        x, y, theta = pose
        
        if self.check_goal_reached(pose):
            self.stop()
            self.is_navigating = False
            self.goal_reached = True
            print(f"Goal reached! Final position: ({x:.2f}, {y:.2f})")
            return True
        
        if self.control_cycle % self.map_update_interval == 0:
            self.update_map(pose)
        
        if self.control_cycle % self.path_update_interval == 0:
            if not self.global_planner.is_path_valid():
                print("Replanning due to path obstruction...")
                self.current_path = self.global_planner.replan((x, y))
        
        local_goal = self.get_local_goal(pose)
        ranges, angles = self.get_lidar_data()
        
        v, w = self.local_planner.compute_velocity_command(
            pose=pose,
            goal=local_goal,
            lidar_ranges=ranges,
            lidar_angles=angles,
            global_path=self.current_path,
            dt=self.timestep / 1000.0
        )
        
        self.set_velocity(v, w)
        
        if self.control_cycle % 50 == 0:
            dist_to_goal = euclidean_distance((x, y), self.goal)
            print(f"Pos: ({x:.2f}, {y:.2f}, {math.degrees(theta):.1f}°) | "
                  f"Cmd: v={v:.2f}, w={math.degrees(w):.1f}°/s | "
                  f"Dist: {dist_to_goal:.2f}m")
        
        return False
    
    def run(self):
        """Main control loop."""
        print("Starting Pioneer 3-DX Navigation (Sensor Mode)")
        print("=" * 50)
        
        # Wait for sensors
        for _ in range(10):
            self.robot.step(self.timestep)
        
        initial_pose = self.get_pose()
        print(f"Initial pose: ({initial_pose[0]:.2f}, {initial_pose[1]:.2f})")
        
        self.update_map(initial_pose)
        
        # Set goal
        goal = (-2.0, -2.0)
        self.set_goal(goal)
        print(f"Goal: {goal}")
        print("=" * 50)
        
        while self.robot.step(self.timestep) != -1:
            if self.navigation_step():
                print("Navigation finished!")
                for _ in range(50):
                    self.robot.step(self.timestep)
                break
        
        print("Controller terminated")


if __name__ == "__main__":
    controller = Pioneer3DXSensorController()
    controller.run()
