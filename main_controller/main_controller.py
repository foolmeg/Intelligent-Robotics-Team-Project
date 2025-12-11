"""
Main Controller for Pioneer 3-DX Robot Navigation System in Webots.

This controller integrates:
- LiDAR-based occupancy grid mapping
- D* Lite global path planning
- Dynamic Window Approach (DWA) local motion planning
- Wheel encoder-based odometry (implemented but supervisor ground truth used)

Robot Pose Estimation:
    We implemented a full wheel-encoder-based odometry module, including:
    - Reading encoder values
    - Computing wheel displacements
    - Estimating the robot's forward motion and rotation
    - Updating the robot's pose (x, y, theta)
    
    However, for the final mapping and navigation module, the system uses the
    Supervisor's ground-truth pose instead of the odometry estimate.
    This choice was made because the standard Robot controller cannot access
    accurate global pose information, and the project requires stable pose
    data for mapping and planning.
    
    The odometry implementation remains part of the project (and works), but
    the Supervisor pose is used during actual execution.
"""

import sys
import math
from typing import Tuple, List, Optional

# Import Webots controller library
from controller import Robot, Supervisor, Lidar, Motor

# Import our navigation modules
from occupancy_grid import OccupancyGrid
from dstar_planner import DStarLitePlanner
from dwa_planner import DWAPlanner
from utils import normalize_angle, euclidean_distance


class OdometryEstimator:
    """
    Wheel encoder-based odometry for pose estimation.
    
    This class implements dead reckoning using wheel encoders to estimate
    the robot's position and orientation. While functional, it accumulates
    drift over time, which is why supervisor ground truth is preferred
    for the navigation system.
    """
    
    def __init__(self, wheel_radius: float, axle_length: float):
        """
        Initialize the odometry estimator.
        
        Args:
            wheel_radius: Radius of the wheels in meters
            axle_length: Distance between the wheels (wheel separation) in meters
        """
        self.wheel_radius = wheel_radius
        self.axle_length = axle_length
        
        # Estimated pose (x, y, theta)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        
        # Previous encoder readings (in radians)
        self.prev_left_position = 0.0
        self.prev_right_position = 0.0
        
        # Flag to track if initialized
        self.initialized = False
        
        # Accumulated distance traveled
        self.total_distance = 0.0
    
    def initialize(self, x: float, y: float, theta: float,
                   left_position: float, right_position: float):
        """
        Initialize odometry with known pose and encoder positions.
        
        Args:
            x, y, theta: Initial pose
            left_position: Initial left wheel encoder position (radians)
            right_position: Initial right wheel encoder position (radians)
        """
        self.x = x
        self.y = y
        self.theta = theta
        self.prev_left_position = left_position
        self.prev_right_position = right_position
        self.initialized = True
        self.total_distance = 0.0
        print(f"Odometry initialized at ({x:.3f}, {y:.3f}, {math.degrees(theta):.1f}°)")
    
    def update(self, left_position: float, right_position: float) -> Tuple[float, float, float]:
        """
        Update pose estimate based on new encoder readings.
        
        Uses differential drive kinematics:
        - Compute wheel displacements from encoder differences
        - Calculate linear and angular displacement
        - Update pose using motion model
        
        Args:
            left_position: Current left wheel encoder position (radians)
            right_position: Current right wheel encoder position (radians)
            
        Returns:
            Updated pose (x, y, theta)
        """
        if not self.initialized:
            # Store initial readings and wait for next update
            self.prev_left_position = left_position
            self.prev_right_position = right_position
            self.initialized = True
            return (self.x, self.y, self.theta)
        
        # Compute wheel position changes (in radians)
        delta_left = left_position - self.prev_left_position
        delta_right = right_position - self.prev_right_position
        
        # Convert to linear distances traveled by each wheel
        dist_left = delta_left * self.wheel_radius
        dist_right = delta_right * self.wheel_radius
        
        # Compute robot displacement using differential drive model
        # Linear displacement (forward motion)
        delta_s = (dist_left + dist_right) / 2.0
        
        # Angular displacement (rotation)
        delta_theta = (dist_right - dist_left) / self.axle_length
        
        # Update pose using midpoint integration
        # This is more accurate than simple Euler integration for curved paths
        if abs(delta_theta) < 1e-6:
            # Approximately straight motion
            self.x += delta_s * math.cos(self.theta)
            self.y += delta_s * math.sin(self.theta)
        else:
            # Curved motion - use arc model
            radius = delta_s / delta_theta
            self.x += radius * (math.sin(self.theta + delta_theta) - math.sin(self.theta))
            self.y -= radius * (math.cos(self.theta + delta_theta) - math.cos(self.theta))
        
        # Update heading
        self.theta = normalize_angle(self.theta + delta_theta)
        
        # Track total distance
        self.total_distance += abs(delta_s)
        
        # Store current positions for next update
        self.prev_left_position = left_position
        self.prev_right_position = right_position
        
        return (self.x, self.y, self.theta)
    
    def get_pose(self) -> Tuple[float, float, float]:
        """Get current estimated pose."""
        return (self.x, self.y, self.theta)
    
    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        """Reset odometry to a known pose."""
        self.x = x
        self.y = y
        self.theta = theta
        self.total_distance = 0.0


class Pioneer3DXController:
    """
    Main navigation controller for the Pioneer 3-DX robot.
    
    Integrates perception, mapping, global planning, and local control
    for autonomous navigation with dynamic obstacle avoidance.
    
    Supports both odometry-based and supervisor ground truth pose estimation,
    with ground truth being used by default for better accuracy.
    """
    
    def __init__(self, use_ground_truth: bool = True):
        """
        Initialize the controller and all components.
        
        Args:
            use_ground_truth: If True, use Supervisor ground truth pose.
                              If False, use wheel encoder odometry.
        """
        self.use_ground_truth = use_ground_truth
        
        # Initialize Webots Supervisor (for ground truth pose)
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # Get robot node for ground truth (Supervisor mode)
        self.robot_node = self.robot.getSelf()
        if self.robot_node is None:
            print("Warning: Could not get robot node. Make sure supervisor=TRUE")
            print("Falling back to odometry-based pose estimation.")
            self.use_ground_truth = False
        
        # Initialize motors
        self.left_motor = self.robot.getDevice('left wheel')
        self.right_motor = self.robot.getDevice('right wheel')
        
        # Set motors to velocity control mode
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        
        # Initialize wheel position sensors (encoders) for odometry
        self.left_encoder = self.robot.getDevice('left wheel sensor')
        self.right_encoder = self.robot.getDevice('right wheel sensor')
        
        if self.left_encoder and self.right_encoder:
            self.left_encoder.enable(self.timestep)
            self.right_encoder.enable(self.timestep)
            print("Wheel encoders initialized")
        else:
            print("Warning: Wheel encoders not found!")
        
        # Pioneer 3-DX wheel parameters
        self.wheel_radius = 0.0975  # meters
        self.axle_length = 0.381    # meters (wheel separation)
        self.max_wheel_speed = 6.28  # rad/s
        
        # Initialize odometry estimator
        self.odometry = OdometryEstimator(self.wheel_radius, self.axle_length)
        
        # Initialize LiDAR
        self.lidar = self.robot.getDevice('lidar')
        if self.lidar:
            self.lidar.enable(self.timestep)
            self.lidar.enablePointCloud()
            print(f"LiDAR: {self.lidar.getHorizontalResolution()} beams, "
                  f"FOV: {math.degrees(self.lidar.getFov()):.1f}°, "
                  f"Range: {self.lidar.getMaxRange()}m")
        else:
            print("ERROR: LiDAR not found! Navigation will fail.")
        
        # Initialize occupancy grid
        self.grid = OccupancyGrid(
            width=10.0,
            height=10.0,
            resolution=0.1,
            origin=(-5.0, -5.0)
        )
        
        # Initialize global planner
        self.global_planner = DStarLitePlanner(self.grid)
        
        # Initialize local planner with tuned parameters
        self.local_planner = DWAPlanner()
        self.local_planner.debug = True
        
        # DWA weights
        self.local_planner.set_weights(
            heading=0.3,
            clearance=1.0,
            velocity=1.0,
            path=0.5
        )
        
        # Safety parameters
        self.local_planner.set_safety_params(
            min_clearance=0.08,
            robot_radius=0.22
        )
        
        # Robot kinematic limits
        self.local_planner.set_robot_params(
            max_v=0.4,
            max_w=1.5,
            max_v_accel=0.5,
            max_w_accel=3.0
        )
        
        # Navigation state
        self.goal: Optional[Tuple[float, float]] = None
        self.current_path: List[Tuple[float, float]] = []
        self.is_navigating = False
        self.goal_reached = False
        
        # Control parameters
        self.goal_tolerance = 0.25
        self.replan_distance = 0.3
        
        # Update frequencies
        self.map_update_interval = 3
        self.path_check_interval = 10
        self.replan_interval = 50
        
        # Counters
        self.control_cycle = 0
        
        # Print pose estimation mode
        pose_mode = "Supervisor Ground Truth" if self.use_ground_truth else "Wheel Encoder Odometry"
        print(f"Pose Estimation: {pose_mode}")
        print("Pioneer 3-DX Navigation Controller initialized")
        print("=" * 50)
    
    def get_ground_truth_pose(self) -> Tuple[float, float, float]:
        """Get ground truth pose from Webots Supervisor."""
        if self.robot_node is None:
            return (0.0, 0.0, 0.0)
        
        position = self.robot_node.getPosition()
        x, y = position[0], position[1]
        
        rotation = self.robot_node.getOrientation()
        theta = math.atan2(rotation[3], rotation[0])
        
        return (x, y, theta)
    
    def get_odometry_pose(self) -> Tuple[float, float, float]:
        """Get pose estimate from wheel encoder odometry."""
        if self.left_encoder and self.right_encoder:
            left_pos = self.left_encoder.getValue()
            right_pos = self.right_encoder.getValue()
            return self.odometry.update(left_pos, right_pos)
        return self.odometry.get_pose()
    
    def get_pose(self) -> Tuple[float, float, float]:
        """
        Get current robot pose.
        
        Uses supervisor ground truth if available and enabled,
        otherwise falls back to odometry.
        
        Returns:
            Robot pose (x, y, theta)
        """
        if self.use_ground_truth and self.robot_node is not None:
            return self.get_ground_truth_pose()
        else:
            return self.get_odometry_pose()
    
    def initialize_odometry(self):
        """
        Initialize odometry with ground truth pose.
        
        This is called at startup to align odometry with the actual robot pose.
        Even when using ground truth for navigation, we keep odometry updated
        for comparison and potential fallback.
        """
        if self.left_encoder and self.right_encoder:
            # Get ground truth pose
            gt_pose = self.get_ground_truth_pose()
            
            # Get current encoder positions
            left_pos = self.left_encoder.getValue()
            right_pos = self.right_encoder.getValue()
            
            # Initialize odometry
            self.odometry.initialize(
                gt_pose[0], gt_pose[1], gt_pose[2],
                left_pos, right_pos
            )
    
    def get_lidar_data(self) -> Tuple[List[float], List[float]]:
        """Get LiDAR range and angle data."""
        if self.lidar is None:
            return ([], [])
        
        ranges = list(self.lidar.getRangeImage())
        
        num_beams = len(ranges)
        fov = self.lidar.getFov()
        
        angles = []
        for i in range(num_beams):
            angle = -fov / 2 + (i / max(num_beams - 1, 1)) * fov
            angles.append(angle)
        
        return (ranges, angles)
    
    def update_map(self, pose: Tuple[float, float, float]) -> bool:
        """Update the occupancy grid with new LiDAR scan."""
        ranges, angles = self.get_lidar_data()
        if not ranges:
            return False
        
        max_range = self.lidar.getMaxRange() if self.lidar else 5.0
        changed_cells = self.grid.update(pose, ranges, angles, max_range)
        
        if changed_cells:
            self.global_planner.update_map(changed_cells)
            return True
        
        return False
    
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
    
    def initialize_map(self, num_scans: int = 20):
        """Initialize the occupancy grid by taking multiple scans."""
        print("Initializing map with LiDAR scans...")
        
        for i in range(num_scans):
            self.robot.step(self.timestep)
            pose = self.get_pose()
            self.update_map(pose)
            
            # Also update odometry during initialization
            if self.left_encoder and self.right_encoder:
                self.get_odometry_pose()
        
        print(f"Map initialized with {num_scans} scans")
    
    def set_goal(self, goal: Tuple[float, float]):
        """Set a new navigation goal."""
        self.goal = goal
        self.is_navigating = True
        self.goal_reached = False
        
        pose = self.get_pose()
        self.update_map(pose)
        
        # Debug info
        start_cell = self.grid.world_to_grid(pose[0], pose[1])
        goal_cell = self.grid.world_to_grid(goal[0], goal[1])
        print(f"Start cell: {start_cell}, Goal cell: {goal_cell}")
        
        # Plan path
        self.current_path = self.global_planner.plan(
            (pose[0], pose[1]), goal
        )
        
        if self.current_path:
            print(f"Path planned: {len(self.current_path)} waypoints")
        else:
            print(f"WARNING: No path found to goal {goal}")
    
    def get_local_goal(self, pose: Tuple[float, float, float]) -> Tuple[float, float]:
        """Get the next local goal from the global path."""
        if not self.current_path or len(self.current_path) < 2:
            return self.goal if self.goal else (pose[0], pose[1])
        
        x, y = pose[0], pose[1]
        
        min_dist = float('inf')
        closest_idx = 0
        
        for i, (px, py) in enumerate(self.current_path):
            dist = euclidean_distance((x, y), (px, py))
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        lookahead_dist = 0.6
        
        for i in range(closest_idx, len(self.current_path)):
            px, py = self.current_path[i]
            if euclidean_distance((x, y), (px, py)) > lookahead_dist:
                return (px, py)
        
        return self.current_path[-1]
    
    def check_path_blocked(self, pose: Tuple[float, float, float]) -> bool:
        """Check if the current path is blocked by obstacles."""
        if not self.current_path:
            return False
        
        for px, py in self.current_path:
            cell = self.grid.world_to_grid(px, py)
            if self.grid.is_occupied_inflated(cell[0], cell[1]):
                return True
        
        return False
    
    def navigation_step(self) -> bool:
        """Execute one navigation control cycle."""
        if not self.is_navigating or self.goal is None:
            return True
        
        self.control_cycle += 1
        
        # Get current pose (uses ground truth or odometry based on config)
        pose = self.get_pose()
        x, y, theta = pose
        
        # Also update odometry even if using ground truth (for comparison)
        if self.use_ground_truth and self.left_encoder and self.right_encoder:
            odom_pose = self.get_odometry_pose()
            
            # Periodically compare odometry vs ground truth
            if self.control_cycle % 100 == 0:
                gt_pose = self.get_ground_truth_pose()
                odom_error = euclidean_distance((odom_pose[0], odom_pose[1]), 
                                                 (gt_pose[0], gt_pose[1]))
                print(f"Odometry drift: {odom_error:.3f}m after {self.odometry.total_distance:.2f}m traveled")
        
        # Check if goal reached
        dist_to_goal = euclidean_distance((x, y), self.goal)
        if dist_to_goal < self.goal_tolerance:
            self.stop()
            self.is_navigating = False
            self.goal_reached = True
            print(f"\n*** GOAL REACHED! ***")
            print(f"Final position: ({x:.2f}, {y:.2f})")
            return True
        
        # Update map periodically
        if self.control_cycle % self.map_update_interval == 0:
            self.update_map(pose)
        
        # Check if path needs replanning
        needs_replan = False
        
        if self.control_cycle % self.path_check_interval == 0:
            if self.check_path_blocked(pose):
                print("Path blocked! Replanning...")
                needs_replan = True
        
        if self.control_cycle % self.replan_interval == 0:
            needs_replan = True
        
        if needs_replan:
            self.current_path = self.global_planner.replan((x, y))
            if self.current_path:
                print(f"Replanned: {len(self.current_path)} waypoints")
        
        # Get local goal for DWA
        local_goal = self.get_local_goal(pose)
        
        # Get LiDAR data
        ranges, angles = self.get_lidar_data()
        
        # Compute velocity command using DWA
        v, w = self.local_planner.compute_velocity_command(
            pose=pose,
            goal=local_goal,
            lidar_ranges=ranges,
            lidar_angles=angles,
            global_path=self.current_path,
            dt=self.timestep / 1000.0
        )
        
        # Apply velocity
        self.set_velocity(v, w)
        
        # Debug output
        if self.control_cycle % 30 == 0:
            print(f"Pos: ({x:.2f}, {y:.2f}) | "
                  f"Cmd: v={v:.2f}, w={math.degrees(w):.0f}°/s | "
                  f"Goal dist: {dist_to_goal:.2f}m")
        
        return False
    
    def run(self):
        """Main control loop."""
        print("\n" + "=" * 50)
        print("PIONEER 3-DX NAVIGATION SYSTEM")
        print("=" * 50)
        
        # Wait for sensors to initialize
        for _ in range(5):
            self.robot.step(self.timestep)
        
        # Initialize odometry with ground truth
        self.initialize_odometry()
        
        # Initialize map
        self.initialize_map(num_scans=10)
        
        # Get initial pose
        initial_pose = self.get_pose()
        print(f"Start: ({initial_pose[0]:.2f}, {initial_pose[1]:.2f})")
        
        # Set goal
        goal = (-2.0, -2.0)
        print(f"Goal:  ({goal[0]:.2f}, {goal[1]:.2f})")
        print("=" * 50 + "\n")
        
        self.set_goal(goal)
        
        # Main loop
        while self.robot.step(self.timestep) != -1:
            if self.navigation_step():
                for _ in range(100):
                    self.robot.step(self.timestep)
                break
        
        print("\nController terminated")


# Entry point
if __name__ == "__main__":
    # Set use_ground_truth=False to use odometry instead
    controller = Pioneer3DXController(use_ground_truth=True)
    controller.run()
