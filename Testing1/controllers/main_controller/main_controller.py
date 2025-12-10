"""
Main Controller for Pioneer 3-DX Robot Navigation System in Webots.

This controller integrates:
- LiDAR-based occupancy grid mapping
- D* Lite global path planning
- Dynamic Window Approach (DWA) local motion planning

The robot navigates to a goal position while dynamically avoiding obstacles.
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


class Pioneer3DXController:
    """
    Main navigation controller for the Pioneer 3-DX robot.
    
    Integrates perception, mapping, global planning, and local control
    for autonomous navigation with dynamic obstacle avoidance.
    """
    
    def __init__(self):
        """Initialize the controller and all components."""
        
        # Initialize Webots Supervisor (for ground truth pose)
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # Get robot node for ground truth (Supervisor mode)
        self.robot_node = self.robot.getSelf()
        if self.robot_node is None:
            print("Warning: Could not get robot node. Make sure supervisor=TRUE")
        
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
        # Grid covers 10m x 10m area centered at origin
        # Using 0.1m resolution for speed (100x100 = 10,000 cells vs 200x200 = 40,000)
        self.grid = OccupancyGrid(
            width=10.0,
            height=10.0,
            resolution=0.1,  # 10cm resolution (faster than 5cm)
            origin=(-5.0, -5.0)
        )
        
        # Initialize global planner
        self.global_planner = DStarLitePlanner(self.grid)
        
        # Initialize local planner with tuned parameters
        self.local_planner = DWAPlanner()
        self.local_planner.debug = True  # Enable debug output
        
        # ============================================================
        # DWA PARAMETER TUNING - ADJUSTED FOR OBSTACLE AVOIDANCE
        # ============================================================
        # Weights: lower heading = willing to turn away from goal to avoid obstacles
        self.local_planner.set_weights(
            heading=0.3,      # LOW - allow turning away from goal
            clearance=1.0,    # Prefer safer paths
            velocity=1.0,     # HIGH - prefer moving forward
            path=0.5          # Somewhat follow global path
        )
        
        # Safety parameters
        self.local_planner.set_safety_params(
            min_clearance=0.08,   # was 0.15
            robot_radius=0.22     # was 0.25
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
        self.goal_tolerance = 0.25  # meters
        self.replan_distance = 0.3  # Replan if robot deviates this much from path
        
        # Update frequencies
        self.map_update_interval = 3     # Update map every N cycles
        self.path_check_interval = 10    # Check path validity every N cycles
        self.replan_interval = 50        # Force replan every N cycles
        
        # Counters
        self.control_cycle = 0
        
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
    
    def get_lidar_data(self) -> Tuple[List[float], List[float]]:
        """Get LiDAR range and angle data."""
        if self.lidar is None:
            return ([], [])
        
        ranges = list(self.lidar.getRangeImage())
        
        num_beams = len(ranges)
        fov = self.lidar.getFov()
        
        angles = []
        for i in range(num_beams):
            # Angles from -fov/2 to +fov/2
            angle = -fov / 2 + (i / max(num_beams - 1, 1)) * fov
            angles.append(angle)
        
        return (ranges, angles)
    
    def update_map(self, pose: Tuple[float, float, float]) -> bool:
        """
        Update the occupancy grid with new LiDAR scan.
        
        Returns:
            True if map changed significantly
        """
        ranges, angles = self.get_lidar_data()
        if not ranges:
            return False
        
        max_range = self.lidar.getMaxRange() if self.lidar else 5.0
        changed_cells = self.grid.update(pose, ranges, angles, max_range)
        
        # Update global planner with map changes
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
        """
        Initialize the occupancy grid by taking multiple scans.
        This helps detect static obstacles before planning.
        """
        print("Initializing map with LiDAR scans...")
        
        for i in range(num_scans):
            self.robot.step(self.timestep)
            pose = self.get_ground_truth_pose()
            self.update_map(pose)
        
        print(f"Map initialized with {num_scans} scans")
    
    def set_goal(self, goal: Tuple[float, float]):
        """Set a new navigation goal."""
        self.goal = goal
        self.is_navigating = True
        self.goal_reached = False
        
        # Make sure map is up to date
        pose = self.get_ground_truth_pose()
        self.update_map(pose)
        
        # Debug: Check start and goal cells
        start_cell = self.grid.world_to_grid(pose[0], pose[1])
        goal_cell = self.grid.world_to_grid(goal[0], goal[1])
        print(f"Start cell: {start_cell}, Goal cell: {goal_cell}")
        print(f"Start status: {self.grid.debug_cell(pose[0], pose[1])}")
        print(f"Goal status: {self.grid.debug_cell(goal[0], goal[1])}")
        
        # Plan path using D* Lite
        self.current_path = self.global_planner.plan(
            (pose[0], pose[1]), goal
        )
        
        if self.current_path:
            print(f"Path planned: {len(self.current_path)} waypoints")
        else:
            print(f"WARNING: No path found to goal {goal}")
            print("Will use DWA to navigate directly toward goal...")
    
    def get_local_goal(self, pose: Tuple[float, float, float]) -> Tuple[float, float]:
        """
        Get the next local goal from the global path.
        Uses lookahead distance to find a point ahead on the path.
        """
        # If no path, navigate directly to final goal
        if not self.current_path or len(self.current_path) < 2:
            return self.goal if self.goal else (pose[0], pose[1])
        
        x, y = pose[0], pose[1]
        
        # Find closest point on path
        min_dist = float('inf')
        closest_idx = 0
        
        for i, (px, py) in enumerate(self.current_path):
            dist = euclidean_distance((x, y), (px, py))
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Look ahead on the path
        lookahead_dist = 0.6  # meters
        
        for i in range(closest_idx, len(self.current_path)):
            px, py = self.current_path[i]
            if euclidean_distance((x, y), (px, py)) > lookahead_dist:
                return (px, py)
        
        # Return last point if near end of path
        return self.current_path[-1]
    
    def check_path_blocked(self, pose: Tuple[float, float, float]) -> bool:
        """Check if the current path is blocked by obstacles."""
        if not self.current_path:
            return False
        
        # Check each path waypoint against inflated obstacles
        for px, py in self.current_path:
            cell = self.grid.world_to_grid(px, py)
            if self.grid.is_occupied_inflated(cell[0], cell[1]):
                return True
        
        return False
    
    def navigation_step(self) -> bool:
        """
        Execute one navigation control cycle.
        
        Returns:
            True if navigation is complete
        """
        if not self.is_navigating or self.goal is None:
            return True
        
        self.control_cycle += 1
        
        # Get current pose
        pose = self.get_ground_truth_pose()
        x, y, theta = pose
        
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
            map_changed = self.update_map(pose)
        
        # Check if path needs replanning
        needs_replan = False
        
        if self.control_cycle % self.path_check_interval == 0:
            if self.check_path_blocked(pose):
                print("Path blocked! Replanning...")
                needs_replan = True
        
        # Periodic replanning
        if self.control_cycle % self.replan_interval == 0:
            needs_replan = True
        
        # Replan if needed
        if needs_replan:
            self.current_path = self.global_planner.replan((x, y))
            if self.current_path:
                print(f"Replanned: {len(self.current_path)} waypoints")
        
        # Get local goal for DWA
        local_goal = self.get_local_goal(pose)
        
        # Get LiDAR data
        ranges, angles = self.get_lidar_data()
        
        # Debug: Find actual closest obstacle in ANY direction
        if self.control_cycle % 30 == 0 and ranges and angles:
            min_dist = float('inf')
            min_angle = 0
            for i, (r, a) in enumerate(zip(ranges, angles)):
                if 0.05 < r < min_dist and not math.isinf(r):
                    min_dist = r
                    min_angle = a
            
            # Also check what's in the direction we're heading (toward goal)
            goal_angle = math.atan2(local_goal[1] - y, local_goal[0] - x) - theta
            goal_angle = math.atan2(math.sin(goal_angle), math.cos(goal_angle))  # normalize
            
            # Find beams near goal direction
            goal_dir_dist = float('inf')
            for r, a in zip(ranges, angles):
                angle_diff = abs(math.atan2(math.sin(a - goal_angle), math.cos(a - goal_angle)))
                if angle_diff < 0.3 and 0.05 < r < goal_dir_dist and not math.isinf(r):  # ~17 degree cone
                    goal_dir_dist = r
            
            print(f"Closest obs: {min_dist:.2f}m at {math.degrees(min_angle):.0f}°, "
                  f"Goal dir obs: {goal_dir_dist:.2f}m, Goal angle: {math.degrees(goal_angle):.0f}°")
        
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
        
        # Initialize map by scanning environment (fewer scans = faster)
        self.initialize_map(num_scans=10)
        
        # Get initial pose
        initial_pose = self.get_ground_truth_pose()
        print(f"Start: ({initial_pose[0]:.2f}, {initial_pose[1]:.2f})")
        
        # ============================================================
        # SET YOUR GOAL HERE
        # ============================================================
        goal = (-2.0, -2.0)
        print(f"Goal:  ({goal[0]:.2f}, {goal[1]:.2f})")
        print("=" * 50 + "\n")
        
        self.set_goal(goal)
        
        # Main loop
        while self.robot.step(self.timestep) != -1:
            if self.navigation_step():
                # Navigation complete - wait and exit
                for _ in range(100):
                    self.robot.step(self.timestep)
                break
        
        print("\nController terminated")


# Entry point
if __name__ == "__main__":
    controller = Pioneer3DXController()
    controller.run()
