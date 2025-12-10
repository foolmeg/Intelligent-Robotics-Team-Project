"""
Dynamic Window Approach (DWA) Local Planner for Pioneer 3-DX Navigation System.
Fixed version with committed detour direction and better obstacle filtering.
"""

import math
import numpy as np
from typing import Tuple, List, Optional
from utils import normalize_angle, euclidean_distance


class DWAPlanner:
    """
    Dynamic Window Approach local planner for real-time obstacle avoidance.
    """
    
    def __init__(self):
        """Initialize DWA planner."""
        
        # Robot parameters
        self.max_linear_velocity = 0.4
        self.min_linear_velocity = -0.1
        self.max_angular_velocity = 1.5
        self.max_linear_accel = 0.5
        self.max_angular_accel = 3.0
        
        # Robot geometry
        self.robot_radius = 0.25
        
        # DWA sampling
        self.v_resolution = 0.05
        self.w_resolution = 0.15
        
        # Trajectory prediction
        self.predict_time = 2.5
        self.dt = 0.15
        
        # Weights
        self.heading_weight = 0.3
        self.clearance_weight = 0.8
        self.velocity_weight = 1.2      # High priority on moving
        self.path_weight = 0.3
        
        # Safety - REDUCED to allow closer approach
        self.min_clearance = 0.08       # Very small - just avoid collision
        self.goal_tolerance = 0.2
        
        # Only consider obstacles within this range (ignore distant walls)
        self.obstacle_detection_range = 2.0   # Reduced from 5.0
        
        # Current state
        self.current_v = 0.0
        self.current_w = 0.0
        
        # COMMITTED detour direction (to avoid oscillation)
        self.committed_detour = None    # +1 for left, -1 for right
        self.detour_commit_count = 0
        self.detour_commit_threshold = 15  # Commit for this many cycles
        
        # Debug flag
        self.debug = False
        
    def compute_velocity_command(self,
                                 pose: Tuple[float, float, float],
                                 goal: Tuple[float, float],
                                 lidar_ranges: List[float],
                                 lidar_angles: List[float],
                                 global_path: Optional[List[Tuple[float, float]]] = None,
                                 dt: float = 0.1) -> Tuple[float, float]:
        """Compute the best velocity command using DWA."""
        x, y, theta = pose
        
        # Check if at goal
        dist_to_goal = euclidean_distance((x, y), goal)
        if dist_to_goal < self.goal_tolerance:
            self.current_v = 0.0
            self.current_w = 0.0
            self.committed_detour = None
            return (0.0, 0.0)
        
        # Get nearby obstacles only (filter out distant walls)
        obstacles = self._get_nearby_obstacles(lidar_ranges, lidar_angles)
        
        # Transform goal to robot frame
        goal_robot = self._world_to_robot(goal[0], goal[1], x, y, theta)
        goal_angle = math.atan2(goal_robot[1], goal_robot[0])
        
        # Check if goal direction is blocked
        goal_blocked, blocking_side = self._check_goal_blocked(goal_angle, obstacles)
        
        # Manage committed detour direction
        if goal_blocked:
            if self.committed_detour is None:
                # Commit to a detour direction based on which side is clearer
                self.committed_detour = blocking_side
                self.detour_commit_count = self.detour_commit_threshold
            elif self.detour_commit_count > 0:
                self.detour_commit_count -= 1
            else:
                # Re-evaluate after commitment expires
                self.committed_detour = blocking_side
                self.detour_commit_count = self.detour_commit_threshold
        else:
            # Goal is clear - gradually release commitment
            if self.detour_commit_count > 0:
                self.detour_commit_count -= 1
            else:
                self.committed_detour = None
        
        # Debug output
        if self.debug:
            status = "BLOCKED" if goal_blocked else "clear"
            detour_str = f"detour={'L' if self.committed_detour == 1 else 'R' if self.committed_detour == -1 else 'none'}"
            print(f"DWA: {len(obstacles)} nearby obs, goal={status}, {detour_str}")
        
        # Dynamic window
        v_min, v_max, w_min, w_max = self._compute_dynamic_window(dt)
        
        best_score = float('-inf')
        best_v = 0.0
        best_w = 0.0
        found_admissible = False
        
        # Sample velocities
        for v in np.arange(v_min, v_max + self.v_resolution, self.v_resolution):
            for w in np.arange(w_min, w_max + self.w_resolution, self.w_resolution):
                
                # Simulate trajectory
                traj = self._simulate_trajectory(v, w)
                
                # Check clearance
                clearance = self._trajectory_clearance(traj, obstacles)
                
                if clearance < self.min_clearance:
                    continue
                
                found_admissible = True
                
                # Compute heading score
                if self.committed_detour is not None:
                    # Steer in committed detour direction
                    detour_angle = goal_angle + self.committed_detour * (math.pi / 3)  # 60° offset
                    heading_score = self._direction_score(traj[-1], detour_angle)
                else:
                    heading_score = self._heading_score(traj[-1], goal_robot)
                
                clearance_score = min(clearance / 0.5, 1.0)
                velocity_score = max(0, v) / self.max_linear_velocity
                
                # Bonus for moving forward
                if v > 0.15:
                    velocity_score += 0.15
                
                score = (self.heading_weight * heading_score +
                        self.clearance_weight * clearance_score +
                        self.velocity_weight * velocity_score)
                
                if score > best_score:
                    best_score = score
                    best_v = v
                    best_w = w
        
        # If no admissible trajectory, use simple reactive avoidance
        if not found_admissible:
            best_v, best_w = self._reactive_avoid(obstacles, goal_robot)
        
        self.current_v = best_v
        self.current_w = best_w
        
        return (best_v, best_w)
    
    def _get_nearby_obstacles(self, ranges: List[float], 
                               angles: List[float]) -> List[Tuple[float, float]]:
        """Get only nearby obstacles, filtering out distant walls."""
        obstacles = []
        
        for dist, angle in zip(ranges, angles):
            # Only include close obstacles
            if dist < 0.08 or dist > self.obstacle_detection_range:
                continue
            if math.isinf(dist) or math.isnan(dist):
                continue
            
            ox = dist * math.cos(angle)
            oy = dist * math.sin(angle)
            obstacles.append((ox, oy))
        
        return obstacles
    
    def _check_goal_blocked(self, goal_angle: float, 
                            obstacles: List[Tuple[float, float]]) -> Tuple[bool, int]:
        """
        Check if goal direction is blocked and determine which way to go around.
        Returns (is_blocked, suggested_side) where side is +1 for left, -1 for right.
        """
        blocked = False
        left_clear = 0.0
        right_clear = 0.0
        
        for ox, oy in obstacles:
            obs_dist = math.sqrt(ox**2 + oy**2)
            obs_angle = math.atan2(oy, ox)
            angle_diff = normalize_angle(obs_angle - goal_angle)
            
            # Check if obstacle is in goal direction (within 25° cone)
            if abs(angle_diff) < 0.45 and obs_dist < 1.2:
                blocked = True
            
            # Accumulate clearance on each side
            if -1.5 < angle_diff < -0.2 and obs_dist < 1.5:
                right_clear -= 1.0 / max(obs_dist, 0.3)
            elif 0.2 < angle_diff < 1.5 and obs_dist < 1.5:
                left_clear -= 1.0 / max(obs_dist, 0.3)
        
        # Choose side with more clearance (less negative = more clear)
        suggested_side = 1 if left_clear > right_clear else -1
        
        return blocked, suggested_side
    
    def _compute_dynamic_window(self, dt: float) -> Tuple[float, float, float, float]:
        """Compute velocity bounds."""
        v_min = max(self.min_linear_velocity, self.current_v - self.max_linear_accel * dt)
        v_max = min(self.max_linear_velocity, self.current_v + self.max_linear_accel * dt)
        w_min = max(-self.max_angular_velocity, self.current_w - self.max_angular_accel * dt)
        w_max = min(self.max_angular_velocity, self.current_w + self.max_angular_accel * dt)
        return (v_min, v_max, w_min, w_max)
    
    def _simulate_trajectory(self, v: float, w: float) -> List[Tuple[float, float, float]]:
        """Simulate trajectory in robot frame."""
        traj = [(0.0, 0.0, 0.0)]
        x, y, th = 0.0, 0.0, 0.0
        steps = int(self.predict_time / self.dt)
        
        for _ in range(steps):
            x += v * math.cos(th) * self.dt
            y += v * math.sin(th) * self.dt
            th += w * self.dt
            traj.append((x, y, th))
        
        return traj
    
    def _trajectory_clearance(self, traj: List[Tuple[float, float, float]],
                              obstacles: List[Tuple[float, float]]) -> float:
        """Compute minimum clearance along trajectory."""
        if not obstacles:
            return float('inf')
        
        min_clear = float('inf')
        
        for tx, ty, _ in traj:
            for ox, oy in obstacles:
                dist = math.sqrt((tx - ox)**2 + (ty - oy)**2) - self.robot_radius
                if dist < min_clear:
                    min_clear = dist
        
        return max(0.0, min_clear)
    
    def _world_to_robot(self, wx: float, wy: float, 
                        rx: float, ry: float, rtheta: float) -> Tuple[float, float]:
        """Transform world point to robot frame."""
        dx = wx - rx
        dy = wy - ry
        cos_t = math.cos(-rtheta)
        sin_t = math.sin(-rtheta)
        return (dx * cos_t - dy * sin_t, dx * sin_t + dy * cos_t)
    
    def _heading_score(self, final_pose: Tuple[float, float, float],
                       goal_robot: Tuple[float, float]) -> float:
        """Score based on heading toward goal."""
        fx, fy, ftheta = final_pose
        gx, gy = goal_robot
        
        dx = gx - fx
        dy = gy - fy
        
        if abs(dx) < 0.01 and abs(dy) < 0.01:
            return 1.0
        
        goal_angle = math.atan2(dy, dx)
        angle_diff = abs(normalize_angle(goal_angle - ftheta))
        
        return (math.pi - angle_diff) / math.pi
    
    def _direction_score(self, final_pose: Tuple[float, float, float],
                         target_direction: float) -> float:
        """Score based on alignment with a target direction."""
        fx, fy, ftheta = final_pose
        angle_diff = abs(normalize_angle(target_direction - ftheta))
        return (math.pi - angle_diff) / math.pi
    
    def _reactive_avoid(self, obstacles: List[Tuple[float, float]],
                        goal_robot: Tuple[float, float]) -> Tuple[float, float]:
        """Simple reactive avoidance when DWA can't find a path."""
        
        if not obstacles:
            # No obstacles - go toward goal
            goal_angle = math.atan2(goal_robot[1], goal_robot[0])
            return (0.25, np.clip(goal_angle, -1.0, 1.0))
        
        # Find closest obstacle
        min_dist = float('inf')
        closest_angle = 0
        
        for ox, oy in obstacles:
            d = math.sqrt(ox**2 + oy**2)
            if d < min_dist:
                min_dist = d
                closest_angle = math.atan2(oy, ox)
        
        # Turn away from closest obstacle
        if self.committed_detour is not None:
            turn_dir = self.committed_detour
        else:
            turn_dir = -1 if closest_angle > 0 else 1
        
        # Speed based on clearance
        if min_dist > 0.5:
            v = 0.2
            w = turn_dir * 0.8
        elif min_dist > 0.35:
            v = 0.12
            w = turn_dir * 1.2
        else:
            v = 0.05
            w = turn_dir * 1.5
        
        return (v, w)
    
    def set_weights(self, heading: float = None, clearance: float = None, 
                   velocity: float = None, path: float = None):
        """Set objective function weights."""
        if heading is not None:
            self.heading_weight = heading
        if clearance is not None:
            self.clearance_weight = clearance
        if velocity is not None:
            self.velocity_weight = velocity
        if path is not None:
            self.path_weight = path
        print(f"DWA weights: heading={self.heading_weight}, clearance={self.clearance_weight}, "
              f"velocity={self.velocity_weight}, path={self.path_weight}")
    
    def set_safety_params(self, min_clearance: float = None, robot_radius: float = None):
        """Set safety parameters."""
        if min_clearance is not None:
            self.min_clearance = min_clearance
        if robot_radius is not None:
            self.robot_radius = robot_radius
        print(f"DWA safety: min_clearance={self.min_clearance}, robot_radius={self.robot_radius}")
    
    def set_robot_params(self, max_v: float = None, max_w: float = None,
                        max_v_accel: float = None, max_w_accel: float = None,
                        robot_radius: float = None):
        """Set robot kinematic parameters."""
        if max_v is not None:
            self.max_linear_velocity = max_v
        if max_w is not None:
            self.max_angular_velocity = max_w
        if max_v_accel is not None:
            self.max_linear_accel = max_v_accel
        if max_w_accel is not None:
            self.max_angular_accel = max_w_accel
        if robot_radius is not None:
            self.robot_radius = robot_radius
    
    def reset(self):
        """Reset planner state."""
        self.current_v = 0.0
        self.current_w = 0.0
        self.committed_detour = None
        self.detour_commit_count = 0
