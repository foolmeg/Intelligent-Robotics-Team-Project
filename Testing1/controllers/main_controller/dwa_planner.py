"""
Dynamic Window Approach (DWA) Local Planner for Pioneer 3-DX Navigation System.
Improved obstacle filtering, more accurate simulation timestep handling,
better reactive fallback (angular velocity units fixed), and optional
path-following objective to follow the global planner's path.

Adjusted to avoid backwards motion by default (min_linear_velocity=0.0)
and to sample the dynamic window with an adaptive linspace so that small
positive forward speeds are considered even when the controller dt is small.
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
        # Prevent reversing by default to avoid moving away from forward goal
        self.min_linear_velocity = 0.0
        self.max_angular_velocity = 1.5
        self.max_linear_accel = 0.5
        self.max_angular_accel = 3.0

        # Robot geometry
        self.robot_radius = 0.25

        # DWA sampling (base resolutions, used as a guideline)
        self.v_resolution = 0.05
        self.w_resolution = 0.15

        # Trajectory prediction
        self.predict_time = 2.5
        self.dt = 0.15  # simulation integration step (seconds)

        # Weights
        self.heading_weight = 0.3
        self.clearance_weight = 0.8
        self.velocity_weight = 1.2      # High priority on moving
        self.path_weight = 0.3

        # Safety
        self.min_clearance = 0.08       # Minimum acceptable clearance
        self.goal_tolerance = 0.2

        # Only consider obstacles within this range (ignore very distant returns)
        self.obstacle_detection_range = 2.0

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
        """Compute the best velocity command using DWA.

        pose: (x, y, theta) world pose
        goal: (x, y) world coordinates (final goal)
        lidar_ranges, lidar_angles: in robot frame (ranges in meters, angles in radians)
        global_path: optional list of (x,y) waypoints in world coords (used by path_weight)
        dt: control loop interval (seconds) - used to compute dynamic window
        """
        x, y, theta = pose

        # Check if at goal
        dist_to_goal = euclidean_distance((x, y), goal)
        if dist_to_goal < self.goal_tolerance:
            self.current_v = 0.0
            self.current_w = 0.0
            self.committed_detour = None
            return (0.0, 0.0)

        # Get nearby obstacles only (filter out distant / invalid returns)
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

        # Adaptive sampling: ensure we sample multiple v and w values even when dynamic window is narrow
        # Number of samples determined from window width and base resolution, with a minimum count
        v_span = max(0.0, v_max - v_min)
        w_span = max(0.0, w_max - w_min)
        num_v = max(3, int(v_span / max(self.v_resolution, 1e-6)) + 1)
        num_w = max(3, int(w_span / max(self.w_resolution, 1e-6)) + 1)

        v_samples = np.linspace(v_min, v_max, num_v)
        w_samples = np.linspace(w_min, w_max, num_w)

        for v in v_samples:
            for w in w_samples:
                # Simulate trajectory using a realistic simulation timestep
                traj = self._simulate_trajectory(v, w, sim_dt=min(self.dt, dt))

                # Check clearance
                clearance = self._trajectory_clearance(traj, obstacles)

                if clearance < self.min_clearance:
                    continue

                found_admissible = True

                # Compute heading score (toward goal or committed detour)
                if self.committed_detour is not None:
                    # Steer in committed detour direction
                    detour_angle = goal_angle + self.committed_detour * (math.pi / 3)  # 60° offset
                    heading_score = self._direction_score(traj[-1], detour_angle)
                else:
                    heading_score = self._heading_score(traj[-1], goal_robot)

                clearance_score = min(clearance / 0.5, 1.0)

                # Velocity score: reward forward motion, penalize reverse strongly
                if v >= 0:
                    velocity_score = (v / max(1e-6, self.max_linear_velocity))
                    # small bonus for reasonable forward speed
                    if v > 0.15:
                        velocity_score += 0.12
                else:
                    # Strong penalty for reversing so it is chosen only as last resort
                    velocity_score = -0.8 * (abs(v) / max(1e-6, self.max_linear_velocity))

                # Path-following score (if a global path is provided)
                path_score = 0.0
                if global_path:
                    path_score = self._path_alignment_score(traj[-1], global_path, pose)

                score = (self.heading_weight * heading_score +
                         self.clearance_weight * clearance_score +
                         self.velocity_weight * velocity_score +
                         self.path_weight * path_score)

                if score > best_score:
                    best_score = score
                    best_v = v
                    best_w = w

        # If no admissible trajectory, use simple reactive avoidance
        if not found_admissible:
            best_v, best_w = self._reactive_avoid(obstacles, goal_robot)

        # Clamp commands to robot limits just in case
        best_v = max(self.min_linear_velocity, min(self.max_linear_velocity, best_v))
        best_w = max(-self.max_angular_velocity, min(self.max_angular_velocity, best_w))

        # Optional debug of final command
        if self.debug:
            print(f"DWA chosen: v={best_v:.3f} m/s, w={math.degrees(best_w):.1f}°/s, score={best_score:.3f}")

        self.current_v = best_v
        self.current_w = best_w

        return (best_v, best_w)

    def _get_nearby_obstacles(self, ranges: List[float],
                               angles: List[float]) -> List[Tuple[float, float]]:
        """Get only nearby obstacles, filtering out distant walls and invalid returns."""
        obstacles = []

        # accept any finite return within detection range, ignore extremely tiny spurious values
        min_valid = 0.02  # meters
        for dist, angle in zip(ranges, angles):
            if not math.isfinite(dist):
                continue
            if dist < min_valid or dist > self.obstacle_detection_range:
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
            obs_dist = math.hypot(ox, oy)
            obs_angle = math.atan2(oy, ox)
            angle_diff = normalize_angle(obs_angle - goal_angle)

            # Check if obstacle is in goal direction (within ~25° cone)
            if abs(angle_diff) < 0.45 and obs_dist < 1.2:
                blocked = True

            # Accumulate clearance on each side (closer obstacles penalize more)
            if -1.5 < angle_diff < -0.2 and obs_dist < 1.5:
                right_clear -= 1.0 / max(obs_dist, 0.3)
            elif 0.2 < angle_diff < 1.5 and obs_dist < 1.5:
                left_clear -= 1.0 / max(obs_dist, 0.3)

        # Choose side with more clearance (less negative = more clear)
        suggested_side = 1 if left_clear > right_clear else -1

        return blocked, suggested_side

    def _compute_dynamic_window(self, dt: float) -> Tuple[float, float, float, float]:
        """Compute velocity bounds based on current state and acceleration limits."""
        v_min = max(self.min_linear_velocity, self.current_v - self.max_linear_accel * dt)
        v_max = min(self.max_linear_velocity, self.current_v + self.max_linear_accel * dt)
        w_min = max(-self.max_angular_velocity, self.current_w - self.max_angular_accel * dt)
        w_max = min(self.max_angular_velocity, self.current_w + self.max_angular_accel * dt)
        return (v_min, v_max, w_min, w_max)

    def _simulate_trajectory(self, v: float, w: float, sim_dt: float = None) -> List[Tuple[float, float, float]]:
        """Simulate trajectory in robot frame using sim_dt integration step."""
        if sim_dt is None or sim_dt <= 0:
            sim_dt = self.dt

        traj = [(0.0, 0.0, 0.0)]
        x, y, th = 0.0, 0.0, 0.0
        steps = max(1, int(self.predict_time / sim_dt))

        for _ in range(steps):
            # simple unicycle integration
            x += v * math.cos(th) * sim_dt
            y += v * math.sin(th) * sim_dt
            th += w * sim_dt
            traj.append((x, y, th))

        return traj

    def _trajectory_clearance(self, traj: List[Tuple[float, float, float]],
                              obstacles: List[Tuple[float, float]]) -> float:
        """Compute minimum clearance along trajectory (meters) from robot body to obstacles."""
        if not obstacles:
            return float('inf')

        min_clear = float('inf')

        # For efficiency, iterate trajectory points and obstacles; exit early if collision detected
        for tx, ty, _ in traj:
            for ox, oy in obstacles:
                # distance between obstacle and robot center minus robot radius
                dist = math.hypot(tx - ox, ty - oy) - self.robot_radius
                if dist < min_clear:
                    min_clear = dist
                    if min_clear <= 0.0:
                        # collision or touching - early exit
                        return min_clear
        return min_clear

    def _world_to_robot(self, wx: float, wy: float,
                        rx: float, ry: float, rtheta: float) -> Tuple[float, float]:
        """Transform world point to robot frame (robot at origin facing +x)."""
        dx = wx - rx
        dy = wy - ry
        cos_t = math.cos(-rtheta)
        sin_t = math.sin(-rtheta)
        return (dx * cos_t - dy * sin_t, dx * sin_t + dy * cos_t)

    def _heading_score(self, final_pose: Tuple[float, float, float],
                       goal_robot: Tuple[float, float]) -> float:
        """Score based on heading toward goal (1.0 best, 0 worst)."""
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
        """Score based on alignment with a target direction (1.0 best)."""
        fx, fy, ftheta = final_pose
        angle_diff = abs(normalize_angle(target_direction - ftheta))
        return (math.pi - angle_diff) / math.pi

    def _path_alignment_score(self, final_pose: Tuple[float, float, float],
                              global_path: List[Tuple[float, float]],
                              robot_pose_world: Tuple[float, float, float]) -> float:
        """
        Compute a path-following score for the trajectory endpoint.
        - final_pose is in robot frame (fx, fy, ftheta)
        - global_path is in world coords
        - robot_pose_world is the robot world pose (x,y,theta)
        Return 1.0 for perfect alignment/small distance, 0.0 for far away.
        """
        fx, fy, _ = final_pose
        rx, ry, rtheta = robot_pose_world

        # Transform global path to robot frame and find closest distance
        min_dist = float('inf')
        for wx, wy in global_path:
            # world -> robot
            dx = wx - rx
            dy = wy - ry
            cos_t = math.cos(-rtheta)
            sin_t = math.sin(-rtheta)
            px = dx * cos_t - dy * sin_t
            py = dx * sin_t + dy * cos_t
            d = math.hypot(px - fx, py - fy)
            if d < min_dist:
                min_dist = d

        # Convert distance to score [0,1]
        score = 1.0 / (1.0 + min_dist)  # close -> ~1, far -> ~0
        return float(max(0.0, min(1.0, score)))

    def _reactive_avoid(self, obstacles: List[Tuple[float, float]],
                        goal_robot: Tuple[float, float]) -> Tuple[float, float]:
        """Simple reactive avoidance when DWA can't find a path."""

        if not obstacles:
            # No obstacles - go toward goal (angular velocity proportional to angle)
            goal_angle = math.atan2(goal_robot[1], goal_robot[0])
            w = max(-self.max_angular_velocity, min(self.max_angular_velocity, 1.5 * goal_angle))
            return (0.25, w)

        # Find closest obstacle
        min_dist = float('inf')
        closest_angle = 0.0

        for ox, oy in obstacles:
            d = math.hypot(ox, oy)
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

        # Clamp angular velocity
        w = max(-self.max_angular_velocity, min(self.max_angular_velocity, w))

        # Ensure forward speed when possible
        v = max(v, 0.0)

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
        if self.debug:
            print(f"DWA weights: heading={self.heading_weight}, clearance={self.clearance_weight}, "
                  f"velocity={self.velocity_weight}, path={self.path_weight}")

    def set_safety_params(self, min_clearance: float = None, robot_radius: float = None):
        """Set safety parameters."""
        if min_clearance is not None:
            self.min_clearance = min_clearance
        if robot_radius is not None:
            self.robot_radius = robot_radius
        if self.debug:
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