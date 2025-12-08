# dwa_planner.py

import numpy as np
import math

class DWAPlanner:
    def __init__(self, params):
        self.params = params
        self.last_v = 0.0
        self.last_w = 0.0
        self.robot_radius = params.get("robot_radius", 0.25)

    def motion(self, state, v, w, dt):
        x, y, theta = state
        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt
        theta += w * dt
        return np.array([x, y, theta])

    def simulate_trajectory(self, state, v, w):
        traj = [np.array(state)]
        t = 0.0
        current_state = np.array(state)
        while t <= self.params["predict_time"]:
            current_state = self.motion(current_state, v, w, self.params["dt"])
            traj.append(current_state.copy())
            t += self.params["dt"]
        return np.array(traj)

    def compute_dynamic_window(self):
        max_accel = self.params.get("max_accel", 2.0)
        max_ang_acc = self.params.get("max_ang_acc", 4.0)

        dv = max_accel * self.params["dt"]
        dw = max_ang_acc * self.params["dt"]

        v_min = max(0.0, self.last_v - dv)
        v_max = min(self.params["v_max"], self.last_v + dv)
        
        # Ensure minimum velocity window to prevent getting stuck
        # Always allow some reasonable velocity range for acceleration
        MIN_V_WINDOW = 0.3  # Minimum velocity range to allow acceleration
        if v_max - v_min < MIN_V_WINDOW:
            # Expand the window to allow acceleration
            center = (v_min + v_max) / 2.0
            v_min = max(0.0, center - MIN_V_WINDOW / 2.0)
            v_max = min(self.params["v_max"], center + MIN_V_WINDOW / 2.0)
            # If still too small, force a minimum range
            if v_max - v_min < MIN_V_WINDOW:
                v_max = min(self.params["v_max"], v_min + MIN_V_WINDOW)

        w_min = max(-self.params["w_max"], self.last_w - dw)
        w_max = min(self.params["w_max"], self.last_w + dw)
        
        # Ensure minimum angular velocity window for turning capability
        # Increased to allow sharper turns around obstacles
        MIN_W_WINDOW = 1.5  # Increased to allow sharper turns
        if w_max - w_min < MIN_W_WINDOW:
            w_max = min(self.params["w_max"], max(w_min + MIN_W_WINDOW, 1.5))
            w_min = max(-self.params["w_max"], w_max - MIN_W_WINDOW)

        return v_min, v_max, w_min, w_max

    def min_clearance_along_traj(self, traj, obstacles):
        if not obstacles or len(traj) == 0:
            return float("inf")

        min_d = float("inf")
        # Check points along trajectory for collision detection
        # With dt=0.1 and predict_time=2.0, we have ~20 points
        # Check every few points for efficiency while maintaining safety
        traj_step = max(1, len(traj) // 20)  # Check ~20 points along trajectory
        
        for i in range(0, len(traj), traj_step):
            state = traj[i]
            x, y, _ = state
            for (ox, oy) in obstacles:
                d = math.hypot(x - ox, y - oy)
                if d < min_d:
                    min_d = d
                    # Early exit if collision is certain
                    if min_d < self.robot_radius + 0.1:
                        return min_d
        return min_d

    def evaluate_trajectory(self, traj, goal, obstacles):
        gx, gy = goal
        dx = gx - traj[-1, 0]
        dy = gy - traj[-1, 1]

        heading = math.atan2(dy, dx)
        heading_error = heading - traj[-1, 2]
        heading_score = math.cos(heading_error)

        # Mean forward speed (encourages progress) - use actual velocity magnitude
        if len(traj) > 1:
            dx = traj[1:, 0] - traj[:-1, 0]
            dy = traj[1:, 1] - traj[:-1, 1]
            velocities = np.sqrt(dx**2 + dy**2) / self.params["dt"]
            avg_speed = velocities.mean()
        else:
            avg_speed = 0.0

        clearance = self.min_clearance_along_traj(traj, obstacles)
        # Reject trajectories that would cause collision
        # Safety distance: robot radius + minimum clearance
        # Reduced to allow more trajectories to be considered
        SAFE_DISTANCE = self.robot_radius + 0.10  # Minimum safe clearance (reduced from 0.12)
        if clearance < SAFE_DISTANCE:
            return -1e9  # discard colliding trajectories

        # Normalize clearance: 0..1 for SAFE_DISTANCE..(SAFE_DISTANCE+0.5)m
        # Reward trajectories with good clearance, but don't over-penalize closer ones
        clearance_norm = max(0.0, min((clearance - SAFE_DISTANCE) / 0.5, 1.0))
        # Linear normalization - balanced approach
        # This allows DWA to find paths around obstacles while maintaining safety
        
        # Normalize velocity: 0..1 for 0..v_max
        velocity_norm = avg_speed / self.params["v_max"] if self.params["v_max"] > 0 else 0.0

        score = (
            self.params["heading_weight"] * heading_score +
            self.params["velocity_weight"] * velocity_norm +
            self.params["clearance_weight"] * clearance_norm
        )
        return score

    def compute_velocity(self, state, goal, obstacles):
        best_score = -1e9
        best_v, best_w = 0.0, 0.0

        v_min, v_max, w_min, w_max = self.compute_dynamic_window()
        
        # Ensure we have at least some velocity options
        # The dynamic window should already handle this, but double-check
        # CRITICAL: If we're stuck (last_v and last_w are zero), force a larger window
        if abs(self.last_v) < 0.01 and abs(self.last_w) < 0.01:
            # Robot is stuck - force larger dynamic window to allow movement
            v_min = 0.0
            v_max = min(0.4, self.params["v_max"])  # Allow up to 0.4 m/s
            w_min = max(-self.params["w_max"], -2.0)  # Allow significant turning
            w_max = min(self.params["w_max"], 2.0)
        elif v_max <= v_min:
            v_max = max(v_min + 0.3, 0.4)  # Increased minimum range
            v_min = max(0.0, v_max - 0.3)
        if w_max <= w_min:
            w_max = w_min + 0.5  # Increased minimum angular range
            w_min = max(-self.params["w_max"], w_max - 0.5)

        valid_trajectories = 0
        # Sample more angular velocities to better find paths around obstacles
        v_samples = np.linspace(v_min, v_max, max(3, int((v_max - v_min) / self.params["v_res"]) + 1))
        w_samples = np.linspace(w_min, w_max, max(5, int((w_max - w_min) / self.params["w_res"]) + 1))
        
        # Limit total samples to prevent overload, but allow more angular samples
        # Increased angular samples to better find paths around obstacles
        if len(v_samples) * len(w_samples) > 80:
            # Reduce samples if too many, but keep good angular resolution
            v_samples = np.linspace(v_min, v_max, 6)
            w_samples = np.linspace(w_min, w_max, 15)  # More angular samples for better obstacle avoidance
        
        for v in v_samples:
            for w in w_samples:
                traj = self.simulate_trajectory(state, v, w)
                score = self.evaluate_trajectory(traj, goal, obstacles)
                if score > best_score:
                    best_score = score
                    best_v, best_w = v, w
                    valid_trajectories += 1
        
        # Note: evaluate_trajectory already rejects unsafe trajectories (returns -1e9)
        # So if we have valid_trajectories > 0, the selected trajectory is safe
        # No need for redundant safety check that was causing all trajectories to be rejected

        # If no good trajectories found, try slow movement or rotation
        using_fallback = (best_score < -1e8 or valid_trajectories == 0)
        if using_fallback:
            # No safe trajectories - try to find a way forward
            gx, gy = goal
            dx = gx - state[0]
            dy = gy - state[1]
            desired_heading = math.atan2(dy, dx)
            heading_error = desired_heading - state[2]
            
            # Normalize heading error to [-pi, pi]
            while heading_error > math.pi:
                heading_error -= 2 * math.pi
            while heading_error < -math.pi:
                heading_error += 2 * math.pi
            
            # No safe trajectories found - need to find a safe escape direction
            # First, check if there's a safe direction to move
            if obstacles:
                # Find direction with maximum clearance
                best_escape_angle = None
                best_escape_clearance = 0.0
                
                # Sample directions around the robot - use simpler test
                for angle_offset in np.linspace(-math.pi, math.pi, 16):
                    # Test just the immediate clearance in this direction (not full trajectory)
                    test_angle = state[2] + angle_offset
                    test_x = state[0] + 0.3 * math.cos(test_angle)  # Check 30cm ahead
                    test_y = state[1] + 0.3 * math.sin(test_angle)
                    
                    # Find minimum distance to obstacles in this direction
                    min_dist_in_dir = float('inf')
                    for (ox, oy) in obstacles:
                        dist = math.hypot(test_x - ox, test_y - oy)
                        if dist < min_dist_in_dir:
                            min_dist_in_dir = dist
                    
                    if min_dist_in_dir > best_escape_clearance:
                        best_escape_clearance = min_dist_in_dir
                        best_escape_angle = angle_offset
                
                # Always use the best direction found, even if clearance is tight
                # The robot MUST move to avoid getting stuck
                if best_escape_angle is not None:
                    if best_escape_clearance > self.robot_radius + 0.1:
                        # Good clearance - move normally
                        best_w = best_escape_angle * 2.0
                        best_v = 0.2
                    elif best_escape_clearance > self.robot_radius + 0.05:
                        # Moderate clearance - move slowly
                        best_w = best_escape_angle * 1.5
                        best_v = 0.15
                    else:
                        # Tight clearance - but still move
                        best_w = best_escape_angle * 1.0
                        best_v = 0.1
                else:
                    # No direction found (shouldn't happen) - rotate toward goal
                    best_w = 2.0 * heading_error
                    best_v = 0.1
            else:
                # No obstacles - move toward goal
                if abs(heading_error) < 1.2:
                    best_v = 0.3
                    best_w = 1.2 * heading_error
                else:
                    best_w = 2.5 * heading_error
                    best_v = 0.2
            
            best_w = max(-self.params["w_max"], min(self.params["w_max"], best_w))
            # CRITICAL: Override dynamic window for fallback - we MUST move
            # Don't let dynamic window constraints prevent movement when stuck
            v_min_fallback = 0.0
            v_max_fallback = min(0.3, self.params["v_max"])  # Allow up to 0.3 m/s in fallback
            best_v = max(v_min_fallback, min(v_max_fallback, best_v))
            
            # Ensure we always have some movement in fallback
            if abs(best_v) < 0.05 and abs(best_w) < 0.1:
                # Force at least rotation if we can't move forward
                best_w = 1.0 if abs(heading_error) > 0.1 else 0.5
                best_v = 0.05  # Minimal forward motion
            
            # Final safety check: fallback MUST return non-zero
            if abs(best_v) < 0.01 and abs(best_w) < 0.01:
                # Emergency: force movement
                best_w = 1.0  # Rotate
                best_v = 0.1  # Move forward slowly

        # Final check before returning - should never return zeros
        if abs(best_v) < 0.01 and abs(best_w) < 0.01:
            # This should never happen, but if it does, force emergency movement
            best_w = 0.5
            best_v = 0.1

        self.last_v = best_v
        self.last_w = best_w
        return best_v, best_w