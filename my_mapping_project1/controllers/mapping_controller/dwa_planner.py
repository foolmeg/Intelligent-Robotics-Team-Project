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
        traj = [state]
        t = 0.0
        while t <= self.params["predict_time"]:
            state = self.motion(state, v, w, self.params["dt"])
            traj.append(state)
            t += self.params["dt"]
        return np.array(traj)

    def compute_dynamic_window(self):
        max_accel = self.params.get("max_accel", 2.0)  # Increased acceleration
        max_ang_acc = self.params.get("max_ang_acc", 4.0)  # Increased angular acceleration

        dv = max_accel * self.params["dt"]
        dw = max_ang_acc * self.params["dt"]

        v_min = max(0.0, self.last_v - dv)
        v_max = min(self.params["v_max"], self.last_v + dv)
        
        # Ensure minimum velocity window for startup - always allow reasonable speeds
        MIN_V_WINDOW = 0.4  # Minimum velocity range
        if v_max - v_min < MIN_V_WINDOW:
            v_max = min(self.params["v_max"], max(v_min + MIN_V_WINDOW, 0.4))
            v_min = max(0.0, v_max - MIN_V_WINDOW)

        w_min = max(-self.params["w_max"], self.last_w - dw)
        w_max = min(self.params["w_max"], self.last_w + dw)
        
        # Ensure minimum angular velocity window
        MIN_W_WINDOW = 1.0
        if w_max - w_min < MIN_W_WINDOW:
            w_max = min(self.params["w_max"], max(w_min + MIN_W_WINDOW, 1.0))
            w_min = max(-self.params["w_max"], w_max - MIN_W_WINDOW)

        return v_min, v_max, w_min, w_max

    def min_clearance_along_traj(self, traj, obstacles):
        if not obstacles:
            return float("inf")

        min_d = float("inf")
        # Sample trajectory points (every other point) for performance
        traj_step = max(1, len(traj) // 10)  # Check ~10 points max
        
        for i in range(0, len(traj), traj_step):
            state = traj[i]
            x, y, _ = state
            for (ox, oy) in obstacles:
                d = math.hypot(x - ox, y - oy)
                if d < min_d:
                    min_d = d
                    # Early exit if too close
                    if min_d < self.robot_radius + 0.05:
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
        if clearance < self.robot_radius + 0.05:
            return -1e9  # discard colliding trajectories

        # Normalize clearance: 0..1 for 0.3..1.0m
        clearance_norm = max(0.0, min((clearance - 0.3) / (1.0 - 0.3), 1.0))
        
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
        if v_max <= v_min:
            v_max = max(v_min + 0.1, 0.2)
        if w_max <= w_min:
            w_max = w_min + 0.2

        valid_trajectories = 0
        # Limit number of velocity samples for performance
        v_samples = np.linspace(v_min, v_max, max(3, int((v_max - v_min) / self.params["v_res"]) + 1))
        w_samples = np.linspace(w_min, w_max, max(3, int((w_max - w_min) / self.params["w_res"]) + 1))
        
        # Limit total samples to prevent overload
        if len(v_samples) * len(w_samples) > 50:
            # Reduce samples if too many
            v_samples = np.linspace(v_min, v_max, 6)
            w_samples = np.linspace(w_min, w_max, 9)
        
        for v in v_samples:
            for w in w_samples:
                traj = self.simulate_trajectory(state, v, w)
                score = self.evaluate_trajectory(traj, goal, obstacles)
                if score > best_score:
                    best_score = score
                    best_v, best_w = v, w
                    valid_trajectories += 1

        # If no good trajectories found, use simple goal-seeking behavior
        if best_score < -1e8 or valid_trajectories == 0:
            # Calculate desired heading
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
            
            # Simple proportional control
            best_w = 2.0 * heading_error  # Turn towards goal
            best_w = max(-self.params["w_max"], min(self.params["w_max"], best_w))
            best_v = 0.5  # Move forward at reasonable speed
        elif abs(best_v) < 0.3:
            # Ensure minimum forward velocity - be more aggressive
            best_v = max(0.3, abs(best_v))

        self.last_v = best_v
        self.last_w = best_w
        return best_v, best_w