# dwa_planner.py

import math
import numpy as np


class DWAPlanner:
    def __init__(self, params):
        self.params = params

        # Basic parameter setting (default if missing)
        self.v_max = params.get("v_max", 1.0)
        self.w_max = params.get("w_max", 3.0)
        self.v_res = params.get("v_res", 0.05)          # Finer resolution
        self.w_res = params.get("w_res", 0.1)
        self.dt = params.get("dt", 0.1)
        self.predict_time = params.get("predict_time", 2.0)

        self.heading_weight = params.get("heading_weight", 3.0)
        self.velocity_weight = params.get("velocity_weight", 5.0)
        self.clearance_weight = params.get("clearance_weight", 4.0)

        self.robot_radius = params.get("robot_radius", 0.18)   # Slightly conservative
        self.max_accel = params.get("max_accel", 2.0)
        self.max_ang_acc = params.get("max_ang_acc", 4.0)

        # Allow slight backward movement
        self.min_speed = params.get("min_speed", -0.2)

        self.last_v = 0.0
        self.last_w = 0.0

    # ------------------------------
    # Unicycle motion model
    # ------------------------------
    def motion(self, state, v, w, dt):
        x, y, theta = state
        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt
        theta += w * dt

        # Normalize to -pi ~ pi
        while theta > math.pi:
            theta -= 2.0 * math.pi
        while theta < -math.pi:
            theta += 2.0 * math.pi

        return [x, y, theta]

    # ------------------------------
    # Dynamic window (Velocity/Acceleration limits)
    # ------------------------------
    def dynamic_window(self):
        # Velocity range (Global)
        vs_min = self.min_speed
        vs_max = self.v_max
        ws_min = -self.w_max
        ws_max = self.w_max

        # Range possible in current step based on acceleration
        vd_min = self.last_v - self.max_accel * self.dt
        vd_max = self.last_v + self.max_accel * self.dt
        wd_min = self.last_w - self.max_ang_acc * self.dt
        wd_max = self.last_w + self.max_ang_acc * self.dt

        v_min = max(vs_min, vd_min)
        v_max = min(vs_max, vd_max)
        w_min = max(ws_min, wd_min)
        w_max = min(ws_max, wd_max)

        return v_min, v_max, w_min, w_max

    # ------------------------------
    # Trajectory Evaluation
    # ------------------------------
    def evaluate_trajectory(self, state, v, w, goal, obstacles):
        x, y, theta = state

        time = 0.0
        min_clearance = float("inf")
        total_speed = 0.0
        steps = 0

        while time < self.predict_time:
            x, y, theta = self.motion([x, y, theta], v, w, self.dt)
            steps += 1
            total_speed += abs(v)

            # Minimum distance to obstacles
            if obstacles:
                for ox, oy in obstacles:
                    d = math.hypot(ox - x, oy - y) - self.robot_radius
                    if d < min_clearance:
                        min_clearance = d
            else:
                # Large value if no obstacles
                min_clearance = max(min_clearance, 5.0)

            time += self.dt

        if steps == 0:
            return -1e9, None

        # Heading error at final state
        gx, gy = goal
        goal_dir = math.atan2(gy - y, gx - x)
        heading_error = goal_dir - theta
        while heading_error > math.pi:
            heading_error -= 2.0 * math.pi
        while heading_error < -math.pi:
            heading_error += 2.0 * math.pi

        # Score Calculation
        heading_score = (math.pi - abs(heading_error)) / math.pi  # 0~1

        avg_speed = total_speed / steps

        # Collision check - Fatal
        if min_clearance < 0.0:
            return -1e9, [x, y, theta]

        # Continuous gradient for clearance (0.0m to 1.0m)
        clearance_norm = min(min_clearance / 1.0, 1.0)

        velocity_norm = avg_speed / self.v_max if self.v_max > 0 else 0.0

        score = (
            self.heading_weight * heading_score +
            self.velocity_weight * velocity_norm +
            self.clearance_weight * clearance_norm
        )

        # Apply penalty for backward motion
        if v < 0:
            score -= 2.0

        final_state = [x, y, theta]
        return score, final_state

    # ------------------------------
    # Main: Velocity Selection
    # ------------------------------
    def compute_velocity(self, state, goal, obstacles):
        # state = [x, y, theta], goal = [gx, gy]
        v_min, v_max, w_min, w_max = self.dynamic_window()

        if v_max < v_min:
            v_min, v_max = v_max, v_min
        if w_max < w_min:
            w_min, w_max = w_max, w_min

        best_score = -1e9
        best_v = 0.0
        best_w = 0.0

        # Velocity sampling
        v = v_min
        while v <= v_max + 1e-6:
            w = w_min
            while w <= w_max + 1e-6:

                # Skip meaningless trajectories (too slow forward + barely rotating)
                if abs(v) < 0.01 and abs(w) < 0.01:
                    w += self.w_res
                    continue

                score, _ = self.evaluate_trajectory(state, v, w, goal, obstacles)

                if score > best_score:
                    best_score = score
                    best_v = v
                    best_w = w

                w += self.w_res
            v += self.v_res

        # --------------------------
        # Fallback: When no trajectory is valid
        # --------------------------
        if best_score < -1e8:
            # Check if there are actually obstacles blocking
            sx, sy, stheta = state
            gx, gy = goal

            heading = math.atan2(gy - sy, gx - sx)
            heading_error = heading - stheta
            while heading_error > math.pi:
                heading_error -= 2.0 * math.pi
            while heading_error < -math.pi:
                heading_error += 2.0 * math.pi

            # Check if obstacles are really blocking the path
            has_close_obstacle = False
            if obstacles:
                for ox, oy in obstacles:
                    dist = math.hypot(ox - sx, oy - sy)
                    if dist < 0.5:  # Very close obstacle
                        has_close_obstacle = True
                        break
            
            if has_close_obstacle:
                # Real obstacle - turn in place
                best_v = 0.0
                best_w = 1.5 * heading_error
            else:
                # No close obstacle - continue forward slowly
                best_v = 0.15  # Small forward velocity
                best_w = 1.0 * heading_error  # Gentle turn
            
            best_w = max(-self.w_max, min(self.w_max, best_w))

        # Do not force "minimum forward velocity" here.
        # Must allow rotation-only to survive in tight spaces.

        self.last_v = best_v
        self.last_w = best_w
        return best_v, best_w
