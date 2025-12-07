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
        max_accel = self.params.get("max_accel", 1.2)
        max_ang_acc = self.params.get("max_ang_acc", 3.0)

        dv = max_accel * self.params["dt"]
        dw = max_ang_acc * self.params["dt"]

        v_min = max(0.0, self.last_v - dv)
        v_max = min(self.params["v_max"], self.last_v + dv)

        w_min = max(-self.params["w_max"], self.last_w - dw)
        w_max = min(self.params["w_max"], self.last_w + dw)

        return v_min, v_max, w_min, w_max

    def min_clearance_along_traj(self, traj, obstacles):
        if not obstacles:
            return float("inf")

        min_d = float("inf")
        for state in traj:
            x, y, _ = state
            for (ox, oy) in obstacles:
                d = math.hypot(x - ox, y - oy)
                if d < min_d:
                    min_d = d
        return min_d

    def evaluate_trajectory(self, traj, goal, obstacles):
        gx, gy = goal
        dx = gx - traj[-1, 0]
        dy = gy - traj[-1, 1]

        heading = math.atan2(dy, dx)
        heading_error = heading - traj[-1, 2]
        heading_score = math.cos(heading_error)

        # Mean forward speed (encourages progress)
        avg_speed = (traj[1:, 0] - traj[:-1, 0]).mean() if len(traj) > 1 else 0.0

        clearance = self.min_clearance_along_traj(traj, obstacles)
        if clearance < self.robot_radius + 0.05:
            return -1e9  # discard colliding trajectories

        # Normalize clearance: 0..1 for 0.3..1.0m
        clearance_norm = max(0.0, min((clearance - 0.3) / (1.0 - 0.3), 1.0))

        score = (
            self.params["heading_weight"] * heading_score +
            self.params["velocity_weight"] * avg_speed +
            self.params["clearance_weight"] * clearance_norm
        )
        return score

    def compute_velocity(self, state, goal, obstacles):
        best_score = -1e9
        best_v, best_w = 0.0, 0.0

        v_min, v_max, w_min, w_max = self.compute_dynamic_window()

        for v in np.arange(v_min, v_max + 1e-6, self.params["v_res"]):
            for w in np.arange(w_min, w_max + 1e-6, self.params["w_res"]):
                traj = self.simulate_trajectory(state, v, w)
                score = self.evaluate_trajectory(traj, goal, obstacles)
                if score > best_score:
                    best_score = score
                    best_v, best_w = v, w

        # small forward bias to avoid creeping
        if abs(best_v) < 0.05:
            best_v = 0.15

        self.last_v = best_v
        self.last_w = best_w
        return best_v, best_w