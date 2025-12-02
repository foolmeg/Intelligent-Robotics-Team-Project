import numpy as np
import math

class DWAPlanner:
    def __init__(self, params):
        self.params = params
        self.last_v = 0.0
        self.last_w = 0.0

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

    # ------------------------------------------------------------
    # Dynamic window based on robot acceleration limits
    # ------------------------------------------------------------
    def compute_dynamic_window(self):
        # max accel = tuned aggressive value
        max_accel = 1.2      # m/s^2
        max_ang_acc = 3.0    # rad/s^2

        dv = max_accel * self.params["dt"]
        dw = max_ang_acc * self.params["dt"]

        v_min = max(0.0, self.last_v - dv)
        v_max = min(self.params["v_max"], self.last_v + dv)

        w_min = max(-self.params["w_max"], self.last_w - dw)
        w_max = min(self.params["w_max"], self.last_w + dw)

        return v_min, v_max, w_min, w_max

    # ------------------------------------------------------------
    # Scoring function
    # ------------------------------------------------------------
    def evaluate_trajectory(self, traj, goal, obstacles):
        # Goal direction score
        dx = goal[0] - traj[-1, 0]
        dy = goal[1] - traj[-1, 1]
        heading = math.atan2(dy, dx)
        heading_error = heading - traj[-1, 2]
        heading_score = math.cos(heading_error)

        # Clearance: MAX distance to nearest obstacle
        if len(obstacles) > 0:
            dists = [np.hypot(traj[-1,0]-ox, traj[-1,1]-oy) for ox, oy in obstacles]
            clearance = min(dists)
        else:
            clearance = 1.0

        # Normalize clearance (0 → dangerous, 1 → safe)
        clearance_norm = min(clearance / 1.0, 1.0)

        # Forward velocity score
        forward_speed = traj[1,0]  # small bias for smoothing

        score = (self.params["heading_weight"] * heading_score +
                 self.params["velocity_weight"] * forward_speed +
                 self.params["clearance_weight"] * clearance_norm)

        return score

    # ------------------------------------------------------------
    # Main DWA loop
    # ------------------------------------------------------------
    def compute_velocity(self, state, goal, obstacles):
        best_score = -1e9
        best_v, best_w = 0.0, 0.0

        v_min, v_max, w_min, w_max = self.compute_dynamic_window()

        # Sample aggressively for speed
        for v in np.arange(v_min, v_max + 1e-6, self.params["v_res"]):
            for w in np.arange(w_min, w_max + 1e-6, self.params["w_res"]):

                traj = self.simulate_trajectory(state, v, w)
                score = self.evaluate_trajectory(traj, goal, obstacles)

                if score > best_score:
                    best_score = score
                    best_v, best_w = v, w

        # Save last velocities for dynamic window
        self.last_v = best_v
        self.last_w = best_w

        return best_v, best_w
