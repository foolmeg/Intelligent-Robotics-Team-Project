import numpy as np
import math

class DWAPlanner:
    def __init__(self, params):
        self.params = params

    def motion(self, state, v, w, dt):
        x, y, theta = state
        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt
        theta += w * dt
        return np.array([x, y, theta])

    def simulate_trajectory(self, state, v, w):
        traj = [state]
        time = 0.0
        while time <= self.params["predict_time"]:
            state = self.motion(state, v, w, self.params["dt"])
            traj.append(state)
            time += self.params["dt"]
        return np.array(traj)

    def evaluate(self, traj, goal, obstacles):
        dx = goal[0] - traj[-1, 0]
        dy = goal[1] - traj[-1, 1]
        heading = math.cos(math.atan2(dy, dx) - traj[-1, 2])

        if len(obstacles) > 0:
            dists = [np.hypot(traj[-1,0]-ox, traj[-1,1]-oy) for ox, oy in obstacles]
            clearance = min(dists)
        else:
            clearance = 1.0

        return (self.params["heading_weight"] * heading +
                self.params["clearance_weight"] * clearance +
                self.params["velocity_weight"] * traj[1,0])

    def compute_velocity(self, state, goal, obstacles):
        best_score = -1e9
        best_v, best_w = 0.0, 0.0

        for v in np.arange(0, self.params["v_max"], self.params["v_res"]):
            for w in np.arange(-self.params["w_max"], self.params["w_max"], self.params["w_res"]):
                traj = self.simulate_trajectory(state, v, w)
                score = self.evaluate(traj, goal, obstacles)
                if score > best_score:
                    best_score = score
                    best_v, best_w = v, w

        return best_v, best_w
