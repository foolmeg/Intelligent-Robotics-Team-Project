# dwa_planner.py

import math
import numpy as np


class DWAPlanner:
    def __init__(self, params):
        self.params = params

        # 기본 파라미터 세팅 (없으면 default)
        self.v_max = params.get("v_max", 1.0)
        self.w_max = params.get("w_max", 3.0)
        self.v_res = params.get("v_res", 0.05)          # 더 촘촘하게
        self.w_res = params.get("w_res", 0.1)
        self.dt = params.get("dt", 0.1)
        self.predict_time = params.get("predict_time", 2.0)

        self.heading_weight = params.get("heading_weight", 3.0)
        self.velocity_weight = params.get("velocity_weight", 5.0)
        self.clearance_weight = params.get("clearance_weight", 4.0)

        self.robot_radius = params.get("robot_radius", 0.18)   # 조금 보수적
        self.max_accel = params.get("max_accel", 2.0)
        self.max_ang_acc = params.get("max_ang_acc", 4.0)

        # 뒤로 살짝 빠질 수 있도록 허용
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

        # -pi ~ pi 정규화
        while theta > math.pi:
            theta -= 2.0 * math.pi
        while theta < -math.pi:
            theta += 2.0 * math.pi

        return [x, y, theta]

    # ------------------------------
    # Dynamic window (속도/가속 제한)
    # ------------------------------
    def dynamic_window(self):
        # 속도 범위 (전역)
        vs_min = self.min_speed
        vs_max = self.v_max
        ws_min = -self.w_max
        ws_max = self.w_max

        # 가속도 기반 현재 step에서 가능한 범위
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
    # Trajectory 평가
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

            # 장애물과의 최소 거리
            if obstacles:
                for ox, oy in obstacles:
                    d = math.hypot(ox - x, oy - y) - self.robot_radius
                    if d < min_clearance:
                        min_clearance = d
            else:
                # 장애물이 없으면 충분히 큰 값
                min_clearance = max(min_clearance, 5.0)

            time += self.dt

        if steps == 0:
            return -1e9, None

        # 최종 상태에서의 heading error
        gx, gy = goal
        goal_dir = math.atan2(gy - y, gx - x)
        heading_error = goal_dir - theta
        while heading_error > math.pi:
            heading_error -= 2.0 * math.pi
        while heading_error < -math.pi:
            heading_error += 2.0 * math.pi

        # 점수 계산
        heading_score = (math.pi - abs(heading_error)) / math.pi  # 0~1

        avg_speed = total_speed / steps

        # clearance가 음수면 충돌
        if min_clearance < 0.0:
            clearance_norm = 0.0
        else:
            # More reasonable safety margin: 0.2m ~ 0.8m 사이를 0~1로 노멀라이즈
            # Allows closer approach while still maintaining safety
            clearance_norm = max(
                0.0,
                min((min_clearance - 0.2) / (0.8 - 0.2), 1.0)
            )

        velocity_norm = avg_speed / self.v_max if self.v_max > 0 else 0.0

        score = (
            self.heading_weight * heading_score +
            self.velocity_weight * velocity_norm +
            self.clearance_weight * clearance_norm
        )

        final_state = [x, y, theta]
        return score, final_state

    # ------------------------------
    # 메인: 속도 선택
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

        # 속도 샘플링
        v = v_min
        while v <= v_max + 1e-6:
            w = w_min
            while w <= w_max + 1e-6:

                # 너무 느린 전진 + 회전 거의 없음 → 의미 없는 궤적은 skip
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
        # Fallback: 어떤 궤적도 유효하지 않을 때
        # --------------------------
        if best_score < -1e8:
            # 전진은 하지 않고, 제자리 회전만 해서 빠져나갈 기회 만든다
            sx, sy, stheta = state
            gx, gy = goal

            heading = math.atan2(gy - sy, gx - sx)
            heading_error = heading - stheta
            while heading_error > math.pi:
                heading_error -= 2.0 * math.pi
            while heading_error < -math.pi:
                heading_error += 2.0 * math.pi

            best_v = 0.0
            best_w = 1.5 * heading_error
            best_w = max(-self.w_max, min(self.w_max, best_w))

        # 여기서는 "최소 전진속도" 같은 거 절대 강제하지 않는다.
        # 회전-only가 가능해야 좁은 곳에서 살아남는다.

        self.last_v = best_v
        self.last_w = best_w
        return best_v, best_w
