from controller import Robot, Node
import math
import heapq

# ===================== 参数配置 =====================

# 目标点（世界坐标，注意是 x, y，第三个坐标是高度）
GOAL_X = 3.2
GOAL_Y = -3.5

# 差速驱动参数（Pioneer 3-DX 近似）
WHEEL_RADIUS = 0.0975   # 轮半径 (m)
AXLE_LENGTH = 0.33      # 轮间距 (m)
MAX_WHEEL_SPEED = 10.0  # 最大轮角速度 (rad/s)

# 线速度（适当偏小，减小撞墙风险）
V_FAST = 0.18
V_MED  = 0.10
V_SLOW = 0.05
V_BACK = -0.03

# 角速度控制
W_MAX = 2.5          # 角速度上限
K_ANG = 1.8          # 转向 P 系数
W_RECOVER = 1.8      # 碰撞恢复时原地转速

# 终点判定距离
GOAL_TOL = 0.15

# 路径跟踪 look-ahead（向前看的栅格数）
LOOKAHEAD_CELLS = 2

# ===================== 地图 & WALLS 定义 =====================

# 场地边界（和 RectangleArena floorSize 10 10 对应）
X_MIN, X_MAX = -5.0, 5.0
Y_MIN, Y_MAX = -5.0, 5.0

CELL_SIZE = 0.05  # 10m / 0.05 = 200 cells
GRID_W = int(round((X_MAX - X_MIN) / CELL_SIZE)) + 1
GRID_H = int(round((Y_MAX - Y_MIN) / CELL_SIZE)) + 1


def world_to_grid(x, y):
    i = int(round((x - X_MIN) / CELL_SIZE))
    j = int(round((y - Y_MIN) / CELL_SIZE))
    i = max(0, min(GRID_W - 1, i))
    j = max(0, min(GRID_H - 1, j))
    return i, j


def grid_to_world(i, j):
    x = X_MIN + i * CELL_SIZE
    y = Y_MIN + j * CELL_SIZE
    return x, y


# 直接从 dstar_map.wbt 解析出的 21 个墙体 (cx, cy, sx, sy, yaw)
# cx, cy 在平面上是 (x, y)，yaw 为绕 z 轴的旋转（0 或 -pi/2）
WALLS = [
    (0.0,   1.87, 0.1, 3.0, 0.0),                  # unnamed
    (2.71,  3.06, 0.1, 3.0, 0.0),                  # wall(20)
    (-0.5, -1.15, 0.1, 3.0, 0.0),                  # wall(8)
    (1.34, -3.24, 0.1, 3.0, 0.0),                  # wall(12)
    (-2.84, 3.07, 0.1, 3.0, 0.0),                  # wall(1)
    (-5.06, 0.54, 0.1, 8.0, 0.0),                  # wall(16)
    (4.9,  -0.08, 0.1, 9.5, 0.0),                  # wall(17)
    (-2.84, -3.48, 0.1, 3.0, 0.0),                 # wall(2)
    (-3.9,  -3.48, 0.1, 2.2, -1.5707953071795862), # wall(3)
    (-1.83, -4.77, 0.1, 2.2, -1.5707953071795862), # wall(9)
    (0.3,   -4.77, 0.1, 2.2, -1.5707953071795862), # wall(10)
    (2.45,  -4.77, 0.1, 2.2, -1.5707953071795862), # wall(11)
    (3.9,   -4.77, 0.1, 2.2, -1.5707953071795862), # wall(15)
    (2.45,  -1.73, 0.1, 2.2, -1.5707953071795862), # wall(13)
    (1.15,   0.52, 0.1, 2.2, -1.5707953071795862), # wall(14)
    (0.97,   3.27, 0.1, 1.8, -1.5707953071795862), # wall(18)
    (3.76,   4.54, 0.1, 2.3, -1.5707953071795862), # wall(19)
    (-1.07,  0.42, 0.1, 2.2, -1.5707953071795862), # wall(5)
    (-3.88,  4.55, 0.1, 2.2, -1.5707953071795862), # wall(4)
    (-1.44,  4.55, 0.1, 2.8, -1.5707953071795862), # wall(6)
    (1.33,   4.55, 0.1, 2.8, -1.5707953071795862), # wall(7)
]

# ===================== 机器人半径 & 膨胀 =====================

ROBOT_RADIUS = 0.22  # 实际机器人半径

PLANNING_MARGIN = 0.20
PLANNING_INFLATION = ROBOT_RADIUS + PLANNING_MARGIN  # 约 0.42m

NEAR_EXTRA = 0.25  # 靠墙多近算 near_obs


def _distance_point_to_wall_rect(x, y, wall):
    cx, cy, sx, sy, yaw = wall
    dx = x - cx
    dy = y - cy

    c = math.cos(-yaw)
    s = math.sin(-yaw)
    lx = c * dx - s * dy
    ly = s * dx + c * dy

    hx = sx * 0.5
    hy = sy * 0.5

    dx_out = max(abs(lx) - hx, 0.0)
    dy_out = max(abs(ly) - hy, 0.0)

    inside = (dx_out <= 0.0 and dy_out <= 0.0)
    if inside:
        return 0.0, True
    dist = math.hypot(dx_out, dy_out)
    return dist, False


def distance_to_walls(x, y):
    # 出界直接视为 0
    if x < X_MIN or x > X_MAX or y < Y_MIN or y > Y_MAX:
        return 0.0

    min_dist = float('inf')

    for wall in WALLS:
        d, inside = _distance_point_to_wall_rect(x, y, wall)
        if inside:
            return 0.0
        if d < min_dist:
            min_dist = d

    # 场地边界
    bx = min(abs(x - X_MIN), abs(x - X_MAX))
    by = min(abs(y - Y_MIN), abs(y - Y_MAX))
    border_dist = min(bx, by)
    if border_dist < min_dist:
        min_dist = border_dist

    return min_dist


def is_in_wall_world(x, y):
    # 真正“撞上”，用 ROBOT_RADIUS
    d = distance_to_walls(x, y)
    return d <= ROBOT_RADIUS


def is_near_obstacle_world(x, y, extra=NEAR_EXTRA):
    d = distance_to_walls(x, y)
    return d <= (ROBOT_RADIUS + extra)


# ===================== 占据栅格（规划用） =====================

OCCUPIED = [[False] * GRID_H for _ in range(GRID_W)]

for i in range(GRID_W):
    for j in range(GRID_H):
        x, y = grid_to_world(i, j)
        if distance_to_walls(x, y) <= PLANNING_INFLATION:
            OCCUPIED[i][j] = True


def is_occupied_cell(i, j):
    if i < 0 or i >= GRID_W or j < 0 or j >= GRID_H:
        return True
    return OCCUPIED[i][j]


# ===================== A* =====================

class AStarPlanner:
    def __init__(self, grid_w, grid_h):
        self.grid_w = grid_w
        self.grid_h = grid_h

    @staticmethod
    def heuristic(a, b):
        (x1, y1) = a
        (x2, y2) = b
        return math.hypot(x1 - x2, y1 - y2)

    def neighbors(self, cell):
        (x, y) = cell
        for dx, dy in [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ]:
            nx = x + dx
            ny = y + dy
            if not (0 <= nx < self.grid_w and 0 <= ny < self.grid_h):
                continue
            if is_occupied_cell(nx, ny):
                continue

            # 对角线时检查 corner-cutting
            if dx != 0 and dy != 0:
                if is_occupied_cell(x + dx, y) or is_occupied_cell(x, y + dy):
                    continue

            yield (nx, ny)

    def plan(self, start, goal):
        if is_occupied_cell(*start):
            print("[WARN] start cell is inside (inflated) obstacle:", start)
        if is_occupied_cell(*goal):
            print("[WARN] goal cell is inside (inflated) obstacle:", goal)

        open_set = []
        heapq.heappush(open_set, (0.0, start))
        came_from = {}
        g_score = {start: 0.0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for nb in self.neighbors(current):
                step_cost = math.hypot(nb[0] - current[0], nb[1] - current[1])
                tentative_g = g_score[current] + step_cost

                if nb not in g_score or tentative_g < g_score[nb]:
                    came_from[nb] = current
                    g_score[nb] = tentative_g
                    f_score[nb] = tentative_g + self.heuristic(nb, goal)
                    heapq.heappush(open_set, (f_score[nb], nb))

        print("[ERROR] A* failed: no path found.")
        return []


# ===================== 工具函数 =====================

def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


# ===================== 控制器 =====================

class AStarController:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        self.left_motor, self.right_motor = self._find_wheel_motors()
        if self.left_motor is None or self.right_motor is None:
            print("[FATAL] Cannot find wheel motors for Pioneer 3-DX.")
            print("[HINT] Available devices:")
            for i in range(self.robot.getNumberOfDevices()):
                dev = self.robot.getDeviceByIndex(i)
                try:
                    node_type = dev.getNodeType()
                except Exception:
                    node_type = None
                print(f"  - {dev.getName()} (nodeType={node_type})")
            raise RuntimeError("Wheel motors not found; please check device names.")

        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)

        self.imu = self.robot.getDevice('inertial unit')
        self.imu.enable(self.timestep)

        self.planner = AStarPlanner(GRID_W, GRID_H)
        self.path = []
        self.path_idx = 0
        self.step_counter = 0
        self.planned = False
        self.done = False

    def _find_wheel_motors(self):
        candidate_name_pairs = [
            ('left wheel', 'right wheel'),
            ('left wheel motor', 'right wheel motor'),
            ('leftWheel', 'rightWheel'),
            ('left_wheel', 'right_wheel'),
            ('left_motor', 'right_motor'),
        ]
        for left_name, right_name in candidate_name_pairs:
            left = right = None
            try:
                left = self.robot.getDevice(left_name)
            except Exception:
                pass
            try:
                right = self.robot.getDevice(right_name)
            except Exception:
                pass
            if left is not None and right is not None:
                print(f"[INIT] Using wheel motors '{left_name}' and '{right_name}'")
                return left, right

        motors = []
        for i in range(self.robot.getNumberOfDevices()):
            dev = self.robot.getDeviceByIndex(i)
            try:
                if dev.getNodeType() == Node.ROTATIONAL_MOTOR:
                    motors.append(dev)
            except Exception:
                continue

        if len(motors) >= 2:
            left = right = None
            for m in motors:
                n = m.getName().lower()
                if 'left' in n and left is None:
                    left = m
                elif 'right' in n and right is None:
                    right = m
            if left is None or right is None:
                left, right = motors[0], motors[1]
            print(f"[INIT] Auto-detected motors: left='{left.getName()}', right='{right.getName()}'")
            return left, right

        return None, None

    def get_pose(self):
        xyz = self.gps.getValues()
        x = xyz[0]
        y = xyz[1]

        rpy = self.imu.getRollPitchYaw()
        theta = normalize_angle(rpy[2])
        return x, y, theta

    def set_speed(self, v, w):
        w = max(-W_MAX, min(W_MAX, w))

        v_r = (2.0 * v + w * AXLE_LENGTH) / (2.0 * WHEEL_RADIUS)
        v_l = (2.0 * v - w * AXLE_LENGTH) / (2.0 * WHEEL_RADIUS)

        v_r = max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, v_r))
        v_l = max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, v_l))

        self.left_motor.setVelocity(v_l)
        self.right_motor.setVelocity(v_r)

    def plan_path_if_needed(self):
        if self.planned:
            return

        x, y, theta = self.get_pose()
        start_cell = world_to_grid(x, y)
        goal_cell = world_to_grid(GOAL_X, GOAL_Y)

        self.path = self.planner.plan(start_cell, goal_cell)
        if not self.path or len(self.path) <= 1:
            print(f"[INIT] Start cell = {start_cell}, Goal cell = {goal_cell}")
            print("[ERROR] A* returned empty or trivial path.")
            self.done = True
            return

        print(f"[INIT] Start cell = {start_cell}, Goal cell = {goal_cell}")
        print(f"[INIT] Planned path length = {len(self.path)}")

        for idx, (ci, cj) in enumerate(self.path[:20]):
            wx, wy = grid_to_world(ci, cj)
            print(f"[PATH] idx={idx} cell=({ci}, {cj}) world=({wx:.2f},{wy:.2f})")

        self.planned = True
        self.path_idx = 0

    def follow_path_step(self):
        if self.done or not self.planned or not self.path:
            self.set_speed(0.0, 0.0)
            return

        x, y, theta = self.get_pose()

        goal_dist = math.hypot(x - GOAL_X, y - GOAL_Y)
        if goal_dist < GOAL_TOL or self.path_idx >= len(self.path) - 1:
            print(f"[DONE] Reached goal (x={x:.3f}, y={y:.3f}), dist={goal_dist:.3f}")
            self.set_speed(0.0, 0.0)
            self.done = True
            return

        self.path_idx = max(0, min(self.path_idx, len(self.path) - 1))
        while True:
            ci, cj = self.path[self.path_idx]
            cx, cy = grid_to_world(ci, cj)
            dist_curr = math.hypot(x - cx, y - cy)
            if dist_curr < 0.15 and self.path_idx < len(self.path) - 1:
                self.path_idx += 1
            else:
                break

        target_idx = min(self.path_idx + LOOKAHEAD_CELLS, len(self.path) - 1)
        tgt_i, tgt_j = self.path[target_idx]
        tgt_x, tgt_y = grid_to_world(tgt_i, tgt_j)

        dx = tgt_x - x
        dy = tgt_y - y
        target_heading = math.atan2(dy, dx)
        heading_err = normalize_angle(target_heading - theta)
        dist = math.hypot(dx, dy)

        in_wall = is_in_wall_world(x, y)
        near_obs = is_near_obstacle_world(x, y)

        # 线速度策略：朝向误差越大先减速甚至原地转
        if abs(heading_err) < 0.3:
            v = V_FAST
        elif abs(heading_err) < 0.7:
            v = V_MED
        elif abs(heading_err) < 1.2:
            v = V_SLOW
        else:
            v = 0.0

        if near_obs:
            v = min(v, V_SLOW)

        if in_wall:
            print(f"[RECOVER] In wall at cell={world_to_grid(x, y)}, backing & turning left.")
            v = V_BACK
            w = W_RECOVER
        else:
            w = K_ANG * heading_err
            w = max(-W_MAX, min(W_MAX, w))

        self.step_counter += 1
        if self.step_counter % 20 == 0:
            ci, cj = world_to_grid(x, y)
            print(
                f"[STEP {self.step_counter}] "
                f"pos=({x:.3f},{y:.3f}) theta={theta:.3f} "
                f"cell=({ci}, {cj}) in_wall={in_wall} near_obs={near_obs} "
                f"path_idx={self.path_idx}/{len(self.path)-1} target_idx={target_idx} "
                f"target_cell=({tgt_i},{tgt_j}) target_world=({tgt_x:.2f},{tgt_y:.2f}) "
                f"dist={dist:.3f} heading_err={heading_err:.3f} v={v:.3f} w={w:.3f}"
            )

        self.set_speed(v, w)

    def run(self):
        print("Static A* path following controller (exact walls, inflated planning, no-corner-cut) started.")
        while self.robot.step(self.timestep) != -1:
            self.plan_path_if_needed()
            self.follow_path_step()
            if self.done:
                self.set_speed(0.0, 0.0)


if __name__ == "__main__":
    controller = AStarController()
    controller.run()
