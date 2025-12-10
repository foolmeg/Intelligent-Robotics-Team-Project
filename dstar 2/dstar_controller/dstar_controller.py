from controller import Robot, Node
import math
import heapq

# ===================== Parameter configuration =====================

# Goal position in world coordinates (x, y). z=0.09 is ignored here.
GOAL_X = 4.0
GOAL_Y = 3.3

# Differential drive parameters (approximate Pioneer 3-DX)
WHEEL_RADIUS = 0.0975   # wheel radius (m)
AXLE_LENGTH = 0.33      # distance between wheels (m)
MAX_WHEEL_SPEED = 10.0  # max wheel angular speed (rad/s)

# Linear speed parameters
V_FAST = 0.18
V_MED  = 0.10
V_SLOW = 0.05
V_BACK = -0.03

# Angular speed control
W_MAX = 2.5          # max angular velocity (rad/s)
K_ANG = 1.8          # P gain for heading control
W_RECOVER = 1.8      # angular velocity during recovery behaviour

# Goal tolerance
GOAL_TOL = 0.15

# Look-ahead for path tracking (in grid cells)
LOOKAHEAD_CELLS = 2

# Dynamic obstacle detection
SONAR_FRONT_MAX_DIST = 1.0   # m, anything closer than this is considered as obstacle
DYNAMIC_INFLATION_CELLS = 2  # extra cells around detected obstacle cell

# ===================== Map and WALLS definition =====================

# Arena boundaries (RectangleArena floorSize 10 10)
X_MIN, X_MAX = -5.0, 5.0
Y_MIN, Y_MAX = -5.0, 5.0

CELL_SIZE = 0.05  # grid resolution: 0.05 m per cell
GRID_W = int(round((X_MAX - X_MIN) / CELL_SIZE)) + 1
GRID_H = int(round((Y_MAX - Y_MIN) / CELL_SIZE)) + 1


def world_to_grid(x, y):
    """Convert world coordinates (x, y) to grid indices (i, j)."""
    i = int(round((x - X_MIN) / CELL_SIZE))
    j = int(round((y - Y_MIN) / CELL_SIZE))
    i = max(0, min(GRID_W - 1, i))
    j = max(0, min(GRID_H - 1, j))
    return i, j


def grid_to_world(i, j):
    """Convert grid indices (i, j) to world coordinates (x, y)."""
    x = X_MIN + i * CELL_SIZE
    y = Y_MIN + j * CELL_SIZE
    return x, y


# Walls (cx, cy, sx, sy, yaw), matched to your latest astar_map.wbt
WALLS = [
    (0.00,   1.43, 0.10, 2.00,  0.0),                  # unnamed wall
    (2.71,   3.52, 0.10, 2.00,  0.0),                  # wall(20)
    (-0.50, -1.15, 0.10, 3.00,  0.0),                  # wall(8)
    (1.34,  -3.24, 0.10, 3.00,  0.0),                  # wall(12)
    (-2.84,  3.03, 0.10, 3.00,  0.0),                  # wall(1)
    (-5.06,  0.54, 0.10, 8.00,  0.0),                  # wall(16), left boundary
    (4.90,  -0.08, 0.10, 9.50,  0.0),                  # wall(17), right boundary
    (-2.84, -3.48, 0.10, 3.00,  0.0),                  # wall(2)
    (-3.90, -3.48, 0.10, 2.20, -1.5707953071795862),   # wall(3)
    (-1.83, -4.77, 0.10, 2.20, -1.5707953071795862),   # wall(9)
    (0.30,  -4.77, 0.10, 2.20, -1.5707953071795862),   # wall(10)
    (2.45,  -4.77, 0.10, 2.20, -1.5707953071795862),   # wall(11)
    (3.90,  -4.77, 0.10, 2.20, -1.5707953071795862),   # wall(15)
    (2.45,  -1.73, 0.10, 2.20, -1.5707953071795862),   # wall(13)
    (1.15,   0.40, 0.10, 2.20, -1.5707953071795862),   # wall(14)
    (3.76,   4.54, 0.10, 2.30, -1.5707953071795862),   # wall(19)
    (-1.07,  0.42, 0.10, 2.20, -1.5707953071795862),   # wall(5)
    (-3.88,  4.55, 0.10, 2.20, -1.5707953071795862),   # wall(4)
    (-1.44,  4.55, 0.10, 2.80, -1.5707953071795862),   # wall(6)
    (1.33,   4.55, 0.10, 2.80, -1.5707953071795862),   # wall(7)
]

# ===================== Robot footprint and inflation =====================

ROBOT_RADIUS = 0.22      # used for real collision checks
PLANNING_MARGIN = 0.20   # extra safety margin in planning
PLANNING_INFLATION = ROBOT_RADIUS + PLANNING_MARGIN

NEAR_EXTRA = 0.25        # near-obstacle extra distance


def _distance_point_to_wall_rect(x, y, wall):
    """
    Compute shortest distance from (x, y) to a rectangular wall in the ground plane.
    Wall is axis-aligned in its own frame: [-sx/2, sx/2] x [-sy/2, sy/2].
    """
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
    """Minimum distance from point (x, y) to all walls and arena boundary."""
    if x < X_MIN or x > X_MAX or y < Y_MIN or y > Y_MAX:
        return 0.0

    min_dist = float('inf')

    for wall in WALLS:
        d, inside = _distance_point_to_wall_rect(x, y, wall)
        if inside:
            return 0.0
        if d < min_dist:
            min_dist = d

    bx = min(abs(x - X_MIN), abs(x - X_MAX))
    by = min(abs(y - Y_MIN), abs(y - Y_MAX))
    border_dist = min(bx, by)
    if border_dist < min_dist:
        min_dist = border_dist

    return min_dist


def is_in_wall_world(x, y):
    """Use physical robot radius to decide if it is colliding with a wall/boundary."""
    d = distance_to_walls(x, y)
    return d <= ROBOT_RADIUS


def is_near_obstacle_world(x, y, extra=NEAR_EXTRA):
    """Return True if robot is considered near an obstacle at (x, y)."""
    d = distance_to_walls(x, y)
    return d <= (ROBOT_RADIUS + extra)


# ===================== Static occupancy grid (for planning) =====================

OCCUPIED = [[False] * GRID_H for _ in range(GRID_W)]

for i in range(GRID_W):
    for j in range(GRID_H):
        x, y = grid_to_world(i, j)
        if distance_to_walls(x, y) <= PLANNING_INFLATION:
            OCCUPIED[i][j] = True


def is_occupied_cell(i, j):
    """Check whether a grid cell is occupied in the inflated map."""
    if i < 0 or i >= GRID_W or j < 0 or j >= GRID_H:
        return True
    return OCCUPIED[i][j]


# ===================== D* Lite planner =====================

class DStarLitePlanner:
    """
    Minimal D* Lite implementation on an 8-connected grid.
    In this controller we call initialise() every time we need to replan,
    so it behaves similarly to repeated A* but keeps a D* Lite structure.
    """

    def __init__(self, grid_w, grid_h):
        self.grid_w = grid_w
        self.grid_h = grid_h

        self.s_start = None
        self.s_goal = None

        self.k_m = 0.0
        self.g = {}
        self.rhs = {}
        self.U = []
        self.in_queue = {}

    @staticmethod
    def heuristic(a, b):
        (x1, y1) = a
        (x2, y2) = b
        return math.hypot(x1 - x2, y1 - y2)

    def neighbours(self, cell):
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
            # No corner cutting
            if dx != 0 and dy != 0:
                if is_occupied_cell(x + dx, y) or is_occupied_cell(x, y + dy):
                    continue
            yield (nx, ny)

    def cost(self, a, b):
        if a == b:
            return 0.0
        if is_occupied_cell(*b):
            return float('inf')
        (x1, y1) = a
        (x2, y2) = b
        return math.hypot(x1 - x2, y1 - y2)

    def get_g(self, s):
        return self.g.get(s, float('inf'))

    def get_rhs(self, s):
        return self.rhs.get(s, float('inf'))

    def set_g(self, s, value):
        self.g[s] = value

    def set_rhs(self, s, value):
        self.rhs[s] = value

    # ----- queue helpers -----

    def calculate_key(self, s):
        g = self.get_g(s)
        rhs = self.get_rhs(s)
        k2 = min(g, rhs)
        k1 = k2 + self.heuristic(self.s_start, s) + self.k_m
        return (k1, k2)

    def _push_queue(self, s):
        key = self.calculate_key(s)
        self.in_queue[s] = key
        heapq.heappush(self.U, (key[0], key[1], s))

    def _remove_from_queue(self, s):
        if s in self.in_queue:
            del self.in_queue[s]

    def _top_queue(self):
        while self.U:
            k1, k2, s = self.U[0]
            key = (k1, k2)
            if s not in self.in_queue or self.in_queue[s] != key:
                heapq.heappop(self.U)
                continue
            return key
        return (float('inf'), float('inf'))

    def _pop_queue(self):
        while self.U:
            k1, k2, s = heapq.heappop(self.U)
            key = (k1, k2)
            if s not in self.in_queue or self.in_queue[s] != key:
                continue
            del self.in_queue[s]
            return s
        return None

    # ----- main D* Lite logic -----

    def initialise(self, start, goal):
        """Initialise for a new start and goal."""
        self.s_start = start
        self.s_goal = goal
        self.k_m = 0.0
        self.g.clear()
        self.rhs.clear()
        self.U.clear()
        self.in_queue.clear()

        self.set_rhs(self.s_goal, 0.0)
        self._push_queue(self.s_goal)
        self.compute_shortest_path()

    def update_vertex(self, u):
        """Update vertex u according to D* Lite rules."""
        if u != self.s_goal:
            min_rhs = float('inf')
            for s in self.neighbours(u):
                val = self.cost(u, s) + self.get_g(s)
                if val < min_rhs:
                    min_rhs = val
            self.set_rhs(u, min_rhs)

        self._remove_from_queue(u)
        if self.get_g(u) != self.get_rhs(u):
            self._push_queue(u)

    def compute_shortest_path(self):
        """Core D* Lite loop."""
        while True:
            top_key = self._top_queue()
            start_key = self.calculate_key(self.s_start)
            if not (top_key < start_key or
                    self.get_rhs(self.s_start) != self.get_g(self.s_start)):
                break

            u = self._pop_queue()
            if u is None:
                break

            if (top_key[0], top_key[1]) < self.calculate_key(u):
                self._push_queue(u)
            elif self.get_g(u) > self.get_rhs(u):
                self.set_g(u, self.get_rhs(u))
                for s in self.neighbours(u):
                    if s != self.s_goal:
                        rhs_s = self.get_rhs(s)
                        val = self.cost(s, u) + self.get_g(u)
                        if val < rhs_s:
                            self.set_rhs(s, val)
                    self.update_vertex(s)
            else:
                g_old = self.get_g(u)
                self.set_g(u, float('inf'))
                to_update = set(self.neighbours(u))
                to_update.add(u)
                for s in to_update:
                    if self.get_rhs(s) == self.cost(s, u) + g_old:
                        if s != self.s_goal:
                            min_rhs = float('inf')
                            for s2 in self.neighbours(s):
                                val = self.cost(s, s2) + self.get_g(s2)
                                if val < min_rhs:
                                    min_rhs = val
                            self.set_rhs(s, min_rhs)
                    self.update_vertex(s)

    def extract_path(self, start, max_steps=10000):
        """Extract a path by greedily following cost + g(s)."""
        if self.get_g(start) == float('inf') and self.get_rhs(start) == float('inf'):
            return []

        path = [start]
        current = start
        steps = 0

        while current != self.s_goal and steps < max_steps:
            best_s = None
            best_val = float('inf')
            for s in self.neighbours(current):
                c = self.cost(current, s)
                g_s = self.get_g(s)
                val = c + g_s
                if val < best_val:
                    best_val = val
                    best_s = s

            if best_s is None or best_val == float('inf'):
                break

            path.append(best_s)
            current = best_s
            steps += 1

        if current != self.s_goal:
            return []
        return path


# ===================== Utility functions =====================

def normalise_angle(angle):
    """Normalise angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


# ===================== Controller =====================

class DStarController:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        # Motors
        self.left_motor, self.right_motor = self._find_wheel_motors()
        if self.left_motor is None or self.right_motor is None:
            print("[FATAL] Cannot find wheel motors for Pioneer 3-DX.")
            for i in range(self.robot.getNumberOfDevices()):
                dev = self.robot.getDeviceByIndex(i)
                try:
                    node_type = dev.getNodeType()
                except Exception:
                    node_type = None
                print(f"  - {dev.getName()} (nodeType={node_type})")
            raise RuntimeError("Wheel motors not found.")

        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Sensors
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)

        self.imu = self.robot.getDevice('inertial unit')
        self.imu.enable(self.timestep)

        # Sonar sensors (so0..so15)
        self.sonars = []
        self._init_sonars()

        # Planner
        self.planner = DStarLitePlanner(GRID_W, GRID_H)
        self.path = []
        self.path_idx = 0
        self.step_counter = 0
        self.planned = False
        self.done = False

        # Dynamic obstacle tracking
        self.dynamic_obstacles = set()  # set of (i, j)
        self.needs_replan = False

    # ---------- device helpers ----------

    def _find_wheel_motors(self):
        candidates = [
            ('left wheel', 'right wheel'),
            ('left wheel motor', 'right wheel motor'),
            ('leftWheel', 'rightWheel'),
            ('left_wheel', 'right_wheel'),
            ('left_motor', 'right_motor'),
        ]
        for ln, rn in candidates:
            left = right = None
            try:
                left = self.robot.getDevice(ln)
            except Exception:
                pass
            try:
                right = self.robot.getDevice(rn)
            except Exception:
                pass
            if left is not None and right is not None:
                print(f"[INIT] Using wheel motors '{ln}' and '{rn}'")
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

    def _init_sonars(self):
        for i in range(16):
            name = f"so{i}"
            try:
                s = self.robot.getDevice(name)
                s.enable(self.timestep)
                self.sonars.append((name, s))
            except Exception:
                continue
        if self.sonars:
            print(f"[INIT] Enabled {len(self.sonars)} sonar sensors.")
        else:
            print("[WARN] No sonar sensors found; dynamic obstacles will not be detected.")

    # ---------- pose & speed ----------

    def get_pose(self):
        xyz = self.gps.getValues()
        x = xyz[0]
        y = xyz[1]
        rpy = self.imu.getRollPitchYaw()
        theta = normalise_angle(rpy[2])
        return x, y, theta

    def set_speed(self, v, w):
        w = max(-W_MAX, min(W_MAX, w))
        v_r = (2.0 * v + w * AXLE_LENGTH) / (2.0 * WHEEL_RADIUS)
        v_l = (2.0 * v - w * AXLE_LENGTH) / (2.0 * WHEEL_RADIUS)
        v_r = max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, v_r))
        v_l = max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, v_l))
        self.left_motor.setVelocity(v_l)
        self.right_motor.setVelocity(v_r)

    # ---------- dynamic map update ----------

    def update_map_from_sensors(self):
        """
        Use front sonar sensors (so0..so7) to detect unknown obstacles in front
        of the robot. When a new obstacle is detected, mark corresponding grid
        cells as occupied and request a replan.
        """
        if not self.sonars:
            return

        front = []
        for name, s in self.sonars:
            try:
                idx = int(name.replace('so', ''))
            except ValueError:
                idx = 0
            if idx <= 7:  # front hemisphere
                front.append((s.getValue(), name))

        if not front:
            return

        min_r, min_name = min(front, key=lambda t: t[0])

        if min_r <= 0.0 or min_r > SONAR_FRONT_MAX_DIST:
            return

        x, y, theta = self.get_pose()
        # Approximate obstacle position along robot heading
        obs_x = x + math.cos(theta) * min_r
        obs_y = y + math.sin(theta) * min_r
        ci, cj = world_to_grid(obs_x, obs_y)

        if (ci, cj) in self.dynamic_obstacles:
            return

        print(f"[MAP] New dynamic obstacle detected by {min_name} at world=({obs_x:.2f},{obs_y:.2f}) cell=({ci},{cj})")

        self.dynamic_obstacles.add((ci, cj))

        # Inflate obstacle in occupancy grid
        for di in range(-DYNAMIC_INFLATION_CELLS, DYNAMIC_INFLATION_CELLS + 1):
            for dj in range(-DYNAMIC_INFLATION_CELLS, DYNAMIC_INFLATION_CELLS + 1):
                ii = ci + di
                jj = cj + dj
                if 0 <= ii < GRID_W and 0 <= jj < GRID_H:
                    OCCUPIED[ii][jj] = True

        self.needs_replan = True

    # ---------- planning & tracking ----------

    def plan_path_if_needed(self):
        """
        Run / rerun D* Lite when there is no path yet or the map has changed
        because of dynamic obstacles.
        """
        if self.done:
            return
        if self.planned and not self.needs_replan:
            return

        x, y, theta = self.get_pose()
        start_cell = world_to_grid(x, y)
        goal_cell = world_to_grid(GOAL_X, GOAL_Y)

        self.planner.initialise(start_cell, goal_cell)
        path = self.planner.extract_path(start_cell)

        if not path or len(path) <= 1:
            print(f"[INIT] Start cell = {start_cell}, Goal cell = {goal_cell}")
            print("[ERROR] D* Lite returned an empty or trivial path.")
            self.done = True
            self.set_speed(0.0, 0.0)
            return

        self.path = path
        self.path_idx = 0
        self.planned = True
        self.needs_replan = False

        print(f"[INIT] Start cell = {start_cell}, Goal cell = {goal_cell}")
        print(f"[INIT] Planned path length = {len(self.path)}")
        for idx, (ci, cj) in enumerate(self.path[:20]):
            wx, wy = grid_to_world(ci, cj)
            print(f"[PATH] idx={idx} cell=({ci}, {cj}) world=({wx:.2f},{wy:.2f})")

    def follow_path_step(self):
        """
        Follow the current path with look-ahead, slow-down near obstacles,
        and simple collision recovery.
        """
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

        # Skip waypoints that are already close
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
        heading_err = normalise_angle(target_heading - theta)
        dist = math.hypot(dx, dy)

        in_wall = is_in_wall_world(x, y)
        near_obs = is_near_obstacle_world(x, y)

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
            print(f"[RECOVER] In wall at cell={world_to_grid(x, y)}, backing and turning left.")
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
                f"path_idx={self.path_idx}/{len(self.path)-1} "
                f"target_idx={target_idx} target_cell=({tgt_i},{tgt_j}) "
                f"target_world=({tgt_x:.2f},{tgt_y:.2f}) "
                f"dist={dist:.3f} heading_err={heading_err:.3f} v={v:.3f} w={w:.3f}"
            )

        self.set_speed(v, w)

    # ---------- main loop ----------

    def run(self):
        print("D* Lite dynamic-obstacle controller started.")
        while self.robot.step(self.timestep) != -1:
            self.update_map_from_sensors()
            self.plan_path_if_needed()
            self.follow_path_step()
            if self.done:
                self.set_speed(0.0, 0.0)


if __name__ == "__main__":
    controller = DStarController()
    controller.run()
