from controller import Robot
import math
import sys
import os

# Import D* Lite from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from d_start_lite import DStarLite

# ======================
# Constants
# ======================
MAX_SPEED = 5.24
MAX_SENSOR_NUMBER = 16

FORWARD = 0
LEFT = 1
RIGHT = 2

# ======================
# Init
# ======================
robot = Robot()
time_step = int(robot.getBasicTimeStep())

# LiDAR
lidar = robot.getDevice("RPlidar A2")
lidar.enable(time_step)
lidar_res = lidar.getHorizontalResolution()
lidar_fov = lidar.getFov()

# LiDAR yaw offset (keep 0 as direction is correct)
LIDAR_YAW_OFFSET = 0.0

# Display
display = robot.getDevice("display")
MAP_W = display.getWidth()
MAP_H = display.getHeight()
MAP_RES = 0.02  # 2cm/pixel

# Occupancy grid
occupancy = [[0 for _ in range(MAP_W)] for _ in range(MAP_H)]
current_obstacles = set()

# Odometry pose (Webots: X forward, Y left)
robot_x = 0.0
robot_y = 0.0
robot_theta = 0.0

map_cx = MAP_W // 2
map_cy = MAP_H // 2

# Motors
left_wheel = robot.getDevice("left wheel")
right_wheel = robot.getDevice("right wheel")
left_wheel.setPosition(float('inf'))
right_wheel.setPosition(float('inf'))

# Use pre-tuned values
WHEEL_RADIUS = 0.0975
WHEEL_BASE   = 0.33

left_encoder = left_wheel.getPositionSensor()
right_encoder = right_wheel.getPositionSensor()
left_encoder.enable(time_step)
right_encoder.enable(time_step)

prev_left_pos = left_encoder.getValue() or 0.0
prev_right_pos = right_encoder.getValue() or 0.0

# IR sensors (not used but kept)
sensors = []
for i in range(MAX_SENSOR_NUMBER):
    s = robot.getDevice(f"so{i}")
    s.enable(time_step)
    sensors.append(s)

# ======================
# Coordinate transformation functions
# ======================
def world_to_grid(wx, wy):
    """
    Webots world (X forward, Y left) -> display grid
    Map x to right (+), y to top (-) for display
    """
    gx = int(map_cx + wx / MAP_RES)
    gy = int(map_cy - wy / MAP_RES)
    return gx, gy

def grid_to_world(gx, gy):
    """
    display grid -> Webots world
    Exact inverse of world_to_grid
    """
    wx = (gx - map_cx) * MAP_RES
    wy = -(gy - map_cy) * MAP_RES
    return wx, wy

def draw_robot_on_display(gx, gy, theta):
    display.setColor(0x0000FF)  # blue
    display.fillOval(gx - 3, gy - 3, 6, 6)

    arrow_len_world = 0.20
    arrow_wx = robot_x + arrow_len_world * math.cos(theta)
    arrow_wy = robot_y + arrow_len_world * math.sin(theta)
    ax, ay = world_to_grid(arrow_wx, arrow_wy)
    display.drawLine(gx, gy, ax, ay)

def draw_goal_on_display(gx, gy):
    display.setColor(0x00FF00)
    display.fillOval(gx - 3, gy - 3, 6, 6)

# ======================
# D* Lite Setup
# ======================
# Goal: world coordinates (1.0, -1.0) near (right, down)
goal_world = (1.0, -1.0)
GOAL_NODE = world_to_grid(*goal_world)

# Clamp to map bounds
GOAL_NODE = (
    max(10, min(MAP_W - 10, GOAL_NODE[0])),
    max(10, min(MAP_H - 10, GOAL_NODE[1]))
)

START_NODE = (map_cx, map_cy)
dstar = DStarLite(MAP_W, MAP_H, START_NODE, GOAL_NODE)

print(f"D* Lite Initialized. Start: {START_NODE}, Goal: {GOAL_NODE}")
dstar.replan(START_NODE)
test_path = dstar.getShortestPath(START_NODE) if hasattr(dstar, "getShortestPath") else dstar.get_shortest_path(START_NODE)
if test_path:
    print(f"D* initial path length: {len(test_path)}")
else:
    print("D* initial path FAILED (empty map). Check START/GOAL.")

# Unify DStarLite interface names
def dstar_get_path(node):
    if hasattr(dstar, "getShortestPath"):
        return dstar.getShortestPath(node)
    else:
        return dstar.get_shortest_path(node)

# ======================
# Util
# ======================
def safe(x, fallback=0.0):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return fallback
    return x

# Inflation: approximately 24cm (12 * 0.02)
INFLATION_RADIUS = 7
inflation_offsets = []
for dy in range(-INFLATION_RADIUS, INFLATION_RADIUS + 1):
    for dx in range(-INFLATION_RADIUS, INFLATION_RADIUS + 1):
        if dx * dx + dy * dy <= INFLATION_RADIUS * INFLATION_RADIUS:
            inflation_offsets.append((dx, dy))

# Protect only around goal
GOAL_SAFE_RADIUS_SQ = int((0.25 / MAP_RES) ** 2)  # 25cm

debug_counter = 0
prev_node = START_NODE
replan_counter = 0

# ======================
# Main loop
# ======================
while robot.step(time_step) != -1:

    # ---------------- Odometry ----------------
    left_pos = safe(left_encoder.getValue(), prev_left_pos)
    right_pos = safe(right_encoder.getValue(), prev_right_pos)

    dleft  = (left_pos  - prev_left_pos)  * WHEEL_RADIUS
    dright = (right_pos - prev_right_pos) * WHEEL_RADIUS

    prev_left_pos = left_pos
    prev_right_pos = right_pos

    ENCODER_NOISE_THRESHOLD = 1e-4
    if abs(dleft) < ENCODER_NOISE_THRESHOLD:
        dleft = 0.0
    if abs(dright) < ENCODER_NOISE_THRESHOLD:
        dright = 0.0

    d_center = (dleft + dright) / 2.0
    d_theta  = (dright - dleft) / WHEEL_BASE

    ANGLE_NOISE_THRESHOLD = 1e-4
    if abs(d_theta) < ANGLE_NOISE_THRESHOLD:
        d_theta = 0.0

    robot_theta += d_theta
    robot_theta = (robot_theta + math.pi) % (2 * math.pi) - math.pi

    robot_x += d_center * math.cos(robot_theta)
    robot_y += d_center * math.sin(robot_theta)

    curr_gx, curr_gy = world_to_grid(robot_x, robot_y)
    curr_gx = max(0, min(MAP_W - 1, curr_gx))
    curr_gy = max(0, min(MAP_H - 1, curr_gy))
    current_node = (curr_gx, curr_gy)

    # Display robot/goal
    draw_robot_on_display(curr_gx, curr_gy, robot_theta)
    draw_goal_on_display(GOAL_NODE[0], GOAL_NODE[1])

    debug_counter += 1
    if debug_counter % 50 == 0:
        gx, gy = GOAL_NODE
        gwx, gwy = grid_to_world(gx, gy)
        dist_goal = math.sqrt((robot_x - gwx) ** 2 + (robot_y - gwy) ** 2)
        print(f"Pose world=({robot_x:.2f},{robot_y:.2f},{math.degrees(robot_theta):.1f}°) "
              f"grid={current_node}, goal_world=({gwx:.2f},{gwy:.2f}), dist={dist_goal:.2f}")

    # ---------------- LiDAR & Map update (ZONE sampling) ----------------
    ranges = lidar.getRangeImage()
    detected_obstacles = set()

    for i in range(0, lidar_res):
        # Webots: -fov/2 ~ +fov/2
        angle_raw = -lidar_fov / 2.0 + (i * lidar_fov / lidar_res)
        angle = angle_raw + LIDAR_YAW_OFFSET

        abs_angle = abs(angle)
        if abs_angle <= math.pi / 3:          # Front ±60°
            ray_step = 1
        elif abs_angle <= 2 * math.pi / 3:    # Side ±60°~±120°
            ray_step = 4
        else:                                 # Back ±120°~180°
            ray_step = 12

        if i % ray_step != 0:
            continue

        r = safe(ranges[i])
        if r < 0.12 or r > 7.5:
            continue

        # LiDAR local (robot frame)
        lx = r * math.cos(angle)
        ly = r * math.sin(angle)

        # Robot coordinates -> World coordinates (keep current correct version)
        wx = robot_x + (lx * math.cos(robot_theta) - ly * math.sin(robot_theta))
        wy = robot_y - (lx * math.sin(robot_theta) + ly * math.cos(robot_theta))

        gx, gy = world_to_grid(wx, wy)
        if gx < 0 or gy < 0 or gx >= MAP_W or gy >= MAP_H:
            continue

        # Inflation
        for dx, dy in inflation_offsets:
            ix = gx + dx
            iy = gy + dy
            if 0 <= ix < MAP_W and 0 <= iy < MAP_H:
                dist_to_goal_sq = (ix - GOAL_NODE[0]) ** 2 + (iy - GOAL_NODE[1]) ** 2
                if dist_to_goal_sq > GOAL_SAFE_RADIUS_SQ:
                    detected_obstacles.add((ix, iy))

    # Diff update
    to_remove = current_obstacles - detected_obstacles
    to_add    = detected_obstacles - current_obstacles

    MAX_MAP_CHANGES = 80
    is_rotating_only = abs(d_center) < 0.001 and abs(d_theta) > 0.001
    if is_rotating_only:
        MAX_MAP_CHANGES = 40

    if len(to_add) + len(to_remove) > MAX_MAP_CHANGES:
        map_changed = False
        if debug_counter % 50 == 0:
            print(f"[Map] Too many changes ({len(to_add)+len(to_remove)}), ignore this frame")
    else:
        map_changed = False
        for ox, oy in to_remove:
            occupancy[oy][ox] = 0
            dstar.clear_obstacle(ox, oy)
            display.setColor(0x000000)
            display.drawPixel(ox, oy)
            map_changed = True

        for ox, oy in to_add:
            occupancy[oy][ox] = 1
            dstar.set_obstacle(ox, oy)
            display.setColor(0xFFFFFF)
            display.drawPixel(ox, oy)
            map_changed = True

    current_obstacles = detected_obstacles

    # ---------------- Path planning (D*) ----------------
    node_moved = (current_node != prev_node)
    replan_counter += 1
    if map_changed or node_moved or replan_counter >= 20:
        dstar.replan(current_node)
        prev_node = current_node
        replan_counter = 0

    path = dstar_get_path(current_node)
    if not path:
        print(f"No path! current_node={current_node}, GOAL={GOAL_NODE}")
        left_wheel.setVelocity(0)
        right_wheel.setVelocity(0)
        continue

    # ---------------- Goal check ----------------
    gwx, gwy = grid_to_world(GOAL_NODE[0], GOAL_NODE[1])
    dist_to_goal = math.sqrt((robot_x - gwx) ** 2 + (robot_y - gwy) ** 2)
    if dist_to_goal < 0.10:
        print("Goal reached.")
        left_wheel.setVelocity(0)
        right_wheel.setVelocity(0)
        continue

    # ---------------- Emergency avoidance (front lidar) ----------------
    front_min_dist = float('inf')
    left_min_dist  = float('inf')
    right_min_dist = float('inf')

    fov_range = math.pi / 3
    center_start = int(lidar_res * (0.5 - fov_range / lidar_fov))
    center_end   = int(lidar_res * (0.5 + fov_range / lidar_fov))

    for i in range(center_start, center_end):
        r = safe(ranges[i])
        if r < 0.1 or r > 7.5:
            continue
        angle_raw = -lidar_fov / 2.0 + (i * lidar_fov / lidar_res)
        angle = angle_raw + LIDAR_YAW_OFFSET
        if angle < -0.1:
            left_min_dist = min(left_min_dist, r)
        elif angle > 0.1:
            right_min_dist = min(right_min_dist, r)
        if abs(angle) < math.pi * 2 / 9:
            front_min_dist = min(front_min_dist, r)

    is_blocked = False
    emergency_turn_dir = 0
    # Force avoidance if within 0.25m ahead
    if front_min_dist < 0.25:
        is_blocked = True
        if left_min_dist > 0.5 and right_min_dist < 0.5:
            emergency_turn_dir = 1
        elif right_min_dist > 0.5 and left_min_dist < 0.5:
            emergency_turn_dir = -1
        elif left_min_dist < right_min_dist:
            emergency_turn_dir = -1
        else:
            emergency_turn_dir = 1
        if debug_counter % 10 == 0:
            print(f"[BLOCK] front={front_min_dist:.2f}, L={left_min_dist:.2f}, "
                  f"R={right_min_dist:.2f}, turn={'L' if emergency_turn_dir>0 else 'R'}")

    # ---------------- Path following ----------------
    speed_left = 0.0
    speed_right = 0.0

    if len(path) > 1:
        # Look slightly ahead
        lookahead = 2
        target_idx = min(len(path) - 1, lookahead)
        next_node = path[target_idx]
        target_wx, target_wy = grid_to_world(next_node[0], next_node[1])

        dx = target_wx - robot_x
        dy = target_wy - robot_y
        target_angle = math.atan2(dy, dx)

        angle_diff = target_angle - robot_theta
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        # Set path_safe = False if obstacle in immediate cell
        path_safe = True
        if len(path) > 1:
            nx = path[1]
            if 0 <= nx[0] < MAP_W and 0 <= nx[1] < MAP_H:
                if occupancy[nx[1]][nx[0]] == 1:
                    path_safe = False

        if is_blocked:
            # Completely blocked: almost in-place rotation + slight forward movement
            base_speed = 0.0
            if front_min_dist < 0.18:
                turn_speed = MAX_SPEED * 0.8 * emergency_turn_dir
            elif front_min_dist < 0.25:
                turn_speed = MAX_SPEED * 0.6 * emergency_turn_dir
            else:
                turn_speed = MAX_SPEED * 0.4 * emergency_turn_dir
        elif not path_safe:
            # If immediate node blocked: slowly rotate while correcting
            base_speed = MAX_SPEED * 0.25
            k_p = 2.0
            turn_speed = k_p * angle_diff
        elif abs(angle_diff) > 0.35:
            # If significantly misaligned, rotate in place
            base_speed = 0.0
            k_p = 3.0
            turn_speed = k_p * angle_diff
        elif abs(angle_diff) > 0.15:
            # When slightly misaligned: slowly advance while rotating
            base_speed = MAX_SPEED * 0.4
            k_p = 2.0
            turn_speed = k_p * angle_diff
        else:
            # If angle well aligned, increase speed
            if front_min_dist < 0.60:
                base_speed = MAX_SPEED * 0.5 * (front_min_dist / 0.60)
            else:
                base_speed = MAX_SPEED * 0.8
            k_p = 1.0
            turn_speed = k_p * angle_diff

        distance_to_target = math.sqrt(dx * dx + dy * dy)
        if distance_to_target < 0.3:
            base_speed = min(base_speed, MAX_SPEED * 0.3)

        max_turn = MAX_SPEED * 0.6
        turn_speed = max(-max_turn, min(max_turn, turn_speed))

        speed_left  = base_speed - turn_speed
        speed_right = base_speed + turn_speed

        speed_left  = max(-MAX_SPEED, min(MAX_SPEED, speed_left))
        speed_right = max(-MAX_SPEED, min(MAX_SPEED, speed_right))
    else:
        if current_node == GOAL_NODE:
            print("Goal reached (path len 1).")
        speed_left = 0.0
        speed_right = 0.0

    left_wheel.setVelocity(speed_left)
    right_wheel.setVelocity(speed_right)
