"""
from controller import Robot
import math

# ======================
# Constants
# ======================
MAX_SPEED = 5.24
MAX_SENSOR_NUMBER = 16
MAX_SENSOR_VALUE = 1024
MIN_DISTANCE = 1.0
WHEEL_WEIGHT_THRESHOLD = 100

SENSOR_WEIGHTS = [
    [150,   0], [200,   0], [300,   0], [600,   0],
    [  0, 600], [  0, 300], [  0, 200], [  0, 150],
    [  0,   0], [  0,   0], [  0,   0], [  0,   0],
    [  0,   0], [  0,   0], [  0,   0], [  0,   0],
]

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

# Display
display = robot.getDevice("display")
MAP_W = display.getWidth()
MAP_H = display.getHeight()
MAP_RES = 0.02  # 2cm/pixel

occupancy = [[0 for _ in range(MAP_W)] for _ in range(MAP_H)]

# Robot pose in odometry frame
robot_x = 0.0
robot_y = 0.0
robot_theta = 0.0

map_cx = MAP_W // 2
map_cy = MAP_H // 2

# Wheels
left_wheel = robot.getDevice("left wheel")
right_wheel = robot.getDevice("right wheel")
left_wheel.setPosition(float('inf'))
right_wheel.setPosition(float('inf'))

# Encoders
WHEEL_RADIUS = 0.0475
WHEEL_BASE = 0.33

left_encoder = left_wheel.getPositionSensor()
right_encoder = right_wheel.getPositionSensor()
left_encoder.enable(time_step)
right_encoder.enable(time_step)

prev_left_pos = left_encoder.getValue() or 0.0
prev_right_pos = right_encoder.getValue() or 0.0

# IR sensors
sensors = []
for i in range(MAX_SENSOR_NUMBER):
    s = robot.getDevice(f"so{i}")
    s.enable(time_step)
    sensors.append(s)

state = FORWARD


# ======================
# Utility
# ======================
def safe(x, fallback=0.0):
    if x is None or isinstance(x, float) and math.isnan(x):
        return fallback
    return x

# Bresenham for free-space marking
def bresenham(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x1, y1))
    return points


# ======================
# MAIN LOOP
# ======================
while robot.step(time_step) != -1:

    # --- ODOMETRY ---
    left_pos = safe(left_encoder.getValue(), prev_left_pos)
    right_pos = safe(right_encoder.getValue(), prev_right_pos)

    dleft = (left_pos - prev_left_pos) * WHEEL_RADIUS
    dright = (right_pos - prev_right_pos) * WHEEL_RADIUS

    prev_left_pos = left_pos
    prev_right_pos = right_pos

    d_center = (dleft + dright) / 2.0
    d_theta = (dright - dleft) / WHEEL_BASE

    robot_theta += d_theta
    robot_theta = (robot_theta + math.pi) % (2 * math.pi) - math.pi

    robot_x += d_center * math.cos(robot_theta)
    robot_y += d_center * math.sin(robot_theta)

    # --- LIDAR → OGM ---
    ranges = lidar.getRangeImage()

    for i in range(lidar_res):
        r = safe(ranges[i])

        if r < 0.15 or r > 4.5:
            continue

        angle = -lidar_fov / 2.0 + (i * lidar_fov / lidar_res)
        # angle += math.pi / 2.0

        lx = r * math.cos(angle)
        ly = r * math.sin(angle)

        wx = robot_x + (lx * math.cos(robot_theta) - ly * math.sin(robot_theta))
        wy = robot_y + (lx * math.sin(robot_theta) + ly * math.cos(robot_theta))

        gx = int(map_cx + wx / MAP_RES)
        gy = int(map_cy - wy / MAP_RES)

        if gx < 0 or gy < 0 or gx >= MAP_W or gy >= MAP_H:
            continue

        # Free space FIRST
        rx = int(map_cx + robot_x / MAP_RES)
        ry = int(map_cy - robot_y / MAP_RES)

        for fx, fy in bresenham(rx, ry, gx, gy):
            if 0 <= fx < MAP_W and 0 <= fy < MAP_H:
                occupancy[fy][fx] = 0
                display.setColor(0x000000)
                display.drawPixel(fx, fy)

        # Occupied cell
        occupancy[gy][gx] = 1
        display.setColor(0xFFFFFF)
        display.drawPixel(gx, gy)

    # --- OBSTACLE AVOIDANCE ---
    wheel_weight_total = [0.0, 0.0]

    for i in range(MAX_SENSOR_NUMBER):
        sv = safe(sensors[i].getValue())
        if sv == 0:
            continue

        dist = 5.0 * (1.0 - sv / MAX_SENSOR_VALUE)
        spd = 1 - (dist / MIN_DISTANCE) if dist < MIN_DISTANCE else 0.0

        wheel_weight_total[0] += SENSOR_WEIGHTS[i][0] * spd
        wheel_weight_total[1] += SENSOR_WEIGHTS[i][1] * spd

    speed_left = MAX_SPEED
    speed_right = MAX_SPEED

    if wheel_weight_total[0] > WHEEL_WEIGHT_THRESHOLD:
        speed_left = 0.7 * MAX_SPEED
        speed_right = -0.7 * MAX_SPEED
    elif wheel_weight_total[1] > WHEEL_WEIGHT_THRESHOLD:
        speed_left = -0.7 * MAX_SPEED
        speed_right = 0.7 * MAX_SPEED

    left_wheel.setVelocity(speed_left)
    right_wheel.setVelocity(speed_right)
"""
from controller import Robot
import math

# ======================
# CONFIG
# ======================
MAP_RES = 0.02          # 2cm per pixel
LIDAR_MIN_RANGE = 0.05
LIDAR_MAX_RANGE = 8.0
LIDAR_OFFSET = math.pi  # LiDAR가 뒤보기 문제 → 180도 보정

LOG_ODD_FREE = -0.8     # free 업데이트
LOG_ODD_OCC = 1.2       # occupied 업데이트
LOG_ODD_MIN = -5.0
LOG_ODD_MAX = 5.0

# ======================
# Init
# ======================
robot = Robot()
time_step = int(robot.getBasicTimeStep())

lidar = robot.getDevice("RPlidar A2")
lidar.enable(time_step)
lidar_res = lidar.getHorizontalResolution()
lidar_fov = lidar.getFov()

display = robot.getDevice("display")
MAP_W = display.getWidth()
MAP_H = display.getHeight()

map_cx = MAP_W // 2
map_cy = MAP_H // 2

# OGM log-odds
grid = [[0.0 for _ in range(MAP_W)] for _ in range(MAP_H)]

# Wheels + odometry
left = robot.getDevice("left wheel")
right = robot.getDevice("right wheel")
left.setPosition(float('inf'))
right.setPosition(float('inf'))

WHEEL_RADIUS = 0.0475
WHEEL_BASE = 0.33

lenc = left.getPositionSensor()
renc = right.getPositionSensor()
lenc.enable(time_step)
renc.enable(time_step)

prev_l = lenc.getValue()
prev_r = renc.getValue()

robot_x = 0.0
robot_y = 0.0
robot_theta = 0.0

# ======================
# Utility
# ======================
def bresenham(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x1, y1))
    return points


def world_to_grid(wx, wy):
    gx = int(map_cx + wx / MAP_RES)
    gy = int(map_cy - wy / MAP_RES)
    return gx, gy


def update_cell(x, y, val):
    if 0 <= x < MAP_W and 0 <= y < MAP_H:
        grid[y][x] = max(LOG_ODD_MIN, min(LOG_ODD_MAX, grid[y][x] + val))


# ======================
# MAIN LOOP
# ======================
while robot.step(time_step) != -1:

    # ======================
    # 1) ODOMETRY
    # ======================
    lpos = lenc.getValue()
    rpos = renc.getValue()

    dL = (lpos - prev_l) * WHEEL_RADIUS
    dR = (rpos - prev_r) * WHEEL_RADIUS

    prev_l = lpos
    prev_r = rpos

    dC = (dL + dR) / 2
    dT = (dR - dL) / WHEEL_BASE

    robot_theta += dT
    robot_theta = (robot_theta + math.pi) % (2*math.pi) - math.pi

    robot_x += dC * math.cos(robot_theta)
    robot_y += dC * math.sin(robot_theta)

    # Grid robot pose
    rx, ry = world_to_grid(robot_x, robot_y)

    # ======================
    # 2) LIDAR SCAN
    # ======================
    ranges = lidar.getRangeImage()

    for i in range(lidar_res):
        r = ranges[i]
        if r < LIDAR_MIN_RANGE or r > LIDAR_MAX_RANGE:
            continue

        angle = -lidar_fov/2 + (i+0.5) * lidar_fov / lidar_res
        beam_theta = robot_theta + angle + LIDAR_OFFSET

        wx = robot_x + r * math.cos(beam_theta)
        wy = robot_y + r * math.sin(beam_theta)

        gx, gy = world_to_grid(wx, wy)

        # ---- Free space update ----
        if 0 <= rx < MAP_W and 0 <= ry < MAP_H:
            free_cells = bresenham(rx, ry, gx, gy)
            for fx, fy in free_cells[:-1]:      # 끝점은 occupied
                update_cell(fx, fy, LOG_ODD_FREE)

        # ---- Occupied update ----
        update_cell(gx, gy, LOG_ODD_OCC)

    # ======================
    # 3) DRAW MAP
    # ======================
    for y in range(MAP_H):
        for x in range(MAP_W):
            val = grid[y][x]

            if val > 1.0:       # occupied
                display.setColor(0xFFFFFF)
            elif val < -1.0:    # free
                display.setColor(0x222222)
            else:               # unknown
                display.setColor(0x000000)

            display.drawPixel(x, y)

    # Robot position indicator
    if 0 <= rx < MAP_W and 0 <= ry < MAP_H:
        display.setColor(0x00FF00)
        display.drawPixel(rx, ry)

    # Heading line
    hx = int(rx + 6 * math.cos(robot_theta))
    hy = int(ry - 6 * math.sin(robot_theta))
    if 0 <= hx < MAP_W and 0 <= hy < MAP_H:
        display.setColor(0x00FF00)
        display.drawLine(rx, ry, hx, hy)

    # Debug print
    if int(robot.getTime()) % 1 == 0:
        print(f"Pose: {robot_x:.2f}, {robot_y:.2f}, {math.degrees(robot_theta):.1f}°")

