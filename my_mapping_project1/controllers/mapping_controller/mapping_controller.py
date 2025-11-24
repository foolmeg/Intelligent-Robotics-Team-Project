# Fully patched mapping_controller.py with DWA integration and NaN safety

from controller import Robot
import math
import numpy as np
from dwa_planner import DWAPlanner

# ======================
# DWA PARAMETERS
# ======================
dwa_params = {
    "v_max": 0.5,
    "w_max": 1.2,
    "v_res": 0.05,
    "w_res": 0.1,
    "dt": 0.1,
    "predict_time": 2.0,
    "heading_weight": 1.2,
    "velocity_weight": 0.2,
    "clearance_weight": 1.0
}

dwa = DWAPlanner(dwa_params)

goal = [1.5, 1.5]  # temporary static goal

# ======================
# Constants
# ======================
MAX_SPEED = 5.24
MAX_SENSOR_NUMBER = 16
MAX_SENSOR_VALUE = 1024
MIN_DISTANCE = 1.0
WHEEL_WEIGHT_THRESHOLD = 100

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
MAP_RES = 0.02  # meters/pixel

occupancy = [[0 for _ in range(MAP_W)] for _ in range(MAP_H)]

# Robot pose
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

WHEEL_RADIUS = 0.04875
WHEEL_BASE = 0.331

# Encoders
left_encoder = left_wheel.getPositionSensor()
right_encoder = right_wheel.getPositionSensor()
left_encoder.enable(time_step)
right_encoder.enable(time_step)

prev_left_pos = 0.0
prev_right_pos = 0.0

# Distance sensors (ignored for DWA but left initialized)
sensors = []
for i in range(MAX_SENSOR_NUMBER):
    s = robot.getDevice(f"so{i}")
    s.enable(time_step)
    sensors.append(s)


def isnan(x):
    return (x is None) or (isinstance(x, float) and math.isnan(x)) or (x != x)


def safe(x, fallback=0.0):
    if isnan(x):
        return fallback
    return x


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


def get_obstacles_from_lidar(ranges, robot_x, robot_y, robot_theta):
    obs = []
    for i, r in enumerate(ranges):
        if isnan(r) or r <= 0.0 or r > 4.5:
            continue

        angle = -lidar_fov/2 + i * lidar_fov/lidar_res
        if isnan(angle):
            continue

        lx = r * math.cos(angle)
        ly = r * math.sin(angle)

        wx = robot_x + (lx * math.cos(robot_theta) - ly * math.sin(robot_theta))
        wy = robot_y + (lx * math.sin(robot_theta) + ly * math.cos(robot_theta))

        if isnan(wx) or isnan(wy):
            continue

        obs.append((wx, wy))
    return obs


# ======================
# MAIN LOOP
# ======================
while robot.step(time_step) != -1:
    # ODOMETRY
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

    if isnan(robot_x) or isnan(robot_y) or isnan(robot_theta):
        robot_x, robot_y, robot_theta = 0.0, 0.0, 0.0
        continue

    # LIDAR MAPPING
    ranges = lidar.getRangeImage()

    for i in range(lidar_res):
        r = ranges[i]
        if isnan(r) or r < 0.15 or r > 4.5:
            continue

        angle = -lidar_fov/2 + (i * lidar_fov / lidar_res)
        if isnan(angle):
            continue

        lx = r * math.cos(angle)
        ly = r * math.sin(angle)

        wx = robot_x + (lx * math.cos(robot_theta) - ly * math.sin(robot_theta))
        wy = robot_y + (lx * math.sin(robot_theta) + ly * math.cos(robot_theta))

        if isnan(wx) or isnan(wy):
            continue

        gx = int(map_cx + wx / MAP_RES)
        gy = int(map_cy - wy / MAP_RES)

        if 0 <= gx < MAP_W and 0 <= gy < MAP_H:
            rx = int(map_cx + robot_x / MAP_RES)
            ry = int(map_cy - robot_y / MAP_RES)

            for fx, fy in bresenham(rx, ry, gx, gy):
                if 0 <= fx < MAP_W and 0 <= fy < MAP_H:
                    occupancy[fy][fx] = 0
                    display.setColor(0x000000)
                    display.drawPixel(fx, fy)

            occupancy[gy][gx] = 1
            display.setColor(0xFFFFFF)
            display.drawPixel(gx, gy)

    # ======================
    # DWA LOCAL PLANNER
    # ======================
    state = [robot_x, robot_y, robot_theta]
    obstacles = get_obstacles_from_lidar(ranges, robot_x, robot_y, robot_theta)

    v, w = dwa.compute_velocity(state, goal, obstacles)

    v_left = (v - w * WHEEL_BASE / 2) / WHEEL_RADIUS
    v_right = (v + w * WHEEL_BASE / 2) / WHEEL_RADIUS

    left_wheel.setVelocity(v_left)
    right_wheel.setVelocity(v_right)
