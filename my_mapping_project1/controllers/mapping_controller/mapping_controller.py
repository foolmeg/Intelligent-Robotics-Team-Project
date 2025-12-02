# Full merged Mapping + fast DWA + Correct RPLidar A2 + robot.getSelf() Pose
# Pioneer 3-DX corrected Controller

from controller import Robot
import math
import numpy as np
from dwa_planner import DWAPlanner

# ======================
# DWA PARAMETERS
# ======================
dwa_params = {
    "v_max": 1.4,             # was 0.5 → much faster
    "w_max": 3.0,             # agile turning
    "v_res": 0.1,             # coarser sampling = faster compute + bolder moves
    "w_res": 0.2,
    "dt": 0.1,
    "predict_time": 1.0,      # short horizon = fast reaction
    "heading_weight": 6.0,    # strong goal seeking
    "velocity_weight": 4.0,   # encourages moving forward quickly
    "clearance_weight": 1.5,  # moderate caution
}

dwa = DWAPlanner(dwa_params)

goal = [1.5, 1.5]  # temporary static goal

# ==============================================================
# Init Robot
# ==============================================================
robot = Robot()
time_step = int(robot.getBasicTimeStep())

# ==============================================================
# Ground-Truth Pose from Webots Node
# ==============================================================
node = robot.getSelf()

# ==============================================================
# LIDAR (RPLidar A2 in extensionSlot)
# ==============================================================
lidar = robot.getDevice("RPlidar A2")
lidar.enable(time_step)
lidar.enablePointCloud()

# Allow lidar to stabilize
for _ in range(10):
    robot.step(time_step)

lidar_res = lidar.getHorizontalResolution()
lidar_fov = lidar.getFov()

# ==============================================================
# Display Map
# ==============================================================
display = robot.getDevice("display")
MAP_W = display.getWidth()
MAP_H = display.getHeight()
MAP_RES = 0.02

occupancy = [[0 for _ in range(MAP_W)] for _ in range(MAP_H)]
map_cx = MAP_W // 2
map_cy = MAP_H // 2

# ==============================================================
# Wheels
# ==============================================================
left_wheel = robot.getDevice("left wheel")
right_wheel = robot.getDevice("right wheel")
left_wheel.setPosition(float('inf'))
right_wheel.setPosition(float('inf'))

WHEEL_RADIUS = 0.04875
WHEEL_BASE = 0.331
MAX_WHEEL_SPEED = 12.3

# ==============================================================
# Helpers
# ==============================================================
def isnan(x):
    return (x is None) or (isinstance(x, float) and math.isnan(x)) or (x != x)


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


def get_robot_pose():
    pos = node.getPosition()  # (X, Y, Z)
    ori = node.getOrientation()  # 3x3 rotation matrix flattened

    robot_x = pos[0]
    robot_y = pos[2]  # Z-axis becomes mapping Y

    R11 = ori[0]
    R13 = ori[2]
    robot_theta = math.atan2(R13, R11)

    return robot_x, robot_y, robot_theta


def get_obstacles_from_lidar(ranges, robot_x, robot_y, robot_theta):
    obs = []
    for i, r in enumerate(ranges):
        if isnan(r) or r < 0.1 or r > 5.0:
            continue

        angle = (i / lidar_res) * lidar_fov - lidar_fov / 2

        lx = r * math.cos(angle)
        ly = r * math.sin(angle)

        wx = robot_x + (lx * math.cos(robot_theta) - ly * math.sin(robot_theta))
        wy = robot_y + (lx * math.sin(robot_theta) + ly * math.cos(robot_theta))

        if isnan(wx) or isnan(wy):
            continue

        obs.append((wx, wy))
    return obs

# ==============================================================
# MAIN LOOP
# ==============================================================
while robot.step(time_step) != -1:
    # Pose
    robot_x, robot_y, robot_theta = get_robot_pose()

    # Lidar scan
    ranges = lidar.getRangeImage()

    # -----------------------------------------------------------
    # Mapping
    # -----------------------------------------------------------
    for i in range(lidar_res):
        r = ranges[i]
        if isnan(r) or r < 0.1 or r > 5.0:
            continue

        angle = (i / lidar_res) * lidar_fov - lidar_fov / 2

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

    # -----------------------------------------------------------
    # DWA
    # -----------------------------------------------------------
    state = [robot_x, robot_y, robot_theta]
    obstacles = get_obstacles_from_lidar(ranges, robot_x, robot_y, robot_theta)

    v, w = dwa.compute_velocity(state, goal, obstacles)

    v_left = (v - w * WHEEL_BASE / 2) / WHEEL_RADIUS
    v_right = (v + w * WHEEL_BASE / 2) / WHEEL_RADIUS

    v_left = max(min(v_left, MAX_WHEEL_SPEED), -MAX_WHEEL_SPEED)
    v_right = max(min(v_right, MAX_WHEEL_SPEED), -MAX_WHEEL_SPEED)

    left_wheel.setVelocity(v_left)
    right_wheel.setVelocity(v_right)


print("is supervisor?", robot.getSupervisor())