# Full merged Mapping + fast DWA + Correct RPLidar A2 + robot.getSelf() Pose
# Pioneer 3-DX corrected Controller

from controller import Supervisor
import math
import numpy as np

from dwa_planner import DWAPlanner
from dstar_lite import DStarLite


# ==============================================================
# DWA PARAMETERS (tuned)
# ==============================================================
dwa_params = {
    "v_max": 1.0,
    "w_max": 3.0,
    "v_res": 0.1,
    "w_res": 0.1,
    "dt": 0.1,
    "predict_time": 2.0,
    "heading_weight": 4.0,
    "velocity_weight": 3.0,
    "clearance_weight": 5.0,
    "robot_radius": 0.25,
}
dwa = DWAPlanner(dwa_params)


# ==============================================================
# Webots supervisor setup
# ==============================================================
robot = Supervisor()
time_step = int(robot.getBasicTimeStep())
node = robot.getSelf()


# ==============================================================
# LIDAR
# ==============================================================
lidar = robot.getDevice("RPlidar A2")
lidar.enable(time_step)
lidar.enablePointCloud()

for _ in range(10):
    robot.step(time_step)

lidar_res = lidar.getHorizontalResolution()
lidar_fov = lidar.getFov()


# ==============================================================
# Display mapping (unchanged)
# ==============================================================
display = robot.getDevice("display")
MAP_W = display.getWidth()
MAP_H = display.getHeight()
MAP_RES = 0.02
map_cx = MAP_W // 2
map_cy = MAP_H // 2

occupancy = [[0 for _ in range(MAP_W)] for _ in range(MAP_H)]


# ==============================================================
# Wheels & robot geometry
# ==============================================================
left_wheel = robot.getDevice("left wheel")
right_wheel = robot.getDevice("right wheel")

left_wheel.setPosition(float('inf'))
right_wheel.setPosition(float('inf'))

WHEEL_RADIUS = 0.04875
WHEEL_BASE = 0.331
MAX_WHEEL_SPEED = 12.3


# ==============================================================
# Global grid (unchanged)
# ==============================================================
GRID_ROWS = 30
GRID_COLS = 30
GRID_CELL = 0.2
GRID_X_MIN = -3.0
GRID_Y_MIN = -3.0

global_grid = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

planner = None
global_path_world = []
current_wp_index = 0


# ==============================================================
# Helpers
# ==============================================================
def get_robot_pose():
    pos = node.getPosition()
    ori = node.getOrientation()

    robot_x = pos[0]
    robot_y = pos[2]

    # CORRECT yaw extraction
    robot_theta = math.atan2(ori[3], ori[0])
    return robot_x, robot_y, robot_theta


def world_to_grid(x, y):
    col = int((x - GRID_X_MIN) / GRID_CELL)
    row = int((y - GRID_Y_MIN) / GRID_CELL)
    if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
        return (row, col)
    return None


def grid_to_world(r, c):
    return (
        GRID_X_MIN + (c + 0.5) * GRID_CELL,
        GRID_Y_MIN + (r + 0.5) * GRID_CELL,
    )


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
        err = dx // 2
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy // 2
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
        if isnan(r) or r < 0.1 or r > 5.0:
            continue
        angle = (i / lidar_res) * lidar_fov - lidar_fov / 2
        lx = r * math.cos(angle)
        ly = r * math.sin(angle)
        wx = robot_x + (lx * math.cos(robot_theta) - ly * math.sin(robot_theta))
        wy = robot_y + (lx * math.sin(robot_theta) + ly * math.cos(robot_theta))

        obs.append((wx, wy))
        obs.append((wx + 0.05, wy))
        obs.append((wx - 0.05, wy))
        obs.append((wx, wy + 0.05))
        obs.append((wx, wy - 0.05))
    return obs


# ==============================================================
# GLOBAL GOAL
# ==============================================================
GOAL_WORLD = (-2.0, -2.0)


# ==============================================================
# Main loop
# ==============================================================
while robot.step(time_step) != -1:

    robot_x, robot_y, robot_theta = get_robot_pose()

    # Stop at goal
    if math.hypot(robot_x - GOAL_WORLD[0], robot_y - GOAL_WORLD[1]) < 0.4:
        left_wheel.setVelocity(0)
        right_wheel.setVelocity(0)
        continue

    ranges = lidar.getRangeImage()

    # mapping to display + global grid
    for i in range(lidar_res):
        r = ranges[i]
        if isnan(r) or r < 0.1 or r > 5.0:
            continue

        angle = (i / lidar_res) * lidar_fov - lidar_fov / 2
        lx = r * math.cos(angle)
        ly = r * math.sin(angle)
        wx = robot_x + (lx * math.cos(robot_theta) - ly * math.sin(robot_theta))
        wy = robot_y + (lx * math.sin(robot_theta) + ly * math.cos(robot_theta))

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

        cell = world_to_grid(wx, wy)
        if cell is not None:
            global_grid[cell[0]][cell[1]] = 1

    # Global planning (only once)
    if planner is None:
        start_cell = world_to_grid(robot_x, robot_y)
        goal_cell = world_to_grid(GOAL_WORLD[0], GOAL_WORLD[1])
        planner = DStarLite(global_grid, start_cell, goal_cell)
        path_cells = planner.plan()
        global_path_world = [grid_to_world(r, c) for (r, c) in path_cells]
        current_wp_index = 0

    # Select waypoint and compute look-ahead
    wp_x, wp_y = global_path_world[current_wp_index]
    dx = wp_x - robot_x
    dy = wp_y - robot_y
    dist_wp = math.hypot(dx, dy)

    if dist_wp < 0.6 and current_wp_index < len(global_path_world) - 1:
        current_wp_index += 1
        wp_x, wp_y = global_path_world[current_wp_index]
        dx = wp_x - robot_x
        dy = wp_y - robot_y
        dist_wp = math.hypot(dx, dy)

    goal_dir = math.atan2(dy, dx)
    LOOKAHEAD_MIN = 0.6
    LOOKAHEAD_MAX = 1.4
    look_ahead = min(LOOKAHEAD_MAX, max(LOOKAHEAD_MIN, dist_wp))
    local_goal = [
        robot_x + look_ahead * math.cos(goal_dir),
        robot_y + look_ahead * math.sin(goal_dir)
    ]

    # DWA
    obstacles = get_obstacles_from_lidar(ranges, robot_x, robot_y, robot_theta)
    state = [robot_x, robot_y, robot_theta]
    v, w = dwa.compute_velocity(state, local_goal, obstacles)

    # Convert to wheel speeds
    v_left = (v - w * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    v_right = (v + w * WHEEL_BASE / 2.0) / WHEEL_RADIUS

    v_left = max(min(v_left, MAX_WHEEL_SPEED), -MAX_WHEEL_SPEED)
    v_right = max(min(v_right, MAX_WHEEL_SPEED), -MAX_WHEEL_SPEED)

    left_wheel.setVelocity(v_left)
    right_wheel.setVelocity(v_right)