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
    "v_res": 0.15,  # Increased resolution (fewer samples) for performance
    "w_res": 0.2,   # Increased resolution (fewer samples) for performance
    "dt": 0.15,     # Larger timestep for fewer trajectory points
    "predict_time": 1.5,  # Shorter prediction time for fewer points
    "heading_weight": 3.0,
    "velocity_weight": 5.0,
    "clearance_weight": 4.0,
    "robot_radius": 0.25,
    "max_accel": 2.0,
    "max_ang_acc": 4.0,
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
step_count = 0
last_replan_step = 0
REPLAN_INTERVAL = 100  # Replan every 100 steps (less frequent for performance)


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
    num_samples = len(ranges)
    if num_samples == 0:
        return obs
    
    # Downsample lidar for performance - process every Nth point
    downsample_factor = max(1, num_samples // 60)  # Limit to ~60 points max
    
    for i in range(0, num_samples, downsample_factor):
        r = ranges[i]
        if isnan(r) or r < 0.1 or r > 5.0:
            continue
        # Correct angle calculation: distribute angles evenly across FOV
        angle = (i / (num_samples - 1)) * lidar_fov - lidar_fov / 2 if num_samples > 1 else 0.0
        # Convert to world coordinates
        lx = r * math.cos(angle)
        ly = r * math.sin(angle)
        # Rotate by robot orientation and translate
        wx = robot_x + (lx * math.cos(robot_theta) - ly * math.sin(robot_theta))
        wy = robot_y + (lx * math.sin(robot_theta) + ly * math.cos(robot_theta))

        # Only add the main obstacle point (removed padding for performance)
        obs.append((wx, wy))
    
    return obs


# ==============================================================
# GLOBAL GOAL
# ==============================================================
GOAL_WORLD = (-2.0, -2.0)


# ==============================================================
# Main loop
# ==============================================================
while robot.step(time_step) != -1:
    step_count += 1

    robot_x, robot_y, robot_theta = get_robot_pose()

    # Stop at goal
    if math.hypot(robot_x - GOAL_WORLD[0], robot_y - GOAL_WORLD[1]) < 0.4:
        left_wheel.setVelocity(0)
        right_wheel.setVelocity(0)
        continue

    ranges = lidar.getRangeImage()

    # mapping to display + global grid (downsampled for performance)
    num_samples = len(ranges)
    map_downsample = max(1, num_samples // 40)  # Process ~40 points for mapping
    
    for i in range(0, min(num_samples, lidar_res), map_downsample):
        r = ranges[i]
        if isnan(r) or r < 0.1 or r > 5.0:
            continue

        # Correct angle calculation: distribute angles evenly across FOV
        angle = (i / (num_samples - 1)) * lidar_fov - lidar_fov / 2 if num_samples > 1 else 0.0
        lx = r * math.cos(angle)
        ly = r * math.sin(angle)
        wx = robot_x + (lx * math.cos(robot_theta) - ly * math.sin(robot_theta))
        wy = robot_y + (lx * math.sin(robot_theta) + ly * math.cos(robot_theta))

        gx = int(map_cx + wx / MAP_RES)
        gy = int(map_cy - wy / MAP_RES)

        if 0 <= gx < MAP_W and 0 <= gy < MAP_H:
            rx = int(map_cx + robot_x / MAP_RES)
            ry = int(map_cy - robot_y / MAP_RES)
            # Downsample bresenham line for performance
            bres_points = bresenham(rx, ry, gx, gy)
            bres_step = max(1, len(bres_points) // 20)  # Limit line points
            for j in range(0, len(bres_points), bres_step):
                fx, fy = bres_points[j]
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

    # Global planning (initial or periodic replanning)
    should_replan = (planner is None) or (step_count - last_replan_step > REPLAN_INTERVAL)
    
    if should_replan:
        start_cell = world_to_grid(robot_x, robot_y)
        goal_cell = world_to_grid(GOAL_WORLD[0], GOAL_WORLD[1])
        if start_cell is None or goal_cell is None:
            # Robot or goal outside grid bounds - use direct navigation
            if planner is None:  # Only stop on first planning attempt
                left_wheel.setVelocity(0)
                right_wheel.setVelocity(0)
                continue
        else:
            # Check if start or goal cells are occupied - if so, mark as free for planning
            # (robot is currently there, so it must be traversable)
            if global_grid[start_cell[0]][start_cell[1]] == 1:
                global_grid[start_cell[0]][start_cell[1]] = 0
            if global_grid[goal_cell[0]][goal_cell[1]] == 1:
                global_grid[goal_cell[0]][goal_cell[1]] = 0
            
            planner = DStarLite(global_grid, start_cell, goal_cell)
            path_cells = planner.plan()
            if path_cells is None or len(path_cells) == 0:
                # Path planning failed - try direct navigation
                if planner is None:  # Only clear on first attempt
                    global_path_world = []
            else:
                global_path_world = [grid_to_world(r, c) for (r, c) in path_cells]
                # Reset waypoint index - find closest waypoint
                if len(global_path_world) > 0:
                    min_dist = float('inf')
                    best_idx = 0
                    for idx, (wx, wy) in enumerate(global_path_world):
                        dist = math.hypot(robot_x - wx, robot_y - wy)
                        if dist < min_dist:
                            min_dist = dist
                            best_idx = idx
                    current_wp_index = best_idx
            last_replan_step = step_count
    
    # Replan if path is empty or we've reached the end
    if len(global_path_world) == 0:
        # Direct navigation to goal
        dx = GOAL_WORLD[0] - robot_x
        dy = GOAL_WORLD[1] - robot_y
        goal_dir = math.atan2(dy, dx)
        LOOKAHEAD_MIN = 0.6
        LOOKAHEAD_MAX = 1.4
        dist_to_goal = math.hypot(dx, dy)
        look_ahead = min(LOOKAHEAD_MAX, max(LOOKAHEAD_MIN, dist_to_goal))
        local_goal = [
            robot_x + look_ahead * math.cos(goal_dir),
            robot_y + look_ahead * math.sin(goal_dir)
        ]
    else:
        # Select waypoint and compute look-ahead
        if current_wp_index >= len(global_path_world):
            current_wp_index = len(global_path_world) - 1
        
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
    
    # Calculate goal direction for fallback
    dx_goal = local_goal[0] - robot_x
    dy_goal = local_goal[1] - robot_y
    goal_angle = math.atan2(dy_goal, dx_goal)
    angle_diff = goal_angle - robot_theta
    # Normalize angle difference
    while angle_diff > math.pi:
        angle_diff -= 2 * math.pi
    while angle_diff < -math.pi:
        angle_diff += 2 * math.pi
    
    # Ensure minimum forward velocity - robot should always be moving
    MIN_FORWARD_V = 0.4  # Minimum forward velocity in m/s
    if abs(v) < MIN_FORWARD_V:
        if abs(angle_diff) > 0.3:
            # Need to turn more - reduce forward speed while turning
            w = 2.0 * angle_diff  # Proportional control for turning
            w = max(-dwa_params["w_max"], min(dwa_params["w_max"], w))
            v = MIN_FORWARD_V * 0.6  # Reduced forward speed while turning
        else:
            # Aligned with goal - move forward at minimum speed
            v = MIN_FORWARD_V
            if abs(angle_diff) > 0.1:
                w = 1.5 * angle_diff  # Small correction
            else:
                w = 0.0
    
    # Ensure reasonable angular velocity limits
    w = max(-dwa_params["w_max"], min(dwa_params["w_max"], w))

    # Convert to wheel speeds (rad/s)
    # v is in m/s, w is in rad/s
    v_left = (v - w * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    v_right = (v + w * WHEEL_BASE / 2.0) / WHEEL_RADIUS

    v_left = max(min(v_left, MAX_WHEEL_SPEED), -MAX_WHEEL_SPEED)
    v_right = max(min(v_right, MAX_WHEEL_SPEED), -MAX_WHEEL_SPEED)

    left_wheel.setVelocity(v_left)
    right_wheel.setVelocity(v_right)