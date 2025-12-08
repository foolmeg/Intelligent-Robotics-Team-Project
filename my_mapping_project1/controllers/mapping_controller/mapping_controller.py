# Full merged Mapping + fast DWA + Correct RPLidar A2 + robot.getSelf() Pose
# Pioneer 3-DX corrected Controller (x-z plane unified)

from controller import Supervisor
import math
import numpy as np

from dwa_planner import DWAPlanner
from dstar_lite import DStarLite


# ==============================================================
# DWA PARAMETERS (tuned)
# ==============================================================
dwa_params = {
    "v_max": 0.4,  # Moderate speed
    "w_max": 2.0,  # Moderate angular velocity
    "v_res": 0.08,  # Finer resolution for better control
    "w_res": 0.12,
    "dt": 0.1,  # Smaller timestep for more accurate prediction
    "predict_time": 2.5,  # Longer prediction time - predicts ~1.0m ahead at max speed

    "heading_weight": 8.0,  # Balanced - follow path but maintain safety
    "velocity_weight": 2.0,  # Moderate - balance speed and safety
    "clearance_weight": 1.8,  # Increased - maintain safe distance from walls/obstacles

    "robot_radius": 0.25,
    "max_accel": 1.0,  # Moderate acceleration
    "max_ang_acc": 2.5,  # Moderate angular acceleration
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
# Display mapping
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
# Global grid (x-z plane)
# ==============================================================
GRID_ROWS = 30
GRID_COLS = 30
GRID_CELL = 0.2
GRID_X_MIN = -3.0
GRID_Z_MIN = -3.0

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
    robot_y = pos[1]  # CHECK: height value
    robot_z = pos[2]

    robot_theta = math.atan2(ori[3], ori[0])
    return robot_x, robot_y, robot_z, robot_theta


def world_to_grid(x, y):
    """
    Convert world coordinates (x, y) to grid coordinates.
    Note: Parameter name was 'z' but actually represents y-coordinate (x-y plane).
    GRID_Z_MIN is actually the y-coordinate minimum.
    """
    col = int((x - GRID_X_MIN) / GRID_CELL)
    row = int((y - GRID_Z_MIN) / GRID_CELL)  # GRID_Z_MIN is actually y-coordinate
    if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
        return (row, col)
    return None


def grid_to_world(r, c):
    """
    Convert grid coordinates to world coordinates (x, y).
    Returns (x, y) tuple where y corresponds to the grid's Z_MIN dimension.
    """
    world_x = GRID_X_MIN + (c + 0.5) * GRID_CELL
    world_y = GRID_Z_MIN + (r + 0.5) * GRID_CELL  # GRID_Z_MIN is actually y-coordinate
    return (world_x, world_y)


def isnan(x):
    return (x is None) or (isinstance(x, float) and math.isnan(x)) or (x != x)


def smooth_path(path, min_distance=0.4):
    """
    Remove redundant waypoints from path to make it more efficient.
    Keeps waypoints that are at least min_distance apart.
    """
    if len(path) <= 2:
        return path
    
    smoothed = [path[0]]  # Always keep start
    for i in range(1, len(path) - 1):
        # Calculate distance from last kept waypoint
        last_x, last_y = smoothed[-1]
        curr_x, curr_y = path[i]
        dist = math.hypot(curr_x - last_x, curr_y - last_y)
        
        # Keep waypoint if it's far enough
        if dist >= min_distance:
            smoothed.append(path[i])
    
    # Always keep goal
    if len(smoothed) == 0 or smoothed[-1] != path[-1]:
        smoothed.append(path[-1])
    
    return smoothed


def filter_obstacles_on_path(obstacles, path, robot_x, robot_y, trust_radius=0.4):
    """
    Filter out obstacles that are on or very close to the planned path.
    This allows DWA to trust D*Lite's path planning and not overreact to obstacles
    that are already accounted for in the global plan.
    """
    if len(path) < 2:
        return obstacles
    
    filtered = []
    for ox, oy in obstacles:
        min_dist_to_path = float('inf')
        
        # Check distance to each path segment
        for i in range(len(path) - 1):
            p1_x, p1_y = path[i]
            p2_x, p2_y = path[i + 1]
            
            # Distance from obstacle to line segment
            # Vector from p1 to p2
            dx = p2_x - p1_x
            dy = p2_y - p1_y
            seg_len_sq = dx*dx + dy*dy
            
            if seg_len_sq < 1e-6:  # Degenerate segment
                dist = math.hypot(ox - p1_x, oy - p1_y)
            else:
                # Vector from p1 to obstacle
                t = max(0, min(1, ((ox - p1_x) * dx + (oy - p1_y) * dy) / seg_len_sq))
                # Closest point on segment
                closest_x = p1_x + t * dx
                closest_y = p1_y + t * dy
                dist = math.hypot(ox - closest_x, oy - closest_y)
            
            min_dist_to_path = min(min_dist_to_path, dist)
        
        # Only keep obstacles that are significantly off the path
        # or very close to the robot (safety)
        dist_to_robot = math.hypot(ox - robot_x, oy - robot_y)
        # Keep obstacles that are far from path OR very close to robot (emergency safety)
        if min_dist_to_path > trust_radius or dist_to_robot < 0.35:  # Balanced emergency distance
            filtered.append((ox, oy))
    
    return filtered


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
        # Increased minimum range check for earlier obstacle detection
        if isnan(r) or r < 0.15 or r > 5.0:
            continue
        # Distribute angles across FOV
        angle = (i / (num_samples - 1)) * lidar_fov - lidar_fov / 2 if num_samples > 1 else 0.0

        # Lidar point in robot frame (x-forward, y-side) mapped to x-z plane
        lx = r * math.cos(angle)
        ly = r * math.sin(angle)

        wx = robot_x + (lx * math.cos(robot_theta) - ly * math.sin(robot_theta))
        wy = robot_y + (lx * math.sin(robot_theta) + ly * math.cos(robot_theta))

        obs.append((wx, wy))

    return obs


# ==============================================================
# GLOBAL GOAL (x, z)
# ==============================================================
GOAL_WORLD = (-1.0, -1.0)


# ==============================================================
# Main loop
# ==============================================================
while robot.step(time_step) != -1:
    step_count += 1

    robot_x, robot_y, robot_z, robot_theta = get_robot_pose()
    
    # ============================
    # DEBUG: POSE + LIDAR MAPPING
    # ============================

    # Stop at goal
    if math.hypot(robot_x - GOAL_WORLD[0], robot_y - GOAL_WORLD[1]) < 0.4:
        left_wheel.setVelocity(0)
        right_wheel.setVelocity(0)
        continue

    ranges = lidar.getRangeImage()
    
    # ============================
    # DEBUG: POSE + LIDAR MAPPING
    # ============================
    print(f"[POSE] x={robot_x:.3f}, y={robot_y:.3f}, z={robot_z:.3f}, theta={robot_theta:.3f}")

    if len(ranges) > 0:
        # 라이다 중앙 근처 하나 샘플링 (정면)
        i = lidar_res // 2
        r = ranges[i]

        if not isnan(r) and 0.15 < r < 5.0:
            # 라이다 각도 계산
            angle = (i / (lidar_res - 1)) * lidar_fov - lidar_fov / 2

            
            lx = r * math.cos(angle)
            ly = r * math.sin(angle)

            # world frame 변환
            wx = robot_x + (lx * math.cos(robot_theta) - ly * math.sin(robot_theta))
            wy = robot_y + (lx * math.sin(robot_theta) + ly * math.cos(robot_theta))

            print(
                f"[LIDAR] r={r:.3f}, angle={angle:.3f} | "
                f"local=({lx:.3f}, {ly:.3f}) | world=({wx:.3f}, {wy:.3f})"
            )
        else:
            print(f"[LIDAR] invalid r={r}")


    

    # ----------------------------------------------------------
    # Mapping to display + global grid (downsampled for performance)
    # ----------------------------------------------------------
    num_samples = len(ranges)
    map_downsample = max(1, num_samples // 40)  # Process ~40 points for mapping

    for i in range(0, min(num_samples, lidar_res), map_downsample):
        r = ranges[i]
        # Increased minimum range check for earlier obstacle detection
        if isnan(r) or r < 0.15 or r > 5.0:
            continue

        angle = (i / (num_samples - 1)) * lidar_fov - lidar_fov / 2 if num_samples > 1 else 0.0
        lx = r * math.cos(angle)
        ly = r * math.sin(angle)

        wx = robot_x + (lx * math.cos(robot_theta) - ly * math.sin(robot_theta))
        wy = robot_y + (lx * math.sin(robot_theta) + ly * math.cos(robot_theta))


        # Display coordinates
        gx = int(map_cx + wx / MAP_RES)
        gy = int(map_cy - wy / MAP_RES)

        if 0 <= gx < MAP_W and 0 <= gy < MAP_H:
            rx = int(map_cx + robot_x / MAP_RES)
            ry = int(map_cy - robot_y / MAP_RES)

            # Free space along the ray
            bres_points = bresenham(rx, ry, gx, gy)
            bres_step = max(1, len(bres_points) // 20)
            for j in range(0, len(bres_points), bres_step):
                fx, fy = bres_points[j]
                if 0 <= fx < MAP_W and 0 <= fy < MAP_H:
                    occupancy[fy][fx] = 0
                    display.setColor(0x000000)
                    display.drawPixel(fx, fy)

            # Occupied cell (endpoint)
            occupancy[gy][gx] = 1
            display.setColor(0xFFFFFF)
            display.drawPixel(gx, gy)

        # Global grid update (x-y plane, but world_to_grid uses x-z parameter names)
        # Only mark as obstacle if distance is reasonable (not too far, not too close)
        cell = world_to_grid(wx, wy)
        if cell is not None:
            # Only mark as obstacle if it's a real obstacle (not just noise)
            dist_to_robot = math.hypot(wx - robot_x, wy - robot_y)
            if 0.2 < dist_to_robot < 4.5:  # Reasonable obstacle distance
                global_grid[cell[0]][cell[1]] = 1

    # ----------------------------------------------------------
    # Global planning (initial or periodic replanning)
    # ----------------------------------------------------------
    should_replan = (planner is None) or (step_count - last_replan_step > REPLAN_INTERVAL)

    if should_replan:
        start_cell = world_to_grid(robot_x, robot_y)
        goal_cell = world_to_grid(GOAL_WORLD[0], GOAL_WORLD[1])

        if start_cell is None or goal_cell is None:
            # Robot or goal outside grid bounds
            if planner is None:
                left_wheel.setVelocity(0)
                right_wheel.setVelocity(0)
                continue
        else:
            # Start/goal must be free for planning
            if global_grid[start_cell[0]][start_cell[1]] == 1:
                global_grid[start_cell[0]][start_cell[1]] = 0
            if global_grid[goal_cell[0]][goal_cell[1]] == 1:
                global_grid[goal_cell[0]][goal_cell[1]] = 0

            # Reuse planner if it exists and goal hasn't changed, otherwise create new one
            if planner is None or planner.goal != goal_cell:
                planner = DStarLite(global_grid, start_cell, goal_cell)
                path_cells = planner.plan()
            else:
                # Update start position, grid, and replan
                planner.start = start_cell
                path_cells = planner.plan(new_grid=global_grid)
            
            if path_cells is None or len(path_cells) == 0:
                # Path planning failed - clear path and try direct navigation
                global_path_world = []
            else:
                global_path_world = [grid_to_world(r, c) for (r, c) in path_cells]
                
                # Smooth path to remove redundant waypoints
                if len(global_path_world) > 2:
                    global_path_world = smooth_path(global_path_world)

                if len(global_path_world) > 0:
                    # Find closest waypoint ahead of current position
                    min_dist = float('inf')
                    best_idx = 0
                    for idx, (wx, wy) in enumerate(global_path_world):
                        dist = math.hypot(robot_x - wx, robot_y - wy)
                        # Prefer waypoints ahead (further along the path)
                        if idx >= current_wp_index and dist < min_dist:
                            min_dist = dist
                            best_idx = idx
                        elif idx < current_wp_index and dist < min_dist * 0.5:
                            # Only go back if significantly closer
                            min_dist = dist
                            best_idx = idx
                    current_wp_index = best_idx

            last_replan_step = step_count

    # ----------------------------------------------------------
    # Local goal selection (from global path or direct goal)
    # ----------------------------------------------------------
    if len(global_path_world) == 0:
        # Direct navigation to goal - but be more careful
        dx = GOAL_WORLD[0] - robot_x
        dy = GOAL_WORLD[1] - robot_y
        dist_to_goal = math.hypot(dx, dy)
        
        # If very close to goal, stop
        if dist_to_goal < 0.3:
            left_wheel.setVelocity(0)
            right_wheel.setVelocity(0)
            continue
        
        goal_dir = math.atan2(dy, dx)
        LOOKAHEAD_MIN = 0.15  # Smaller lookahead for direct navigation
        LOOKAHEAD_MAX = 0.4   # Smaller max for safety
        look_ahead = min(LOOKAHEAD_MAX, max(LOOKAHEAD_MIN, dist_to_goal))

        local_goal = [
            robot_x + look_ahead * math.cos(goal_dir),
            robot_y + look_ahead * math.sin(goal_dir)
        ]
    else:
        if current_wp_index >= len(global_path_world):
            current_wp_index = len(global_path_world) - 1

        wp_x, wp_y = global_path_world[current_wp_index]
        dx = wp_x - robot_x
        dy = wp_y - robot_y
        dist_wp = math.hypot(dx, dy)

        # Adaptive waypoint threshold based on path curvature
        waypoint_threshold = 0.5  # Base threshold
        if current_wp_index < len(global_path_world) - 1:
            # Look ahead to next waypoint
            next_wp_x, next_wp_y = global_path_world[current_wp_index + 1]
            # Calculate angle between current->wp and wp->next_wp
            angle1 = math.atan2(dy, dx)
            angle2 = math.atan2(next_wp_y - wp_y, next_wp_x - wp_x)
            angle_diff = abs(angle1 - angle2)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            # If path is straight, use larger threshold for smoother movement
            if angle_diff < 0.3:  # ~17 degrees
                waypoint_threshold = 0.7
        
        if dist_wp < waypoint_threshold and current_wp_index < len(global_path_world) - 1:
            current_wp_index += 1
            wp_x, wp_y = global_path_world[current_wp_index]
            dx = wp_x - robot_x
            dy = wp_y - robot_y
            dist_wp = math.hypot(dx, dy)

        goal_dir = math.atan2(dy, dx)
        LOOKAHEAD_MIN = 0.2  # Reduced for slower movement
        LOOKAHEAD_MAX = 0.5  # Reduced for slower movement
        look_ahead = min(LOOKAHEAD_MAX, max(LOOKAHEAD_MIN, dist_wp))

        local_goal = [
            robot_x + look_ahead * math.cos(goal_dir),
            robot_y + look_ahead * math.sin(goal_dir)
        ]

    # ----------------------------------------------------------
    # Direct path following (DWA removed for testing)
    # ----------------------------------------------------------
    # Simple proportional control to follow the local goal
    dx_goal = local_goal[0] - robot_x
    dy_goal = local_goal[1] - robot_y
    dist_to_goal = math.hypot(dx_goal, dy_goal)
    
    # Desired heading
    desired_heading = math.atan2(dy_goal, dx_goal)
    heading_error = desired_heading - robot_theta
    
    # Normalize angle to [-pi, pi]
    while heading_error > math.pi:
        heading_error -= 2 * math.pi
    while heading_error < -math.pi:
        heading_error += 2 * math.pi
    
    # Simple proportional control
    # Forward velocity based on distance and heading alignment
    v_base = 0.3  # Base forward velocity
    v = v_base * (1.0 - abs(heading_error) / math.pi) * min(1.0, dist_to_goal / 0.5)
    v = max(0.0, min(v, 0.4))  # Clamp to reasonable range
    
    # Angular velocity to correct heading
    w = 2.0 * heading_error  # Proportional gain
    w = max(-2.0, min(2.0, w))  # Clamp angular velocity
    
    # If very close to goal, slow down
    if dist_to_goal < 0.2:
        v *= 0.5
        w *= 0.5

    # 바퀴 속도로 변환
    v_left = (v - w * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    v_right = (v + w * WHEEL_BASE / 2.0) / WHEEL_RADIUS

    v_left = max(min(v_left, MAX_WHEEL_SPEED), -MAX_WHEEL_SPEED)
    v_right = max(min(v_right, MAX_WHEEL_SPEED), -MAX_WHEEL_SPEED)

    left_wheel.setVelocity(v_left)
    right_wheel.setVelocity(v_right)

