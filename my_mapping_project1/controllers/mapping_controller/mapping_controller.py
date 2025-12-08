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
    "v_max": 0.35,  # Reduced speed for conservative navigation
    "w_max": 1.8,  # Reduced angular velocity
    "v_res": 0.08,  # Finer resolution for better control
    "w_res": 0.12,
    "dt": 0.1,  # Smaller timestep for more accurate prediction
    "predict_time": 2.5,  # Longer prediction time - predicts ~0.9m ahead at max speed

    "heading_weight": 4.0,  # Balanced
    "velocity_weight": 1.0,  # Increased to encourage movement
    "clearance_weight": 6.0,  # Slightly relaxed from 6.0 to allow passing gaps

    "robot_radius": 0.25,
    "max_accel": 0.8,  # Reduced acceleration for smoother, safer movement
    "max_ang_acc": 2.0,  # Reduced angular acceleration
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
stuck_count = 0  # Logic for escape maneuver
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
        # Filter out invalid readings and very close noise
        # Only consider obstacles at reasonable distance (not too close, not too far)
        # CRITICAL FIX: Reduced min from 0.3 to 0.15. 
        # 0.3 made the robot blind to obstacles closer than 30cm, causing crashes.
        if isnan(r) or r < 0.15 or r > 4.5:
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
GOAL_WORLD = (-1.0, 2.0)


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


    if len(ranges) > 0:
        # Sample one point near lidar center (front)
        i = lidar_res // 2
        r = ranges[i]

        if not isnan(r) and 0.15 < r < 5.0:
            # Calculate lidar angle
            angle = (i / (lidar_res - 1)) * lidar_fov - lidar_fov / 2

            
            lx = r * math.cos(angle)
            ly = r * math.sin(angle)

            # Convert to world frame
            wx = robot_x + (lx * math.cos(robot_theta) - ly * math.sin(robot_theta))
            wy = robot_y + (lx * math.sin(robot_theta) + ly * math.cos(robot_theta))

            print(
                #f"[LIDAR] r={r:.3f}, angle={angle:.3f} | "
                #f"local=({lx:.3f}, {ly:.3f}) | world=({wx:.3f}, {wy:.3f})"
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
                new_path_cells = planner.plan()
            else:
                # Update start position, grid, and replan
                planner.start = start_cell
                new_path_cells = planner.plan(new_grid=global_grid)
            
            # Fix 1: Only update path if new path is valid, otherwise keep previous path
            if new_path_cells is not None and len(new_path_cells) > 0:
                global_path_world = [grid_to_world(r, c) for (r, c) in new_path_cells]
                
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
            # If path planning failed, keep previous global_path_world (don't clear it)

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
    # DWA local planning (conservative obstacle avoidance)
    # ----------------------------------------------------------
    obstacles = get_obstacles_from_lidar(ranges, robot_x, robot_y, robot_theta)
    
    # Debug: Check obstacles in front direction
    front_obstacles = []
    for ox, oy in obstacles:
        dx = ox - robot_x
        dy = oy - robot_y
        dist = math.hypot(dx, dy)
        angle_to_obs = math.atan2(dy, dx)
        angle_diff = abs(angle_to_obs - robot_theta)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        # Check if obstacle is in front (within 60 degrees) and close
        if angle_diff < math.pi / 3 and dist < 1.5:
            front_obstacles.append((dist, angle_diff))
    
    if len(front_obstacles) > 0:
        closest = min(front_obstacles, key=lambda x: x[0])
        print(f"[DWA] Front obstacles: {len(front_obstacles)}, closest: {closest[0]:.3f}m at {math.degrees(closest[1]):.1f}deg")
    
    # Filter obstacles: trust D*Lite path but maintain safety
    original_clearance_weight = dwa_params["clearance_weight"]
    
    if len(global_path_world) > 0:
        # Filter obstacles on path - trust the global planner
        # FIX: Reduced trust_radius from 0.6 to 0.1.
        # 0.6 caused the robot to ignore real obstacles if they were near the path.
        filtered_obstacles = filter_obstacles_on_path(obstacles, global_path_world, robot_x, robot_y, trust_radius=0.4)
        # More reduction when following path - trust the path more
        dwa_params["clearance_weight"] = original_clearance_weight * 0.8  # Slight reduction only
        print(f"[DWA] Following path: {len(obstacles)} -> {len(filtered_obstacles)} obstacles, clearance_weight: {dwa_params['clearance_weight']:.2f}")
    else:
        # No path - use all obstacles but be conservative
        filtered_obstacles = obstacles
        dwa_params["clearance_weight"] = original_clearance_weight
        print(f"[DWA] No path: {len(obstacles)} obstacles, clearance_weight: {dwa_params['clearance_weight']:.2f}")
    
    state = [robot_x, robot_y, robot_theta]
    v, w = dwa.compute_velocity(state, local_goal, filtered_obstacles)
    
    print(f"[DWA] Output: v={v:.3f}, w={w:.3f}, obstacles={len(filtered_obstacles)}")
    
    # Restore original clearance weight
    dwa_params["clearance_weight"] = original_clearance_weight
    
    # Check if there are real obstacles blocking forward movement
    has_blocking_obstacle = False
    for ox, oy in filtered_obstacles:
        dx = ox - robot_x
        dy = oy - robot_y
        dist = math.hypot(dx, dy)
        angle_to_obs = math.atan2(dy, dx)
        angle_diff = abs(angle_to_obs - robot_theta)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        # Obstacle is blocking if it's in front (wider check to prevent unsafe forcing)
        if angle_diff < math.pi / 2.5 and dist < 1.0:  # Increased range 0.6->1.0, Angle 45->72 deg
            has_blocking_obstacle = True
            print(f"[DWA] Blocking obstacle found: dist={dist:.3f}, angle={math.degrees(angle_diff):.1f}deg")
            break
    
    # If no blocking obstacle and velocity is too low, force forward movement
    # ----------------------------------------
    # Fallback: Strong Escape Maneuver
    # ----------------------------------------
    if abs(v) < 0.05:
        stuck_count += 1
    else:
        stuck_count = 0

    if stuck_count > 15:  # Stuck for 1.5s
        # Execute escape: Rotate in place ~120 degrees
        print(f"[Fallback] Stuck detected ({stuck_count}). Rotating to escape.")
        v = 0.0
        w = 1.5  # Fast rotation
        
        # Reset after enough rotation (~15 steps * 0.1s * 1.5 rad/s = 2.25 rad = 128 deg)
        if stuck_count > 30:
            stuck_count = 0
            
    elif not has_blocking_obstacle and v < 0.1:
        # Just a nudge if purely hesitant, but don't force collision
        pass
    
    # Ensure minimum forward velocity to prevent backing up unnecessarily
    # Only allow backward movement if there's a real obstacle very close
    if v < 0:
        # Check if there's an obstacle very close in front
        front_obstacle_close = False
        for ox, oy in filtered_obstacles:
            dx = ox - robot_x
            dy = oy - robot_y
            dist = math.hypot(dx, dy)
            # Check if obstacle is in front (within 90 degrees)
            angle_to_obs = math.atan2(dy, dx)
            angle_diff = abs(angle_to_obs - robot_theta)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            if dist < 0.4 and angle_diff < math.pi / 2:  # Very close and in front
                front_obstacle_close = True
                break
        
        if not front_obstacle_close:
            # No close obstacle in front - don't go backward
            v = max(0.0, v * 0.3)  # Reduce backward velocity significantly

    # Angular velocity clamp (safety)
    w = max(-dwa_params["w_max"], min(dwa_params["w_max"], w))

    # Convert to wheel speeds
    v_left = (v - w * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    v_right = (v + w * WHEEL_BASE / 2.0) / WHEEL_RADIUS

    v_left = max(min(v_left, MAX_WHEEL_SPEED), -MAX_WHEEL_SPEED)
    v_right = max(min(v_right, MAX_WHEEL_SPEED), -MAX_WHEEL_SPEED)

    left_wheel.setVelocity(v_left)
    right_wheel.setVelocity(v_right)

