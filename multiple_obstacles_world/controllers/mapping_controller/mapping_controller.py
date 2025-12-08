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
    "v_max": 0.8,  # Max velocity
    "w_max": 3.5,  # Max angular velocity for maneuverability
    "v_res": 0.1,  # Velocity sampling resolution
    "w_res": 0.12,  # Angular velocity sampling resolution
    "dt": 0.1,     # Timestep for trajectory simulation
    "predict_time": 2.0,  # Prediction time for obstacle avoidance
    "heading_weight": 3.0,  # Weight for heading toward goal
    "velocity_weight": 4.0,  # Weight for forward progress
    "clearance_weight": 3.0,  # Weight for obstacle clearance (balanced with progress)
    "robot_radius": 0.25,
    "max_accel": 2.5,  # Max linear acceleration
    "max_ang_acc": 5.0,  # Max angular acceleration
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
GRID_ROWS = 40  # Increased grid size to cover larger area
GRID_COLS = 40  # Increased grid size to cover larger area
GRID_CELL = 0.2
GRID_X_MIN = -4.0  # Expanded to cover goal at -2.7
GRID_Y_MIN = -4.0  # Expanded to cover goal at -2.7

global_grid = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

planner = None
global_path_world = []
current_wp_index = 0
step_count = 0
last_replan_step = 0
REPLAN_INTERVAL = 50  # Replan more frequently for better responsiveness
last_local_goal_adjustment = None


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
        
        # Webots lidar typically scans from -FOV/2 to +FOV/2 relative to robot forward direction
        # Calculate angle relative to robot's forward direction
        if num_samples > 1:
            # Map index to angle: from -FOV/2 to +FOV/2
            angle = (i / (num_samples - 1) - 0.5) * lidar_fov
        else:
            angle = 0.0
        
        # Convert to local robot coordinates (lidar frame)
        # In Webots, forward is typically +x, left is +y
        lx = r * math.cos(angle)
        ly = r * math.sin(angle)
        
        # Transform to world coordinates: rotate by robot orientation and translate
        wx = robot_x + (lx * math.cos(robot_theta) - ly * math.sin(robot_theta))
        wy = robot_y + (lx * math.sin(robot_theta) + ly * math.cos(robot_theta))

        obs.append((wx, wy))
    
    return obs


# ==============================================================
# GLOBAL GOAL
# ==============================================================
GOAL_WORLD = (-2.7, -2.7)


# ==============================================================
# Main loop
# ==============================================================
last_robot_pos = None
stuck_counter = 0

last_local_goal_adjustment = None

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

        # Calculate angle relative to robot's forward direction
        if num_samples > 1:
            # Map index to angle: from -FOV/2 to +FOV/2
            angle = (i / (num_samples - 1) - 0.5) * lidar_fov
        else:
            angle = 0.0
        
        # Convert to local robot coordinates (lidar frame)
        lx = r * math.cos(angle)
        ly = r * math.sin(angle)
        
        # Transform to world coordinates
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
                    occupancy[fy][fx] = 0  # Free space for display
                    display.setColor(0x000000)
                    display.drawPixel(fx, fy)
            # Mark obstacle at endpoint
            occupancy[gy][gx] = 1
            display.setColor(0xFFFFFF)
            display.drawPixel(gx, gy)

        # Update global grid - mark obstacle at detected point with inflation
        # Use less aggressive inflation to allow path planning to find routes
        cell = world_to_grid(wx, wy)
        if cell is not None:
            # Don't mark cells too close to robot as obstacles (robot's own position)
            robot_cell = world_to_grid(robot_x, robot_y)
            if robot_cell:
                dist_to_robot_cell = math.hypot(cell[0] - robot_cell[0], cell[1] - robot_cell[1]) * GRID_CELL
                # Don't mark cells within robot radius as obstacles (robot's own space)
                if dist_to_robot_cell < dwa_params["robot_radius"] + 0.1:
                    continue  # Skip marking cells too close to robot
            
            # Inflate obstacles by robot radius + smaller safety margin
            # Reduced inflation to allow more paths through narrow passages
            inflation_radius_cells = int((dwa_params["robot_radius"] + 0.08) / GRID_CELL) + 1
            for dr in range(-inflation_radius_cells, inflation_radius_cells + 1):
                for dc in range(-inflation_radius_cells, inflation_radius_cells + 1):
                    r, c = cell[0] + dr, cell[1] + dc
                    if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
                        # Don't mark cells too close to robot
                        if robot_cell:
                            dist_to_robot = math.hypot((r - robot_cell[0]) * GRID_CELL, (c - robot_cell[1]) * GRID_CELL)
                            if dist_to_robot < dwa_params["robot_radius"] + 0.15:
                                continue  # Skip cells in robot's immediate space
                        dist = math.hypot(dr * GRID_CELL, dc * GRID_CELL)
                        if dist <= (dwa_params["robot_radius"] + 0.08):
                            global_grid[r][c] = 1

    # Global planning (initial or periodic replanning)
    # Improved stuck detection: check if robot hasn't made progress
    stuck_threshold = 30  # Replan if stuck for 30 steps (more responsive)
    
    # Track robot position for stuck detection
    if last_robot_pos is None:
        last_robot_pos = (robot_x, robot_y)
        stuck_counter = 0
    else:
        dist_moved = math.hypot(robot_x - last_robot_pos[0], robot_y - last_robot_pos[1])
        if dist_moved < 0.05:  # Less than 5cm movement
            stuck_counter = stuck_counter + 1
        else:
            stuck_counter = 0
            last_robot_pos = (robot_x, robot_y)
    
    is_stuck = (stuck_counter > stuck_threshold) or (len(global_path_world) > 0 and 
                current_wp_index < len(global_path_world) and
                step_count - last_replan_step > stuck_threshold * 2)
    
    should_replan = (planner is None) or (step_count - last_replan_step > REPLAN_INTERVAL) or is_stuck or (len(global_path_world) == 0)
    
    if should_replan:
        start_cell = world_to_grid(robot_x, robot_y)
        goal_cell = world_to_grid(GOAL_WORLD[0], GOAL_WORLD[1])
        
        if start_cell is None or goal_cell is None:
            # Robot or goal outside grid bounds - use direct navigation
            global_path_world = []  # Clear path to use direct navigation
            if planner is None:  # Only stop on first planning attempt
                # Don't stop - use direct navigation instead
                pass
        else:
            # Create a copy of the grid for planning (to avoid modifying the original)
            planning_grid = [row[:] for row in global_grid]
            
            # Ensure start and goal cells are free for planning
            # (robot is currently at start, goal should be reachable)
            # Also clear a larger radius around start/goal to ensure planning can begin
            # This is critical - robot needs space to start planning
            # Clear even more aggressively to ensure path planning works
            clear_radius_start = 4  # Larger radius around robot (start)
            clear_radius_goal = 3  # Smaller radius around goal
            for cell, clear_rad in [(start_cell, clear_radius_start), (goal_cell, clear_radius_goal)]:
                if cell is not None:
                    for dr in range(-clear_rad, clear_rad + 1):
                        for dc in range(-clear_rad, clear_rad + 1):
                            r, c = cell[0] + dr, cell[1] + dc
                            if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
                                if math.hypot(dr, dc) <= clear_rad:
                                    planning_grid[r][c] = 0
            
            # Create or update planner
            if planner is None:
                planner = DStarLite(planning_grid, start_cell, goal_cell)
                changed = []  # No changes on first plan
            else:
                # Update grid and replan
                planner.grid = planning_grid
                planner.start = start_cell
                planner.goal = goal_cell
                # Find changed cells (cells that became occupied since last plan)
                # For efficiency, check cells in a radius around robot and goal
                changed = []
                check_radius = 8  # Check larger radius for better replanning
                for center in [start_cell, goal_cell]:
                    for dr in range(-check_radius, check_radius + 1):
                        for dc in range(-check_radius, check_radius + 1):
                            r, c = center[0] + dr, center[1] + dc
                            if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
                                changed.append((r, c))
                # Remove duplicates
                changed = list(set(changed))
            
            path_cells = planner.plan(changed)
            if path_cells is None or len(path_cells) == 0:
                # Path planning failed - debug why
                if step_count % 50 == 0:  # Print debug every 50 steps
                    # Check if start/goal are in bounds
                    start_ok = (start_cell is not None and 
                               0 <= start_cell[0] < GRID_ROWS and 
                               0 <= start_cell[1] < GRID_COLS)
                    goal_ok = (goal_cell is not None and 
                              0 <= goal_cell[0] < GRID_ROWS and 
                              0 <= goal_cell[1] < GRID_COLS)
                    # Count occupied cells around start/goal
                    occupied_near_start = 0
                    occupied_near_goal = 0
                    if start_cell:
                        for dr in range(-3, 4):
                            for dc in range(-3, 4):
                                r, c = start_cell[0] + dr, start_cell[1] + dc
                                if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
                                    if planning_grid[r][c] == 1:
                                        occupied_near_start += 1
                    if goal_cell:
                        for dr in range(-3, 4):
                            for dc in range(-3, 4):
                                r, c = goal_cell[0] + dr, goal_cell[1] + dc
                                if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
                                    if planning_grid[r][c] == 1:
                                        occupied_near_goal += 1
                    
                    print(f"Step {step_count}: PATH PLANNING FAILED - start_cell={start_cell}, goal_cell={goal_cell}, start_ok={start_ok}, goal_ok={goal_ok}, occupied_near_start={occupied_near_start}, occupied_near_goal={occupied_near_goal}")
                
                # Path planning failed - clear path to use direct navigation
                global_path_world = []
                # Try to expand the grid or use a larger search if goal is far
                dist_to_goal = math.hypot(robot_x - GOAL_WORLD[0], robot_y - GOAL_WORLD[1])
                if dist_to_goal > 2.0:  # Goal is far, might be outside mapped area
                    # Use direct navigation - DWA will handle obstacles
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
                    # Don't start at the first waypoint if we're already past it
                    if current_wp_index > 0 and min_dist > 0.6:
                        # We might have overshot - find the next waypoint ahead
                        for idx in range(current_wp_index, len(global_path_world)):
                            wp_x, wp_y = global_path_world[idx]
                            dist = math.hypot(robot_x - wp_x, robot_y - wp_y)
                            if dist < min_dist:
                                min_dist = dist
                                best_idx = idx
                        current_wp_index = best_idx
            last_replan_step = step_count
    
    # Get obstacles early for use in local goal calculation and DWA
    obstacles = get_obstacles_from_lidar(ranges, robot_x, robot_y, robot_theta)
    
    # Replan if path is empty or we've reached the end
    if len(global_path_world) == 0:
        # Direct navigation to goal - but check if direct path is blocked
        dx = GOAL_WORLD[0] - robot_x
        dy = GOAL_WORLD[1] - robot_y
        goal_dir = math.atan2(dy, dx)
        dist_to_goal = math.hypot(dx, dy)
        
        # Check if direct path to goal is blocked by obstacles
        direct_path_blocked = False
        if obstacles:
            # Sample points along direct path to goal
            num_samples = 15
            for i in range(1, num_samples + 1):
                t = i / num_samples
                sample_x = robot_x + t * dx
                sample_y = robot_y + t * dy
                
                for (ox, oy) in obstacles:
                    dist = math.hypot(sample_x - ox, sample_y - oy)
                    if dist < dwa_params["robot_radius"] + 0.3:
                        direct_path_blocked = True
                        break
                if direct_path_blocked:
                    break
        
        if direct_path_blocked:
            # Direct path is blocked - find a way around
            # Use a shorter lookahead and let DWA find the way
            LOOKAHEAD_MIN = 0.5
            LOOKAHEAD_MAX = 1.0
            look_ahead = min(LOOKAHEAD_MAX, max(LOOKAHEAD_MIN, min(dist_to_goal * 0.3, 1.0)))
            # Set local goal in general direction of goal, but closer
            local_goal = [
                robot_x + look_ahead * math.cos(goal_dir),
                robot_y + look_ahead * math.sin(goal_dir)
            ]
        else:
            # Direct path is clear - use normal lookahead
            LOOKAHEAD_MIN = 0.8
            LOOKAHEAD_MAX = 1.5
            look_ahead = min(LOOKAHEAD_MAX, max(LOOKAHEAD_MIN, min(dist_to_goal * 0.5, 1.5)))
            local_goal = [
                robot_x + look_ahead * math.cos(goal_dir),
                robot_y + look_ahead * math.sin(goal_dir)
            ]
    else:
        # Select waypoint and compute look-ahead
        if current_wp_index >= len(global_path_world):
            current_wp_index = len(global_path_world) - 1
        
        # Advance waypoint if we're close to current one or have passed it
        # More aggressive waypoint advancement to prevent getting stuck
        while current_wp_index < len(global_path_world) - 1:
            wp_x, wp_y = global_path_world[current_wp_index]
            dx = wp_x - robot_x
            dy = wp_y - robot_y
            dist_wp = math.hypot(dx, dy)
            
            # Check if we've passed the waypoint (projection test)
            if current_wp_index < len(global_path_world) - 1:
                next_wp_x, next_wp_y = global_path_world[current_wp_index + 1]
                # Vector from current waypoint to next
                dx_next = next_wp_x - wp_x
                dy_next = next_wp_y - wp_y
                # Vector from current waypoint to robot
                dx_robot = robot_x - wp_x
                dy_robot = robot_y - wp_y
                # If robot is closer to next waypoint or has passed current one
                dist_to_next = math.hypot(robot_x - next_wp_x, robot_y - next_wp_y)
                # More lenient threshold - advance if close OR if we've passed it
                if dist_wp < 0.6 or (dist_to_next < dist_wp * 1.2 and dx_robot * dx_next + dy_robot * dy_next > -0.1):
                    current_wp_index += 1
                else:
                    break
            else:
                if dist_wp < 0.6:
                    current_wp_index += 1
                else:
                    break
        
        # Get current target waypoint
        wp_x, wp_y = global_path_world[current_wp_index]
        dx = wp_x - robot_x
        dy = wp_y - robot_y
        dist_wp = math.hypot(dx, dy)

        goal_dir = math.atan2(dy, dx)
        LOOKAHEAD_MIN = 0.8  # Increased for better navigation
        LOOKAHEAD_MAX = 1.5  # Increased for longer lookahead
        look_ahead = min(LOOKAHEAD_MAX, max(LOOKAHEAD_MIN, min(dist_wp * 0.7, 1.5)))
        local_goal = [
            robot_x + look_ahead * math.cos(goal_dir),
            robot_y + look_ahead * math.sin(goal_dir)
        ]

    # Check if local goal or path to it is blocked by obstacles - adjust if needed
    # This prevents DWA from trying to reach an unreachable goal
    local_goal_blocked = False
    if obstacles:
        # Check if local goal itself is too close to obstacles
        min_dist_to_goal = float('inf')
        blocking_obstacle = None
        for (ox, oy) in obstacles:
            dist_to_goal = math.hypot(local_goal[0] - ox, local_goal[1] - oy)
            if dist_to_goal < min_dist_to_goal:
                min_dist_to_goal = dist_to_goal
                blocking_obstacle = (ox, oy)
        
        # Check if path to local goal is blocked
        path_blocked = False
        dx_lg = local_goal[0] - robot_x
        dy_lg = local_goal[1] - robot_y
        lg_dist = math.hypot(dx_lg, dy_lg)
        
        if lg_dist > 0:
            # Sample points along path to local goal
            num_samples = 10
            for i in range(1, num_samples + 1):
                t = i / num_samples
                sample_x = robot_x + t * dx_lg
                sample_y = robot_y + t * dy_lg
                
                for (ox, oy) in obstacles:
                    dist = math.hypot(sample_x - ox, sample_y - oy)
                    if dist < dwa_params["robot_radius"] + 0.2:
                        path_blocked = True
                        blocking_obstacle = (ox, oy)
                        break
                if path_blocked:
                    break
        
        # Adjust local goal if blocked - but be VERY conservative to prevent oscillation
        # Only adjust if really necessary and use a stable adjustment
        # Add hysteresis to prevent rapid switching
        should_adjust = False
        if (min_dist_to_goal < dwa_params["robot_radius"] + 0.12 or path_blocked) and blocking_obstacle:
            # Only adjust if significantly blocked (hysteresis) - prevent rapid switching
            if last_local_goal_adjustment is None or step_count - last_local_goal_adjustment > 10:
                should_adjust = True
                last_local_goal_adjustment = step_count
        
        if should_adjust:
            local_goal_blocked = True
            # Find direction to global goal
            dx_global = GOAL_WORLD[0] - robot_x
            dy_global = GOAL_WORLD[1] - robot_y
            global_dir = math.atan2(dy_global, dx_global)
            
            # Find direction to obstacle
            dx_obs = blocking_obstacle[0] - robot_x
            dy_obs = blocking_obstacle[1] - robot_y
            obs_dir = math.atan2(dy_obs, dx_obs)
            obs_dist = math.hypot(dx_obs, dy_obs)
            
            # Calculate angle from robot to obstacle relative to global goal
            angle_diff = obs_dir - global_dir
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            # If obstacle is in the way, adjust local goal to go around it
            # Choose direction that's closer to global goal
            if abs(angle_diff) < math.pi / 2:  # Obstacle is in front
                # Go around obstacle - choose side closer to global goal
                perp_angle1 = obs_dir + math.pi / 2
                perp_angle2 = obs_dir - math.pi / 2
                
                # Check which perpendicular is closer to global goal
                diff1 = abs(perp_angle1 - global_dir)
                if diff1 > math.pi:
                    diff1 = 2 * math.pi - diff1
                diff2 = abs(perp_angle2 - global_dir)
                if diff2 > math.pi:
                    diff2 = 2 * math.pi - diff2
                
                if diff1 < diff2:
                    avoid_angle = perp_angle1
                else:
                    avoid_angle = perp_angle2
                
                # Set local goal in avoid direction, but still toward global goal
                # Use distance that avoids obstacle but makes progress
                # Use a more conservative distance to prevent oscillation
                avoid_dist = min(obs_dist * 0.6, 0.8)  # Shorter distance, more stable
                local_goal[0] = robot_x + avoid_dist * math.cos(avoid_angle)
                local_goal[1] = robot_y + avoid_dist * math.sin(avoid_angle)
                
                # Verify the adjusted goal is actually better
                # If it's still too close to obstacles, don't use it
                new_min_dist = float('inf')
                for (ox, oy) in obstacles:
                    dist = math.hypot(local_goal[0] - ox, local_goal[1] - oy)
                    if dist < new_min_dist:
                        new_min_dist = dist
                if new_min_dist < dwa_params["robot_radius"] + 0.1:
                    # Adjusted goal is still too close - use original with shorter lookahead
                    local_goal[0] = robot_x + 0.5 * math.cos(global_dir)
                    local_goal[1] = robot_y + 0.5 * math.sin(global_dir)
            else:
                # Obstacle is to the side - just push goal away from it
                dx = local_goal[0] - blocking_obstacle[0]
                dy = local_goal[1] - blocking_obstacle[1]
                dist = math.hypot(dx, dy)
                if dist > 0:
                    push_dist = dwa_params["robot_radius"] + 0.3
                    local_goal[0] = blocking_obstacle[0] + (dx / dist) * push_dist
                    local_goal[1] = blocking_obstacle[1] + (dy / dist) * push_dist
    
    # DWA - Let it handle all obstacle avoidance
    # DWA computes dynamic window, evaluates trajectories, and selects optimal velocity
    state = [robot_x, robot_y, robot_theta]
    v, w = dwa.compute_velocity(state, local_goal, obstacles)
    
    # Debug: Print DWA output periodically (every 50 steps) to verify it's working
    if step_count % 50 == 0:
        min_obs_dist = float('inf')
        if obstacles:
            for (ox, oy) in obstacles:
                dist = math.hypot(robot_x - ox, robot_y - oy)
                if dist < min_obs_dist:
                    min_obs_dist = dist
        
        # Calculate distance to global goal
        dist_to_goal = math.hypot(robot_x - GOAL_WORLD[0], robot_y - GOAL_WORLD[1])
        
        # Check if DWA returned zeros (shouldn't happen with fallback)
        if abs(v) < 0.01 and abs(w) < 0.01:
            print(f"Step {step_count}: WARNING - DWA returned zeros! robot=({robot_x:.2f}, {robot_y:.2f}), goal_dist={dist_to_goal:.2f}, v={v:.3f}, w={w:.3f}, obstacles={len(obstacles)}, min_dist={min_obs_dist:.3f}")
        else:
            print(f"Step {step_count}: robot=({robot_x:.2f}, {robot_y:.2f}), goal_dist={dist_to_goal:.2f}, DWA v={v:.3f}, w={w:.3f}, obstacles={len(obstacles)}, min_dist={min_obs_dist:.3f}, local_goal=({local_goal[0]:.2f}, {local_goal[1]:.2f}), blocked={local_goal_blocked}, path_len={len(global_path_world)}")
    
    # Prevent oscillation: if robot is stuck and oscillating, force a recovery
    # Check if robot is oscillating (alternating angular velocities)
    if step_count > 100 and abs(w) > 3.0 and stuck_counter > 10:
        # Robot is oscillating - force a recovery direction
        dx_goal = GOAL_WORLD[0] - robot_x
        dy_goal = GOAL_WORLD[1] - robot_y
        goal_angle = math.atan2(dy_goal, dx_goal)
        angle_diff = goal_angle - robot_theta
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # Force rotation toward goal, reduce forward velocity
        w = 1.5 * angle_diff
        v = max(0.0, v * 0.5)  # Reduce forward velocity when oscillating
    
    # Minimal emergency override: only for truly critical situations where DWA fails
    # This should rarely trigger if DWA is working correctly
    CRITICAL_DISTANCE = dwa_params["robot_radius"] + 0.05  # Extremely close - immediate collision risk
    min_obstacle_dist = float('inf')
    
    if obstacles:
        for (ox, oy) in obstacles:
            dist = math.hypot(robot_x - ox, robot_y - oy)
            if dist < min_obstacle_dist:
                min_obstacle_dist = dist
    
    # Only override if we're about to collide AND DWA returned zero velocity AND we're stuck
    # This is a last resort safety mechanism
    if min_obstacle_dist < CRITICAL_DISTANCE and abs(v) < 0.01 and stuck_counter > 20:
        # Emergency: rotate away from closest obstacle
        if obstacles:
            closest_obs = None
            for (ox, oy) in obstacles:
                dist = math.hypot(robot_x - ox, robot_y - oy)
                if dist == min_obstacle_dist:
                    closest_obs = (ox, oy)
                    break
            
            if closest_obs:
                dx_obs = robot_x - closest_obs[0]
                dy_obs = robot_y - closest_obs[1]
                escape_angle = math.atan2(dy_obs, dx_obs)
                escape_error = escape_angle - robot_theta
                while escape_error > math.pi:
                    escape_error -= 2 * math.pi
                while escape_error < -math.pi:
                    escape_error += 2 * math.pi
                w = 2.0 * escape_error
                v = -0.05  # Small backward motion
    # Otherwise, trust DWA completely - it handles obstacle avoidance
    
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