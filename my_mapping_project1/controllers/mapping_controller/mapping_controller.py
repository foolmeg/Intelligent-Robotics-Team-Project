from controller import Robot
import math
import sys
import os

# Add parent directory to path to import d_start_lite
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from d_start_lite import DStarLite

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
MAP_RES = 0.02  # 2cm/pixel

# Initialize Occupancy Grid
occupancy = [[0 for _ in range(MAP_W)] for _ in range(MAP_H)]

# Set of obstacles detected in the current frame (to handle drift by clearing old ones)
current_obstacles = set()

# Robot pose in odometry frame
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

WHEEL_RADIUS = 0.0975     
WHEEL_BASE   = 0.33        # Reduced to increase rotation sensitivity (correcting under-rotation)

left_encoder = left_wheel.getPositionSensor()
right_encoder = right_wheel.getPositionSensor()
left_encoder.enable(time_step)
right_encoder.enable(time_step)

prev_left_pos = left_encoder.getValue() or 0.0
prev_right_pos = right_encoder.getValue() or 0.0

# Sensors (kept for safety/fallback if needed, but main logic is D*)
sensors = []
for i in range(MAX_SENSOR_NUMBER):
    s = robot.getDevice(f"so{i}")
    s.enable(time_step)
    sensors.append(s)

# ======================
# D* Lite Setup
# ======================
# Goal: Navigate to a point on the map
# Box is at (-1.49, -0.353). Setting goal to (1.0, -1.0) which is clear space (below center)
# Goal Node (Grid Coordinates)
# world_to_grid: gx = map_cx + wx/MAP_RES, gy = map_cy - wy/MAP_RES
# For goal at (1.0, -1.0): 
#   gx = map_cx + 1.0/0.02 = map_cx + 50
#   gy = map_cy - (-1.0)/0.02 = map_cy + 50
# World (1.0, -1.0) -> Grid: (map_cx+50, map_cy+50) ✓
GOAL_NODE = (map_cx + 50, map_cy + 50)  # World: (1.0, -1.0) - right and DOWN from center

# Ensure goal is within bounds
GOAL_NODE = (
    max(10, min(MAP_W - 10, GOAL_NODE[0])),
    max(10, min(MAP_H - 10, GOAL_NODE[1]))
)

# Start node (center of map initially)
START_NODE = (map_cx, map_cy)

dstar = DStarLite(MAP_W, MAP_H, START_NODE, GOAL_NODE)

print(f"D* Lite Initialized. Start: {START_NODE}, Goal: {GOAL_NODE}")

# Immediate test: Can D* find a path on empty map?
print("Testing D* Lite on empty map...")
dstar.replan(START_NODE)
test_path = dstar.get_shortest_path(START_NODE)
if test_path:
    print(f"SUCCESS! Initial path length: {len(test_path)}")
else:
    print("FAILED! D* Lite cannot find path even on empty map!")
    print(f"Check: START={START_NODE}, GOAL={GOAL_NODE}")
    print(f"Map size: {MAP_W}x{MAP_H}")



def safe(x, fallback=0.0):
    if x is None or isinstance(x, float) and math.isnan(x):
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


def world_to_grid(wx, wy):
    """Convert world coordinates to grid coordinates.
    Webots: y increases upward, but we want grid y to increase downward for correct display.
    So we invert y: positive world y -> smaller grid y (which displays as higher on screen after flip).
    """
    gx = int(map_cx + wx / MAP_RES)
    gy = int(map_cy - wy / MAP_RES)  # Invert y: world +y -> grid -y (displays as +y after flip)
    return gx, gy

def grid_to_world(gx, gy):
    """Convert grid coordinates to world coordinates.
    Inverse of world_to_grid: invert y back.
    """
    wx = (gx - map_cx) * MAP_RES
    wy = -(gy - map_cy) * MAP_RES  # Invert y back: grid y -> world y
    return wx, wy

def display_y(gy):
    """Convert grid y coordinate to display y coordinate (flip for display)."""
    return MAP_H - 1 - gy  # Display y increases downward, so flip


# Main Loop
# Initialize global variables
debug_counter = 0

while robot.step(time_step) != -1:
    # 1. Odometry Update
    left_pos = safe(left_encoder.getValue(), prev_left_pos)
    right_pos = safe(right_encoder.getValue(), prev_right_pos)

    # Encoder values: check if direction needs to be inverted
    # If robot moves forward but position goes backward, invert encoder signs
    dleft = (left_pos - prev_left_pos) * WHEEL_RADIUS
    dright = (right_pos - prev_right_pos) * WHEEL_RADIUS
    
    # If robot moves in opposite direction, try inverting encoder signs
    # Uncomment the next two lines if robot moves backward when it should move forward:
    # dleft = -dleft
    # dright = -dright

    # Filter out noise: ignore very small encoder changes (likely noise)
    # Threshold: 0.1mm movement
    ENCODER_NOISE_THRESHOLD = 0.0001  # 0.1mm
    if abs(dleft) < ENCODER_NOISE_THRESHOLD:
        dleft = 0.0
    if abs(dright) < ENCODER_NOISE_THRESHOLD:
        dright = 0.0

    prev_left_pos = left_pos
    prev_right_pos = right_pos

    d_center = (dleft + dright) / 2.0
    d_theta = (dright - dleft) / WHEEL_BASE
    
    # Filter out very small angle changes (likely noise)
    ANGLE_NOISE_THRESHOLD = 0.0001  # ~0.006 degrees
    if abs(d_theta) < ANGLE_NOISE_THRESHOLD:
        d_theta = 0.0
    
    # If axis is wrong, try inverting theta sign
    # Standard: positive theta = counterclockwise (left turn)
    # If robot rotates left but theta decreases, invert the sign
    # d_theta = -(dright - dleft) / WHEEL_BASE  # Uncomment if axis is inverted

    # Always update position (even if movement is small, for debugging)
    # Runge-Kutta 2nd order integration
    robot_theta += d_theta / 2.0
    
    # If robot moves in opposite direction, try inverting y-axis movement
    # Webots coordinate system: X forward, Y left, Z up
    # If robot should go up but goes down, invert y movement
    robot_x += d_center * math.cos(robot_theta)
    robot_y -= d_center * math.sin(robot_theta)  # Inverted y: robot moves up when it should

    robot_theta = (robot_theta + math.pi) % (2 * math.pi) - math.pi

    # Current Grid Position
    curr_gx, curr_gy = world_to_grid(robot_x, robot_y)
    
    # Ensure robot is within map bounds for D*
    curr_gx = max(0, min(MAP_W - 1, curr_gx))
    curr_gy = max(0, min(MAP_H - 1, curr_gy))
    
    current_node = (curr_gx, curr_gy)
    
    # DEBUG: Draw robot position and orientation on map (BLUE circle + direction line)
    # Flip y coordinate for display (display y increases downward)
    display_gy = display_y(curr_gy)
    display.setColor(0x0000FF)  # Blue for robot
    display.fillOval(curr_gx - 3, display_gy - 3, 6, 6)  # Draw robot as circle
    # Draw orientation arrow
    arrow_len = 10
    arrow_x = int(curr_gx + arrow_len * math.cos(robot_theta))
    arrow_gy = int(curr_gy + arrow_len * math.sin(robot_theta))  # No inversion in grid space
    display_arrow_gy = display_y(arrow_gy)  # Flip for display
    display.drawLine(curr_gx, display_gy, arrow_x, display_arrow_gy)
    
    # DEBUG: Draw goal position (GREEN circle)
    display.setColor(0x00FF00)  # Green for goal
    display_goal_gy = display_y(GOAL_NODE[1])
    display.fillOval(GOAL_NODE[0] - 3, display_goal_gy - 3, 6, 6)
    
    # Print position every 50 loops
    debug_counter += 1
    if debug_counter % 50 == 0:
        goal_world = grid_to_world(GOAL_NODE[0], GOAL_NODE[1])
        dist_to_goal = math.sqrt((robot_x - goal_world[0])**2 + (robot_y - goal_world[1])**2)
        print(f"Robot Pos: ({robot_x:.2f}, {robot_y:.2f}, θ={math.degrees(robot_theta):.1f}°) Grid: {current_node}")
        print(f"Goal World: ({goal_world[0]:.2f}, {goal_world[1]:.2f}) Grid: {GOAL_NODE}")
        print(f"Distance to goal: {dist_to_goal:.2f}m")
        print(f"Encoder: L={left_pos:.4f}, R={right_pos:.4f}, dL={dleft*1000:.2f}mm, dR={dright*1000:.2f}mm, d_center={d_center*1000:.2f}mm")

    # 2. LiDAR & Map Update (OPTIMIZED - DIFF STRATEGY)
    ranges = lidar.getRangeImage()
    
    # Set of obstacles detected in THIS frame
    detected_obstacles = set()
    
    # OPTIMIZATION: Process every 8th ray (balance between detection and performance)
    RAY_STEP = 8  # Increased back to 8 to reduce map instability
    INFLATION_RADIUS = 12  # Reduced to 12 pixels (24cm) - more reasonable
    
    # Pre-compute inflation offsets
    if 'inflation_offsets' not in globals():
        global inflation_offsets
        inflation_offsets = []
        for dy in range(-INFLATION_RADIUS, INFLATION_RADIUS + 1):
            for dx in range(-INFLATION_RADIUS, INFLATION_RADIUS + 1):
                if dx*dx + dy*dy <= INFLATION_RADIUS*INFLATION_RADIUS:
                    inflation_offsets.append((dx, dy))
    
    # Safety radii - increased for better collision avoidance
    ROBOT_SAFE_RADIUS_SQ = 400  # 20^2 = 400 (20cm safe zone, increased for safety)
    GOAL_SAFE_RADIUS_SQ = 625   # 25^2 = 625 (25cm safe zone - increased to prevent obstacles near goal)
    
    # CRITICAL FIX: Force clear robot's current position area
    # Standardized to 10 (20cm)
    CLEAR_RADIUS = 10
    
    # Process LiDAR
    for i in range(0, lidar_res, RAY_STEP):
        r = safe(ranges[i])
        # Filter out noise:
        # 1. Too close (< 0.12m): Likely robot self-body or sensor noise
        # 2. Too far (> 4.5m): Out of reliable range
        if r < 0.12 or r > 4.5:
            continue

        # LiDAR angle calculation
        # Webots getRangeImage() typically has index 0 at left side (-fov/2)
        # and increases to right side (+fov/2)
        # Try inverting if axis is wrong
        angle = -lidar_fov / 2.0 + (i * lidar_fov / lidar_res)
        
        # Convert LiDAR local coordinates to robot frame
        # Standard: angle 0 = front (+X), positive = left (+Y, counterclockwise)
        lx = r * math.cos(angle)
        ly = r * math.sin(angle)

        # Transform from robot frame to world frame
        # Standard rotation matrix for counterclockwise rotation:
        # x' = x*cos(θ) - y*sin(θ)
        # y' = x*sin(θ) + y*cos(θ)
        # Webots: X forward, Y left, Z up (right-handed)
        # robot_theta: 0 = +X (forward), positive = counterclockwise (left turn)
        wx = robot_x + (lx * math.cos(robot_theta) - ly * math.sin(robot_theta))
        wy = robot_y + (lx * math.sin(robot_theta) + ly * math.cos(robot_theta))
        
        # If LiDAR coordinates are still wrong, check:
        # 1. LiDAR angle definition (might need to invert angle)
        # 2. Webots Y-axis direction (might need to invert wy)

        gx, gy = world_to_grid(wx, wy)

        if gx < 0 or gy < 0 or gx >= MAP_W or gy >= MAP_H:
            continue

        # Add inflated obstacles to detected_obstacles set
        for dx, dy in inflation_offsets:
            ix, iy = gx + dx, gy + dy
            if 0 <= ix < MAP_W and 0 <= iy < MAP_H:
                # Safety Check: Filter out robot and goal areas
                dist_to_robot = (ix - curr_gx)**2 + (iy - curr_gy)**2
                dist_to_goal = (ix - GOAL_NODE[0])**2 + (iy - GOAL_NODE[1])**2
                
                if dist_to_robot > ROBOT_SAFE_RADIUS_SQ and dist_to_goal > GOAL_SAFE_RADIUS_SQ:
                    detected_obstacles.add((ix, iy))
                    
    # --- DIFFING & UPDATE ---
    # Calculate what changed
    to_remove = current_obstacles - detected_obstacles
    to_add = detected_obstacles - current_obstacles
    
    # Stability check: ignore updates if too many changes (likely noise or drift)
    # Also reduce updates when robot is only rotating (not translating)
    MAX_MAP_CHANGES = 100  # Reduced from 200 - more strict
    is_rotating_only = abs(d_center) < 0.001 and abs(d_theta) > 0.001
    
    # If robot is only rotating, be more conservative with map updates
    if is_rotating_only:
        MAX_MAP_CHANGES = 50  # Even stricter when rotating only
    
    if len(to_add) + len(to_remove) > MAX_MAP_CHANGES:
        if debug_counter % 50 == 0:
            print(f"⚠️ Too many map changes ({len(to_add) + len(to_remove)}), ignoring (likely noise/drift)")
        # Don't update map, keep current state
        map_changed = False
    else:
        map_changed = False
        
        # 1. Remove disappeared obstacles (only if reasonable number)
        if to_remove and len(to_remove) < MAX_MAP_CHANGES:
            for ox, oy in to_remove:
                occupancy[oy][ox] = 0
                dstar.clear_obstacle(ox, oy)
                display.setColor(0x000000)
                display.drawPixel(ox, display_y(oy))  # Flip y for display
            map_changed = True
            
        # 2. Add new obstacles (only if reasonable number)
        if to_add and len(to_add) < MAX_MAP_CHANGES:
            for ox, oy in to_add:
                occupancy[oy][ox] = 1
                dstar.set_obstacle(ox, oy)
                display.setColor(0xFFFFFF)
                display.drawPixel(ox, display_y(oy))  # Flip y for display
            map_changed = True
            
        # Only print if significant changes
        if map_changed and (len(to_add) > 50 or len(to_remove) > 50):
            print(f"Map Update: +{len(to_add)} / -{len(to_remove)} cells")

    # Update current state
    current_obstacles = detected_obstacles

    # 3. Path Planning (OPTIMIZED)
    # Replan frequently to avoid following stale paths
    
    # Track previous position for movement detection
    if 'prev_node' not in globals():
        global prev_node, replan_counter
        prev_node = current_node
        replan_counter = 0
    
    node_moved = (current_node != prev_node)
    replan_counter += 1
    
    # Replan if: map changed, robot moved, OR every 20 loops (to avoid stale paths)
    if map_changed or node_moved or replan_counter >= 20:
        dstar.replan(current_node)
        prev_node = current_node
        replan_counter = 0
    
    path = dstar.get_shortest_path(current_node)
    
    # Debug output
    if path is None or len(path) == 0:
        print(f"No path! current_node={current_node}, GOAL={GOAL_NODE}")
        # STOP robot if no path
        left_wheel.setVelocity(0)
        right_wheel.setVelocity(0)
        continue

    # 4. Path Following Control
    
    # --- GOAL CHECK FIRST ---
    dist_to_goal = math.sqrt((robot_x - grid_to_world(GOAL_NODE[0], GOAL_NODE[1])[0])**2 + 
                             (robot_y - grid_to_world(GOAL_NODE[0], GOAL_NODE[1])[1])**2)
    
    if dist_to_goal < 0.1: # 10cm tolerance
        print("Goal Reached!")
        left_wheel.setVelocity(0)
        right_wheel.setVelocity(0)
        continue
    # ------------------------
    
    # --- EMERGENCY COLLISION AVOIDANCE ---
    # Check for obstacles in front (wider FOV for better detection)
    front_min_dist = float('inf')
    left_min_dist = float('inf')
    right_min_dist = float('inf')
    
    # Check front rays (approx -60 to +60 degrees) - wider FOV
    fov_range = math.pi / 3  # 60 degrees
    center_start = int(lidar_res * (0.5 - fov_range/lidar_fov))
    center_end = int(lidar_res * (0.5 + fov_range/lidar_fov))
    
    # Check every ray for better detection
    for i in range(center_start, center_end, 2):
        r = safe(ranges[i])
        if r < 0.1 or r > 4.5:  # Ignore self-noise and too far
            continue
            
        # Determine if left or right side
        angle = -lidar_fov / 2.0 + (i * lidar_fov / lidar_res)
        if angle < -0.1:  # Left side (with margin)
            left_min_dist = min(left_min_dist, r)
        elif angle > 0.1:  # Right side (with margin)
            right_min_dist = min(right_min_dist, r)
            
        # Front center (within 40 degrees)
        if abs(angle) < math.pi * 2 / 9:  # Within ~40 degrees
            front_min_dist = min(front_min_dist, r)
            
    # If obstacle is TOO CLOSE (< 30cm), FORCE STOP & TURN
    # Balanced threshold - not too conservative, not too aggressive
    is_blocked = False
    emergency_turn_dir = 0  # 0 = no turn, 1 = left, -1 = right
    if front_min_dist < 0.30:  # 30cm - balanced threshold
        is_blocked = True
        # Choose turn direction: turn away from closer obstacle
        # But prefer direction with more clearance
        if left_min_dist > 0.5 and right_min_dist < 0.5:
            emergency_turn_dir = 1   # Turn left (left side is clear)
        elif right_min_dist > 0.5 and left_min_dist < 0.5:
            emergency_turn_dir = -1  # Turn right (right side is clear)
        elif left_min_dist < right_min_dist:
            emergency_turn_dir = -1  # Turn right (away from left obstacle)
        else:
            emergency_turn_dir = 1   # Turn left (away from right obstacle)
        if debug_counter % 10 == 0:  # Print more frequently when blocked
            print(f"🚨 BLOCKED! Front: {front_min_dist:.2f}m, L: {left_min_dist:.2f}m, R: {right_min_dist:.2f}m, turning {'left' if emergency_turn_dir > 0 else 'right'}")
    # -------------------------------------

    speed_left = 0.0
    speed_right = 0.0

    if path and len(path) > 1:
        # Lookahead: Use 2 nodes (4cm) for more cautious path following
        lookahead_dist = 2  
        target_idx = min(len(path) - 1, lookahead_dist)
        next_node = path[target_idx]
        
        target_wx, target_wy = grid_to_world(next_node[0], next_node[1])
        
        # Calculate angle to target
        dx = target_wx - robot_x
        dy = target_wy - robot_y
        target_angle = math.atan2(dy, dx)
        
        angle_diff = target_angle - robot_theta
        # Normalize angle diff to [-pi, pi]
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        
        # Check if path is safe: verify next few nodes aren't obstacles
        # More lenient check - only check immediate next node, not neighbors
        path_safe = True
        if len(path) > 1:
            next_check_node = path[1]  # Only check immediate next node
            if (0 <= next_check_node[0] < MAP_W and 0 <= next_check_node[1] < MAP_H):
                if occupancy[next_check_node[1]][next_check_node[0]] == 1:
                    path_safe = False
        
        # IMPROVED: Point Turn Logic (More precise for tight spaces)
        # 1. If angle error is large, turn in place (no forward motion)
        # 2. If angle is aligned, move forward
        
        if is_blocked:
            # FORCE TURN to escape deadlock!
            # Ignore path, rotate away from obstacle with more aggressive turning
            base_speed = 0.0
            # More aggressive turning when very close
            if front_min_dist < 0.20:  # Very close - turn faster
                turn_speed = MAX_SPEED * 0.8 * emergency_turn_dir
            elif front_min_dist < 0.30:  # Close - turn moderately
                turn_speed = MAX_SPEED * 0.7 * emergency_turn_dir
            else:
                turn_speed = MAX_SPEED * 0.6 * emergency_turn_dir
            
            # After turning for a bit, try to move forward slightly if side is clear
            # This helps escape from corners
            if emergency_turn_dir == 1 and left_min_dist > 0.6:  # Left turn and left side clear
                base_speed = MAX_SPEED * 0.1  # Small forward motion while turning
            elif emergency_turn_dir == -1 and right_min_dist > 0.6:  # Right turn and right side clear
                base_speed = MAX_SPEED * 0.1  # Small forward motion while turning
        elif not path_safe:
            # Path is not safe, slow down and be more cautious
            base_speed = MAX_SPEED * 0.2  # Very slow
            k_p = 2.0  # Stronger correction
            turn_speed = k_p * angle_diff
            if debug_counter % 50 == 0:
                print(f"⚠️ Path unsafe! Slowing down...")
        elif abs(angle_diff) > 0.35:  # If error > 20 degrees, turn in place
            base_speed = 0.0
            k_p = 3.0  # Stronger turn gain for point turn
            turn_speed = k_p * angle_diff
        elif abs(angle_diff) > 0.15:  # If error > 8.6 degrees, slow forward with correction
            base_speed = MAX_SPEED * 0.3  # Moderate forward while correcting
            k_p = 2.0  # Moderate correction
            turn_speed = k_p * angle_diff
            if debug_counter % 50 == 0:
                print(f"🔄 Angle error: {math.degrees(angle_diff):.1f}°, correcting while moving...")
        else:
            # Aligned! Move forward
            # Reduce speed if obstacle is close (even if not blocked)
            if front_min_dist < 0.60:  # If obstacle within 60cm, slow down
                base_speed = MAX_SPEED * 0.4 * (front_min_dist / 0.60)  # Scale speed by distance
            else:
                base_speed = MAX_SPEED * 0.5  # Normal speed when clear
            k_p = 1.0  # Gentle correction while moving
            turn_speed = k_p * angle_diff
            if debug_counter % 50 == 0:
                print(f"✅ Moving forward: base={base_speed:.2f}, turn={turn_speed:.2f}, angle_err={math.degrees(angle_diff):.1f}°, front_dist={front_min_dist:.2f}m")
            
        # Slow down near goal
        distance_to_target = math.sqrt(dx*dx + dy*dy) # This was moved from above
        if distance_to_target < 0.3:
            base_speed = min(base_speed, MAX_SPEED * 0.2)
            
        # Limit turn speed
        max_turn = MAX_SPEED * 0.5
        turn_speed = max(-max_turn, min(max_turn, turn_speed))
        
        speed_left = base_speed - turn_speed
        speed_right = base_speed + turn_speed
        
        # Clamp speeds
        speed_left = max(-MAX_SPEED, min(MAX_SPEED, speed_left))
        speed_right = max(-MAX_SPEED, min(MAX_SPEED, speed_right))
            
    else:
        # No path found or already at goal (len=1)
        if current_node == GOAL_NODE:
             print("Goal Reached!")
        else:
             print("No path found!")
        speed_left = 0.0
        speed_right = 0.0

    # Apply calculated velocities
    left_wheel.setVelocity(speed_left)
    right_wheel.setVelocity(speed_right)
