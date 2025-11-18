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


left_wheel = robot.getDevice("left wheel")
right_wheel = robot.getDevice("right wheel")
left_wheel.setPosition(float('inf'))
right_wheel.setPosition(float('inf'))

WHEEL_RADIUS = 0.04875     
WHEEL_BASE   = 0.331       

left_encoder = left_wheel.getPositionSensor()
right_encoder = right_wheel.getPositionSensor()
left_encoder.enable(time_step)
right_encoder.enable(time_step)

prev_left_pos = left_encoder.getValue() or 0.0
prev_right_pos = right_encoder.getValue() or 0.0


sensors = []
for i in range(MAX_SENSOR_NUMBER):
    s = robot.getDevice(f"so{i}")
    s.enable(time_step)
    sensors.append(s)

state = FORWARD



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



while robot.step(time_step) != -1:

    
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

    
    ranges = lidar.getRangeImage()

    for i in range(lidar_res):
        r = safe(ranges[i])

        if r < 0.15 or r > 4.5:
            continue

        angle = -lidar_fov / 2.0 + (i * lidar_fov / lidar_res)
        

        lx = r * math.cos(angle)
        ly = r * math.sin(angle)

        wx = robot_x + (lx * math.cos(robot_theta) - ly * math.sin(robot_theta))
        wy = robot_y + (lx * math.sin(robot_theta) + ly * math.cos(robot_theta))

        gx = int(map_cx + wx / MAP_RES)
        gy = int(map_cy - wy / MAP_RES)

        if gx < 0 or gy < 0 or gx >= MAP_W or gy >= MAP_H:
            continue

       
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
