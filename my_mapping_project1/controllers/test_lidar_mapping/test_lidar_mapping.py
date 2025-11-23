from controller import Robot
import math

robot = Robot()
timestep = int(robot.getBasicTimeStep())

lidar = robot.getDevice("RPlidar A2")
lidar.enable(timestep)
res = lidar.getHorizontalResolution()
fov = lidar.getFov()

display = robot.getDevice("display")
W = display.getWidth()
H = display.getHeight()
RES = 0.02   # 2cm/pixel

cx = W // 2
cy = H // 2

x = 0.0
y = 0.0
theta = 0.0

# Wheels & Encoders
left_wheel = robot.getDevice("left wheel")
right_wheel = robot.getDevice("right wheel")
left_wheel.setPosition(float('inf'))
right_wheel.setPosition(float('inf'))

WHEEL_RADIUS = 0.0975
WHEEL_BASE   = 0.35

l_enc = left_wheel.getPositionSensor()
r_enc = right_wheel.getPositionSensor()
l_enc.enable(timestep)
r_enc.enable(timestep)

prev_l = l_enc.getValue()
prev_r = r_enc.getValue()

# Rotate in place
left_wheel.setVelocity(-1.0)
right_wheel.setVelocity(1.0)

def bres(x0,y0,x1,y1):
    pts=[]
    dx=abs(x1-x0); dy=abs(y1-y0)
    sx=1 if x0<x1 else -1
    sy=1 if y0<y1 else -1
    err=dx-dy
    while True:
        pts.append((x0,y0))
        if x0==x1 and y0==y1: break
        e2=2*err
        if e2>-dy: err-=dy; x0+=sx
        if e2<dx: err+=dx; y0+=sy
    return pts

while robot.step(timestep) != -1:
    # Odometry Update
    l = l_enc.getValue()
    r = r_enc.getValue()

    dL = (l - prev_l) * WHEEL_RADIUS
    dR = (r - prev_r) * WHEEL_RADIUS

    prev_l = l
    prev_r = r

    d_center = (dL + dR) / 2
    d_theta = (dR - dL) / WHEEL_BASE

    theta += d_theta / 2.0
    x += d_center * math.cos(theta)
    y += d_center * math.sin(theta)
    theta += d_theta / 2.0
    
    # Normalize theta
    theta = (theta + math.pi) % (2 * math.pi) - math.pi

    print(f"Theta: {math.degrees(theta):.1f}")

    scan = lidar.getRangeImage()

    # OGM clear pass
    for i in range(res):
        r = scan[i]
        if r < 0.15 or r > 4.5:
            continue

        angle = -fov/2 + i*(fov/res)


        if 0 <= gx < W and 0 <= gy < H:
            display.setColor(0xFFFFFF)
            display.drawPixel(gx,gy)
