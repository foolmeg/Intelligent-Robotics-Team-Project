from controller import Robot
import math

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Wheels
left_wheel = robot.getDevice("left wheel")
right_wheel = robot.getDevice("right wheel")
left_wheel.setPosition(float('inf'))
right_wheel.setPosition(float('inf'))

WHEEL_RADIUS = 0.0975  # Correct radius for Pioneer 3-DX
WHEEL_BASE   = 0.331

left_enc = left_wheel.getPositionSensor()
right_enc = right_wheel.getPositionSensor()
left_enc.enable(timestep)
right_enc.enable(timestep)

prev_left = left_enc.getValue()
prev_right = right_enc.getValue()

# robot pose (starts at 0,0 relative to starting point)
x = 0.0
z = 0.0 # Webots uses Z for the "side" axis on the ground
theta = 0.0

# Calibration Step 1: Linear Distance
# Drive forward until Odom X reaches 2.0 meters
TARGET_DIST = 2.0

left_wheel.setVelocity(2.0)
right_wheel.setVelocity(2.0) # Drive straight

print("Driving forward 2.0m...")

while robot.step(timestep) != -1:
    l = left_enc.getValue()
    r = right_enc.getValue()

    # Revert inversion: Encoders increase when moving forward
    dL = (l - prev_left) * WHEEL_RADIUS
    dR = (r - prev_right) * WHEEL_RADIUS

    prev_left = l
    prev_right = r

    d_center = (dL + dR) / 2
    d_theta = (dR - dL) / WHEEL_BASE

    theta += d_theta / 2.0
    x += d_center * math.cos(theta)
    z += d_center * math.sin(theta)
    theta += d_theta / 2.0

    print(f"Odom: x={x:.3f}, z={z:.3f}")

    if x >= TARGET_DIST:
        left_wheel.setVelocity(0)
        right_wheel.setVelocity(0)
        print(f"Stopped. Odom X: {x:.4f}m")
        print("Please check Webots 'Translation X' field.")
        print("Real Distance = (End X - Start X)")
        print("Correction Factor = Real Distance / Odom Distance")
        print("New Radius = Current Radius * Correction Factor")
        break
