from controller import Robot
import math

robot = Robot()
timestep = int(robot.getBasicTimeStep())

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

theta = 0.0

# 제자리 회전 시도 (왼바퀴 뒤로, 오른바퀴 앞으로)
left_wheel.setVelocity(-2.0)
right_wheel.setVelocity(2.0)

print("Rotating in place...")

target_rad = 2 * math.pi  # 360 degrees
accum = 0.0

while robot.step(timestep) != -1:
    l = l_enc.getValue()
    r = r_enc.getValue()

    dL = (l - prev_l) * WHEEL_RADIUS
    dR = (r - prev_r) * WHEEL_RADIUS

    prev_l = l
    prev_r = r

    d_theta = (dR - dL) / WHEEL_BASE
    accum += d_theta

    deg = math.degrees(accum % (2 * math.pi))
    print(f"Theta: {deg:.1f}°")

    if abs(accum) >= target_rad:
        left_wheel.setVelocity(0)
        right_wheel.setVelocity(0)
        print("Stopped. Check Webots 'Rotation -> yaw'")
        print("Correction Factor = TrueYaw / OdomYaw")
        print("New WHEEL_BASE = OldBase * CorrectionFactor")
        break
