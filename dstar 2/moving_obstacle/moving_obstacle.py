from controller import Robot, Node

# Simple controller to move a Pioneer3dx back and forth along a straight line.

MAX_WHEEL_SPEED = 6.0     # rad/s, slower than main robot
WHEEL_RADIUS = 0.0975
AXLE_LENGTH = 0.33

class MovingObstacle:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        # Motors
        self.left_motor = self.robot.getDevice('left wheel')
        self.right_motor = self.robot.getDevice('right wheel')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Move forward for a while, then backward, repeatedly
        self.direction = 1.0   # +1 forward, -1 backward
        self.steps_in_direction = 0
        self.max_steps_per_direction = 400  # adjust to change travel distance

    def set_speed(self, v, w):
        v_r = (2.0 * v + w * AXLE_LENGTH) / (2.0 * WHEEL_RADIUS)
        v_l = (2.0 * v - w * AXLE_LENGTH) / (2.0 * WHEEL_RADIUS)
        v_r = max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, v_r))
        v_l = max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, v_l))
        self.left_motor.setVelocity(v_l)
        self.right_motor.setVelocity(v_r)

    def run(self):
        v_linear = 0.15  # m/s
        while self.robot.step(self.timestep) != -1:
            # Move straight in current direction
            self.set_speed(self.direction * v_linear, 0.0)

            self.steps_in_direction += 1
            if self.steps_in_direction >= self.max_steps_per_direction:
                # Reverse direction periodically
                self.direction *= -1.0
                self.steps_in_direction = 0


if __name__ == "__main__":
    obstacle = MovingObstacle()
    obstacle.run()
