from controller import Supervisor
import math
import random


class Pedestrian(Supervisor):
    def __init__(self):
        Supervisor.__init__(self)

        self.speed = 1.0

        # movement bounds (10x10 area centered at origin)
        self.MIN_X = -5
        self.MAX_X = 5
        self.MIN_Y = -5
        self.MAX_Y = 5

        self.robot = self.getSelf()
        self.translation_field = self.robot.getField("translation")
        self.rotation_field = self.robot.getField("rotation")

    def generate_random_waypoint(self, x, y, max_d=4.0):
        """
        Generates a waypoint that NEVER leaves the allowed boundary.
        Directions that would leave the area are automatically removed.
        """

        valid_dirs = []

        # Possible directions (filtered by bounds)
        if x < self.MAX_X:
            valid_dirs.append((1, 0))
        if x > self.MIN_X:
            valid_dirs.append((-1, 0))
        if y < self.MAX_Y:
            valid_dirs.append((0, 1))
        if y > self.MIN_Y:
            valid_dirs.append((0, -1))

        dx, dy = random.choice(valid_dirs)

        # Calculate maximum safe movement in the chosen direction
        if dx == 1:
            max_step = self.MAX_X - x
        elif dx == -1:
            max_step = x - self.MIN_X
        elif dy == 1:
            max_step = self.MAX_Y - y
        else:  # dy == -1
            max_step = y - self.MIN_Y

        # Actual travel distance (0.5 ~ max_d, but not exceeding max_step)
        step = random.uniform(0.5, min(max_d, max_step))

        tx = x + dx * step
        ty = y + dy * step

        return tx, ty

    def clamp(self, val, lo, hi):
        """Ensures the robot NEVER leaves allowed bounds."""
        return max(lo, min(hi, val))

    def run(self):
        time_step = int(self.getBasicTimeStep())

        px, py, pz = self.translation_field.getSFVec3f()
        self.curr_x, self.curr_y = px, py

        # First target
        self.target_x, self.target_y = self.generate_random_waypoint(self.curr_x, self.curr_y)
        self.start_time = self.getTime()
        self.dist_target = math.dist([self.curr_x, self.curr_y],
                                     [self.target_x, self.target_y])

        while self.step(time_step) != -1:
            t = self.getTime()
            traveled = (t - self.start_time) * self.speed

            # Arrived → pick new waypoint inside boundary
            if traveled >= self.dist_target:
                self.curr_x = self.target_x
                self.curr_y = self.target_y

                self.target_x, self.target_y = self.generate_random_waypoint(self.curr_x, self.curr_y)
                self.dist_target = math.dist([self.curr_x, self.curr_y],
                                             [self.target_x, self.target_y])
                self.start_time = t
                traveled = 0.0

            # Linear interpolation
            progress = traveled / self.dist_target if self.dist_target > 0 else 0
            x = self.curr_x + (self.target_x - self.curr_x) * progress
            y = self.curr_y + (self.target_y - self.curr_y) * progress

            # Final safety clamp (belt + suspenders)
            x = self.clamp(x, self.MIN_X, self.MAX_X)
            y = self.clamp(y, self.MIN_Y, self.MAX_Y)

            # Orientation update
            angle = math.atan2(self.target_y - self.curr_y,
                               self.target_x - self.curr_x)

            self.translation_field.setSFVec3f([x, y, pz])
            self.rotation_field.setSFRotation([0, 0, 1, angle])


controller = Pedestrian()
controller.run()
