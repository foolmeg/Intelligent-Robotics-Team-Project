from controller import Supervisor
import math
import random


class Pedestrian(Supervisor):
    def __init__(self):
        Supervisor.__init__(self)

        self.speed = 1.0

        # movement bounds (6x6 arena centered at origin, with small margin for pedestrian size)
        # Arena is 6x6 meters, so bounds are -3 to 3, but we use slightly smaller to keep pedestrian inside
        self.MIN_X = -2.8
        self.MAX_X = 2.8
        self.MIN_Y = -2.8
        self.MAX_Y = 2.8

        self.robot = self.getSelf()
        self.translation_field = self.robot.getField("translation")
        self.rotation_field = self.robot.getField("rotation")

    def generate_random_waypoint(self, x, y, max_d=2.0):
        """
        Generates a waypoint that NEVER leaves the allowed boundary.
        Generates a random direction and distance, then clamps to stay within bounds.
        """
        # Generate random angle (0 to 2*pi)
        angle = random.uniform(0, 2 * math.pi)
        
        # Generate random distance
        distance = random.uniform(0.5, max_d)
        
        # Calculate target position
        tx = x + distance * math.cos(angle)
        ty = y + distance * math.sin(angle)
        
        # Clamp to boundaries
        tx = self.clamp(tx, self.MIN_X, self.MAX_X)
        ty = self.clamp(ty, self.MIN_Y, self.MAX_Y)
        
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