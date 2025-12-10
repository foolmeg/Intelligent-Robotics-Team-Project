from controller import Supervisor
import math
import random

class Pedestrian(Supervisor):
    def __init__(self):
        Supervisor.__init__(self)
        self.time_step = int(self.getBasicTimeStep())
        self.robot = self.getSelf()
        self.translation_field = self.robot.getField("translation")
        self.rotation_field = self.robot.getField("rotation")
        
        # Capture initial position as "Home"
        pos = self.translation_field.getSFVec3f()
        self.home_x = pos[0]
        self.home_y = pos[1]
        self.height = pos[2]  # Keep Z constant
        
        self.speed = 0.6       # Walking speed (m/s)
        self.walk_dist = 1   # Distance to walk from origin
        self.pause_time = 0.5  # Seconds to wait at endpoints
        
        # Safety bounds (Global coordinates)
        self.BOUND_MIN = -5.0
        self.BOUND_MAX = 5.0

    def wait(self, duration):
        """Waits for a specific duration in seconds."""
        start = self.getTime()
        while self.step(self.time_step) != -1:
            if self.getTime() - start >= duration:
                return True
        return False

    def move_to(self, target_x, target_y):
        """ Linearly interpolates position to target. Returns False if simulation stops. """
        start_pos = self.translation_field.getSFVec3f()
        start_x, start_y = start_pos[0], start_pos[1]
        
        dist = math.hypot(target_x - start_x, target_y - start_y)
        if dist < 0.001:
            return True

        duration = dist / self.speed
        start_time = self.getTime()
        
        # Face the target (Z-axis rotation)
        angle = math.atan2(target_y - start_y, target_x - start_x)
        self.rotation_field.setSFRotation([0, 0, 1, angle])
        
        while self.step(self.time_step) != -1:
            now = self.getTime()
            dt = now - start_time
            
            if dt >= duration:
                # Arrived
                self.translation_field.setSFVec3f([target_x, target_y, self.height])
                return True
                
            # Interpolate
            ratio = dt / duration
            cx = start_x + (target_x - start_x) * ratio
            cy = start_y + (target_y - start_y) * ratio
            self.translation_field.setSFVec3f([cx, cy, self.height])
            
        return False

    def run(self):
        # Make sure we perform at least one step to initialize
        if self.step(self.time_step) == -1:
            return

        # Directions: Right, Left, Up, Down
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        while True:
            # 1. Pick a valid random direction
            valid_targets = []
            for dx, dy in directions:
                tx = self.home_x + dx * self.walk_dist
                ty = self.home_y + dy * self.walk_dist
                # Check bounds
                if (self.BOUND_MIN <= tx <= self.BOUND_MAX and 
                    self.BOUND_MIN <= ty <= self.BOUND_MAX):
                    valid_targets.append((tx, ty))
            
            if not valid_targets:
                # If stuck or no valid options, just wait
                if not self.wait(1.0): break
                continue

            target_x, target_y = random.choice(valid_targets)
            
            # 2. Move OUT to target
            if not self.move_to(target_x, target_y): break
            
            # 3. Wait briefly
            if not self.wait(self.pause_time): break
            
            # 4. Move BACK to Home
            if not self.move_to(self.home_x, self.home_y): break
            
            # 5. Wait at Home
            if not self.wait(self.pause_time): break

# Start controller
controller = Pedestrian()
controller.run()
