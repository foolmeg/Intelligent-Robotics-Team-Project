import sys
import os
sys.path.append(os.path.abspath('..'))  # Add parent directory to path for DWA module

import math
import numpy as np
from controller import Robot, Motor, Lidar, Camera, RangeFinder, DistanceSensor, Accelerometer, Gyro, Compass, PositionSensor

# Import DWA planner (once created it)
# from dwa_planner import DWAPlanner

class RosbotDwaController:
    def __init__(self):
        # Initialize Webots robot
        self.robot = Robot()
        self.timestep = 32  # Same as original controller
        
        # Robot parameters (need to verify these)
        self.max_linear_speed = 26.0  # From MAX_VELOCITY in C code controller
        self.max_angular_speed = 5.0   # Need to determine this
        self.robot_radius = 0.2        # Approximate radius in meters
        
        # Initialize devices
        self.motors = {}
        self.sensors = {}
        self.setup_devices()
        
        # Initialize DWA planner (uncomment when ready)
        # self.dwa_planner = DWAPlanner(
        #     max_speed=self.max_linear_speed,
        #     max_angular_speed=self.max_angular_speed,
        #     max_accel=5.0,  # Tune this
        #     max_angular_accel=3.0,  # Tune this
        #     robot_radius=self.robot_radius
        # )
        
        # State variables
        self.current_velocity = [0.0, 0.0]  # [v, w]
        self.current_pose = [0.0, 0.0, 0.0]  # [x, y, theta] - need odometry
        
    def setup_devices(self):
        """Initialize all robot devices similar to the C controller"""
        
        # Motors - same as C controller
        motor_names = ['fl_wheel_joint', 'fr_wheel_joint', 'rl_wheel_joint', 'rr_wheel_joint']
        for name in motor_names:
            self.motors[name] = self.robot.getDevice(name)
            self.motors[name].setPosition(float('inf'))  # Velocity control
            self.motors[name].setVelocity(0.0)
        
        # LiDAR - crucial for DWA
        self.sensors['lidar'] = self.robot.getDevice("laser")
        self.sensors['lidar'].enable(self.timestep)
        self.sensors['lidar'].enablePointCloud()  # For obstacle detection
        
        # Distance sensors for emergency backup
        dist_sensor_names = ['fl_range', 'rl_range', 'fr_range', 'rr_range']
        self.sensors['distance'] = {}
        for name in dist_sensor_names:
            self.sensors['distance'][name] = self.robot.getDevice(name)
            self.sensors['distance'][name].enable(self.timestep)
        
        # Cameras (optional for DWA, but good to have)
        self.sensors['camera_rgb'] = self.robot.getDevice("camera rgb")
        self.sensors['camera_depth'] = self.robot.getDevice("camera depth")
        self.sensors['camera_rgb'].enable(self.timestep)
        self.sensors['camera_depth'].enable(self.timestep)
        
        # IMU (for orientation/odometry)
        self.sensors['accelerometer'] = self.robot.getDevice("imu accelerometer")
        self.sensors['gyro'] = self.robot.getDevice("imu gyro")
        self.sensors['compass'] = self.robot.getDevice("imu compass")
        self.sensors['accelerometer'].enable(self.timestep)
        self.sensors['gyro'].enable(self.timestep)
        self.sensors['compass'].enable(self.timestep)
    
    def get_lidar_obstacles(self):
        """Convert LiDAR point cloud to obstacle positions in robot frame"""
        point_cloud = self.sensors['lidar'].getPointCloud()
        obstacles = []
        
        for point in point_cloud:
            if not math.isinf(point.x) and not math.isinf(point.y):
                # Convert to 2D points (ignore height for ground robot)
                obstacles.append([point.x, point.y])
        
        return obstacles
    
    def get_robot_pose(self):
        """Estimate robot pose using odometry (simplified - you may need more sophisticated approach)"""
        # This is a simplified version - you'll need proper odometry
        # For now, you can use a placeholder or integrate wheel encoders
        return self.current_pose
    
    def set_robot_velocity(self, v, w):
        """Convert linear and angular velocity to wheel velocities"""
        # Kinematics for differential drive robot
        # Assuming wheel radius = 0.1m and wheel base = 0.3m (verify these)
        wheel_radius = 0.1
        wheel_base = 0.3
        
        # Convert [v, w] to left and right wheel velocities
        left_wheel_velocity = (v - (w * wheel_base / 2)) / wheel_radius
        right_wheel_velocity = (v + (w * wheel_base / 2)) / wheel_radius
        
        # Set motor velocities
        self.motors['fl_wheel_joint'].setVelocity(left_wheel_velocity)
        self.motors['rl_wheel_joint'].setVelocity(left_wheel_velocity)
        self.motors['fr_wheel_joint'].setVelocity(right_wheel_velocity)
        self.motors['rr_wheel_joint'].setVelocity(right_wheel_velocity)
        
        self.current_velocity = [v, w]
    
    def emergency_stop(self):
        """Stop the robot if obstacles are too close"""
        # Use distance sensors as backup safety
        for sensor in self.sensors['distance'].values():
            if sensor.getValue() < 0.3:  # 30cm threshold
                self.set_robot_velocity(0, 0)
                return True
        return False
    
    def run(self):
        """Main control loop"""
        # Temporary: simple test movement
        test_global_plan = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]  # Straight line
        
        while self.robot.step(self.timestep) != -1:
            # Skip if emergency stop triggered
            if self.emergency_stop():
                continue
            
            # Get sensor data
            obstacles = self.get_lidar_obstacles()
            current_pose = self.get_robot_pose()
            
            # TODO: Replace with actual DWA planning
            # For now, implement a simple test behavior
            if len(obstacles) > 10:  # If many obstacles detected
                # Simple obstacle avoidance: turn right
                self.set_robot_velocity(1.0, -1.0)
            else:
                # Move forward
                self.set_robot_velocity(2.0, 0.0)
            
            # Once DWA is ready, uncomment:
            # v, w = self.dwa_planner.plan(
            #     current_pose,
            #     self.current_velocity,
            #     test_global_plan,  # From global planner
            #     obstacles
            # )
            # self.set_robot_velocity(v, w)

# Create and run controller
if __name__ == "__main__":
    controller = RosbotDwaController()
    controller.run()