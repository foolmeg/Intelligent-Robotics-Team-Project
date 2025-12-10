"""
Utility functions for the Pioneer 3-DX navigation system.
Contains common helpers for coordinate transformations, math operations, etc.
"""

import math
import numpy as np
from typing import Tuple, List, Optional


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to [-pi, pi] range.
    
    Args:
        angle: Angle in radians
        
    Returns:
        Normalized angle in radians
    """
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
        
    Returns:
        Distance between points
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def angle_to_point(from_point: Tuple[float, float], to_point: Tuple[float, float]) -> float:
    """
    Calculate angle from one point to another.
    
    Args:
        from_point: Starting point (x, y)
        to_point: Target point (x, y)
        
    Returns:
        Angle in radians
    """
    dx = to_point[0] - from_point[0]
    dy = to_point[1] - from_point[1]
    return math.atan2(dy, dx)


def world_to_grid(world_x: float, world_y: float, 
                  grid_resolution: float, 
                  grid_origin: Tuple[float, float]) -> Tuple[int, int]:
    """
    Convert world coordinates to grid cell indices.
    
    Args:
        world_x: X coordinate in world frame
        world_y: Y coordinate in world frame
        grid_resolution: Size of each grid cell in meters
        grid_origin: World coordinates of grid origin (bottom-left corner)
        
    Returns:
        Grid cell indices (row, col)
    """
    col = int((world_x - grid_origin[0]) / grid_resolution)
    row = int((world_y - grid_origin[1]) / grid_resolution)
    return (row, col)


def grid_to_world(row: int, col: int,
                  grid_resolution: float,
                  grid_origin: Tuple[float, float]) -> Tuple[float, float]:
    """
    Convert grid cell indices to world coordinates (cell center).
    
    Args:
        row: Grid row index
        col: Grid column index
        grid_resolution: Size of each grid cell in meters
        grid_origin: World coordinates of grid origin (bottom-left corner)
        
    Returns:
        World coordinates (x, y) of cell center
    """
    world_x = grid_origin[0] + (col + 0.5) * grid_resolution
    world_y = grid_origin[1] + (row + 0.5) * grid_resolution
    return (world_x, world_y)


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """
    Bresenham's line algorithm for ray casting.
    Returns all grid cells along a line from (x0, y0) to (x1, y1).
    
    Args:
        x0, y0: Starting cell
        x1, y1: Ending cell
        
    Returns:
        List of grid cells (row, col) along the line
    """
    cells = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    
    if dx > dy:
        err = dx / 2
        while x != x1:
            cells.append((y, x))  # (row, col)
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
        cells.append((y, x))
    else:
        err = dy / 2
        while y != y1:
            cells.append((y, x))  # (row, col)
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
        cells.append((y, x))
    
    return cells


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a value to a specified range.
    
    Args:
        value: Value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


def interpolate_path(path: List[Tuple[float, float]], 
                     resolution: float = 0.1) -> List[Tuple[float, float]]:
    """
    Interpolate a path to have finer resolution.
    
    Args:
        path: List of waypoints (x, y)
        resolution: Desired distance between interpolated points
        
    Returns:
        Interpolated path with finer resolution
    """
    if len(path) < 2:
        return path
    
    interpolated = [path[0]]
    
    for i in range(1, len(path)):
        p1 = path[i - 1]
        p2 = path[i]
        dist = euclidean_distance(p1, p2)
        
        if dist > resolution:
            n_points = int(dist / resolution)
            for j in range(1, n_points + 1):
                t = j / (n_points + 1)
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                interpolated.append((x, y))
        
        interpolated.append(p2)
    
    return interpolated


def smooth_path(path: List[Tuple[float, float]], 
                weight_data: float = 0.5,
                weight_smooth: float = 0.1,
                tolerance: float = 0.001) -> List[Tuple[float, float]]:
    """
    Smooth a path using gradient descent.
    
    Args:
        path: Original path
        weight_data: Weight for staying close to original path
        weight_smooth: Weight for smoothness
        tolerance: Convergence tolerance
        
    Returns:
        Smoothed path
    """
    if len(path) < 3:
        return path
    
    # Convert to numpy for easier manipulation
    path_array = np.array(path, dtype=float)
    smoothed = np.copy(path_array)
    
    change = tolerance + 1
    while change >= tolerance:
        change = 0
        for i in range(1, len(path) - 1):
            for j in range(2):
                orig = smoothed[i, j]
                smoothed[i, j] += weight_data * (path_array[i, j] - smoothed[i, j])
                smoothed[i, j] += weight_smooth * (smoothed[i-1, j] + smoothed[i+1, j] - 2 * smoothed[i, j])
                change += abs(orig - smoothed[i, j])
    
    return [(smoothed[i, 0], smoothed[i, 1]) for i in range(len(smoothed))]


class PIDController:
    """Simple PID controller for heading control."""
    
    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.1):
        """
        Initialize PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
    
    def compute(self, error: float, dt: float) -> float:
        """
        Compute PID control output.
        
        Args:
            error: Current error
            dt: Time step
            
        Returns:
            Control output
        """
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        
        return self.kp * error + self.ki * self.integral + self.kd * derivative
    
    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.prev_error = 0.0
