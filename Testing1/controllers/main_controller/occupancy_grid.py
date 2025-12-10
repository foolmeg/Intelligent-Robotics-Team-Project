"""
Occupancy Grid Mapping Module for Pioneer 3-DX Navigation System.

This module implements a 2D occupancy grid that is updated using LiDAR
measurements via ray-casting. The grid stores probability values for
each cell being occupied.
"""

import math
import numpy as np
from typing import Tuple, List, Optional, Set
from utils import bresenham_line, world_to_grid, grid_to_world


# Occupancy grid cell states
FREE = 0
OCCUPIED = 1
UNKNOWN = -1

# Log-odds parameters for Bayesian updating
LOG_ODDS_FREE = -0.3
LOG_ODDS_OCCUPIED = 0.7
LOG_ODDS_PRIOR = 0.0
LOG_ODDS_MAX = 3.0
LOG_ODDS_MIN = -3.0


class OccupancyGrid:
    """
    2D Occupancy Grid for mapping the environment.
    
    Uses log-odds representation for Bayesian updating of cell occupancy
    probabilities based on LiDAR measurements.
    """
    
    def __init__(self, 
                 width: float = 10.0,
                 height: float = 10.0,
                 resolution: float = 0.1,
                 origin: Tuple[float, float] = (-5.0, -5.0)):
        """
        Initialize the occupancy grid.
        
        Args:
            width: Width of the grid in meters
            height: Height of the grid in meters
            resolution: Size of each cell in meters (larger = faster but less precise)
            origin: World coordinates of the grid origin (bottom-left corner)
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin = origin
        
        # Calculate grid dimensions
        self.cols = int(width / resolution)
        self.rows = int(height / resolution)
        
        print(f"Grid: {self.rows}x{self.cols} cells, resolution={resolution}m")
        
        # Initialize grid with log-odds prior (unknown)
        self.log_odds = np.full((self.rows, self.cols), LOG_ODDS_PRIOR, dtype=np.float32)
        
        # Binary occupancy grid for planning (thresholded)
        self.occupancy_threshold = 0.6
        
        # Inflation radius for robot safety margin (in cells)
        # With 0.1m resolution: 2 cells = 20cm inflation
        self.inflation_radius = 2
        
        # CACHED inflated grid for fast lookup
        self._inflated_grid: Optional[np.ndarray] = None
        self._needs_inflation_update = True
        
        # Track changed cells for incremental planning
        self.changed_cells: Set[Tuple[int, int]] = set()
        
    def update(self, pose: Tuple[float, float, float], 
               lidar_ranges: List[float],
               lidar_angles: List[float],
               max_range: float = 5.0) -> Set[Tuple[int, int]]:
        """
        Update the occupancy grid with a new LiDAR scan.
        """
        x, y, theta = pose
        self.changed_cells.clear()
        
        # Subsample LiDAR for speed (use every Nth beam)
        step = max(1, len(lidar_ranges) // 60)  # Use ~60 beams max
        
        for i in range(0, len(lidar_ranges), step):
            distance = lidar_ranges[i]
            angle = lidar_angles[i]
            
            # Skip invalid measurements
            if distance <= 0.05 or distance >= max_range or math.isinf(distance):
                continue
            
            # Calculate beam endpoint in world coordinates
            beam_angle = theta + angle
            end_x = x + distance * math.cos(beam_angle)
            end_y = y + distance * math.sin(beam_angle)
            
            # Convert to grid coordinates
            start_cell = self.world_to_grid(x, y)
            end_cell = self.world_to_grid(end_x, end_y)
            
            # Skip if outside grid
            if not self._is_valid_cell(start_cell[0], start_cell[1]):
                continue
            if not self._is_valid_cell(end_cell[0], end_cell[1]):
                continue
            
            # Get cells along the ray using Bresenham's algorithm
            ray_cells = bresenham_line(start_cell[1], start_cell[0], 
                                       end_cell[1], end_cell[0])
            
            # Update cells along the ray
            for j, (row, col) in enumerate(ray_cells):
                if not self._is_valid_cell(row, col):
                    continue
                
                old_value = self.log_odds[row, col]
                
                if j < len(ray_cells) - 1:
                    # Free space along the ray
                    self.log_odds[row, col] += LOG_ODDS_FREE
                else:
                    # Occupied at the endpoint
                    self.log_odds[row, col] += LOG_ODDS_OCCUPIED
                
                # Clamp log-odds
                self.log_odds[row, col] = np.clip(self.log_odds[row, col], 
                                                  LOG_ODDS_MIN, LOG_ODDS_MAX)
                
                # Track changed cells
                if abs(self.log_odds[row, col] - old_value) > 0.05:
                    self.changed_cells.add((row, col))
        
        # Mark inflated grid as needing update
        if self.changed_cells:
            self._needs_inflation_update = True
        
        return self.changed_cells
    
    def _is_valid_cell(self, row: int, col: int) -> bool:
        """Check if a cell index is within grid bounds."""
        return 0 <= row < self.rows and 0 <= col < self.cols
    
    def get_probability(self, row: int, col: int) -> float:
        """Get occupancy probability for a cell."""
        if not self._is_valid_cell(row, col):
            return 1.0  # Out of bounds treated as occupied
        
        log_odds = self.log_odds[row, col]
        return 1.0 / (1.0 + math.exp(-log_odds))
    
    def is_occupied(self, row: int, col: int) -> bool:
        """Check if a cell is occupied (raw, no inflation)."""
        return self.get_probability(row, col) > self.occupancy_threshold
    
    def _update_inflated_grid(self):
        """Update the cached inflated grid."""
        # Get binary grid
        probabilities = 1.0 / (1.0 + np.exp(-self.log_odds))
        binary_grid = (probabilities > self.occupancy_threshold).astype(np.uint8)
        
        # Use scipy-style dilation if available, otherwise manual
        self._inflated_grid = np.copy(binary_grid)
        
        # Find all occupied cells
        occupied_cells = np.argwhere(binary_grid == 1)
        
        # Inflate around each occupied cell
        for r, c in occupied_cells:
            for dr in range(-self.inflation_radius, self.inflation_radius + 1):
                for dc in range(-self.inflation_radius, self.inflation_radius + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        dist_sq = dr * dr + dc * dc
                        if dist_sq <= self.inflation_radius * self.inflation_radius:
                            self._inflated_grid[nr, nc] = 1
        
        self._needs_inflation_update = False
    
    def is_occupied_inflated(self, row: int, col: int) -> bool:
        """Check if a cell is occupied using CACHED inflated grid (fast!)."""
        if not self._is_valid_cell(row, col):
            return True
        
        # Update cache if needed
        if self._needs_inflation_update or self._inflated_grid is None:
            self._update_inflated_grid()
        
        return self._inflated_grid[row, col] == 1
    
    def is_free(self, row: int, col: int) -> bool:
        """Check if a cell is free (below threshold)."""
        return self.get_probability(row, col) < (1.0 - self.occupancy_threshold)
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid cell."""
        col = int((x - self.origin[0]) / self.resolution)
        row = int((y - self.origin[1]) / self.resolution)
        return (row, col)
    
    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid cell to world coordinates (cell center)."""
        world_x = self.origin[0] + (col + 0.5) * self.resolution
        world_y = self.origin[1] + (row + 0.5) * self.resolution
        return (world_x, world_y)
    
    def get_neighbors(self, row: int, col: int, 
                      include_diagonal: bool = True) -> List[Tuple[int, int]]:
        """Get valid neighboring cells."""
        neighbors = []
        
        if include_diagonal:
            directions = [(-1, -1), (-1, 0), (-1, 1),
                         (0, -1),          (0, 1),
                         (1, -1),  (1, 0),  (1, 1)]
        else:
            directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if self._is_valid_cell(nr, nc):
                neighbors.append((nr, nc))
        
        return neighbors
    
    def get_nearest_free_cell(self, row: int, col: int) -> Optional[Tuple[int, int]]:
        """Find the nearest free cell to a given cell using BFS."""
        if self._is_valid_cell(row, col) and not self.is_occupied_inflated(row, col):
            return (row, col)
        
        from collections import deque
        visited = set()
        queue = deque([(row, col, 0)])  # (row, col, distance)
        
        while queue:
            r, c, dist = queue.popleft()
            
            if (r, c) in visited:
                continue
            visited.add((r, c))
            
            # Limit search radius
            if dist > 20:
                continue
            
            if self._is_valid_cell(r, c) and not self.is_occupied_inflated(r, c):
                return (r, c)
            
            for nr, nc in self.get_neighbors(r, c, include_diagonal=False):
                if (nr, nc) not in visited:
                    queue.append((nr, nc, dist + 1))
        
        return None
    
    def debug_cell(self, x: float, y: float) -> str:
        """Debug helper to check cell status at world coordinates."""
        cell = self.world_to_grid(x, y)
        prob = self.get_probability(cell[0], cell[1])
        occupied = self.is_occupied(cell[0], cell[1])
        inflated = self.is_occupied_inflated(cell[0], cell[1])
        return f"Cell {cell}: prob={prob:.2f}, occupied={occupied}, inflated={inflated}"
    
    def clear(self):
        """Reset the occupancy grid to unknown state."""
        self.log_odds.fill(LOG_ODDS_PRIOR)
        self.changed_cells.clear()
        self._needs_inflation_update = True
        self._inflated_grid = None
