"""
D* Lite Global Planner for Pioneer 3-DX Navigation System.

This module implements the D* Lite incremental search algorithm for
efficient path planning in partially known environments. Unlike A*,
D* Lite can efficiently update paths when the environment changes
without recomputing from scratch.

Reference: Koenig, S., & Likhachev, M. (2002). D* Lite.
"""

import math
import heapq
from typing import Tuple, List, Optional, Dict, Set
from occupancy_grid import OccupancyGrid


class DStarLitePlanner:
    """
    D* Lite incremental path planning algorithm.
    
    Maintains a priority queue and can efficiently update paths when
    obstacles are discovered or removed.
    """
    
    def __init__(self, grid: OccupancyGrid):
        """
        Initialize the D* Lite planner.
        
        Args:
            grid: Occupancy grid for the environment
        """
        self.grid = grid
        
        # D* Lite state variables
        self.g: Dict[Tuple[int, int], float] = {}  # Cost-to-goal
        self.rhs: Dict[Tuple[int, int], float] = {}  # One-step lookahead
        self.U: List[Tuple[Tuple[float, float], Tuple[int, int]]] = []  # Priority queue
        self.U_set: Set[Tuple[int, int]] = set()  # For O(1) membership check
        
        # Start and goal positions
        self.start: Optional[Tuple[int, int]] = None
        self.goal: Optional[Tuple[int, int]] = None
        
        # km offset for when start position changes
        self.km: float = 0.0
        self.last_start: Optional[Tuple[int, int]] = None
        
        # Current path
        self.path: List[Tuple[int, int]] = []
        
        # Use 8-connected grid for diagonal movement
        self.use_diagonal = True
    
    def initialize(self, start: Tuple[int, int], goal: Tuple[int, int]):
        """
        Initialize the planner with start and goal positions.
        
        Args:
            start: Start cell (row, col) in grid coordinates
            goal: Goal cell (row, col) in grid coordinates
        """
        self.start = start
        self.goal = goal
        self.last_start = start
        self.km = 0.0
        
        # Clear data structures
        self.g.clear()
        self.rhs.clear()
        self.U = []
        self.U_set.clear()
        
        # Initialize goal
        self.rhs[goal] = 0.0
        self._insert(goal, self._calculate_key(goal))
    
    def _heuristic(self, s1: Tuple[int, int], s2: Tuple[int, int]) -> float:
        """
        Heuristic function (Euclidean distance).
        
        Args:
            s1: First cell
            s2: Second cell
            
        Returns:
            Estimated cost between cells
        """
        return math.sqrt((s1[0] - s2[0])**2 + (s1[1] - s2[1])**2)
    
    def _get_g(self, s: Tuple[int, int]) -> float:
        """Get g-value for a state (infinity if not set)."""
        return self.g.get(s, float('inf'))
    
    def _get_rhs(self, s: Tuple[int, int]) -> float:
        """Get rhs-value for a state (infinity if not set)."""
        return self.rhs.get(s, float('inf'))
    
    def _calculate_key(self, s: Tuple[int, int]) -> Tuple[float, float]:
        """
        Calculate the priority key for a state.
        
        Args:
            s: State (row, col)
            
        Returns:
            Priority key (k1, k2)
        """
        g_val = self._get_g(s)
        rhs_val = self._get_rhs(s)
        min_val = min(g_val, rhs_val)
        
        k1 = min_val + self._heuristic(self.start, s) + self.km
        k2 = min_val
        
        return (k1, k2)
    
    def _insert(self, s: Tuple[int, int], key: Tuple[float, float]):
        """Insert a state into the priority queue."""
        if s not in self.U_set:
            heapq.heappush(self.U, (key, s))
            self.U_set.add(s)
    
    def _remove(self, s: Tuple[int, int]):
        """Mark a state as removed from the queue."""
        if s in self.U_set:
            self.U_set.remove(s)
    
    def _top_key(self) -> Optional[Tuple[float, float]]:
        """Get the key of the top element in the queue."""
        while self.U:
            key, s = self.U[0]
            if s in self.U_set:
                return key
            else:
                heapq.heappop(self.U)
        return None
    
    def _pop(self) -> Optional[Tuple[int, int]]:
        """Pop and return the top element from the queue."""
        while self.U:
            key, s = heapq.heappop(self.U)
            if s in self.U_set:
                self.U_set.remove(s)
                return s
        return None
    
    def _get_cost(self, s1: Tuple[int, int], s2: Tuple[int, int]) -> float:
        """
        Get the cost of moving from s1 to s2.
        
        Args:
            s1: Source cell
            s2: Destination cell
            
        Returns:
            Movement cost (infinity if blocked)
        """
        # Check if destination is occupied (using inflated check for safety margin)
        if self.grid.is_occupied_inflated(s2[0], s2[1]):
            return float('inf')
        
        # Check if source is occupied
        if self.grid.is_occupied_inflated(s1[0], s1[1]):
            return float('inf')
        
        # Calculate distance
        dr = abs(s1[0] - s2[0])
        dc = abs(s1[1] - s2[1])
        
        if dr + dc == 2:  # Diagonal
            return math.sqrt(2.0)
        elif dr + dc == 1:  # Cardinal
            return 1.0
        else:
            return float('inf')  # Not adjacent
    
    def _get_successors(self, s: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get successors (neighbors) of a state."""
        return self.grid.get_neighbors(s[0], s[1], self.use_diagonal)
    
    def _get_predecessors(self, s: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get predecessors (also neighbors for undirected graph)."""
        return self.grid.get_neighbors(s[0], s[1], self.use_diagonal)
    
    def _update_vertex(self, u: Tuple[int, int]):
        """
        Update the rhs value and queue position for a vertex.
        
        Args:
            u: Vertex to update
        """
        if u != self.goal:
            # Calculate minimum rhs from successors
            min_rhs = float('inf')
            for s in self._get_successors(u):
                cost = self._get_cost(u, s)
                if cost < float('inf'):
                    min_rhs = min(min_rhs, cost + self._get_g(s))
            self.rhs[u] = min_rhs
        
        # Remove from queue if present
        self._remove(u)
        
        # Re-insert if inconsistent
        if self._get_g(u) != self._get_rhs(u):
            self._insert(u, self._calculate_key(u))
    
    def _compute_shortest_path(self) -> bool:
        """
        Compute or repair the shortest path.
        
        Returns:
            True if a valid path was found
        """
        max_iterations = self.grid.rows * self.grid.cols * 2
        iterations = 0
        
        start_key = self._calculate_key(self.start)
        
        while True:
            iterations += 1
            if iterations > max_iterations:
                print("D* Lite: Max iterations reached")
                return False
            
            top_key = self._top_key()
            
            # Check termination conditions
            if top_key is None:
                break
            
            start_consistent = (self._get_g(self.start) == self._get_rhs(self.start))
            key_ok = top_key >= start_key
            
            if key_ok and start_consistent:
                break
            
            # Pop the top element
            u = self._pop()
            if u is None:
                break
            
            old_key = top_key
            new_key = self._calculate_key(u)
            
            if old_key < new_key:
                # Key has changed, reinsert with new key
                self._insert(u, new_key)
            elif self._get_g(u) > self._get_rhs(u):
                # Overconsistent: make consistent
                self.g[u] = self._get_rhs(u)
                for s in self._get_predecessors(u):
                    self._update_vertex(s)
            else:
                # Underconsistent: make infinite and update
                self.g[u] = float('inf')
                for s in self._get_predecessors(u):
                    self._update_vertex(s)
                self._update_vertex(u)
            
            # Update start key for next iteration
            start_key = self._calculate_key(self.start)
        
        # Check if path was found
        return self._get_g(self.start) < float('inf')
    
    def plan(self, start_world: Tuple[float, float], 
             goal_world: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Plan a path from start to goal in world coordinates.
        
        Args:
            start_world: Start position (x, y) in world coordinates
            goal_world: Goal position (x, y) in world coordinates
            
        Returns:
            Path as list of world coordinates [(x, y), ...]
        """
        # Convert to grid coordinates
        start_grid = self.grid.world_to_grid(start_world[0], start_world[1])
        goal_grid = self.grid.world_to_grid(goal_world[0], goal_world[1])
        
        # Ensure start and goal are in free space
        if self.grid.is_occupied(start_grid[0], start_grid[1]):
            start_grid = self.grid.get_nearest_free_cell(start_grid[0], start_grid[1])
            if start_grid is None:
                print("D* Lite: No free cell near start")
                return []
        
        if self.grid.is_occupied(goal_grid[0], goal_grid[1]):
            goal_grid = self.grid.get_nearest_free_cell(goal_grid[0], goal_grid[1])
            if goal_grid is None:
                print("D* Lite: No free cell near goal")
                return []
        
        # Initialize if needed
        if self.goal != goal_grid or self.start is None:
            self.initialize(start_grid, goal_grid)
        
        # Update start position and km if moved
        if self.start != start_grid:
            self.km += self._heuristic(self.last_start, start_grid)
            self.last_start = start_grid
            self.start = start_grid
        
        # Compute path
        if not self._compute_shortest_path():
            print("D* Lite: No path found")
            return []
        
        # Extract path by following the gradient
        path_grid = self._extract_path()
        
        # Convert to world coordinates
        path_world = []
        for cell in path_grid:
            world_pos = self.grid.grid_to_world(cell[0], cell[1])
            path_world.append(world_pos)
        
        self.path = path_grid
        return path_world
    
    def _extract_path(self) -> List[Tuple[int, int]]:
        """
        Extract the path from start to goal by following the gradient.
        
        Returns:
            Path as list of grid cells
        """
        path = [self.start]
        current = self.start
        max_steps = self.grid.rows * self.grid.cols
        
        for _ in range(max_steps):
            if current == self.goal:
                break
            
            # Find the best successor
            best_next = None
            best_cost = float('inf')
            
            for s in self._get_successors(current):
                cost = self._get_cost(current, s) + self._get_g(s)
                if cost < best_cost:
                    best_cost = cost
                    best_next = s
            
            if best_next is None or best_cost >= float('inf'):
                print("D* Lite: Path extraction failed")
                break
            
            path.append(best_next)
            current = best_next
        
        return path
    
    def update_map(self, changed_cells: Set[Tuple[int, int]]):
        """
        Update the planner when map cells have changed.
        
        Args:
            changed_cells: Set of cells that have changed
        """
        if not changed_cells or self.start is None:
            return
        
        # Update km based on current start
        if self.last_start != self.start:
            self.km += self._heuristic(self.last_start, self.start)
            self.last_start = self.start
        
        # Update affected vertices
        for cell in changed_cells:
            # Update the changed cell
            self._update_vertex(cell)
            
            # Update all neighbors
            for neighbor in self._get_predecessors(cell):
                self._update_vertex(neighbor)
    
    def replan(self, current_pos: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Replan from current position after map update.
        
        Args:
            current_pos: Current robot position (x, y) in world coordinates
            
        Returns:
            Updated path in world coordinates
        """
        if self.goal is None:
            return []
        
        # Update start position
        new_start = self.grid.world_to_grid(current_pos[0], current_pos[1])
        
        if self.grid.is_occupied(new_start[0], new_start[1]):
            new_start = self.grid.get_nearest_free_cell(new_start[0], new_start[1])
            if new_start is None:
                return []
        
        if self.start != new_start:
            self.km += self._heuristic(self.last_start, new_start)
            self.last_start = new_start
            self.start = new_start
        
        # Recompute shortest path
        if not self._compute_shortest_path():
            return []
        
        # Extract and convert path
        path_grid = self._extract_path()
        path_world = []
        for cell in path_grid:
            world_pos = self.grid.grid_to_world(cell[0], cell[1])
            path_world.append(world_pos)
        
        self.path = path_grid
        return path_world
    
    def get_current_path(self) -> List[Tuple[int, int]]:
        """Get the current path in grid coordinates."""
        return self.path
    
    def is_path_valid(self) -> bool:
        """
        Check if the current path is still valid (no obstacles on it).
        
        Returns:
            True if path is valid
        """
        for cell in self.path:
            if self.grid.is_occupied(cell[0], cell[1]):
                return False
        return True
