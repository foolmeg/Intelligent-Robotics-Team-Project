# dstar_lite.py
import heapq
import math

class DStarLite:
    def __init__(self, grid, start, goal):
        """
        grid: 2D list of 0 (free) and 1 (occupied)
        start, goal: (row, col) tuples
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.start = start
        self.goal = goal
        self.s_last = start
        self.km = 0.0
        self.U = []  # priority queue
        self.g = {}
        self.rhs = {}

        # Initialize all cells
        for i in range(self.rows):
            for j in range(self.cols):
                pos = (i, j)
                self.g[pos] = float('inf')
                self.rhs[pos] = float('inf')

        # Goal has rhs = 0 (it's the destination)
        if 0 <= self.goal[0] < self.rows and 0 <= self.goal[1] < self.cols:
            self.rhs[self.goal] = 0.0
            heapq.heappush(self.U, (self.calculate_key(self.goal), self.goal))

    def calculate_key(self, s):
        g_rhs = min(self.g[s], self.rhs[s])
        return (g_rhs + self.h(self.start, s) + self.km, g_rhs)

    def h(self, s1, s2):
        # Euclidean heuristic
        return math.hypot(s1[0] - s2[0], s1[1] - s2[1])

    def get_neighbors(self, s):
        neighbors = []
        # 8-connected grid (including diagonals for better paths)
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            nx, ny = s[0] + dx, s[1] + dy
            # Check bounds first
            if 0 <= nx < self.rows and 0 <= ny < self.cols:
                # Allow neighbors even if slightly occupied (for dynamic environments)
                # Only block if definitely occupied
                if self.grid[nx][ny] == 0:
                    neighbors.append((nx, ny))
        return neighbors

    def cost(self, s1, s2):
        # Euclidean distance for cost
        base_cost = self.h(s1, s2)
        # Add penalty if target cell is occupied (for dynamic obstacles)
        if 0 <= s2[0] < self.rows and 0 <= s2[1] < self.cols:
            if self.grid[s2[0]][s2[1]] == 1:
                base_cost *= 10.0  # High penalty for occupied cells
        return base_cost

    def update_vertex(self, u):
        if u != self.goal:
            min_rhs = float('inf')
            # Check all neighbors (including potentially occupied ones for dynamic planning)
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                nx, ny = u[0] + dx, u[1] + dy
                if 0 <= nx < self.rows and 0 <= ny < self.cols:
                    s = (nx, ny)
                    if s in self.g:
                        cost_val = self.cost(u, s)
                        candidate = self.g[s] + cost_val
                        if candidate < min_rhs:
                            min_rhs = candidate
            self.rhs[u] = min_rhs
        # remove u from U if present (more efficient removal)
        self.U = [pair for pair in self.U if pair[1] != u]
        heapq.heapify(self.U)
        if self.g[u] != self.rhs[u]:
            heapq.heappush(self.U, (self.calculate_key(u), u))

    def compute_shortest_path(self):
        while self.U and (
            self.U[0][0] < self.calculate_key(self.start) or
            self.rhs[self.start] != self.g[self.start]
        ):
            k_old = self.U[0][0]
            u = heapq.heappop(self.U)[1]
            if k_old < self.calculate_key(u):
                heapq.heappush(self.U, (self.calculate_key(u), u))
                continue
            if self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for s in self.get_neighbors(u):
                    self.update_vertex(s)
            else:
                old_g = self.g[u]
                self.g[u] = float('inf')
                neighbors = self.get_neighbors(u)
                neighbors.append(u)
                for s in neighbors:
                    self.update_vertex(s)

    def rescan(self, changed_cells):
        # changed_cells: list of (row, col) where cost changed
        # Update vertices for all changed cells and their neighbors
        cells_to_update = set()
        for u in changed_cells:
            cells_to_update.add(u)
            # Add neighbors of changed cells
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                nx, ny = u[0] + dx, u[1] + dy
                if 0 <= nx < self.rows and 0 <= ny < self.cols:
                    cells_to_update.add((nx, ny))
        
        # Update all affected vertices
        for s in cells_to_update:
            self.update_vertex(s)

    def plan(self, changed=[]):
        # dynamic replanning entry point
        self.km += self.h(self.s_last, self.start)
        self.s_last = self.start
        
        # Ensure start and goal are initialized
        if self.start not in self.g:
            self.g[self.start] = float('inf')
        if self.start not in self.rhs:
            self.rhs[self.start] = float('inf')
        if self.goal not in self.g:
            self.g[self.goal] = float('inf')
        if self.goal not in self.rhs:
            self.rhs[self.goal] = 0.0
        
        self.rescan(changed)
        self.compute_shortest_path()

        path = []
        current = self.start
        seen = set()
        max_iterations = self.rows * self.cols  # Prevent infinite loops
        
        # Check if goal is reachable
        if self.g[self.goal] == float('inf'):
            return None  # Goal is unreachable
        
        while current != self.goal and len(seen) < max_iterations:
            if current in seen:
                return None  # Loop detected
            seen.add(current)
            path.append(current)
            
            # Get all potential neighbors (including diagonals)
            neighbors = []
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                nx, ny = current[0] + dx, current[1] + dy
                if 0 <= nx < self.rows and 0 <= ny < self.cols:
                    neighbors.append((nx, ny))
            
            if not neighbors:
                return None  # No valid neighbors
            
            # Find best neighbor based on g-value + cost
            best_neighbor = None
            best_cost = float('inf')
            for s in neighbors:
                if s in self.g and self.g[s] < float('inf'):
                    total_cost = self.g[s] + self.cost(current, s)
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_neighbor = s
            
            if best_neighbor is None:
                # Try to find any reachable neighbor (even with high cost)
                for s in neighbors:
                    if s in self.g and self.g[s] < float('inf'):
                        best_neighbor = s
                        break
                if best_neighbor is None:
                    return None  # No reachable path
            
            current = best_neighbor
        
        if current == self.goal:
            path.append(self.goal)
            return path
        else:
            return None  # Could not reach goal
