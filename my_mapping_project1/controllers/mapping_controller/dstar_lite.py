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

        for i in range(self.rows):
            for j in range(self.cols):
                pos = (i, j)
                self.g[pos] = float('inf')
                self.rhs[pos] = float('inf')

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
            if 0 <= nx < self.rows and 0 <= ny < self.cols and self.grid[nx][ny] == 0:
                neighbors.append((nx, ny))
        return neighbors

    def cost(self, s1, s2):
        return self.h(s1, s2)

    def update_vertex(self, u):
        if u != self.goal:
            self.rhs[u] = min(
                (self.g[s] + self.cost(u, s) for s in self.get_neighbors(u)),
                default=float('inf')
            )
        # remove u from U if present
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
        for u in changed_cells:
            neighbors = self.get_neighbors(u)
            neighbors.append(u)
            for s in neighbors:
                self.update_vertex(s)

    def plan(self, changed=[]):
        # dynamic replanning entry point
        self.km += self.h(self.s_last, self.start)
        self.s_last = self.start
        self.rescan(changed)
        self.compute_shortest_path()

        path = []
        current = self.start
        seen = set()
        max_iterations = self.rows * self.cols  # Prevent infinite loops
        
        while current != self.goal and len(seen) < max_iterations:
            if current in seen:
                return None  # Loop detected
            seen.add(current)
            path.append(current)
            
            neighbors = self.get_neighbors(current)
            if not neighbors:
                return None  # No valid neighbors
            
            # Find best neighbor based on g-value + cost
            best_neighbor = None
            best_cost = float('inf')
            for s in neighbors:
                if self.g[s] < float('inf'):
                    total_cost = self.g[s] + self.cost(current, s)
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_neighbor = s
            
            if best_neighbor is None:
                return None  # No reachable path
            
            current = best_neighbor
        
        if current == self.goal:
            path.append(self.goal)
            return path
        else:
            return None  # Could not reach goal
