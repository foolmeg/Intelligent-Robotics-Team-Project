import heapq
import math


class DStarLite:
    """
    Grid-based D* Lite implementation.

    - Grid coordinates: (x, y) with 0 <= x < width, 0 <= y < height
    - occupancy[y][x] == 1  -> obstacle (blocked)
      occupancy[y][x] != 1  -> free (traversable)

    Typical usage:
        dstar = DStarLite(width, height, start, goal)
        # optional: set obstacles
        dstar.set_obstacle(x, y)
        dstar.replan(start)
        path = dstar.get_shortest_path(start)
    """

    def __init__(self, width, height, start, goal, occupancy=None):
        self.width = width
        self.height = height

        self.start = start  # current robot grid position
        self.last_start = start
        self.goal = goal    # fixed goal cell

        # key modifier (for moving start)
        self.km = 0.0

        self.INF = float("inf")

        # cost-to-go and one-step-lookahead
        self.g = {}
        self.rhs = {}

        # priority queue: list of (k1, k2, x, y)
        self.open_heap = []
        # active entries: (x,y) -> (k1, k2)
        self.open_dict = {}

        # occupancy grid
        if occupancy is None:
            self.occupancy = [[0 for _ in range(width)] for _ in range(height)]
        else:
            # assume occupancy is [height][width]
            self.occupancy = occupancy

        self._initialize()

    # -----------------------------
    # Public API
    # -----------------------------

    def set_obstacle(self, x, y):
        """Mark cell (x, y) as obstacle and update D* Lite."""
        if not self._in_bounds((x, y)):
            return
        if self.occupancy[y][x] == 1:
            return  # already obstacle
        self.occupancy[y][x] = 1
        # affected: this cell and its neighbors
        self._update_vertex((x, y))
        for n in self._neighbors((x, y)):
            self._update_vertex(n)

    def clear_obstacle(self, x, y):
        """Mark cell (x, y) as free and update D* Lite."""
        if not self._in_bounds((x, y)):
            return
        if self.occupancy[y][x] != 1:
            return  # already free
        self.occupancy[y][x] = 0
        self._update_vertex((x, y))
        for n in self._neighbors((x, y)):
            self._update_vertex(n)

    def replan(self, new_start):
        """
        Update start position (robot moved) and recompute shortest path.
        Call this whenever:
        - robot start cell changed, or
        - obstacles updated.
        """
        if new_start != self.start:
            self._move_start(new_start)
        self._compute_shortest_path()

    def get_shortest_path(self, start, max_steps=10000):
        """
        Extract a path from 'start' to 'goal' using current g-values.
        Returns: list of (x, y) grid cells, including start & goal,
                 or None if no path.
        """
        if self.rhs.get(start, self.INF) == self.INF:
            # no known path from start to goal
            return None

        path = [start]
        current = start

        for _ in range(max_steps):
            if current == self.goal:
                break

            # among free neighbors, choose successor with minimal (cost + g)
            neighs = [n for n in self._neighbors(current)
                      if self.occupancy[n[1]][n[0]] != 1]

            if not neighs:
                return None

            best = None
            best_val = self.INF
            for n in neighs:
                c = self._cost(current, n)
                val = c + self.g.get(n, self.INF)
                if val < best_val:
                    best_val = val
                    best = n

            if best is None or best_val == self.INF:
                return None

            # avoid infinite loops
            if best in path:
                return None

            path.append(best)
            current = best

        if current != self.goal:
            return None

        return path

    # -----------------------------
    # Core D* Lite methods
    # -----------------------------

    def _initialize(self):
        """Initialize g, rhs, and OPEN with goal state."""
        self.km = 0.0
        self.g.clear()
        self.rhs.clear()
        self.open_heap.clear()
        self.open_dict.clear()

        # initialize all nodes with INF
        for x in range(self.width):
            for y in range(self.height):
                self.g[(x, y)] = self.INF
                self.rhs[(x, y)] = self.INF

        # goal state initialization
        self.rhs[self.goal] = 0.0
        self._insert(self.goal, self._calculate_key(self.goal))

    def _move_start(self, new_start):
        """Update km and start when robot moves."""
        old_start = self.start
        # km += heuristic(old_start, new_start)
        self.km += self._heuristic(old_start, new_start)
        self.last_start = old_start
        self.start = new_start

    def _compute_shortest_path(self):
        """
        Main D* Lite loop.
        Updates g, rhs until start is consistent with OPEN.
        """
        while True:
            k_min = self._get_min_key()
            k_start = self._calculate_key(self.start)

            if (k_min >= k_start and
                    self.rhs[self.start] == self.g[self.start]):
                break

            k_old, u = self._pop()
            if u is None:
                break  # OPEN empty

            k_new = self._calculate_key(u)
            if self._key_less(k_old, k_new):
                # key is out-of-date -> reinsert with new key
                self._insert(u, k_new)
            elif self.g[u] > self.rhs[u]:
                # need to decrease g
                self.g[u] = self.rhs[u]
                for p in self._neighbors(u):
                    self._update_vertex(p)
            else:
                # need to increase g
                g_old = self.g[u]
                self.g[u] = self.INF
                self._update_vertex(u)
                for p in self._neighbors(u):
                    self._update_vertex(p)

    def _update_vertex(self, u):
        """Update rhs(u) and its presence in OPEN."""
        if u != self.goal:
            min_rhs = self.INF
            for s in self._neighbors(u):
                if self.occupancy[s[1]][s[0]] == 1:
                    continue
                c = self._cost(u, s)
                val = c + self.g.get(s, self.INF)
                if val < min_rhs:
                    min_rhs = val
            self.rhs[u] = min_rhs

        # remove from OPEN if present
        if u in self.open_dict:
            del self.open_dict[u]
            # heap entry는 lazy 삭제 (pop할 때 무시)

        if self.g[u] != self.rhs[u]:
            self._insert(u, self._calculate_key(u))

    # -----------------------------
    # Priority queue helpers
    # -----------------------------

    def _insert(self, u, key):
        """Insert/update node u in OPEN with given key."""
        self.open_dict[u] = key
        k1, k2 = key
        x, y = u
        heapq.heappush(self.open_heap, (k1, k2, x, y))

    def _get_min_key(self):
        """Return minimum key in OPEN without removing it."""
        while self.open_heap:
            k1, k2, x, y = self.open_heap[0]
            u = (x, y)
            if u in self.open_dict:
                current_key = self.open_dict[u]
                # if stored key differs, heap entry is stale
                if current_key == (k1, k2):
                    return current_key
                else:
                    heapq.heappop(self.open_heap)
                    continue
            else:
                heapq.heappop(self.open_heap)
        return (self.INF, self.INF)

    def _pop(self):
        """Pop node with minimum key from OPEN."""
        while self.open_heap:
            k1, k2, x, y = heapq.heappop(self.open_heap)
            u = (x, y)
            if u in self.open_dict:
                key = self.open_dict[u]
                if key == (k1, k2):
                    del self.open_dict[u]
                    return (key, u)
                # else: stale entry, skip
        return ((self.INF, self.INF), None)

    # -----------------------------
    # Key / heuristic / neighbors
    # -----------------------------

    def _calculate_key(self, u):
        """Key(u) = [min(g, rhs) + h(start,u) + km, min(g,rhs)]."""
        g_u = self.g.get(u, self.INF)
        rhs_u = self.rhs.get(u, self.INF)
        val = min(g_u, rhs_u)
        return (val + self._heuristic(self.start, u) + self.km, val)

    def _heuristic(self, a, b):
        """Manhattan distance heuristic for 4-connected grid."""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return dx + dy  # Manhattan distance for 4-connected

    @staticmethod
    def _key_less(a, b):
        """Lexicographic compare for keys."""
        return a[0] < b[0] or (a[0] == b[0] and a[1] < b[1])

    def _neighbors(self, u):
        """4-connected neighbors inside grid bounds."""
        x, y = u
        # 4 directions: Right, Left, Up, Down (no diagonals)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                yield (nx, ny)

    def _cost(self, u, v):
        """Edge cost from u to v. Infinite if v is obstacle."""
        x, y = v
        if not self._in_bounds(v):
            return self.INF
        if self.occupancy[y][x] == 1:
            return self.INF
        return 1.0  # All moves cost 1 for 4-connected

    def _in_bounds(self, u):
        x, y = u
        return 0 <= x < self.width and 0 <= y < self.height
