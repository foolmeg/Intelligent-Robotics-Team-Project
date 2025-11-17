import heapq
import matplotlib.pyplot as plt
import random
import time


class DStarSweeper:
    def __init__(self, grid_size=20, start=(0, 0), goal=(19, 19), init_obstacle_rate=0.08):
        # 地图与核心参数
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.current_pos = start
        self.grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]  # 0=passable, 1=static obstacle, 2=dynamic obstacle

        # D*算法核心数据结构
        self.open_list = []
        self.g = {}
        self.h = {}
        self.path = []

        # 传感器参数
        self.sensor_range = 1

        # 初始化
        self._init_costs()
        self._add_static_obstacles(init_obstacle_rate)

        # 可视化配置
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self._init_visualization()

    def _init_costs(self):
        """反向初始化：从终点计算所有节点的h(n)，初始化g(n)"""
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.h[(x, y)] = abs(x - self.goal[0]) + abs(y - self.goal[1])
                self.g[(x, y)] = float('inf')
        self.g[self.goal] = 0
        heapq.heappush(self.open_list, (self.g[self.goal] + self.h[self.goal], self.goal))

    def _add_static_obstacles(self, rate):
        """生成静态障碍物（避免覆盖起点、终点）"""
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) not in [self.start, self.goal] and random.random() < rate:
                    self.grid[x][y] = 1

    def _sense_obstacles(self):
        """模拟传感器：感知当前位置周围3x3区域的障碍物"""
        x, y = self.current_pos
        sensed = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx = x + dx
                ny = y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if self.grid[nx][ny] in [1, 2]:
                        sensed.append((nx, ny))
        return sensed

    def _add_dynamic_obstacle(self):
        """生成动态障碍物（规则：不阻挡终点，不生成在机器人感知范围内）"""
        max_attempts = 50
        for _ in range(max_attempts):
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            if (x, y) not in [self.start, self.goal, self.current_pos] and self.grid[x][y] == 0:
                dx = abs(x - self.current_pos[0])
                dy = abs(y - self.current_pos[1])
                if dx > self.sensor_range or dy > self.sensor_range:
                    self.grid[x][y] = 2
                    print(f"\n⚠️  Dynamic obstacle detected by sensor: ({x}, {y})")
                    return (x, y)
        return None

    def _get_valid_neighbors(self, node):
        """获取节点的四邻域（边界+可通行检查）"""
        x, y = node
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        valid = []
        for nx, ny in neighbors:
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                if self.grid[nx][ny] == 0:
                    valid.append((nx, ny))
        return valid

    def _update_node_cost(self, node):
        """反向传播更新代价（D*核心：仅更新受影响节点）"""
        x, y = node
        for neighbor in self._get_valid_neighbors(node):
            nx, ny = neighbor
            new_g = self.g[(x, y)] + 1
            if new_g < self.g[(nx, ny)]:
                self.g[(nx, ny)] = new_g
                heapq.heappush(self.open_list, (new_g + self.h[(nx, ny)], (nx, ny)))

    def plan_path(self):
        """D*路径规划：从终点反向更新，直到起点"""
        while self.open_list:
            current_f, current_node = heapq.heappop(self.open_list)
            if current_f > self.g[current_node] + self.h[current_node]:
                continue
            if current_node == self.start:
                return self._smooth_path(self._reconstruct_path())
            self._update_node_cost(current_node)
        return None

    def _reconstruct_path(self):
        """从起点回溯到终点，生成原始路径"""
        path = []
        current = self.start
        while current != self.goal:
            path.append(current)
            neighbors = self._get_valid_neighbors(current)
            if not neighbors:
                return None
            current = min(neighbors, key=lambda n: self.g[n] + self.h[n])
        path.append(self.goal)
        return path

    def _smooth_path(self, path):
        """路径平滑：移除连续重复方向的节点（可选，提升移动流畅度）"""
        if not path or len(path) <= 2:
            return path
        smooth = [path[0]]
        for i in range(1, len(path) - 1):
            prev = path[i - 1]
            curr = path[i]
            next_ = path[i + 1]
            if (prev[0] == curr[0] == next_[0]) or (prev[1] == curr[1] == next_[1]):
                continue
            smooth.append(curr)
        smooth.append(path[-1])
        return smooth

    def _init_visualization(self):
        """初始化可视化参数"""
        self.ax.set_xlim(-1, self.grid_size)
        self.ax.set_ylim(-1, self.grid_size)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xticks(range(self.grid_size))
        self.ax.set_yticks(range(self.grid_size))
        self.ax.set_title("Sweeping Robot D* Dynamic Path Planning")

    def visualize(self, step):
        """实时可视化当前状态"""
        self.ax.clear()
        self._init_visualization()

        # 绘制障碍物：静态（灰色）、动态（红色）
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.grid[x][y] == 1:
                    self.ax.scatter(x, y, c='#888888', s=300, marker='s', alpha=0.7)
                elif self.grid[x][y] == 2:
                    self.ax.scatter(x, y, c='#ff3333', s=300, marker='s', alpha=0.8)

        # 绘制传感器范围（蓝色虚线框）
        x, y = self.current_pos
        sx1, sx2 = x - self.sensor_range, x + self.sensor_range
        sy1, sy2 = y - self.sensor_range, y + self.sensor_range
        self.ax.plot([sx1, sx2, sx2, sx1, sx1], [sy1, sy1, sy2, sy2, sy1],
                    c='#0099ff', linestyle='--', linewidth=2, alpha=0.5)

        # 绘制路径（蓝色实线）
        if self.path:
            path_x = [p[0] for p in self.path]
            path_y = [p[1] for p in self.path]
            self.ax.plot(path_x, path_y, c='#0066ff', linewidth=3, marker='o', markersize=4)

        # 绘制关键节点：起点（绿色）、终点（黄色）、机器人（黑色）
        self.ax.scatter(self.start[0], self.start[1], c='#33cc33', s=500, marker='*', label='Start')
        self.ax.scatter(self.goal[0], self.goal[1], c='#ffff33', s=500, marker='*', label='Goal')
        self.ax.scatter(self.current_pos[0], self.current_pos[1], c='#000000', s=400, marker='^', label='Robot')

        self.ax.legend(fontsize=12)
        self.ax.set_title(f"Sweeping Robot D* Dynamic Path Planning (Step {step} | Position: {self.current_pos})")
        plt.pause(0.3)

    def run(self):
        """机器人运行主逻辑"""
        step = 0
        print("🚀 Sweeping robot started, initializing path...")

        # 初始路径规划
        self.path = self.plan_path()
        if not self.path:
            print("❌ Initialization failed: No feasible path!")
            return

        # 主循环：移动→感知→避障→重规划
        while self.current_pos != self.goal:
            step += 1
            self.visualize(step)

            # 沿规划路径移动一步
            curr_idx = self.path.index(self.current_pos)
            next_pos = self.path[curr_idx + 1]
            self.current_pos = next_pos
            print(f"📌 Step {step}: Robot moves to ({self.current_pos[0]}, {self.current_pos[1]})")

            # 15%概率生成动态障碍物
            if random.random() < 0.15:
                obstacle = self._add_dynamic_obstacle()
                if obstacle:
                    if any(self.grid[p[0]][p[1]] in [1, 2] for p in self.path):
                        print("🔄 Path blocked, starting D* incremental replanning...")
                        self.open_list = []
                        heapq.heappush(self.open_list, (self.g[self.goal] + self.h[self.goal], self.goal))
                        self.path = self.plan_path()
                        if not self.path:
                            print("❌ Replanning failed: No feasible path!")
                            return

        # 到达终点
        step += 1
        self.visualize(step)
        print("🎉 Mission completed! Robot successfully reached the goal!")
        plt.show()


# 运行入口
if __name__ == "__main__":
    robot = DStarSweeper(
        grid_size=20,
        start=(0, 0),
        goal=(19, 19),
        init_obstacle_rate=0.08
    )
    robot.run()