from d_start_lite import DStarLite

WIDTH = 200
HEIGHT = 200

START = (100, 100)
GOAL  = (150, 50)

# 빈 occupancy grid 생성
occupancy = [[0 for _ in range(WIDTH)] for _ in range(HEIGHT)]

# D* 초기화
dstar = DStarLite(WIDTH, HEIGHT, START, GOAL, occupancy)

dstar.replan(START)
path = dstar.get_shortest_path(START)

print("Path length:", None if path is None else len(path))
print("First nodes:", path[:10] if path else None)
print("Last nodes:", path[-10:] if path else None)
