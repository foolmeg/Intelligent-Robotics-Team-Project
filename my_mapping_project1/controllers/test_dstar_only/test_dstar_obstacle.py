from d_start_lite import DStarLite

WIDTH = 200
HEIGHT = 200

START = (100, 100)
GOAL  = (150, 50)

# 빈 occupancy grid
occupancy = [[0 for _ in range(WIDTH)] for _ in range(HEIGHT)]

# D* 초기화
dstar = DStarLite(WIDTH, HEIGHT, START, GOAL, occupancy)

# -----------------------------
# 1) 초기 경로 계산
# -----------------------------
dstar.replan(START)
path1 = dstar.get_shortest_path(START)

print("=== Initial Path ===")
print("Path length:", len(path1))
print("First:", path1[:10])
print("Last:", path1[-10:])
print()

# -----------------------------
# 2) 장애물 추가 (벽 세우기)
# -----------------------------
# (120, 90) ~ (120, 110) 까지 벽을 세움
print("=== Adding a wall at x=120, y=[90, 110] ===")
for y in range(90, 111):
    dstar.set_obstacle(120, y)

# -----------------------------
# 3) 리플랜 후 새로운 경로 계산
# -----------------------------
dstar.replan(START)
path2 = dstar.get_shortest_path(START)

print("=== New Path After Obstacle ===")
if path2 is None:
    print("No new path! Something went wrong.")
else:
    print("New Path length:", len(path2))
    print("First:", path2[:10])
    
    # Check where it crosses x=120
    crossing = [p for p in path2 if p[0] == 120]
    print("Crossing x=120 at:", crossing)
    
    print("Last:", path2[-10:])
