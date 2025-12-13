
from controller import Robot
import heapq
import math
from math import atan2,pi,sqrt

class DStarLite:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.start = start
        self.goal = goal
        self.s_last = start
        self.km = 0
        self.U = [] 
        self.g = {}  
        self.rhs = {}  
        for i in range(self.rows):
            for j in range(self.cols):
                pos = (i, j)
                self.g[pos] = float('inf')
                self.rhs[pos] = float('inf')
        self.rhs[self.goal] = 0
        heapq.heappush(self.U, (self.calculate_key(self.goal), self.goal))

    def calculate_key(self, s):
        
        return (min(self.g[s], self.rhs[s]) + self.h(self.start, s) + self.km, min(self.g[s], self.rhs[s]))

    def h(self, s1, s2):
        
        return math.hypot(s1[0] - s2[0], s1[1] - s2[1])

    def get_neighbors(self, s):
        
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = s[0] + dx, s[1] + dy
            if 0 <= nx < self.rows and 0 <= ny < self.cols and self.grid[nx][ny] == 0:
                neighbors.append((nx, ny))
        return neighbors

    def cost(self, s1, s2):
        
        return self.h(s1, s2)

    def update_vertex(self, u):
        
        if u != self.goal:
            self.rhs[u] = min((self.g[s] + self.cost(u, s) for s in self.get_neighbors(u)), default=float('inf'))
        self.U = [pair for pair in self.U if pair[1] != u]
        heapq.heapify(self.U)
        if self.g[u] != self.rhs[u]:
            heapq.heappush(self.U, (self.calculate_key(u), u))

    def compute_shortest_path(self):
        
        while self.U and (self.U[0][0] < self.calculate_key(self.start) or self.rhs[self.start] != self.g[self.start]):
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
                for s in self.get_neighbors(u) + [u]:
                    self.update_vertex(s)

    def rescan(self, changed_cells):
        
        for u in changed_cells:
            for s in self.get_neighbors(u) + [u]:
                self.update_vertex(s)

    def plan(self, changed=[]):
        
        self.km += self.h(self.s_last, self.start)
        self.s_last = self.start
        self.rescan(changed)
        self.compute_shortest_path()
        
        path = []
        current = self.start
        seen = set()
        while current != self.goal:
            if current in seen:
                return None  
            seen.add(current)
            path.append(current)
            current = min(self.get_neighbors(current), key=lambda s: self.g[s] + self.cost(current, s), default=None)
            if current is None:
                return None
        path.append(self.goal)
        return path

MAX_SPEED = 5.24

robot = Robot()

# Get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

leftWheel = robot.getDevice('left wheel')
rightWheel = robot.getDevice('right wheel')

leftWheel.setPosition(float('inf'))
rightWheel.setPosition(float('inf'))

leftWheel.setVelocity(0)
rightWheel.setVelocity(0)

gps = robot.getDevice("gps")
imu = robot.getDevice("imu")

gps.enable(timestep)
imu.enable(timestep)

box_size = 0.5
map_size = 18

def from_w2m():
    
    gps_values = gps.getValues()
    
    x = gps_values[2] + 4.25
    y = -gps_values[0] + 4.25
    
    r = int(y/box_size)
    c = int(x/box_size) 
    
    return r,c

def from_m2w(num):
    
    
    r = int(num/map_size)
    c = num%map_size

    print(r,c)

    x = c*box_size - 4.25
    y = 4.25 - r*box_size 

    return x,y

def trans(degree):

    if degree < -pi:
        degree += 2*pi
    if degree > pi:
        degree -= 2*pi
    
    return degree


def swapped(t):

    a, b = t 
    return (b, a) 

def find_target(target_x,target_y):

    while True:

        robot.step(timestep)

        linear_speed = 0.0

        (x,y,z) = gps.getValues()

        tar = atan2(target_y - y,target_x - x)

        yaw = imu.getRollPitchYaw()[2]

        degree = tar - yaw

        degree = trans(degree)

        d = sqrt((target_x - x)**2 + (target_y - y)**2)

        #print(d,degree)
        if d < 0.2:
            break

        angular_speed = 2.0 * degree
        if abs(degree) < 0.15:
            linear_speed = 2.0

        left_speed = linear_speed - angular_speed 
        right_speed = linear_speed + angular_speed 

        leftWheel.setVelocity(left_speed)
        rightWheel.setVelocity(right_speed)



grid = [[0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],
        [0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0],
        [0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],
        [0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,0],
        [0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],
        [0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],
        [0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],
        [0,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,1,0],
        [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
        [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
        [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
        [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]]

#grid = grid.T
grid = [[grid[j][i] for j in range(len(grid))] for i in range(len(grid[0]))]


start = (15, 3) 
goal = (2, 14) #2 14

start = swapped(start)
goal = swapped(goal)

dstar = DStarLite(grid, start, goal)


# Main loop.
while robot.step(timestep) != -1:

    #find_target(1.0,1.0)
    
    path = dstar.plan()
    print(path)

    for (i,(x,y)) in enumerate(path):
        print(i,x,y)
    

        x = x*box_size - 4.25
        y = 4.25 - y*box_size 
        find_target(x,y)

    leftWheel.setVelocity(0)
    rightWheel.setVelocity(0)

    while robot.step(timestep) != -1:

        print("pass")

