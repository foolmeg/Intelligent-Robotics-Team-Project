# Webots Pioneer 3-DX Path Planning (A* & D* Lite)

This repository is a **path-planning branch** of a larger project.  
It uses the Webots simulator and the **Pioneer 3-DX** mobile robot.

It contains three subfolders:

- `astar/`   – Early version: static A* path planning  
- `dstar 1/` – D* Lite structure but no real dynamic obstacle handling  
- `dstar 2/` – D* Lite + known map + online replanning (with dynamic obstacles)

> Note: This branch focuses only on motion planning and controllers, not on other project modules.

---

## Repository Layout

```text
.
├── astar/
│   ├── astar_controller/   # A* controller (Webots controller folder)
│   └── astar_map.wbt       # World for the A* experiments
├── dstar 1/
│   ├── my_controller.py    # D* Lite structure (no real dynamic avoidance)
│   └── w3.wbt              # World for the first D* Lite test
├── dstar 2/
│   ├── dstar_controller/   # D* Lite + online replanning controller
│   ├── moving_obstacle/    # Controller for a moving obstacle
│   └── dstar_map.wbt       # World for D* Lite with dynamic obstacle
└── README.md

1. astar/ – Static A* Path Planning (Baseline)
The astar folder contains the baseline implementation using a static A* algorithm.
The world astar_map.wbt defines the environment (walls and obstacles).
The astar_controller builds an occupancy grid from the world geometry at startup.
At the beginning of the simulation, the controller runs A* once to compute a complete path from start to goal.
During the main loop, the robot simply follows this fixed path using a differential-drive controller and basic speed/heading control.
There is no global replanning when the environment changes.

2. dstar 1/ – D* Lite Structure (Static Map, No True Dynamic Obstacles)
The dstar 1 folder contains a first D* Lite-based implementation.
Controller: my_controller.py
World: w3.wbt
The code includes the core D* Lite data structures and functions:
g / rhs values
Priority queue U
Heuristic h
Incremental update routines such as update_vertex() and compute_shortest_path()
However, in this version:
The start state is not continuously updated to the robot’s current grid cell during execution.
The occupancy grid is essentially static (no sensor-based updates).
plan() is typically called with an empty set of changed cells, so no meaningful incremental replanning happens.
As a result, even though the algorithm looks like D* Lite, it still behaves like static planning with no real dynamic obstacle avoidance

3. dstar 2/ – D* Lite with Known Map and Online Replanning (Dynamic Obstacles)
The dstar 2 folder contains the improved implementation with full D* Lite + online replanning.
World: dstar_map.wbt
Main robot controller: dstar_controller
Dynamic obstacle controller: moving_obstacle
Compared to dstar 1, this version:
Still builds an initial occupancy grid from the static walls in dstar_map.wbt.
Initializes D* Lite at the goal cell (rhs(goal) = 0, others = ∞, goal inserted into the priority queue).
In every control step:
Converts the robot’s current world pose to a grid cell and treats that cell as the new start.
Uses sonar / distance / contact information to update the occupancy grid (e.g., mark newly detected obstacles as occupied).
Passes the set of changed cells into the D* Lite update routines and calls compute_shortest_path() to incrementally replan only where needed.
Extracts the updated shortest path from the current start to the goal and drives the robot along this path.
With the moving_obstacle controller attached to an obstacle node, the world contains dynamic obstacles. When the obstacle moves or is newly detected, the planner updates its map and replans around it.
In this way, dstar 2 is the first folder that implements true D* Lite behavior with dynamic replanning.

Summary Comparison
astar/
Pure static A* planning.
Single global plan computed at startup.
No genuine dynamic obstacle handling or online replanning.
dstar 1/
Uses D* Lite data structures and functions.
Start state and map are effectively static.
Behaves like static planning – no true dynamic obstacle avoidance.
dstar 2/
Full D* Lite pipeline: known map + sensor-based updates + incremental replanning.
Supports dynamic obstacles (with moving_obstacle) and online path updates.
