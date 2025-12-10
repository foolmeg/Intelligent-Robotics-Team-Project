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
