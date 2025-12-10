## What I Implemented Myself  

### 1. Static A* Path Planner on a Known Grid Map  

I implemented a classic grid-based **A\*** planner that assumes a known, static map:

- The planner runs on a 2D occupancy / cost grid.  
- Start and goal states are given in grid or world coordinates and converted into grid indices.  
- A standard A\* search is performed once on the static grid to compute the full path.  
- The output is a discrete sequence of grid cells from start to goal, which can be converted back to world coordinates by other modules.  

This A\* implementation serves as the **baseline** for comparing with the D\* Lite algorithms.

---

### 2. D* Lite Core on a Known Map  

I implemented the core logic of **D\* Lite** for incremental path planning on a known grid:

- Maintains `g` and `rhs` values for each grid cell.  
- Defines the D\* Lite key function and uses a priority queue to manage the open list.  
- Implements the main routines:
  - `update_vertex()` to update local consistency when a cell cost changes.  
  - `compute_shortest_path()` to incrementally repair the shortest-path tree from the goal.  
- Supports extracting a shortest path from the current start state to the fixed goal state using the maintained value functions.  

This version already behaves like D\* Lite on a static map and provides the algorithmic foundation for online replanning.

---

### 3. D* Lite with Online Replanning Under Map Updates  

I extended the D\* Lite implementation to support **online replanning** when the grid map changes:

- The planner is initialized with:
  - A grid map (with free/occupied cells).  
  - The goal state on the grid.  
- During execution, a set of **changed grid cells** (for example, when new obstacles are detected or cells become free) is passed to the planner.  
- For each batch of changed cells:
  - Their costs or occupancy states are updated in the internal grid.  
  - `update_vertex()` is called for those affected states.  
  - `compute_shortest_path()` is run to repair the shortest-path tree incrementally.  
- After each repair, the planner can provide a new path from the current start to the goal without recomputing everything from scratch.  

This realizes the core idea of D\* Lite: **incremental path planning in a changing environment**, with efficient reuse of previous search results.

---

## What Pre-Programmed Packages Were Used  

### Programming Environment and Libraries  

For the planning logic itself, I used only basic Python features and standard libraries:

- **Python data structures**  
  - Lists, dictionaries, and sets to store states, neighbors, and map data.  
- **`heapq` / `collections`**  
  - To implement the priority queue required by D\* Lite.  
- **`math`**  
  - For heuristic calculations (e.g., Manhattan or Euclidean distance) and basic numeric operations.  
