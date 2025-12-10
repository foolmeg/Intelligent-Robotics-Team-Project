# Pioneer 3-DX Navigation System for Webots

A complete navigation pipeline with D* Lite global planning and DWA local obstacle avoidance.

## Quick Setup

1. Copy all `.py` files to: `<webots_project>/controllers/pioneer_navigation/`
2. Set robot controller to `pioneer_navigation`
3. Make sure robot has `supervisor="TRUE"` in the world file
4. Run simulation

## DWA PARAMETERS - WHERE TO TUNE

### In `main_controller.py` (lines ~75-95):

```python
# ============================================================
# DWA PARAMETER TUNING - ADJUST THESE FOR YOUR SCENARIO
# ============================================================
# Weights for objective function
self.local_planner.set_weights(
    heading=0.6,      # Lower = less aggressive toward goal
    clearance=2.5,    # Higher = more cautious around obstacles
    velocity=0.2,     # Lower = doesn't prioritize speed
    path=1.2          # Higher = follows global path more closely
)

# Safety parameters
self.local_planner.set_safety_params(
    min_clearance=0.4,   # Minimum distance to obstacles (meters)
    robot_radius=0.3     # Robot footprint radius
)

# Robot kinematic limits
self.local_planner.set_robot_params(
    max_v=0.35,          # Max forward speed (m/s)
    max_w=1.2,           # Max turn rate (rad/s)
    max_v_accel=0.3,     # Linear acceleration
    max_w_accel=2.0      # Angular acceleration
)
```

### In `dwa_planner.py` (lines ~29-66):

```python
# ============================================================
# ROBOT KINEMATIC PARAMETERS
# ============================================================
self.max_linear_velocity = 0.4      # m/s - max forward speed
self.min_linear_velocity = -0.1     # m/s - allow backing up
self.max_angular_velocity = 1.5     # rad/s - max turn rate

# ============================================================
# DWA SAMPLING PARAMETERS
# ============================================================
self.v_resolution = 0.02            # Linear velocity step size
self.w_resolution = 0.05            # Angular velocity step size

# ============================================================
# TRAJECTORY PREDICTION PARAMETERS
# ============================================================
self.predict_time = 3.0             # Seconds to simulate forward

# ============================================================
# OBJECTIVE FUNCTION WEIGHTS (TUNE THESE!)
# ============================================================
self.heading_weight = 0.8           # Alignment with goal direction
self.clearance_weight = 2.0         # Distance from obstacles
self.velocity_weight = 0.3          # Preference for faster motion
self.path_weight = 1.0              # Following the global path

# ============================================================
# SAFETY PARAMETERS
# ============================================================
self.min_clearance = 0.35           # Minimum distance to obstacles
self.obstacle_detection_range = 3.0 # Range to consider for obstacles
```

## Tuning Tips

### Robot won't go around obstacles:
- Increase `clearance_weight` to 3.0 or higher
- Decrease `heading_weight` to 0.4
- Increase `min_clearance` to 0.5

### Robot is too slow/cautious:
- Decrease `clearance_weight` to 1.5
- Increase `velocity_weight` to 0.5
- Increase `max_linear_velocity`

### Robot oscillates:
- Decrease `w_resolution` to 0.03
- Increase `predict_time` to 4.0
- Decrease `max_angular_velocity`

### Path planning fails:
- Check inflation radius in `occupancy_grid.py` (default: 6 cells = 30cm)
- Reduce `inflation_radius` if environment is tight
- Increase `initialize_map()` scans

## Files

- `main_controller.py` - Main navigation controller
- `dwa_planner.py` - Dynamic Window Approach local planner
- `dstar_planner.py` - D* Lite global path planner
- `occupancy_grid.py` - LiDAR-based occupancy mapping
- `utils.py` - Utility functions

## Goal Position

Change the goal in `main_controller.py` line ~245:

```python
goal = (-2.0, -2.0)  # Change to your desired (x, y)
```
