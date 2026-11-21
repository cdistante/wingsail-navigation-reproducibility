"""Constants used by the imported legacy harmonic planners.

The old workspace pulled these values from ``inputs_var.py``.  Keeping only
planner-relevant constants here removes the dependency on simulator_2 and its
mutable GUI/simulation state.
"""

EMPTY_CELL = 0
OBSTACLE_CELL = 1
BOAT_CELL = 2
GOAL_CELL = 3
WAYPOINT_CELL = 4
MOVE_CELL = 5
PATH_CELL = 6
WIND_OBSTACLE_CELL = 7
VESSEL_CELL = 8
VESSEL2_CELL = 9
LOCAL_GRID_SIZE = 41
