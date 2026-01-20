# Comprehensive Objectives from Time Series

## Overview
The `compute_objectives_from_time_series` function now extracts 11 different objectives from episode time series data, enabling multi-faceted analysis of scenario safety.

## Objectives

### Primary Objectives
1. **crash_count** (0 or 1)
   - Binary indicator: 1 if any collision occurred, 0 otherwise
   - Most critical objective

2. **min_distance** (float, meters)
   - Minimum Euclidean distance between ego and any other vehicle
   - Across entire episode
   - Smaller = more dangerous

### Lane-Specific Distances
3. **min_distance_same_lane** (float)
   - Minimum distance to vehicles in the same lane as ego
   - Most relevant for head-on or rear-end collisions

4. **min_distance_left_lane** (float)
   - Minimum distance to vehicles in the left adjacent lane
   - Relevant for lane-change collisions

5. **min_distance_right_lane** (float)
   - Minimum distance to vehicles in the right adjacent lane
   - Relevant for lane-change collisions

### Temporal Objectives
6. **collision_time** (int, frames)
   - Time step at which collision occurred
   - If no collision: equals total_frames
   - Earlier collision = more dangerous

7. **time_to_close_call** (int, frames)
   - Time step at which distance first drops below 2.0m
   - If never: equals total_frames
   - Earlier close call = more dangerous

### Aggregate Objectives
8. **avg_distance** (float, meters)
   - Average minimum distance per frame
   - Captures overall scenario safety level

9. **max_vehicles_nearby** (int)
   - Maximum number of vehicles within 10m at any frame
   - Higher = more congestion = more risk

10. **frames_with_close_call** (int)
    - Number of frames where min_distance < 3.0m
    - Counts "near-miss" situations

11. **close_call_ratio** (float, 0-1)
    - Proportion of frames with close calls
    - Normalized: frames_with_close_call / total_frames

## Fitness Function Strategy

The fitness function combines these objectives into a single scalar to minimize:

```
if crash_count == 1:
    fitness = -1.0  (best possible)
elif min_distance < 2.0:
    fitness = -0.5 to -0.25  (close call penalty)
elif close_call_ratio > 0.3:
    fitness = -0.1 to 0.0  (many close calls)
else:
    fitness = 0.6 * min_distance + 0.3 * avg_distance - 0.1 * max_vehicles_nearby
```

### Fitness Hierarchy
1. **Crashes** (-1.0): Best outcome
2. **Close calls** (-0.5 to -0.25): Very dangerous
3. **Many near-misses** (-0.1 to 0.0): Dangerous
4. **Distance-based** (0+): Safer scenarios

## Usage in Hill Climbing

The algorithm uses these objectives to:
1. **Identify crashes**: Primary goal
2. **Find close calls**: Secondary goal when crashes not found
3. **Minimize distances**: Tertiary goal for safer scenarios
4. **Track progress**: Monitor multiple metrics simultaneously

## Example Output

```
Initial: fitness=-0.234, crashed=0, min_dist=2.15, avg_dist=4.32, vehicles_nearby=3
Iter 0: Improved to -0.456, crashed=0, min_dist=1.87, close_calls=12
Iter 1: Improved to -0.892, crashed=0, min_dist=0.95, close_calls=28
💥 Crash found at iteration 2!
```

## Benefits

- **Multi-objective insight**: See crash, distance, and congestion metrics
- **Better convergence**: Multiple objectives guide search more effectively
- **Interpretability**: Easy to understand what makes a scenario dangerous
- **Flexibility**: Can weight objectives differently for different goals
