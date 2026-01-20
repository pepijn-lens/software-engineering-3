# Hill Climbing Implementation Summary

## Overview
Implemented a Hill Climbing algorithm for finding failure-inducing scenarios in RL agents. The algorithm uses multiple-parameter mutation and is biased towards crash-prone configurations.

## Key Features

### 1. Objective Computation (`compute_objectives_from_time_series`)
- **crash_count**: 1 if collision occurred, 0 otherwise
- **min_distance**: Minimum distance between ego vehicle and any other vehicle across the episode
- Handles missing ego/other vehicle data gracefully

### 2. Fitness Function (`compute_fitness`)
- Crashes get fitness = -1.0 (best possible)
- Non-crashes get fitness = min_distance (smaller is better)
- Ensures any crash is strictly better than any non-crash

### 3. Multiple-Parameter Mutation (`mutate_config`)
- Mutates 1-3 parameters randomly per neighbor
- **Crash-biased mutations**:
  - `vehicles_count`: 70% chance to increase (more vehicles = higher crash risk)
  - `initial_spacing`/`ego_spacing`: 70% chance to decrease (tighter spacing = higher crash risk)
  - `lanes_count`: 50/50 increase/decrease
  - `initial_lane_id`: Random valid lane
- Maintains parameter bounds and lane consistency
- Returns copy (doesn't modify original)

### 4. Hill Climbing Algorithm (`hill_climb`)
- **Initialization**: Random scenario biased towards crash-prone settings
- **Search Loop**: 
  - Generate N neighbors per iteration
  - Evaluate all neighbors
  - Accept best neighbor if it improves fitness
  - Track global best throughout search
- **Early stopping**: Stops when crash is found
- **Returns**: Best configuration, objectives, fitness, seed, history, evaluations

### 5. Search Class (`HillClimbingSearch`)
- Compatible interface with `RandomSearch`
- Automatically records crash videos
- Configurable iterations and neighbors per iteration

## Usage

```python
from search.hill_climbing import HillClimbingSearch

search = HillClimbingSearch(env_id, base_cfg, param_spec, policy, defaults)
crashes = search.run_search(iterations=50, neighbors_per_iter=8, seed=42)
```

## Strategy
The implementation prioritizes finding crashes by:
1. Starting with crash-prone initial configurations
2. Biasing mutations towards dangerous scenarios (more vehicles, less spacing)
3. Using greedy hill climbing to quickly converge to local optima
4. Evaluating multiple neighbors per iteration for better exploration