"""
Assignment 3 — Scenario-Based Testing of an RL Agent (Hill Climbing)

You MUST implement:
    - compute_objectives_from_time_series
    - compute_fitness
    - mutate_config
    - hill_climb

DO NOT change function signatures.
You MAY add helper functions.

Goal
----
Find a scenario (environment configuration) that triggers a collision.
If you cannot trigger a collision, minimize the minimum distance between the ego
vehicle and any other vehicle across the episode.

Black-box requirement
---------------------
Your evaluation must rely only on observable behavior during execution:
- crashed flag from the environment
- time-series data returned by run_episode (positions, lane_id, etc.)
No internal policy/model details beyond calling policy(obs, info).
"""

import copy
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from tqdm import trange
from envs.highway_env_utils import run_episode, record_video_episode


# ============================================================
# 1) OBJECTIVES FROM TIME SERIES
# ============================================================

def compute_objectives_from_time_series(time_series: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute comprehensive objective values from the recorded time-series.

    The time_series is a list of frames. Each frame typically contains:
      - frame["crashed"]: bool
      - frame["ego"]: dict or None, e.g. {"pos":[x,y], "lane_id":..., "length":..., "width":...}
      - frame["others"]: list of dicts with positions, lane_id, etc.

    Return a dictionary with multiple objectives for multi-objective optimization.
    """
    crash_count = 0
    min_distance = float('inf')
    min_distance_same_lane = float('inf')
    min_distance_left_lane = float('inf')
    min_distance_right_lane = float('inf')
    avg_distance = 0.0
    collision_time = float('inf')
    time_to_close_call = float('inf')  # Time until distance < 2.0
    max_vehicles_nearby = 0  # Max number of vehicles within 10m
    total_frames = len(time_series)
    frames_with_close_call = 0  # Frames where min_distance < 3.0
    
    for frame_idx, frame in enumerate(time_series):
        if frame.get("crashed", False):
            crash_count = 1
            if collision_time == float('inf'):
                collision_time = frame_idx
            
        ego = frame.get("ego")
        others = frame.get("others", [])
        
        if ego and others:
            ego_pos = np.array(ego["pos"])
            ego_lane = ego.get("lane_id")
            vehicles_nearby = 0
            frame_min_distance = float('inf')
            
            for other in others:
                other_pos = np.array(other["pos"])
                other_lane = other.get("lane_id")
                
                # Euclidean distance between centers
                distance = np.linalg.norm(ego_pos - other_pos)
                frame_min_distance = min(frame_min_distance, distance)
                min_distance = min(min_distance, distance)
                
                # Track vehicles within 10m
                if distance < 10.0:
                    vehicles_nearby += 1
                
                # Lane-specific distances
                if ego_lane is not None and other_lane is not None:
                    if ego_lane == other_lane:
                        min_distance_same_lane = min(min_distance_same_lane, distance)
                    elif other_lane == ego_lane - 1:
                        min_distance_left_lane = min(min_distance_left_lane, distance)
                    elif other_lane == ego_lane + 1:
                        min_distance_right_lane = min(min_distance_right_lane, distance)
                
                # Time to close call (distance < 2.0)
                if distance < 2.0 and time_to_close_call == float('inf'):
                    time_to_close_call = frame_idx
            
            # Track close calls (distance < 3.0)
            if frame_min_distance < 3.0:
                frames_with_close_call += 1
            
            max_vehicles_nearby = max(max_vehicles_nearby, vehicles_nearby)
            avg_distance += frame_min_distance if frame_min_distance != float('inf') else 0
    
    # Normalize average distance
    if total_frames > 0:
        avg_distance /= total_frames
    else:
        avg_distance = 100.0
    
    # Handle cases where no other vehicles were encountered
    if min_distance == float('inf'):
        min_distance = 100.0
    if min_distance_same_lane == float('inf'):
        min_distance_same_lane = 100.0
    if min_distance_left_lane == float('inf'):
        min_distance_left_lane = 100.0
    if min_distance_right_lane == float('inf'):
        min_distance_right_lane = 100.0
    if time_to_close_call == float('inf'):
        time_to_close_call = total_frames
    if collision_time == float('inf'):
        collision_time = total_frames
        
    return {
        # Primary objectives
        "crash_count": crash_count,
        "min_distance": min_distance,
        
        # Lane-specific distances
        "min_distance_same_lane": min_distance_same_lane,
        "min_distance_left_lane": min_distance_left_lane,
        "min_distance_right_lane": min_distance_right_lane,
        
        # Temporal objectives
        "collision_time": collision_time,
        "time_to_close_call": time_to_close_call,
        
        # Aggregate objectives
        "avg_distance": avg_distance,
        "max_vehicles_nearby": max_vehicles_nearby,
        "frames_with_close_call": frames_with_close_call,
        "close_call_ratio": frames_with_close_call / total_frames if total_frames > 0 else 0.0,
    }


def compute_fitness(objectives: Dict[str, Any]) -> float:
    """
    Convert objectives into ONE scalar fitness value to MINIMIZE.

    Requirement:
    - Any crashing scenario must be strictly better than any non-crashing scenario.

    Strategy:
    - Crashes get fitness = -1.0 (best)
    - Close calls (min_distance < 2.0) get fitness = -0.5 to -0.1
    - Otherwise use combination of distance metrics
    """
    if objectives["crash_count"] == 1:
        return -1.0  # Best possible fitness (crash found)
    
    # Penalize close calls (distance < 2.0)
    if objectives["min_distance"] < 2.0:
        return -0.5 + (objectives["min_distance"] / 4.0)  # Range: -0.5 to -0.25
    
    # Penalize scenarios with many close calls
    if objectives["close_call_ratio"] > 0.3:
        return -0.1 + (objectives["min_distance"] / 100.0)
    
    # Primary objective: minimize minimum distance
    # Secondary: minimize average distance
    # Tertiary: maximize vehicles nearby (more interaction = more risk)
    fitness = (
        objectives["min_distance"] * 0.6 +
        objectives["avg_distance"] * 0.3 -
        objectives["max_vehicles_nearby"] * 0.1
    )
    
    return fitness


# ============================================================
# 2) MUTATION / NEIGHBOR GENERATION
# ============================================================

def mutate_config(
    cfg: Dict[str, Any],
    param_spec: Dict[str, Any],
    rng: np.random.Generator
) -> Dict[str, Any]:
    """
    Generate ONE neighbor configuration by mutating the current scenario.
    
    IMPROVED LOGIC:
    - Reduced Random Resets: 10% was too high and caused easy scenarios (Eval 5, Eval 7). Reduced to 2%.
    - Biased Random Walk: 'vehicles_count' trends up, 'lanes_count' neutral, but both can move in either direction.
      This allows finding specific crash configs (like 10 lanes, 59 vehicles) that strict monotonic logic missed.
    """
    new_cfg = copy.deepcopy(cfg)
    
    keys = list(param_spec.keys())
    # Mutate 1 to n parameters (more aggression = more parameters changed)
    num_mutations = rng.integers(1, min(5, len(keys) + 1))
    params_to_mutate = rng.choice(keys, size=num_mutations, replace=False)
    
    for param in params_to_mutate:
        
        spec = param_spec[param]
        curr_val = new_cfg.get(param, spec["min"])
        min_val = spec["min"]
        max_val = spec["max"]
        
        # 2% chance for random reset (Reduced from 10% to prevent sabotaging progress)
        if rng.random() < 0.02:
            if spec["type"] == "int":
                new_cfg[param] = int(rng.integers(min_val, max_val + 1))
            else:
                new_cfg[param] = float(rng.uniform(min_val, max_val))
            continue

        # Normal Mutation Logic
        if spec["type"] == "int":
            step = 0
            if param == "vehicles_count":
                # Upward Bias, but allow backing off
                # [-2, 1, 2, 3] -> mostly increase, but can decrease
                step = rng.choice([-2, 1, 2, 3])
            
            elif param == "lanes_count":
                # Neutral/Exploratory
                # Allow increasing lanes to find high-speed/chaotic 10-lane scenarios
                step = rng.choice([-3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8])
                
            elif param == "initial_lane_id":
                # 30% chance: Jump to a random lane (to find pockets of traffic)
                if rng.random() < 0.3:
                     max_lane = new_cfg.get("lanes_count", 4) - 1
                     new_cfg[param] = rng.integers(0, max_lane + 1)
                     continue
                else:
                    step = rng.choice([-1, 1])
            
            else:
                step = rng.choice([-1, 1]) * rng.integers(1, 4)

            new_val = int(curr_val + step)

        else: # Float
            delta = 0.0
            if "spacing" in param:
                # Moderate negative bias (tighter), but allow drift
                # loc=-0.5 allows finding sweet spots like 1.4 or 2.6
                delta = rng.normal(loc=-0.5, scale=1.5)
            elif "speed" in param:
                delta = rng.normal(loc=0.0, scale=2.0)
            else:
                range_span = max_val - min_val
                delta = rng.normal(loc=0.0, scale=range_span * 0.15)
                
            new_val = curr_val + delta

        # Clamp
        if spec["type"] == "int":
            new_val = int(np.clip(new_val, min_val, max_val))
        else:
            new_val = float(np.clip(new_val, min_val, max_val))
            
        print(f'Old value {param}={curr_val}, New value={new_val}')
        new_cfg[param] = new_val

    # Consistency Check
    if "lanes_count" in new_cfg and "initial_lane_id" in new_cfg:
        if new_cfg["initial_lane_id"] >= new_cfg["lanes_count"]:
             new_cfg["initial_lane_id"] = max(0, new_cfg["lanes_count"] - 1)

    return new_cfg


# ============================================================
# 3) HILL CLIMBING SEARCH
# ============================================================

def hill_climb(
    env_id: str,
    base_cfg: Dict[str, Any],
    param_spec: Dict[str, Any],
    policy,
    defaults: Dict[str, Any],
    seed: int = 0,
    iterations: int = 100,
    neighbors_per_iter: int = 10,
) -> Dict[str, Any]:
    """
    Hill climbing loop.

    You should:
      1) Start from an initial scenario (base_cfg or random sample).
      2) Evaluate it by running:
            crashed, ts = run_episode(env_id, cfg, policy, defaults, seed_base)
         Then compute objectives + fitness.
      3) For each iteration:
            - Generate neighbors_per_iter neighbors using mutate_config
            - Evaluate each neighbor
            - Select the best neighbor
            - Accept it if it improves fitness (or implement another acceptance rule)
            - Optionally stop early if a crash is found
      4) Return the best scenario found and enough info to reproduce.

    Return dict MUST contain at least:
        {
          "best_cfg": Dict[str, Any],
          "best_objectives": Dict[str, Any],
          "best_fitness": float,
          "best_seed_base": int,
          "history": List[float]
        }

    Optional but useful:
        - "best_time_series": ts
        - "evaluations": int
    """
    rng = np.random.default_rng(seed)

    # Initialize with aggressive crash-prone configuration
    from search.base_search import ScenarioSearch
    search_base = ScenarioSearch(env_id, base_cfg, param_spec, policy, defaults)
    current_cfg = search_base.sample_random_config(rng)
    
    # Aggressively bias initial config towards crash-prone settings
    # current_cfg["vehicles_count"] = int(0.9 * param_spec["vehicles_count"]["max"])
    # current_cfg["initial_spacing"] = param_spec["initial_spacing"]["min"]
    # current_cfg["ego_spacing"] = param_spec["ego_spacing"]["min"]
    # current_cfg["lanes_count"] = param_spec["lanes_count"]["min"]
    # current_cfg["initial_lane_id"] = 0

    # Evaluate initial solution
    seed_base = int(rng.integers(1e9))
    crashed, ts = run_episode(env_id, current_cfg, policy, defaults, seed_base)
    obj = compute_objectives_from_time_series(ts)
    cur_fit = compute_fitness(obj)

    best_cfg = copy.deepcopy(current_cfg)
    best_obj = dict(obj)
    best_fit = float(cur_fit)
    best_seed_base = seed_base
    best_ts = ts

    history = [best_fit]
    evaluations = 1

    print(f"Initial: fitness={cur_fit:.3f}, crashed={obj['crash_count']}, min_dist={obj['min_distance']:.3f}, avg_dist={obj['avg_distance']:.3f}, vehicles_nearby={obj['max_vehicles_nearby']}")

    for iteration in trange(iterations, desc="Hill climber"):
        # Early stop if crash found
        if best_obj["crash_count"] == 1:
            print(f"💥 Crash found at iteration {iteration}!")
            break
            
        neighbors = []
        
        # Generate and evaluate neighbors
        for j in range(neighbors_per_iter):
            neighbor_cfg = mutate_config(current_cfg, param_spec, rng)
            neighbor_seed = int(rng.integers(1e9))
            
            crashed, neighbor_ts = run_episode(env_id, neighbor_cfg, policy, defaults, neighbor_seed)
            neighbor_obj = compute_objectives_from_time_series(neighbor_ts)
            neighbor_fit = compute_fitness(neighbor_obj)
            
            neighbors.append({
                "cfg": neighbor_cfg,
                "obj": neighbor_obj,
                "fit": neighbor_fit,
                "seed": neighbor_seed,
                "ts": neighbor_ts
            })
            evaluations += 1
            
            print('='*30)
            print('Eval ', j)
            print('Fitness ', neighbor_fit)
            print('Objectives ', neighbor_obj)
            print('Config ', neighbor_cfg)
            print('='*30, end='\n\n\n\n')
            
            # Update global best if this neighbor is better
            if neighbor_fit < best_fit:
                best_cfg = copy.deepcopy(neighbor_cfg)
                best_obj = dict(neighbor_obj)
                best_fit = float(neighbor_fit)
                best_seed_base = neighbor_seed
                best_ts = neighbor_ts
        
        # Find best neighbor
        best_neighbor = min(neighbors, key=lambda x: x["fit"])
        
        # Accept if better than current (greedy hill climbing)
        if best_neighbor["fit"] < cur_fit:
            current_cfg = best_neighbor["cfg"]
            cur_fit = best_neighbor["fit"]
            print(f"Iter {iteration}: Improved to {cur_fit:.3f}, crashed={best_neighbor['obj']['crash_count']}, min_dist={best_neighbor['obj']['min_distance']:.3f}, close_calls={best_neighbor['obj']['frames_with_close_call']}")
        else:
            print(f"Iter {iteration}: No improvement, current={cur_fit:.3f}")
        
        history.append(best_fit)

    print(f"Hill climbing completed. Best fitness: {best_fit:.3f}, evaluations: {evaluations}")
    
    return {
        "best_cfg": best_cfg,
        "best_objectives": best_obj,
        "best_fitness": best_fit,
        "best_seed_base": best_seed_base,
        "best_time_series": best_ts,
        "history": history,
        "evaluations": evaluations
    }


# ============================================================
# 4) HILL CLIMBING SEARCH CLASS
# ============================================================

class HillClimbingSearch:
    """Hill Climbing search for scenario generation."""
    
    def __init__(self, env_id, base_cfg, param_spec, policy, defaults):
        self.env_id = env_id
        self.base_cfg = base_cfg
        self.param_spec = param_spec
        self.policy = policy
        self.defaults = defaults

    def run_search(self, iterations=100, neighbors_per_iter=10, seed=42):
        """Run hill climbing search and return crashes found."""
        print(f"Running Hill Climbing Search for {iterations} iterations...")
        
        result = hill_climb(
            self.env_id, 
            self.base_cfg, 
            self.param_spec, 
            self.policy, 
            self.defaults,
            seed=seed,
            iterations=iterations,
            neighbors_per_iter=neighbors_per_iter
        )
        
        crash_log = []
        
        # If we found a crash, record it
        if result["best_objectives"]["crash_count"] == 1:
            crash_entry = {
                "cfg": result["best_cfg"],
                "seed": result["best_seed_base"]
            }
            crash_log.append(crash_entry)
            
            # Record video of the crash
            record_video_episode(
                self.env_id, 
                result["best_cfg"], 
                self.policy, 
                self.defaults, 
                result["best_seed_base"], 
                out_dir="videos"
            )
            
            print(f"💥 Crash found! Fitness: {result['best_fitness']:.3f}")
        else:
            print(f"No crash found. Best min_distance: {result['best_objectives']['min_distance']:.3f}")
        
        return crash_log
