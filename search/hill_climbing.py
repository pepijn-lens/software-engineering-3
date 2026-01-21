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
from tqdm import tqdm

from envs.highway_env_utils import run_episode


# ============================================================
# 1) OBJECTIVES FROM TIME SERIES
# ============================================================

def compute_objectives_from_time_series(time_series: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute your objective values from the recorded time-series.

    The time_series is a list of frames. Each frame typically contains:
      - frame["crashed"]: bool
      - frame["ego"]: dict or None, e.g. {"pos":[x,y], "lane_id":..., "length":..., "width":...}
      - frame["others"]: list of dicts with positions, lane_id, etc.

    Minimum requirements (suggested):
      - crash_count: 1 if any collision happened, else 0
      - min_distance: minimum distance between ego and any other vehicle over time (float)

    Return a dictionary, e.g.:
        {
          "crash_count": 0 or 1,
          "min_distance": float
        }

    NOTE: If you want, you can add more objectives (lane-specific distances, time-to-crash, etc.)
    but keep the keys above at least.
    """

    crash_count = 0
    min_distance = float('inf')
    for idx, frame in enumerate(time_series):
      if frame["crashed"]:
        crash_count += 1
        break
      else:
        for other in frame["others"]:
          distance = np.linalg.norm(np.array(frame["ego"]["pos"]) - np.array(other["pos"]))
          min_distance = min(min_distance, distance)
    return {"crash_count": crash_count, "min_distance": min_distance}


def compute_fitness(objectives: Dict[str, Any]) -> float:
    """
    Convert objectives into ONE scalar fitness value to MINIMIZE.

    Requirement:
    - Any crashing scenario must be strictly better than any non-crashing scenario.

    Examples:
    - If crash_count==1: fitness = -1 (best)
    - Else: fitness = min_distance (smaller is better)

    You can design a more refined scalarization if desired.
    """

    if objectives["crash_count"] > 0:
      return -1
    else:
      return np.log(objectives["min_distance"] + 1)

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

    Inputs:
      - cfg: current scenario dict (e.g., vehicles_count, initial_spacing, ego_spacing, initial_lane_id)
      - param_spec: search space bounds, types (int/float), min/max
      - rng: random generator

    Requirements:
      - Do NOT modify cfg in-place (return a copy).
      - Keep mutated values within [min, max] from param_spec.
      - If you mutate lanes_count, keep initial_lane_id valid (0..lanes_count-1).

    Students can implement:
      - single-parameter mutation (recommended baseline)
      - multiple-parameter mutation
      - adaptive step sizes, etc.
    """
    
    # param_spec = {
    # "vehicles_count":   {"type": "int",   "min": 5,   "max": 60},
    # "lanes_count":      {"type": "int",   "min": 3,   "max": 10},
    # "initial_spacing":  {"type": "float", "min": 0.5, "max": 5.0},
    # "ego_spacing":      {"type": "float", "min": 1.0, "max": 4.0},
    # "initial_lane_id":  {"type": "int",   "min": 0,   "max": 4},
    # }

    new_cfg = copy.deepcopy(cfg)

    # randomly choose a key from param_spec
    key = rng.choice(list(param_spec.keys()))
    
    # Initialize the key if it doesn't exist
    if key not in new_cfg:
      spec = param_spec[key]
      if spec["type"] == "int":
        new_cfg[key] = int(rng.integers(spec["min"], spec["max"] + 1))
      elif spec["type"] == "float":
        new_cfg[key] = float(rng.uniform(spec["min"], spec["max"]))

    # Mutate the chosen parameter
    if param_spec[key]["type"] == "int":
      new_cfg[key] += int(rng.integers(-1, 2))  # -1, 0, or 1
    elif param_spec[key]["type"] == "float":
      max_change = 0.1 * new_cfg[key]
      new_cfg[key] += float(rng.uniform(-max_change, max_change))

    # clamp the values to the min and max
    new_cfg[key] = np.clip(new_cfg[key], param_spec[key]["min"], param_spec[key]["max"])
    
    # Ensure initial_lane_id is valid for the current lanes_count
    if "initial_lane_id" in new_cfg and "lanes_count" in new_cfg:
      if new_cfg["initial_lane_id"] >= new_cfg["lanes_count"]:
        new_cfg["initial_lane_id"] = new_cfg["initial_lane_id"] % new_cfg["lanes_count"]

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

    # (students): choose initialization (base_cfg or random scenario)
    current_cfg = {
      "vehicles_count": 59,
      "lanes_count": 10,
      "initial_lane_id": 0,
      "initial_spacing": 0.6,
      "ego_spacing": 1.1,
    }

    # Evaluate initial solution (seed_base used for reproducibility)
    seed_base = int(rng.integers(1e9))
    crashed, ts = run_episode(env_id, current_cfg, policy, defaults, seed_base)
    obj = compute_objectives_from_time_series(ts)
    cur_fit = compute_fitness(obj)

    best_cfg = copy.deepcopy(current_cfg)
    best_obj = dict(obj)
    best_fit = float(cur_fit)
    best_seed_base = seed_base

    history = [best_fit]

    #  (students): implement HC loop
    # - generate neighbors
    # - evaluate
    # - pick best
    # - accept if improved
    # - early stop on crash (optional)
    print(f"number of iterations: {iterations}")
    pbar = tqdm(range(iterations), desc=f"Hill Climbing (best fitness: {best_fit:.4f})")
    for i in pbar:
      neighbors = [mutate_config(current_cfg, param_spec, rng) for _ in range(neighbors_per_iter)]
      improved = False
      
      for neighbor in tqdm(neighbors, desc=f"Evaluating neighbors", leave=False):
        crashed, ts = run_episode(env_id, neighbor, policy, defaults, seed_base)
        obj = compute_objectives_from_time_series(ts)
        fit = compute_fitness(obj)
        
        if fit < best_fit:
          best_cfg = copy.deepcopy(neighbor)
          best_obj = dict(obj)
          best_fit = fit
          best_seed_base = seed_base
          current_cfg = copy.deepcopy(best_cfg)
          improved = True
          pbar.set_description(f"Hill Climbing (best fitness: {best_fit:.4f})")
          
          # Early stop if crash found
          if obj["crash_count"] > 0:
            history.append(best_fit)
            print(f"\nCrash found at iteration {i}!")
            return {
              "best_cfg": best_cfg,
              "best_objectives": best_obj,
              "best_fitness": best_fit,
              "best_seed_base": best_seed_base,
              "history": history
            }
      
      if improved:
        history.append(best_fit)

    return {
      "best_cfg": best_cfg,
      "best_objectives": best_obj,
      "best_fitness": best_fit,
      "best_seed_base": best_seed_base,
      "history": history
    }

class HillClimbSearch:
  def __init__(self, env_id, base_cfg, param_spec, policy, defaults):
    self.env_id = env_id
    self.base_cfg = base_cfg
    self.param_spec = param_spec
    self.policy = policy
    self.defaults = defaults

  def run_search(self, iterations=100, neighbors_per_iter=10, seed=0):
    return hill_climb(self.env_id, self.base_cfg, self.param_spec, self.policy, self.defaults, 
                      seed=seed, iterations=iterations, neighbors_per_iter=neighbors_per_iter)
