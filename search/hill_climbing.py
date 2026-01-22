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
import time
from typing import Dict, Any, List, Tuple, Optional
from multiprocessing import Pool, cpu_count, TimeoutError as MPTimeoutError

import numpy as np
from tqdm import tqdm

from envs.highway_env_utils import run_episode, record_video_episode


# ============================================================
# 1) OBJECTIVES FROM TIME SERIES
# ============================================================

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
    
    for frame in time_series:
        if frame["crashed"]:
            crash_count += 1
            break
        
        ego = frame["ego"]
        others = frame["others"]
        
        if ego is None or len(others) == 0:
            continue
        
        for other in others:
            # Distance to ego
            distance = np.linalg.norm(np.array(ego["pos"]) - np.array(other["pos"]))
            min_distance = min(min_distance, distance)
    
    return {
        "crash_count": crash_count,
        "min_distance": min_distance,
    }


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
        return -1.0  # Best possible fitness
    
    # Base fitness: smaller distance is better
    fitness = objectives["min_distance"]

    return fitness

# ============================================================
# 2) MUTATION / NEIGHBOR GENERATION
# ============================================================

def mutate_config(
    cfg: Dict[str, Any],
    param_spec: Dict[str, Any],
    rng: np.random.Generator,
    mutation_rate: float = 0.2,
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
    
    new_cfg = copy.deepcopy(cfg)

    # Select parameter randomly
    key = rng.choice(list(param_spec.keys()))

    min_value = param_spec[key]["min"]
    max_value = param_spec[key]["max"]

    # Mutate the selected parameter
    if param_spec[key]["type"] == "int":
        spec_range = max_value - min_value
        max_step = max(1, int(spec_range * mutation_rate))
        step = int(rng.integers(-max_step, max_step + 1))
        new_cfg[key] += step
    elif param_spec[key]["type"] == "float":
        spec_range = max_value - min_value
        max_change = spec_range * mutation_rate
        new_cfg[key] += float(rng.uniform(-max_change, max_change))

    # Clamp the values to the min and max
    new_cfg[key] = np.clip(new_cfg[key], min_value, max_value)
    
    # Ensure initial_lane_id is valid for the current lanes_count
    if "initial_lane_id" in new_cfg and "lanes_count" in new_cfg:
        if new_cfg["initial_lane_id"] >= new_cfg["lanes_count"]:
            new_cfg["initial_lane_id"] = new_cfg["initial_lane_id"] % new_cfg["lanes_count"]

    return new_cfg

# ============================================================
# 3) HILL CLIMBING SEARCH
# ============================================================

# Module-level variable to store policy per worker process
_worker_policy = None

def _init_worker():
    """Initialize worker process - load policy once per worker."""
    global _worker_policy
    from policies.pretrained_policy import load_pretrained_policy
    _worker_policy = load_pretrained_policy("agents/model")

def _evaluate_neighbor(args):
    """
    Helper function for parallel neighbor evaluation.
    
    Optimization: If crashed==True, skip objective computation.
    We already know fitness=-1.0 for crashes, no need to process time series.
    """
    global _worker_policy
    neighbor_cfg, env_id, defaults, seed_base = args
    # Use pre-loaded policy (loaded once per worker via _init_worker)
    crashed, ts = run_episode(env_id, neighbor_cfg, _worker_policy, defaults, seed_base)
    
    # Fast path: If crashed, skip objective computation
    if crashed:
        obj = {
            "crash_count": 1,
            "min_distance": 0.0,
        }
        fit = -1.0  # Best possible fitness
    else:
        obj = compute_objectives_from_time_series(ts)
        fit = compute_fitness(obj)
    
    return neighbor_cfg, obj, fit, seed_base


class HillClimbSearch:
    def __init__(self, env_id, base_cfg, param_spec, policy, defaults):
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

        self.env_id = env_id
        self.base_cfg = base_cfg
        self.param_spec = param_spec
        self.policy = policy
        self.defaults = defaults

    def run_search(
        self,
        seed: int = 0,
        iterations: int = 100,
        neighbors_per_iter: int = 10,
        mutation_rate: float = 0.2,
        disable_tqdm: bool = False,
    ) -> Dict[str, Any]:

        from search.base_search import ScenarioSearch
        
        if not disable_tqdm:
            print(f"Running Hill Climbing Search for {iterations} iterations...")
        rng = np.random.default_rng(seed)
        
        # Initialize with random configuration
        current_cfg = ScenarioSearch.sample_random_config(self, rng)
        seed_base = int(rng.integers(1e9))
        
        # Evaluate initial configuration (Random Search evaluation)
        initial_eval_start = time.time()
        crashed, ts = run_episode(self.env_id, current_cfg, self.policy, self.defaults, seed_base)
        initial_eval_time = time.time() - initial_eval_start
        if crashed:
            current_obj = {"crash_count": 1, "min_distance": 0.0}
            current_fitness = -1.0
        else:
            current_obj = compute_objectives_from_time_series(ts)
            current_fitness = compute_fitness(current_obj)
        
        # Store initial evaluation (this is our "random search" baseline)
        initial_cfg = copy.deepcopy(current_cfg)
        initial_obj = copy.deepcopy(current_obj)
        initial_fitness = current_fitness
        initial_seed = seed_base
        
        # Track best found
        best_cfg = copy.deepcopy(current_cfg)
        best_obj = copy.deepcopy(current_obj)
        best_fitness = current_fitness
        best_seed_base = seed_base
        
        # Track evaluation history
        evaluation_history = [{
            "evaluation_num": 0,
            "iteration": -1,  # -1 indicates initial evaluation
            "config": copy.deepcopy(initial_cfg),
            "seed": initial_seed,
            "objectives": copy.deepcopy(initial_obj),
            "fitness": initial_fitness,
            "crashed": crashed
        }]
        total_evaluations = 1
        
        # If we found a crash immediately, record it
        if crashed:
            # print(f"💥 Collision found in initial configuration!")
            # record_video_episode(self.env_id, best_cfg, self.policy, self.defaults, best_seed_base, out_dir="videos")
            return {
                "iteration": 0,
                "initial_cfg": initial_cfg,
                "initial_objectives": initial_obj,
                "initial_fitness": initial_fitness,
                "initial_seed": initial_seed,
                "best_fitness": best_fitness,
                "best_objectives": best_obj,
                "best_cfg": best_cfg,
                "best_seed_base": best_seed_base,
                "total_evaluations": total_evaluations,
                "evaluation_history": evaluation_history,
                "video_folder": "videos",
                "initial_eval_time_seconds": initial_eval_time,
                "hill_climbing_eval_time_seconds": 0.0
            }
        
        # Hill climbing iterations
        hill_climbing_eval_time = 0.0
        
        # Create pool once and reuse it across all iterations
        # This avoids the overhead of creating/destroying workers repeatedly
        n_workers = min(neighbors_per_iter, int(np.floor(cpu_count()*0.75)))
        pool = Pool(processes=n_workers, initializer=_init_worker)
        
        try:
            iter_range = range(iterations)
            if not disable_tqdm:
                iter_range = tqdm(iter_range, desc="Hill climbing")
            for i in iter_range:
                # Generate neighbors
                neighbors = []
                for _ in range(neighbors_per_iter):
                    neighbor_cfg = mutate_config(current_cfg, self.param_spec, rng, mutation_rate)
                    neighbor_seed = int(rng.integers(1e9))
                    neighbors.append((neighbor_cfg, neighbor_seed))
                
                # Evaluate neighbors in parallel
                best_neighbor_cfg = None
                best_neighbor_obj = None
                best_neighbor_fitness = float('inf')
                best_neighbor_seed = None
                
                # Prepare arguments for parallel evaluation
                eval_args = [
                    (neighbor_cfg, self.env_id, self.defaults, neighbor_seed)
                    for neighbor_cfg, neighbor_seed in neighbors
                ]

                # Time the hill climbing evaluation
                hc_iter_start = time.time()
                # Reuse the same pool - workers already have policy loaded
                # Use map_async with timeout to detect hung workers
                # Timeout: 30 seconds per evaluation 
                timeout_per_eval = 20
                timeout = timeout_per_eval * neighbors_per_iter
                async_result = pool.map_async(_evaluate_neighbor, eval_args)
                try:
                    results = async_result.get(timeout=timeout)
                except MPTimeoutError:
                    if not disable_tqdm:
                        print(f"\n⚠️  Warning: Worker timeout at iteration {i} after {timeout}s. Terminating hung workers...")
                    # Terminate the pool - some workers are hung
                    pool.terminate()
                    pool.join()
                    # Recreate pool for next iteration
                    pool = Pool(processes=n_workers, initializer=_init_worker)
                    if not disable_tqdm:
                        print(f"   Recreated pool. Retrying iteration {i}...")
                    # Retry this iteration with a fresh pool
                    async_result = pool.map_async(_evaluate_neighbor, eval_args)
                    try:
                        results = async_result.get(timeout=timeout)
                        if not disable_tqdm:
                            print(f"   Retry successful.")
                    except MPTimeoutError:
                        if not disable_tqdm:
                            print(f"\n❌ Error: Workers still timing out after retry. Skipping iteration {i}.")
                        # Skip this iteration - use current config as best neighbor
                        results = []
                        best_neighbor_fitness = current_fitness
                        best_neighbor_cfg = current_cfg
                        best_neighbor_obj = current_obj
                        best_neighbor_seed = seed_base
                hill_climbing_eval_time += time.time() - hc_iter_start
                
                # Process results and find best neighbor
                for neighbor_cfg, neighbor_obj, neighbor_fitness, neighbor_seed in results:
                    # Track this evaluation in history
                    total_evaluations += 1
                    evaluation_history.append({
                        "evaluation_num": total_evaluations - 1,
                        "iteration": i,
                        "config": copy.deepcopy(neighbor_cfg),
                        "seed": neighbor_seed,
                        "objectives": copy.deepcopy(neighbor_obj),
                        "fitness": neighbor_fitness,
                        "crashed": neighbor_obj["crash_count"] > 0
                    })
                    
                    # Update best neighbor if better
                    if neighbor_fitness < best_neighbor_fitness:
                        best_neighbor_cfg = neighbor_cfg
                        best_neighbor_obj = neighbor_obj
                        best_neighbor_fitness = neighbor_fitness
                        best_neighbor_seed = neighbor_seed
                    
                    # If we found a crash, record it and return
                    if neighbor_obj["crash_count"] > 0:
                        # print(f"Collision found at iteration {i}!")
                        # record_video_episode(self.env_id, neighbor_cfg, self.policy, self.defaults, neighbor_seed, out_dir="videos")
                        # Pool will be closed by finally block
                        return {
                            "iteration": i,
                            "initial_cfg": initial_cfg,
                            "initial_objectives": initial_obj,
                            "initial_fitness": initial_fitness,
                            "initial_seed": initial_seed,
                            "best_fitness": neighbor_fitness,
                            "best_objectives": neighbor_obj,
                            "best_cfg": neighbor_cfg,
                            "best_seed_base": neighbor_seed,
                            "total_evaluations": total_evaluations,
                            "evaluation_history": evaluation_history,
                            "video_folder": "videos",
                            "initial_eval_time_seconds": initial_eval_time,
                            "hill_climbing_eval_time_seconds": hill_climbing_eval_time
                        }
                
                # Move to best neighbor if it's better
                if best_neighbor_fitness < current_fitness:
                    current_cfg = best_neighbor_cfg
                    current_obj = best_neighbor_obj
                    current_fitness = best_neighbor_fitness
                    seed_base = best_neighbor_seed
                    
                    # Update global best
                    if best_neighbor_fitness < best_fitness:
                        best_cfg = copy.deepcopy(best_neighbor_cfg)
                        best_obj = copy.deepcopy(best_neighbor_obj)
                        best_fitness = best_neighbor_fitness
                        best_seed_base = best_neighbor_seed
                        if not disable_tqdm:
                            print(f"Iteration {i}: Improved fitness to {best_fitness:.4f}")
        
        finally:
            # Always close the pool, even if we exit early
            pool.close()
            pool.join()
        
        # # Record video of best configuration found
        # record_video_episode(self.env_id, best_cfg, self.policy, self.defaults, best_seed_base, out_dir="videos")
        
        return {
            "iteration": iterations,
            "initial_cfg": initial_cfg,
            "initial_objectives": initial_obj,
            "initial_fitness": initial_fitness,
            "initial_seed": initial_seed,
            "best_fitness": best_fitness,
            "best_objectives": best_obj,
            "best_cfg": best_cfg,
            "best_seed_base": best_seed_base,
            "total_evaluations": total_evaluations,
            "evaluation_history": evaluation_history,
            "video_folder": "videos",
            "initial_eval_time_seconds": initial_eval_time,
            "hill_climbing_eval_time_seconds": hill_climbing_eval_time
        }
    