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
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

from envs.highway_env_utils import run_episode, record_video_episode


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
    
    # Handle edge case where no valid distance was measured
    if min_distance == float('inf'):
        min_distance = 0.0
    
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
      return objectives["min_distance"]

# ============================================================
# 2) MUTATION / NEIGHBOR GENERATION
# ============================================================

def initialize_config(
    base_cfg: Dict[str, Any],
    param_spec: Dict[str, Any],
    rng: np.random.Generator
) -> Dict[str, Any]:
    """
    Initialize configuration with random values within search space bounds.
    
    Randomly samples all parameters uniformly within their specified ranges.
    
    Args:
        base_cfg: Base configuration (e.g., duration, frequencies)
        param_spec: Search space specification
        rng: Random number generator
        
    Returns:
        Fully initialized configuration ready for mutation
    """
    cfg = copy.deepcopy(base_cfg)
  
    for param_key, spec in param_spec.items():
        if param_key not in cfg:
            if spec["type"] == "int":
                cfg[param_key] = int(rng.integers(spec["min"], spec["max"] + 1))
            elif spec["type"] == "float":
                cfg[param_key] = float(rng.uniform(spec["min"], spec["max"]))
    
    # Ensure initial_lane_id is valid for lanes_count
    if "initial_lane_id" in cfg and "lanes_count" in cfg:
        if cfg["initial_lane_id"] >= cfg["lanes_count"]:
            cfg["initial_lane_id"] = cfg["initial_lane_id"] % cfg["lanes_count"]
    
    return cfg


def mutate_config(
    cfg: Dict[str, Any],
    param_spec: Dict[str, Any],
    rng: np.random.Generator,
    mutation_rate: float = 0.2
) -> Dict[str, Any]:
    """
    Generate ONE neighbor configuration by mutating the current scenario.
    
    Uses single-parameter mutation: randomly selects ONE parameter to mutate.
    This is more effective than mutating all parameters at once.

    Inputs:
      - cfg: current scenario dict (must have all parameters initialized)
      - param_spec: search space bounds, types (int/float), min/max
      - rng: random generator
      - mutation_rate: size of mutations (0.0-1.0), default 0.2 = 20% of range

    Requirements:
      - Do NOT modify cfg in-place (return a copy).
      - Keep mutated values within [min, max] from param_spec.
      - Keep initial_lane_id valid (0..lanes_count-1).
    """
    
    new_cfg = copy.deepcopy(cfg)

    # Single-parameter mutation: randomly select ONE parameter to mutate
    key = rng.choice(list(param_spec.keys()))

    # Mutate the selected parameter
    if param_spec[key]["type"] == "int":
        spec_range = param_spec[key]["max"] - param_spec[key]["min"]
        max_step = max(1, int(spec_range * mutation_rate))
        step = int(rng.integers(-max_step, max_step + 1))
        new_cfg[key] += step
    elif param_spec[key]["type"] == "float":
        spec_range = param_spec[key]["max"] - param_spec[key]["min"]
        max_change = spec_range * mutation_rate
        new_cfg[key] += float(rng.uniform(-max_change, max_change))

    # Clamp the values to the min and max
    new_cfg[key] = np.clip(new_cfg[key], param_spec[key]["min"], param_spec[key]["max"])
    
    # Ensure initial_lane_id is valid for the current lanes_count
    if "initial_lane_id" in new_cfg and "lanes_count" in new_cfg:
        if new_cfg["initial_lane_id"] >= new_cfg["lanes_count"]:
            new_cfg["initial_lane_id"] = new_cfg["initial_lane_id"] % new_cfg["lanes_count"]

    return new_cfg

# ============================================================
# 3) HILL CLIMBING SEARCH
# ============================================================

def _evaluate_neighbor(args):
    """Helper function for parallel neighbor evaluation."""
    from policies.pretrained_policy import load_pretrained_policy
    neighbor_cfg, env_id, defaults, seed_base = args
    # Load policy in each worker process (can't pickle the policy object)
    policy = load_pretrained_policy("agents/model")
    crashed, ts = run_episode(env_id, neighbor_cfg, policy, defaults, seed_base)
    obj = compute_objectives_from_time_series(ts)
    fit = compute_fitness(obj)
    return neighbor_cfg, obj, fit


def hill_climb(
    env_id: str,
    base_cfg: Dict[str, Any],
    param_spec: Dict[str, Any],
    policy,
    defaults: Dict[str, Any],
    seed: int = 0,
    iterations: int = 100,
    neighbors_per_iter: int = 10,
    mutation_rate: float = 0.2,
) -> Dict[str, Any]:
    """
    Hill climbing loop with Simulated Annealing.
    
    Uses:
    - Aggressive initialization (high vehicles, low spacing, more lanes)
    - Single-parameter mutation 
    - Linear temperature decay for simulated annealing
    - Parallel neighbor evaluation

    You should:
      1) Start from an initial scenario (base_cfg or random sample).
      2) Evaluate it by running:
            crashed, ts = run_episode(env_id, cfg, policy, defaults, seed_base)
         Then compute objectives + fitness.
      3) For each iteration:
            - Generate neighbors_per_iter neighbors using mutate_config
            - Evaluate each neighbor in parallel
            - Select the best neighbor
            - Accept it if it improves fitness OR probabilistically (simulated annealing)
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

    # Initialize configuration once with aggressive crash-prone values
    current_cfg = initialize_config(base_cfg, param_spec, rng)

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
    iterations_without_improvement = 0

    #  Hill climbing with simulated annealing, restarts, and parallel evaluation
    # - Simulated annealing: accept worse solutions with decreasing probability
    # - Single-parameter mutations: more focused search
    # - Restart mechanism: reinitialize to new random config after 20 iterations without improvement
    # - Parallel evaluation: speed up neighbor evaluation using all CPU cores
    
    n_cores = cpu_count()
    print(f"Starting Hill Climbing with {iterations} iterations")
    print(f"Initial fitness: {best_fit:.4f}")
    print(f"Initial config: lanes={current_cfg.get('lanes_count')}, vehicles={current_cfg.get('vehicles_count')}, "
          f"spacing={current_cfg.get('initial_spacing'):.1f}, lane_id={current_cfg.get('initial_lane_id')}")
    print(f"Using {n_cores} CPU cores for parallel evaluation")
    
    pbar = tqdm(range(iterations), desc=f"HC (fit={best_fit:.4f})")
    
    with Pool(processes=n_cores) as pool:
      for i in pbar:
        # Restart if stuck in local minimum for too long
        if iterations_without_improvement > 20:
          print(f"\n⚠️  Stuck in local minimum! Restarting with new random config...")
          current_cfg = initialize_config(base_cfg, param_spec, rng)
          seed_base = int(rng.integers(1e9))
          crashed, ts = run_episode(env_id, current_cfg, policy, defaults, seed_base)
          obj = compute_objectives_from_time_series(ts)
          cur_fit = compute_fitness(obj)
          print(f"   New config: lanes={current_cfg.get('lanes_count')}, vehicles={current_cfg.get('vehicles_count')}, "
                f"spacing={current_cfg.get('initial_spacing'):.1f}, lane_id={current_cfg.get('initial_lane_id')}, fitness={cur_fit:.4f}")
          iterations_without_improvement = 0
          
        # Simulated annealing temperature (slower exponential decay)
        temperature = 5.0 * (0.99 ** i)  # Starts at 5.0, decays slowly
        
        # Generate new seed for this iteration to explore stochastic variations
        iter_seed = int(rng.integers(1e9))
        
        # Generate neighbors with single-parameter mutations
        neighbors = [mutate_config(current_cfg, param_spec, rng, mutation_rate=mutation_rate) 
                     for _ in range(neighbors_per_iter)]
        
        # Parallel evaluation of neighbors with iteration-specific seed
        eval_args = [(neighbor, env_id, defaults, iter_seed) for neighbor in neighbors]
        results = pool.map(_evaluate_neighbor, eval_args)
        
        # Find best neighbor and check for crashes
        best_neighbor_fit = float('inf')
        best_neighbor = None
        best_neighbor_obj = None
        
        for neighbor_cfg, obj, fit in results:
          if fit < best_neighbor_fit:
            best_neighbor_fit = fit
            best_neighbor = neighbor_cfg
            best_neighbor_obj = obj
          
          # Early stop if crash found
          if obj["crash_count"] > 0:
            best_cfg = copy.deepcopy(neighbor_cfg)
            best_obj = dict(obj)
            best_fit = fit
            best_seed_base = iter_seed  # Update to the seed that found the crash
            history.append(best_fit)
            print(f"\n🎯 Crash found at iteration {i}!")
            print(f"Final config: {best_cfg}")
            print(f"Seed: {best_seed_base}")
            
            # Record video of crash scenario
            print("📹 Recording video of crash scenario...")
            _, video_folder = record_video_episode(
                env_id, best_cfg, policy, defaults, best_seed_base, out_dir="videos"
            )
            print(f"✅ Video saved to: {video_folder}")
            
            return {
              "best_cfg": best_cfg,
              "best_objectives": best_obj,
              "best_fitness": best_fit,
              "best_seed_base": best_seed_base,
              "history": history,
              "video_folder": video_folder
            }
        
        # Update global best if found
        if best_neighbor_fit < best_fit:
          best_cfg = copy.deepcopy(best_neighbor)
          best_obj = dict(best_neighbor_obj)
          best_fit = best_neighbor_fit
          best_seed_base = iter_seed  # Update to the seed that produced this result
          history.append(best_fit)
          iterations_without_improvement = 0
        else:
          iterations_without_improvement += 1
        
        # Simulated annealing: decide whether to move to best neighbor
        delta = best_neighbor_fit - cur_fit
        
        if delta < 0:
          # Neighbor is better than current, always accept
          current_cfg = copy.deepcopy(best_neighbor)
          cur_fit = best_neighbor_fit
          pbar.set_description(f"HC (fit={best_fit:.4f}, cur={cur_fit:.4f})")
        else:
          # Neighbor is worse, accept with probability based on temperature
          acceptance_prob = np.exp(-delta / temperature) if temperature > 0 else 0.0
          
          if rng.random() < acceptance_prob:
            current_cfg = copy.deepcopy(best_neighbor)
            cur_fit = best_neighbor_fit
            pbar.set_description(f"HC (fit={best_fit:.4f}, cur={cur_fit:.4f}, T={temperature:.2f}, SA✓)")
          else:
            pbar.set_description(f"HC (fit={best_fit:.4f}, cur={cur_fit:.4f}, T={temperature:.2f})")
    
    # Record video of best scenario found
    print("\n📹 Recording video of best scenario found...")
    _, video_folder = record_video_episode(
        env_id, best_cfg, policy, defaults, best_seed_base, out_dir="videos"
    )
    print(f"✅ Video saved to: {video_folder}")
    
    return {
      "best_cfg": best_cfg,
      "best_objectives": best_obj,
      "best_fitness": best_fit,
      "best_seed_base": best_seed_base,
      "history": history,
      "video_folder": video_folder
    }

class HillClimbSearch:
  def __init__(self, env_id, base_cfg, param_spec, policy, defaults):
    self.env_id = env_id
    self.base_cfg = base_cfg
    self.param_spec = param_spec
    self.policy = policy
    self.defaults = defaults

  def run_search(self, iterations=100, neighbors_per_iter=10, seed=0, mutation_rate=0.2):
    """
    Run hill climbing search with simulated annealing.
    
    Args:
        iterations: Number of hill climbing iterations
        neighbors_per_iter: Number of neighbors to generate per iteration
        seed: Random seed
        mutation_rate: Size of mutations (0.0-1.0). Higher = larger jumps in search space.
    """
    return hill_climb(
        self.env_id, self.base_cfg, self.param_spec, self.policy, self.defaults, 
        seed=seed, iterations=iterations, neighbors_per_iter=neighbors_per_iter,
        mutation_rate=mutation_rate
    )
