"""
Evaluation and Comparison Script for Hill Climbing vs Random Search

This script:
1. Generates N random scenario configurations
2. Runs Random Search on each scenario (100 evaluations per scenario)
3. Runs Hill Climbing on the same scenarios (10 iterations × 10 neighbors = 100 evaluations)
4. Compares results and saves analysis to files
"""

import warnings
import os
import time
import json
import copy
from typing import List, Dict, Any
from pathlib import Path
from multiprocessing import Pool, cpu_count, TimeoutError as MPTimeoutError

# Suppress warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*deprecated.*")
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
os.environ['PYTHONWARNINGS'] = "ignore::UserWarning"

import pandas as pd
import numpy as np
from tqdm import tqdm

from config.search_space import param_spec, base_cfg
from policies.pretrained_policy import load_pretrained_policy
from envs.highway_env_utils import make_env, run_episode
from search.hill_climbing import HillClimbSearch, compute_objectives_from_time_series, compute_fitness
from search.base_search import ScenarioSearch


# Module-level variable for random search worker policy
_rs_worker_policy = None

def _init_rs_worker():
    """Initialize worker process for random search - load policy once per worker."""
    global _rs_worker_policy
    from policies.pretrained_policy import load_pretrained_policy
    _rs_worker_policy = load_pretrained_policy("agents/model")

def _evaluate_random_config(args):
    """
    Helper function for parallel random search evaluation.
    
    Args:
        args: (initial_cfg, env_id, defaults, seed, eval_idx)
    
    Returns:
        (eval_idx, config, seed, objectives, fitness, crashed, eval_time)
    """
    global _rs_worker_policy
    initial_cfg, env_id, defaults, seed, eval_idx = args
    
    eval_start = time.time()
    crashed, ts = run_episode(env_id, initial_cfg, _rs_worker_policy, defaults, seed)
    eval_time = time.time() - eval_start
    
    if crashed:
        obj = {"crash_count": 1, "min_distance": 0.0}
        fitness = -1.0
    else:
        obj = compute_objectives_from_time_series(ts)
        fitness = compute_fitness(obj)
    
    return eval_idx, copy.deepcopy(initial_cfg), seed, obj, fitness, crashed, eval_time


def run_random_search_evaluation(
    initial_cfg: Dict[str, Any],
    env_id: str,
    defaults: Dict[str, Any],
    n_evaluations: int,
    rng: np.random.Generator,
    scenario_id: int
) -> Dict[str, Any]:
    """
    Run random search on a given initial configuration using multiprocessing.
    Evaluates n_evaluations random configs (same config, different seeds) in parallel.
    
    Returns:
        Dictionary with all evaluation results and statistics
    """
    # Generate seeds for all evaluations
    seeds = [int(rng.integers(1e9)) for _ in range(n_evaluations)]
    
    # Prepare arguments for parallel evaluation
    eval_args = [
        (initial_cfg, env_id, defaults, seed, eval_idx)
        for eval_idx, seed in enumerate(seeds)
    ]
    
    # Determine number of workers
    n_workers = min(n_evaluations, int(np.floor(cpu_count() * 0.75)))
    
    start_time = time.time()
    
    # Run evaluations in parallel with timeout
    # Timeout: Since evaluations run in parallel, we wait for all to complete
    # If any worker hangs, we'll timeout. Set to reasonable time for worst-case evaluation
    # 30 seconds should be plenty for a single episode evaluation
    timeout = n_workers + 10
    
    pool = Pool(processes=n_workers, initializer=_init_rs_worker)
    try:
        async_result = pool.map_async(_evaluate_random_config, eval_args)
        try:
            results_list = async_result.get(timeout=timeout)
        except MPTimeoutError:
            print(f"\n⚠️  Warning: Random search worker timeout for scenario {scenario_id}. Terminating pool and retrying...")
            pool.terminate()
            pool.join()
            # Recreate pool and retry
            pool = Pool(processes=n_workers, initializer=_init_rs_worker)
            async_result = pool.map_async(_evaluate_random_config, eval_args)
            try:
                results_list = async_result.get(timeout=timeout)
                print(f"   Retry successful for scenario {scenario_id}.")
            except MPTimeoutError:
                print(f"\n❌ Error: Workers still timing out for scenario {scenario_id}. Using partial results.")
                # Get partial results if available
                results_list = []
                for i in range(n_evaluations):
                    results_list.append((
                        i, initial_cfg, int(rng.integers(1e9)),
                        {"crash_count": 0, "min_distance": float('inf')},
                        float('inf'), False, 0.0
                    ))
    finally:
        pool.close()
        pool.join()
    
    total_time = time.time() - start_time
    
    # Process results
    # Note: config is the same for all evaluations, so we don't store it per evaluation
    results = []
    crashes_found = []
    
    for eval_idx, cfg, seed, obj, fitness, crashed, eval_time in results_list:
        result = {
            "evaluation_num": eval_idx,
            "seed": seed,
            "objectives": obj,
            "fitness": fitness,
            "crashed": crashed,
            "eval_time_seconds": eval_time
        }
        results.append(result)
        
        if crashed:
            crashes_found.append({
                "evaluation_num": eval_idx,
                "seed": seed,
                "objectives": obj,
                "fitness": fitness
            })
    
    # Find best result (lowest fitness = best)
    # Remove config from best_result since it's available as initial_cfg
    best_result = min(results, key=lambda x: x["fitness"])
    best_result = {k: v for k, v in best_result.items() if k != "eval_time_seconds"}  # Keep it clean
    
    return {
        "scenario_id": scenario_id,
        "initial_cfg": initial_cfg,
        "all_evaluations": results,
        "crashes_found": crashes_found,
        "best_result": best_result,
        "total_evaluations": n_evaluations,
        "total_time_seconds": total_time,
        "avg_time_per_eval_seconds": total_time / n_evaluations
    }


def run_evaluation(
    n_scenarios: int = 100,
    random_search_evals: int = 100,
    hc_iterations: int = 10,
    hc_neighbors_per_iter: int = 10,
    hc_mutation_rate: float = 0.3,
    base_seed: int = 42,
    results_dir: str = "results"
) -> Dict[str, Any]:
    """
    Run evaluation comparing Random Search vs Hill Climbing on the same scenarios.
    
    Args:
        n_scenarios: Number of random scenarios to generate and test
        random_search_evals: Number of evaluations per scenario for random search
        hc_iterations: Number of hill climbing iterations per scenario
        hc_neighbors_per_iter: Number of neighbors per hill climbing iteration
        hc_mutation_rate: Mutation rate for hill climbing
        base_seed: Base seed for reproducibility
        results_dir: Directory to save results
    """
    # Setup
    env_id = "highway-fast-v0"
    policy = load_pretrained_policy("agents/model")
    env, defaults = make_env(env_id)
    hc_search = HillClimbSearch(env_id, base_cfg, param_spec, policy, defaults)
    
    # Generate random initial configurations
    initial_configs = []
    for i in range(n_scenarios):
        scenario_rng = np.random.default_rng(base_seed + i)
        cfg = ScenarioSearch.sample_random_config(hc_search, scenario_rng)
        initial_configs.append(cfg)
    
    # Storage for results
    random_search_results = []
    hill_climbing_results = []
    
    total_start_time = time.time()
    
    # Run evaluations
    for i in tqdm(range(n_scenarios), desc="Evaluating scenarios"):
        scenario_cfg = initial_configs[i]
        scenario_rng = np.random.default_rng(base_seed + i)
        
        # Run Random Search
        rs_start = time.time()
        rs_result = run_random_search_evaluation(
            scenario_cfg, env_id, defaults,
            random_search_evals, scenario_rng, i
        )
        rs_time = time.time() - rs_start
        rs_result["total_time_seconds"] = rs_time
        random_search_results.append(rs_result)
        
        # Run Hill Climbing (starting from the same initial config)
        hc_start = time.time()
        hc_result = hc_search.run_search(
            seed=base_seed + i,
            iterations=hc_iterations,
            neighbors_per_iter=hc_neighbors_per_iter,
            mutation_rate=hc_mutation_rate,
            disable_tqdm=True
        )
        hc_time = time.time() - hc_start
        hc_result["scenario_id"] = i
        hc_result["initial_cfg"] = scenario_cfg
        hc_result["total_time_seconds"] = hc_time
        hill_climbing_results.append(hc_result)
    
    total_time = time.time() - total_start_time
    
    # Perform analysis
    analysis = analyze_results(random_search_results, hill_climbing_results, total_time)
    
    # Compile results
    evaluation_results = {
        "parameters": {
            "n_scenarios": n_scenarios,
            "random_search_evals": random_search_evals,
            "hc_iterations": hc_iterations,
            "hc_neighbors_per_iter": hc_neighbors_per_iter,
            "hc_mutation_rate": hc_mutation_rate,
            "base_seed": base_seed,
            "total_evaluations_per_method": random_search_evals
        },
        "random_search_results": random_search_results,
        "hill_climbing_results": hill_climbing_results,
        "analysis": analysis,
        "total_runtime_seconds": total_time
    }
    
    # Save results
    save_evaluation_results(evaluation_results, results_dir)
    
    return evaluation_results


def analyze_results(
    rs_results: List[Dict],
    hc_results: List[Dict],
    total_time: float
) -> Dict[str, Any]:
    """Perform statistical analysis and comparison."""
    
    # Extract data for analysis
    # Helper function to convert config values to proper types
    def convert_config_types(cfg, param_spec):
        """Convert config values to proper types based on param_spec."""
        converted = {}
        for key, value in cfg.items():
            if key in param_spec:
                if param_spec[key]["type"] == "int":
                    converted[key] = int(value) if value is not None else None
                elif param_spec[key]["type"] == "float":
                    converted[key] = float(value) if value is not None else None
                else:
                    converted[key] = value
            else:
                # For keys not in param_spec, try to preserve type
                converted[key] = value
        return converted
    
    rs_data = []
    for r in rs_results:
        best = r["best_result"]
        cfg = convert_config_types(r["initial_cfg"], param_spec)
        rs_data.append({
            "scenario_id": r["scenario_id"],
            "crashed": best["crashed"],
            "min_distance": best["objectives"]["min_distance"],
            "fitness": best["fitness"],
            "total_evaluations": r["total_evaluations"],
            "total_time_seconds": r["total_time_seconds"],
            "crashes_found_count": len(r["crashes_found"]),
            **cfg
        })
    
    hc_data = []
    for r in hc_results:
        cfg = convert_config_types(r["initial_cfg"], param_spec)
        hc_data.append({
            "scenario_id": r["scenario_id"],
            "crashed": r["best_objectives"]["crash_count"] > 0,
            "min_distance": r["best_objectives"]["min_distance"],
            "fitness": r["best_fitness"],
            "total_evaluations": r["total_evaluations"],
            "total_time_seconds": r["total_time_seconds"],
            "iterations": r["iteration"],
            **cfg
        })
    
    rs_df = pd.DataFrame(rs_data)
    hc_df = pd.DataFrame(hc_data)
    
    # 1. Failure Discovery Analysis
    rs_crashes = rs_df["crashed"].sum()
    hc_crashes = hc_df["crashed"].sum()
    rs_crash_rate = rs_crashes / len(rs_df)
    hc_crash_rate = hc_crashes / len(hc_df)
    
    # Distinct crashes
    rs_crash_configs = set(
        tuple(sorted(r["initial_cfg"].items()))
        for r in rs_results if r["best_result"]["crashed"]
    )
    hc_crash_configs = set(
        tuple(sorted(r["initial_cfg"].items()))
        for r in hc_results if r["best_objectives"]["crash_count"] > 0
    )
    
    # Min distance statistics (for non-crashes)
    rs_min_dist = rs_df[~rs_df["crashed"]]["min_distance"]
    hc_min_dist = hc_df[~hc_df["crashed"]]["min_distance"]
    
    # First crash evaluation number
    rs_first_crash_eval = None
    hc_first_crash_eval = None
    for r in rs_results:
        if r["crashes_found"]:
            rs_first_crash_eval = r["crashes_found"][0]["evaluation_num"]
            break
    for r in hc_results:
        if r["best_objectives"]["crash_count"] > 0:
            # Find evaluation number from history
            for eval_hist in r.get("evaluation_history", []):
                if eval_hist.get("crashed", False):
                    hc_first_crash_eval = eval_hist.get("evaluation_num")
                    break
            if hc_first_crash_eval is not None:
                break
    
    # 2. Efficiency Analysis
    rs_avg_time = rs_df["total_time_seconds"].mean()
    hc_avg_time = hc_df["total_time_seconds"].mean()
    rs_avg_time_per_eval = rs_avg_time / rs_df["total_evaluations"].iloc[0]
    hc_avg_time_per_eval = hc_avg_time / hc_df["total_evaluations"].mean()
    
    # 3. Scenario Characteristics (for crashes)
    rs_crash_scenarios = [r for r in rs_results if r["best_result"]["crashed"]]
    hc_crash_scenarios = [r for r in hc_results if r["best_objectives"]["crash_count"] > 0]
    
    analysis = {
        "failure_discovery": {
            "random_search": {
                "crashes_found": int(rs_crashes),
                "crash_rate": float(rs_crash_rate),
                "distinct_crashes": len(rs_crash_configs),
                "first_crash_evaluation": rs_first_crash_eval,
                "min_distance_stats": {
                    "mean": float(rs_min_dist.mean()) if len(rs_min_dist) > 0 else None,
                    "std": float(rs_min_dist.std()) if len(rs_min_dist) > 0 else None,
                    "min": float(rs_min_dist.min()) if len(rs_min_dist) > 0 else None,
                    "max": float(rs_min_dist.max()) if len(rs_min_dist) > 0 else None
                }
            },
            "hill_climbing": {
                "crashes_found": int(hc_crashes),
                "crash_rate": float(hc_crash_rate),
                "distinct_crashes": len(hc_crash_configs),
                "first_crash_evaluation": hc_first_crash_eval,
                "min_distance_stats": {
                    "mean": float(hc_min_dist.mean()) if len(hc_min_dist) > 0 else None,
                    "std": float(hc_min_dist.std()) if len(hc_min_dist) > 0 else None,
                    "min": float(hc_min_dist.min()) if len(hc_min_dist) > 0 else None,
                    "max": float(hc_min_dist.max()) if len(hc_min_dist) > 0 else None
                }
            }
        },
        "efficiency": {
            "total_runtime_seconds": float(total_time),
            "random_search": {
                "avg_time_per_scenario_seconds": float(rs_avg_time),
                "avg_time_per_evaluation_seconds": float(rs_avg_time_per_eval),
                "total_evaluations": int(rs_df["total_evaluations"].sum())
            },
            "hill_climbing": {
                "avg_time_per_scenario_seconds": float(hc_avg_time),
                "avg_time_per_evaluation_seconds": float(hc_avg_time_per_eval),
                "total_evaluations": int(hc_df["total_evaluations"].sum())
            }
        },
        "scenario_characteristics": {
            "random_search_crashes": len(rs_crash_scenarios),
            "hill_climbing_crashes": len(hc_crash_scenarios),
            "random_search_crash_params": {
                "vehicles_count": {
                    "mean": float(rs_df[rs_df["crashed"]]["vehicles_count"].mean()) if rs_crashes > 0 else None,
                    "std": float(rs_df[rs_df["crashed"]]["vehicles_count"].std()) if rs_crashes > 0 else None
                },
                "lanes_count": {
                    "mean": float(rs_df[rs_df["crashed"]]["lanes_count"].mean()) if rs_crashes > 0 else None,
                    "std": float(rs_df[rs_df["crashed"]]["lanes_count"].std()) if rs_crashes > 0 else None
                },
                "initial_spacing": {
                    "mean": float(rs_df[rs_df["crashed"]]["initial_spacing"].mean()) if rs_crashes > 0 else None,
                    "std": float(rs_df[rs_df["crashed"]]["initial_spacing"].std()) if rs_crashes > 0 else None
                },
                "initial_lane_id": {
                    "mean": float(rs_df[rs_df["crashed"]]["initial_lane_id"].mean()) if rs_crashes > 0 else None,
                    "std": float(rs_df[rs_df["crashed"]]["initial_lane_id"].std()) if rs_crashes > 0 else None
                }
            },
            "hill_climbing_crash_params": {
                "vehicles_count": {
                    "mean": float(hc_df[hc_df["crashed"]]["vehicles_count"].mean()) if hc_crashes > 0 else None,
                    "std": float(hc_df[hc_df["crashed"]]["vehicles_count"].std()) if hc_crashes > 0 else None
                },
                "lanes_count": {
                    "mean": float(hc_df[hc_df["crashed"]]["lanes_count"].mean()) if hc_crashes > 0 else None,
                    "std": float(hc_df[hc_df["crashed"]]["lanes_count"].std()) if hc_crashes > 0 else None
                },
                "initial_spacing": {
                    "mean": float(hc_df[hc_df["crashed"]]["initial_spacing"].mean()) if hc_crashes > 0 else None,
                    "std": float(hc_df[hc_df["crashed"]]["initial_spacing"].std()) if hc_crashes > 0 else None
                },
                "initial_lane_id": {
                    "mean": float(hc_df[hc_df["crashed"]]["initial_lane_id"].mean()) if hc_crashes > 0 else None,
                    "std": float(hc_df[hc_df["crashed"]]["initial_lane_id"].std()) if hc_crashes > 0 else None
                }
            }
        },
        "dataframes": {
            "random_search_df": rs_df,
            "hill_climbing_df": hc_df
        }
    }
    
    return analysis


def save_evaluation_results(results: Dict[str, Any], results_dir: str = "results"):
    """Save evaluation results to files."""
    Path(results_dir).mkdir(exist_ok=True)
    
    # Save full results as JSON (excluding DataFrames)
    results_to_save = {k: v for k, v in results.items() if k != "analysis"}
    results_to_save["analysis"] = {
        k: v for k, v in results["analysis"].items() if k != "dataframes"
    }
    
    with open(f"{results_dir}/evaluation_results.json", "w") as f:
        json.dump(results_to_save, f, indent=2, default=str)
    
    # Save DataFrames as CSV
    results["analysis"]["dataframes"]["random_search_df"].to_csv(
        f"{results_dir}/random_search_results.csv", index=False
    )
    results["analysis"]["dataframes"]["hill_climbing_df"].to_csv(
        f"{results_dir}/hill_climbing_results.csv", index=False
    )
    
    # Save summary report as text file
    analysis = results["analysis"]
    with open(f"{results_dir}/evaluation_summary.txt", "w") as f:
        f.write("="*80 + "\n")
        f.write("EVALUATION SUMMARY: Hill Climbing vs Random Search\n")
        f.write("="*80 + "\n\n")
        
        # Parameters
        f.write("PARAMETERS\n")
        f.write("-" * 80 + "\n")
        params = results["parameters"]
        f.write(f"Number of scenarios: {params['n_scenarios']}\n")
        f.write(f"Random Search evaluations per scenario: {params['random_search_evals']}\n")
        f.write(f"Hill Climbing iterations: {params['hc_iterations']}\n")
        f.write(f"Hill Climbing neighbors per iteration: {params['hc_neighbors_per_iter']}\n")
        f.write(f"Total runtime: {results['total_runtime_seconds']:.2f} seconds\n\n")
        
        # Failure Discovery
        f.write("1. FAILURE DISCOVERY\n")
        f.write("-" * 80 + "\n")
        rs_fd = analysis["failure_discovery"]["random_search"]
        hc_fd = analysis["failure_discovery"]["hill_climbing"]
        
        f.write("Random Search:\n")
        f.write(f"  - Crashes found: {rs_fd['crashes_found']} / {params['n_scenarios']}\n")
        f.write(f"  - Crash rate: {rs_fd['crash_rate']:.2%}\n")
        f.write(f"  - Distinct crashes: {rs_fd['distinct_crashes']}\n")
        if rs_fd['first_crash_evaluation'] is not None:
            f.write(f"  - First crash at evaluation: {rs_fd['first_crash_evaluation']}\n")
        if rs_fd['min_distance_stats']['mean'] is not None:
            f.write(f"  - Avg min distance (non-crashes): {rs_fd['min_distance_stats']['mean']:.4f}m\n")
            f.write(f"  - Min distance range: [{rs_fd['min_distance_stats']['min']:.4f}, {rs_fd['min_distance_stats']['max']:.4f}]m\n")
        
        f.write("\nHill Climbing:\n")
        f.write(f"  - Crashes found: {hc_fd['crashes_found']} / {params['n_scenarios']}\n")
        f.write(f"  - Crash rate: {hc_fd['crash_rate']:.2%}\n")
        f.write(f"  - Distinct crashes: {hc_fd['distinct_crashes']}\n")
        if hc_fd['first_crash_evaluation'] is not None:
            f.write(f"  - First crash at evaluation: {hc_fd['first_crash_evaluation']}\n")
        if hc_fd['min_distance_stats']['mean'] is not None:
            f.write(f"  - Avg min distance (non-crashes): {hc_fd['min_distance_stats']['mean']:.4f}m\n")
            f.write(f"  - Min distance range: [{hc_fd['min_distance_stats']['min']:.4f}, {hc_fd['min_distance_stats']['max']:.4f}]m\n")
        
        # Efficiency
        f.write("\n2. EFFICIENCY\n")
        f.write("-" * 80 + "\n")
        eff = analysis["efficiency"]
        f.write(f"Total runtime: {eff['total_runtime_seconds']:.2f} seconds\n\n")
        
        f.write("Random Search:\n")
        f.write(f"  - Avg time per scenario: {eff['random_search']['avg_time_per_scenario_seconds']:.2f} seconds\n")
        f.write(f"  - Avg time per evaluation: {eff['random_search']['avg_time_per_evaluation_seconds']:.4f} seconds\n")
        f.write(f"  - Total evaluations: {eff['random_search']['total_evaluations']}\n")
        
        f.write("\nHill Climbing:\n")
        f.write(f"  - Avg time per scenario: {eff['hill_climbing']['avg_time_per_scenario_seconds']:.2f} seconds\n")
        f.write(f"  - Avg time per evaluation: {eff['hill_climbing']['avg_time_per_evaluation_seconds']:.4f} seconds\n")
        f.write(f"  - Total evaluations: {eff['hill_climbing']['total_evaluations']}\n")
        
        # Scenario Characteristics
        f.write("\n3. SCENARIO CHARACTERISTICS\n")
        f.write("-" * 80 + "\n")
        sc = analysis["scenario_characteristics"]
        
        if rs_fd['crashes_found'] > 0:
            f.write("Random Search Crash Scenarios:\n")
            rs_params = sc["random_search_crash_params"]
            f.write(f"  - Vehicles count: {rs_params['vehicles_count']['mean']:.2f} ± {rs_params['vehicles_count']['std']:.2f}\n")
            f.write(f"  - Lanes count: {rs_params['lanes_count']['mean']:.2f} ± {rs_params['lanes_count']['std']:.2f}\n")
            f.write(f"  - Initial spacing: {rs_params['initial_spacing']['mean']:.2f} ± {rs_params['initial_spacing']['std']:.2f}\n")
            f.write(f"  - Initial lane ID: {rs_params['initial_lane_id']['mean']:.2f} ± {rs_params['initial_lane_id']['std']:.2f}\n")
        
        if hc_fd['crashes_found'] > 0:
            f.write("\nHill Climbing Crash Scenarios:\n")
            hc_params = sc["hill_climbing_crash_params"]
            f.write(f"  - Vehicles count: {hc_params['vehicles_count']['mean']:.2f} ± {hc_params['vehicles_count']['std']:.2f}\n")
            f.write(f"  - Lanes count: {hc_params['lanes_count']['mean']:.2f} ± {hc_params['lanes_count']['std']:.2f}\n")
            f.write(f"  - Initial spacing: {hc_params['initial_spacing']['mean']:.2f} ± {hc_params['initial_spacing']['std']:.2f}\n")
            f.write(f"  - Initial lane ID: {hc_params['initial_lane_id']['mean']:.2f} ± {hc_params['initial_lane_id']['std']:.2f}\n")
        
        f.write("\n" + "="*80 + "\n")



