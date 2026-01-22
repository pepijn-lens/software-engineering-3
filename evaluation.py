"""
Evaluation and Comparison Script for Hill Climbing vs Random Search

This script runs multiple Hill Climbing searches and extracts:
- Initial random configurations → Random Search baseline
- Final best configurations → Hill Climbing results

Then performs statistical analysis and comparison.
"""

import warnings
import os
import time
import json
from typing import List, Dict, Any
from pathlib import Path

# Suppress warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*deprecated.*")
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
os.environ['PYTHONWARNINGS'] = "ignore::UserWarning"

import pandas as pd
from tqdm import tqdm

from config.search_space import param_spec, base_cfg
from policies.pretrained_policy import load_pretrained_policy
from envs.highway_env_utils import make_env
from search.hill_climbing import HillClimbSearch


def run_evaluation(
    n_scenarios: int = 100,
    iterations: int = 10,
    neighbors_per_iter: int = 10,
    mutation_rate: float = 0.3,
    base_seed: int = 0,
    save_results: bool = True,
    results_dir: str = "results"
) -> Dict[str, Any]:
    """
    Run evaluation comparing Random Search (initial configs) vs Hill Climbing (best configs).
    
    Args:
        n_scenarios: Number of scenarios to run (each starts with a random config)
        iterations: Number of hill climbing iterations per scenario
        neighbors_per_iter: Number of neighbors to evaluate per iteration
        mutation_rate: Mutation rate for hill climbing
        base_seed: Base seed for reproducibility
        save_results: Whether to save results to files
        results_dir: Directory to save results
        
    Returns:
        Dictionary containing all results and analysis
    """
    print("="*80)
    print("EVALUATION: Hill Climbing vs Random Search")
    print("="*80)
    print(f"Running {n_scenarios} scenarios...")
    print(f"Hill Climbing: {iterations} iterations, {neighbors_per_iter} neighbors/iter")
    print(f"Total evaluations per scenario: ~{1 + iterations * neighbors_per_iter}")
    print("="*80)
    
    # Setup
    env_id = "highway-fast-v0"
    policy = load_pretrained_policy("agents/model")
    env, defaults = make_env(env_id)
    search = HillClimbSearch(env_id, base_cfg, param_spec, policy, defaults)
    
    # Storage for results
    initial_results = []  # Random Search baseline
    hill_climbing_results = []  # Hill Climbing results
    all_runs = []  # Complete run data
    
    # Run scenarios
    start_time = time.time()
    
    for i in tqdm(range(n_scenarios), desc="Running scenarios"):
        scenario_start = time.time()
        
        # Run hill climbing search
        results = search.run_search(
            seed=base_seed + i,  # Different seed for each scenario
            iterations=iterations,
            neighbors_per_iter=neighbors_per_iter,
            mutation_rate=mutation_rate,
            disable_tqdm=True  # Disable inner progress bar to avoid reprinting
        )
        
        scenario_time = time.time() - scenario_start
        
        # Extract computation times (separate for random search and hill climbing)
        initial_eval_time = results.get("initial_eval_time_seconds", 0.0)
        hc_eval_time = results.get("hill_climbing_eval_time_seconds", 0.0)
        
        # Extract initial (Random Search equivalent)
        initial_data = {
            "scenario_id": i,
            "config": results["initial_cfg"],
            "objectives": results["initial_objectives"],
            "fitness": results["initial_fitness"],
            "seed": results["initial_seed"],
            "crashed": results["initial_objectives"]["crash_count"] > 0,
            "min_distance": results["initial_objectives"]["min_distance"],
            "eval_time_seconds": initial_eval_time,
            "runtime_seconds": scenario_time  # Total scenario time for reference
        }
        initial_results.append(initial_data)
        
        # Extract best (Hill Climbing result)
        hc_data = {
            "scenario_id": i,
            "config": results["best_cfg"],
            "objectives": results["best_objectives"],
            "fitness": results["best_fitness"],
            "seed": results["best_seed_base"],
            "crashed": results["best_objectives"]["crash_count"] > 0,
            "min_distance": results["best_objectives"]["min_distance"],
            "iterations": results["iteration"],
            "total_evaluations": results["total_evaluations"],
            "eval_time_seconds": hc_eval_time,
            "runtime_seconds": scenario_time  # Total scenario time for reference
        }
        hill_climbing_results.append(hc_data)
        
        # Store complete run data
        all_runs.append({
            "scenario_id": i,
            "initial": initial_data,
            "hill_climbing": hc_data,
            "evaluation_history": results.get("evaluation_history", [])
        })
    
    total_time = time.time() - start_time
    
    # Perform analysis
    analysis = analyze_results(initial_results, hill_climbing_results, total_time)
    
    # Compile results
    evaluation_results = {
        "parameters": {
            "n_scenarios": n_scenarios,
            "iterations": iterations,
            "neighbors_per_iter": neighbors_per_iter,
            "mutation_rate": mutation_rate,
            "base_seed": base_seed
        },
        "initial_results": initial_results,  # Random Search baseline
        "hill_climbing_results": hill_climbing_results,  # Hill Climbing results
        "all_runs": all_runs,  # Complete data
        "analysis": analysis,
        "total_runtime_seconds": total_time
    }
    
    # Save results if requested
    if save_results:
        save_evaluation_results(evaluation_results, results_dir)
    
    return evaluation_results


def analyze_results(
    initial_results: List[Dict],
    hc_results: List[Dict],
    total_time: float
) -> Dict[str, Any]:
    """Perform statistical analysis and comparison."""
    
    # Convert to DataFrames for easier analysis
    initial_df = pd.DataFrame([
        {
            "scenario_id": r["scenario_id"],
            "crashed": r["crashed"],
            "min_distance": r["min_distance"],
            "fitness": r["fitness"],
            "eval_time_seconds": r["eval_time_seconds"],
            **r["config"]  # Flatten config parameters
        }
        for r in initial_results
    ])
    
    hc_df = pd.DataFrame([
        {
            "scenario_id": r["scenario_id"],
            "crashed": r["crashed"],
            "min_distance": r["min_distance"],
            "fitness": r["fitness"],
            "iterations": r["iterations"],
            "total_evaluations": r["total_evaluations"],
            "eval_time_seconds": r["eval_time_seconds"],
            **r["config"]  # Flatten config parameters
        }
        for r in hc_results
    ])
    
    # 1. Failure Discovery Analysis
    initial_crashes = initial_df["crashed"].sum()
    hc_crashes = hc_df["crashed"].sum()
    
    initial_crash_rate = initial_crashes / len(initial_df)
    hc_crash_rate = hc_crashes / len(hc_df)
    
    # Distinct crashes (unique configs that crashed)
    initial_crash_configs = set(
        tuple(sorted(r["config"].items()))
        for r in initial_results if r["crashed"]
    )
    hc_crash_configs = set(
        tuple(sorted(r["config"].items()))
        for r in hc_results if r["crashed"]
    )
    
    # Min distance statistics (for non-crashes)
    initial_min_dist = initial_df[~initial_df["crashed"]]["min_distance"]
    hc_min_dist = hc_df[~hc_df["crashed"]]["min_distance"]
    
    # 2. Efficiency Analysis
    avg_evaluations = hc_df["total_evaluations"].mean()
    avg_runtime_per_scenario = total_time / len(hc_results)
    evaluations_per_second = avg_evaluations / avg_runtime_per_scenario if avg_runtime_per_scenario > 0 else 0
    
    # Computation time analysis
    avg_initial_eval_time = initial_df["eval_time_seconds"].mean()
    avg_hc_eval_time = hc_df["eval_time_seconds"].mean()
    total_initial_eval_time = initial_df["eval_time_seconds"].sum()
    total_hc_eval_time = hc_df["eval_time_seconds"].sum()
    
    # Time per evaluation
    avg_time_per_initial_eval = avg_initial_eval_time  # 1 evaluation per initial
    avg_time_per_hc_eval = avg_hc_eval_time / avg_evaluations if avg_evaluations > 0 else 0
    
    # First crash evaluation (if any)
    initial_first_crash = None
    hc_first_crash = None
    for i, r in enumerate(initial_results):
        if r["crashed"]:
            initial_first_crash = i + 1
            break
    for i, r in enumerate(hc_results):
        if r["crashed"]:
            hc_first_crash = i + 1
            break
    
    # 3. Scenario Characteristics (for crashes)
    initial_crash_scenarios = [r for r in initial_results if r["crashed"]]
    hc_crash_scenarios = [r for r in hc_results if r["crashed"]]
    
    analysis = {
        "failure_discovery": {
            "random_search": {
                "crashes_found": int(initial_crashes),
                "crash_rate": float(initial_crash_rate),
                "distinct_crashes": len(initial_crash_configs),
                "first_crash_scenario": initial_first_crash,
                "min_distance_stats": {
                    "mean": float(initial_min_dist.mean()) if len(initial_min_dist) > 0 else None,
                    "std": float(initial_min_dist.std()) if len(initial_min_dist) > 0 else None,
                    "min": float(initial_min_dist.min()) if len(initial_min_dist) > 0 else None,
                    "max": float(initial_min_dist.max()) if len(initial_min_dist) > 0 else None
                }
            },
            "hill_climbing": {
                "crashes_found": int(hc_crashes),
                "crash_rate": float(hc_crash_rate),
                "distinct_crashes": len(hc_crash_configs),
                "first_crash_scenario": hc_first_crash,
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
            "avg_runtime_per_scenario": float(avg_runtime_per_scenario),
            "avg_evaluations_per_scenario": float(avg_evaluations),
            "evaluations_per_second": float(evaluations_per_second),
            "total_evaluations": int(hc_df["total_evaluations"].sum()),
            "computation_time": {
                "random_search": {
                    "total_eval_time_seconds": float(total_initial_eval_time),
                    "avg_eval_time_seconds": float(avg_initial_eval_time),
                    "avg_time_per_evaluation_seconds": float(avg_time_per_initial_eval),
                    "total_evaluations": len(initial_results)
                },
                "hill_climbing": {
                    "total_eval_time_seconds": float(total_hc_eval_time),
                    "avg_eval_time_seconds": float(avg_hc_eval_time),
                    "avg_time_per_evaluation_seconds": float(avg_time_per_hc_eval),
                    "total_evaluations": int(hc_df["total_evaluations"].sum())
                }
            }
        },
        "scenario_characteristics": {
            "random_search_crashes": initial_crash_scenarios,
            "hill_climbing_crashes": hc_crash_scenarios,
            "random_search_crash_params": {
                "vehicles_count": {
                    "mean": float(initial_df[initial_df["crashed"]]["vehicles_count"].mean()) if initial_crashes > 0 else None,
                    "std": float(initial_df[initial_df["crashed"]]["vehicles_count"].std()) if initial_crashes > 0 else None
                },
                "lanes_count": {
                    "mean": float(initial_df[initial_df["crashed"]]["lanes_count"].mean()) if initial_crashes > 0 else None,
                    "std": float(initial_df[initial_df["crashed"]]["lanes_count"].std()) if initial_crashes > 0 else None
                },
                "initial_spacing": {
                    "mean": float(initial_df[initial_df["crashed"]]["initial_spacing"].mean()) if initial_crashes > 0 else None,
                    "std": float(initial_df[initial_df["crashed"]]["initial_spacing"].std()) if initial_crashes > 0 else None
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
                }
            }
        },
        "dataframes": {
            "initial_df": initial_df,
            "hc_df": hc_df
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
    results["analysis"]["dataframes"]["initial_df"].to_csv(
        f"{results_dir}/random_search_results.csv", index=False
    )
    results["analysis"]["dataframes"]["hc_df"].to_csv(
        f"{results_dir}/hill_climbing_results.csv", index=False
    )
    
    print(f"\nResults saved to {results_dir}/")


def print_summary(analysis: Dict[str, Any]):
    """Print a summary of the analysis."""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    rs = analysis["failure_discovery"]["random_search"]
    hc = analysis["failure_discovery"]["hill_climbing"]
    eff = analysis["efficiency"]
    
    print("\n1. FAILURE DISCOVERY")
    print("-" * 80)
    print(f"Random Search (Initial Configs):")
    print(f"  - Crashes found: {rs['crashes_found']} / {rs['crashes_found'] + len(analysis['dataframes']['initial_df']) - rs['crashes_found']}")
    print(f"  - Crash rate: {rs['crash_rate']:.2%}")
    print(f"  - Distinct crashes: {rs['distinct_crashes']}")
    if rs['min_distance_stats']['mean'] is not None:
        print(f"  - Avg min distance (non-crashes): {rs['min_distance_stats']['mean']:.4f}m")
    
    print(f"\nHill Climbing (After Mutations):")
    print(f"  - Crashes found: {hc['crashes_found']}")
    print(f"  - Crash rate: {hc['crash_rate']:.2%}")
    print(f"  - Distinct crashes: {hc['distinct_crashes']}")
    if hc['min_distance_stats']['mean'] is not None:
        print(f"  - Avg min distance (non-crashes): {hc['min_distance_stats']['mean']:.4f}m")
    
    print("\n2. EFFICIENCY")
    print("-" * 80)
    print(f"  - Total runtime: {eff['total_runtime_seconds']:.2f} seconds")
    print(f"  - Avg runtime per scenario: {eff['avg_runtime_per_scenario']:.2f} seconds")
    print(f"  - Avg evaluations per scenario: {eff['avg_evaluations_per_scenario']:.1f}")
    print(f"  - Evaluations per second: {eff['evaluations_per_second']:.2f}")
    print(f"  - Total evaluations: {eff['total_evaluations']}")
    
    print("\n3. COMPUTATION TIME")
    print("-" * 80)
    rs_time = eff['computation_time']['random_search']
    hc_time = eff['computation_time']['hill_climbing']
    print(f"Random Search Evaluation:")
    print(f"  - Total eval time: {rs_time['total_eval_time_seconds']:.2f} seconds")
    print(f"  - Avg eval time per scenario: {rs_time['avg_eval_time_seconds']:.4f} seconds")
    print(f"  - Avg time per evaluation: {rs_time['avg_time_per_evaluation_seconds']:.4f} seconds")
    print(f"  - Total evaluations: {rs_time['total_evaluations']}")
    print(f"\nHill Climbing Evaluation:")
    print(f"  - Total eval time: {hc_time['total_eval_time_seconds']:.2f} seconds")
    print(f"  - Avg eval time per scenario: {hc_time['avg_eval_time_seconds']:.2f} seconds")
    print(f"  - Avg time per evaluation: {hc_time['avg_time_per_evaluation_seconds']:.4f} seconds")
    print(f"  - Total evaluations: {hc_time['total_evaluations']}")
    
    print("\n4. SCENARIO CHARACTERISTICS")
    print("-" * 80)
    rs_params = analysis["scenario_characteristics"]["random_search_crash_params"]
    hc_params = analysis["scenario_characteristics"]["hill_climbing_crash_params"]
    
    if rs['crashes_found'] > 0:
        print("Random Search Crash Scenarios:")
        print(f"  - Avg vehicles_count: {rs_params['vehicles_count']['mean']:.2f} ± {rs_params['vehicles_count']['std']:.2f}")
        print(f"  - Avg lanes_count: {rs_params['lanes_count']['mean']:.2f} ± {rs_params['lanes_count']['std']:.2f}")
        print(f"  - Avg initial_spacing: {rs_params['initial_spacing']['mean']:.2f} ± {rs_params['initial_spacing']['std']:.2f}")
    
    if hc['crashes_found'] > 0:
        print("\nHill Climbing Crash Scenarios:")
        print(f"  - Avg vehicles_count: {hc_params['vehicles_count']['mean']:.2f} ± {hc_params['vehicles_count']['std']:.2f}")
        print(f"  - Avg lanes_count: {hc_params['lanes_count']['mean']:.2f} ± {hc_params['lanes_count']['std']:.2f}")
        print(f"  - Avg initial_spacing: {hc_params['initial_spacing']['mean']:.2f} ± {hc_params['initial_spacing']['std']:.2f}")
    
    print("\n" + "="*80)


def main():
    """Main entry point for evaluation."""
    results = run_evaluation(
        n_scenarios=100,  # Number of scenarios to run
        iterations=10,    # Hill climbing iterations per scenario
        neighbors_per_iter=10,  # Neighbors per iteration
        mutation_rate=0.3,  # Mutation rate
        base_seed=0,  # Base seed for reproducibility
        save_results=True
    )
    
    # Print summary
    print_summary(results["analysis"])
    
    return results


if __name__ == "__main__":
    main()
