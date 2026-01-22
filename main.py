import warnings
import os

# Suppress warnings for cleaner output - must be set before any imports
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*deprecated.*")
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
os.environ['PYTHONWARNINGS'] = "ignore::UserWarning"

import pandas as pd

from config.search_space import param_spec, base_cfg
from policies.pretrained_policy import load_pretrained_policy
from envs.highway_env_utils import make_env
from search.random_search import RandomSearch
from search.hill_climbing import HillClimbSearch

def main():
    env_id = "highway-fast-v0"
    policy = load_pretrained_policy("agents/model")
    env, defaults = make_env(env_id)

    # Initialize search
    # search = HillClimbSearch(env_id, base_cfg, param_spec, policy, defaults)
    search = HillClimbSearch(env_id, base_cfg, param_spec, policy, defaults)

    best_cfgs = []
    n_scenarios = 50

    for i in range(n_scenarios):
        results = search.run_search(
            seed=i,  # Different seed for each scenario to get different initial configurations
            iterations=15,
            neighbors_per_iter=10,    # More neighbors with parallel = better exploration
            mutation_rate=0.3         # Mutation size: 0.3 = 30% of range (higher helps escape local minima)
        )
        best_cfgs.append(results['best_cfg'])
    
    results_df = pd.DataFrame(best_cfgs)
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(results_df.describe())
    

    # print("\n" + "="*60)
    # print("RESULTS")
    # print("="*60)
    # print(f"Best fitness: {results['best_fitness']:.4f}")
    # print(f"Crash found: {results['best_objectives']['crash_count'] > 0}")
    # print(f"Min distance: {results['best_objectives']['min_distance']:.4f}m")
    # print(f"Config: {results['best_cfg']}")
    # print(f"Seed: {results['best_seed_base']}")
    # print(f"Video saved to: {results['video_folder']}")

if __name__ == "__main__":
    main()