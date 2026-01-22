"""
This file contains helper functions for the evaluation of the search algorithms. We test Random Search against Hill Climbing
by running N scenarios, saving the configurations of the outcomes and then doing a statistical analysis of the results. 

"""

from search.random_search import RandomSearch
from search.hill_climbing import HillClimbSearch
import pandas as pd

N_SCENARIOS = 50
N_EVAL = 1
SEED = 42

def run_random_search(env_id, base_cfg, param_spec, policy, defaults, n_scenarios=N_SCENARIOS, n_eval=N_EVAL, seed=SEED):
    search = RandomSearch(env_id, base_cfg, param_spec, policy, defaults)
    results = search.run_search(n_scenarios=n_scenarios, n_eval=n_eval, seed=seed)
    return results

def run_hill_climbing_search(env_id, base_cfg, param_spec, policy, defaults, n_scenarios=N_SCENARIOS, n_eval=N_EVAL, seed=SEED):
    search = HillClimbSearch(env_id, base_cfg, param_spec, policy, defaults)
    results = search.run_search(n_scenarios=n_scenarios, n_eval=n_eval, seed=seed)
    return results