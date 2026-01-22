# Usage Guide

## hill_climbing.py

Hill climbing search algorithm for finding scenario configurations that trigger collisions in an RL agent.

### Purpose
Finds environment configurations that cause the ego vehicle to crash. If no crash is found, minimizes the minimum distance between the ego vehicle and other vehicles.

### Key Components

#### `HillClimbSearch` Class

Main search class that implements hill climbing optimization.

**Initialization:**
```python
from search.hill_climbing import HillClimbSearch
from config.search_space import param_spec, base_cfg
from policies.pretrained_policy import load_pretrained_policy
from envs.highway_env_utils import make_env

env_id = "highway-fast-v0"
policy = load_pretrained_policy("agents/model")
env, defaults = make_env(env_id)

search = HillClimbSearch(env_id, base_cfg, param_spec, policy, defaults)
```

**Running Search:**
```python
results = search.run_search(
    seed=0,                    # Random seed for reproducibility
    iterations=100,            # Number of hill climbing iterations
    neighbors_per_iter=10,     # Neighbors to evaluate per iteration
    mutation_rate=0.2,         # Mutation step size (0.0-1.0)
    disable_tqdm=False         # Disable progress bar
)
```

**Return Value:**
```python
{
    "iteration": int,                    # Iteration where best was found
    "initial_cfg": Dict,                 # Starting configuration
    "initial_objectives": Dict,          # Objectives from initial config
    "initial_fitness": float,            # Fitness of initial config
    "initial_seed": int,                 # Seed for initial evaluation
    "best_fitness": float,               # Best fitness found (lower is better)
    "best_objectives": {                 # Best objectives found
        "crash_count": int,              # 1 if crash found, 0 otherwise
        "min_distance": float            # Minimum distance to other vehicles
    },
    "best_cfg": Dict,                    # Best configuration found
    "best_seed_base": int,               # Seed for best configuration
    "total_evaluations": int,            # Total number of evaluations
    "evaluation_history": List[Dict],    # Full evaluation history
    "initial_eval_time_seconds": float,  # Time for initial evaluation
    "hill_climbing_eval_time_seconds": float  # Time for hill climbing
}
```

### Helper Functions

- **`compute_objectives_from_time_series(time_series)`**: Extracts objectives (crash_count, min_distance) from episode time-series data.
- **`compute_fitness(objectives)`**: Converts objectives to a single scalar fitness value (minimize). Crashes have fitness = -1.0 (best).
- **`mutate_config(cfg, param_spec, rng, mutation_rate)`**: Generates a neighbor configuration by mutating one parameter.

### Example Usage

```python
from search.hill_climbing import HillClimbSearch
from config.search_space import param_spec, base_cfg
from policies.pretrained_policy import load_pretrained_policy
from envs.highway_env_utils import make_env

# Setup
env_id = "highway-fast-v0"
policy = load_pretrained_policy("agents/model")
env, defaults = make_env(env_id)
search = HillClimbSearch(env_id, base_cfg, param_spec, policy, defaults)

# Run search
results = search.run_search(
    seed=42,
    iterations=10,
    neighbors_per_iter=10,
    mutation_rate=0.3
)

# Check results
if results['best_objectives']['crash_count'] > 0:
    print(f"Crash found! Config: {results['best_cfg']}")
else:
    print(f"Best min distance: {results['best_objectives']['min_distance']:.4f}m")
```

---

## evaluation.py

Comparison script that evaluates Hill Climbing vs Random Search on multiple scenarios.

### Purpose
Runs both Random Search and Hill Climbing on the same set of scenarios and compares their performance in finding crashes.

### Main Function

#### `run_evaluation()`

Runs the complete evaluation comparing both methods.

**Parameters:**
```python
run_evaluation(
    n_scenarios=100,              # Number of random scenarios to test
    random_search_evals=100,      # Evaluations per scenario for random search
    hc_iterations=10,             # Hill climbing iterations per scenario
    hc_neighbors_per_iter=10,     # Neighbors per hill climbing iteration
    hc_mutation_rate=0.3,         # Hill climbing mutation rate
    base_seed=42,                 # Base seed for reproducibility
    results_dir="results"         # Directory to save results
)
```

**Returns:**
Dictionary containing:
- `parameters`: Input parameters used
- `random_search_results`: List of random search results per scenario
- `hill_climbing_results`: List of hill climbing results per scenario
- `analysis`: Statistical analysis comparing both methods
- `total_runtime_seconds`: Total execution time

### Output Files

Results are saved to the specified `results_dir`:

1. **`evaluation_results.json`**: Complete results in JSON format
2. **`random_search_results.csv`**: DataFrame with random search results
3. **`hill_climbing_results.csv`**: DataFrame with hill climbing results
4. **`evaluation_summary.txt`**: Human-readable summary report

### Analysis Includes

- **Failure Discovery**: Crash rates, distinct crashes found, first crash evaluation number
- **Efficiency**: Runtime statistics, time per evaluation
- **Scenario Characteristics**: Parameter statistics for crash scenarios

### Example Usage

```python
from evaluation import run_evaluation

# Run evaluation
results = run_evaluation(
    n_scenarios=10,
    random_search_evals=20,
    hc_iterations=10,
    hc_neighbors_per_iter=10,
    hc_mutation_rate=0.3,
    base_seed=0,
    results_dir="results"
)

# Access analysis
analysis = results['analysis']
print(f"Random Search crashes: {analysis['failure_discovery']['random_search']['crashes_found']}")
print(f"Hill Climbing crashes: {analysis['failure_discovery']['hill_climbing']['crashes_found']}")
```

### Command Line Usage

```bash
python evaluation.py
```

Runs with default parameters (5 scenarios, 20 random search evals, 10 HC iterations).

---

## Notes

- Both scripts use multiprocessing for parallel evaluation
- Hill climbing stops early if a crash is found
- Fitness values: crashes have fitness = -1.0 (best), non-crashes use min_distance (lower is better)
