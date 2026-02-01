# Reproducing the Report Results

## Setup

```bash
pip install -r requirements.txt
```

The pre-trained agent must be present in `agents/` (e.g. `model.zip`, `vec_normalize.pkl`).

## Run

```bash
python main.py
```

This runs the experiment reported in the assignment: Hill Climbing vs Random Search on 100 scenarios.

**Parameters** (in `main.py`): `n_scenarios=100`, `random_search_evals=20`, `hc_iterations=10`, `hc_neighbors_per_iter=10`, `hc_mutation_rate=0.3`, `base_seed=0`.

## Outputs

- **`results/`** — `evaluation_summary.txt`, `random_search_results.csv`, `hill_climbing_results.csv`, `evaluation_results.json`
- **`videos/hill_climbing/`** — Crash videos from Hill Climbing (when a crash is found)
- **`videos/random_search/`** — Crash videos from Random Search (one per scenario that crashed)
