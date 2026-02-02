import warnings
import os

# Suppress warnings for cleaner output - must be set before any imports
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*deprecated.*")
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
os.environ['PYTHONWARNINGS'] = "ignore::UserWarning"

from evaluation import run_evaluation

def main():
    results = run_evaluation(
        n_scenarios=50,
        hc_iterations=10,
        hc_neighbors_per_iter=10,
        hc_mutation_rate=0.1,
        base_seed=1,
        results_dir="results"
    )



if __name__ == "__main__":
    main()