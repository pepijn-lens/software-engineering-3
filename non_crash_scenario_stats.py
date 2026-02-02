"""
Print parameter statistics for ALL non-crash evaluations (every eval that did not crash).
- Hill Climbing: from evaluation_history, every entry with crashed=False.
- Random Search: from all_evaluations, every entry with crashed=False (config stored per eval).
From results/evaluation_results.json. Re-run main.py to get RS per-eval configs if missing.
"""

import json
from pathlib import Path

PARAMS = [
    ("vehicles_count", "Vehicles count"),
    ("lanes_count", "Lanes count"),
    ("initial_spacing", "Initial spacing"),
    ("initial_lane_id", "Initial lane ID"),
    ("ego_spacing", "Ego spacing"),
]


def stats(cfgs, name):
    if not cfgs:
        print(f"{name}: no non-crash evaluations")
        return
    n = len(cfgs)
    lines = [f"{name} Non-Crash Evaluations (n={n}):"]
    for key, label in PARAMS:
        raw = [c[key] for c in cfgs if key in c]
        if not raw:
            continue
        vals = [float(x) for x in raw]
        mean = sum(vals) / len(vals)
        var = sum((x - mean) ** 2 for x in vals) / len(vals)
        std = var ** 0.5
        lines.append(f"  - {label}: {mean:.2f} ± {std:.2f}")
    print("\n".join(lines))


def main():
    path = Path(__file__).parent / "results" / "evaluation_results.json"
    with open(path) as f:
        data = json.load(f)

    rs = data["random_search_results"]
    hc = data["hill_climbing_results"]

    # RS: all evaluations that did not crash (use config from each eval when present)
    rs_non = []
    for r in rs:
        for ev in r.get("all_evaluations", []):
            if ev.get("crashed", True):
                continue
            cfg = ev.get("config")
            if cfg is not None:
                rs_non.append(cfg)
    if not rs_non and any(not r["best_result"]["crashed"] for r in rs):
        rs_non = [r["best_cfg"] for r in rs if not r["best_result"]["crashed"]]
        print("(RS: no per-eval config in JSON; using best_cfg per non-crash scenario. Re-run main.py to get all non-crash evals.)\n")

    # HC: all evaluation_history entries that did not crash
    hc_non = []
    for r in hc:
        for eh in r.get("evaluation_history", []):
            if not eh.get("crashed", True):
                hc_non.append(eh["config"])

    print("4. NON-CRASH EVALUATION CHARACTERISTICS (all evals without crash)")
    print("-" * 60)
    stats(rs_non, "Random Search")
    print()
    stats(hc_non, "Hill Climbing")


if __name__ == "__main__":
    main()
