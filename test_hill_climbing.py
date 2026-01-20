#!/usr/bin/env python3
"""
Simple test script for hill climbing functions
"""

import numpy as np
from search.hill_climbing import compute_objectives_from_time_series, compute_fitness, mutate_config
from config.search_space import param_spec

def test_objectives():
    """Test objective computation"""
    # Test case 1: No crash, some distance
    time_series = [
        {
            "crashed": False,
            "ego": {"pos": [0, 0]},
            "others": [{"pos": [5, 0]}, {"pos": [10, 0]}]
        },
        {
            "crashed": False,
            "ego": {"pos": [1, 0]},
            "others": [{"pos": [4, 0]}, {"pos": [9, 0]}]
        }
    ]
    
    obj = compute_objectives_from_time_series(time_series)
    print(f"Test 1 - No crash: {obj}")
    assert obj["crash_count"] == 0
    assert obj["min_distance"] == 3.0  # minimum distance should be 3
    
    # Test case 2: Crash
    time_series_crash = [
        {
            "crashed": False,
            "ego": {"pos": [0, 0]},
            "others": [{"pos": [5, 0]}]
        },
        {
            "crashed": True,
            "ego": {"pos": [1, 0]},
            "others": [{"pos": [1, 0]}]
        }
    ]
    
    obj_crash = compute_objectives_from_time_series(time_series_crash)
    print(f"Test 2 - Crash: {obj_crash}")
    assert obj_crash["crash_count"] == 1
    
    print("✅ Objectives tests passed!")

def test_fitness():
    """Test fitness computation"""
    # Crash should have best fitness
    obj_crash = {"crash_count": 1, "min_distance": 10.0}
    fit_crash = compute_fitness(obj_crash)
    print(f"Crash fitness: {fit_crash}")
    assert fit_crash == -1.0
    
    # No crash fitness should be min_distance
    obj_no_crash = {"crash_count": 0, "min_distance": 2.5}
    fit_no_crash = compute_fitness(obj_no_crash)
    print(f"No crash fitness: {fit_no_crash}")
    assert fit_no_crash == 2.5
    
    # Crash should always be better than no crash
    assert fit_crash < fit_no_crash
    
    print("✅ Fitness tests passed!")

def test_mutation():
    """Test configuration mutation"""
    rng = np.random.default_rng(42)
    
    cfg = {
        "vehicles_count": 20,
        "lanes_count": 5,
        "initial_spacing": 2.0,
        "ego_spacing": 2.5,
        "initial_lane_id": 2,
        "duration": 30
    }
    
    # Test multiple mutations
    for i in range(5):
        mutated = mutate_config(cfg, param_spec, rng)
        print(f"Mutation {i+1}: {mutated}")
        
        # Check bounds
        for param, spec in param_spec.items():
            if param in mutated:
                assert spec["min"] <= mutated[param] <= spec["max"], f"{param} out of bounds"
        
        # Check lane consistency
        if "lanes_count" in mutated and "initial_lane_id" in mutated:
            assert 0 <= mutated["initial_lane_id"] < mutated["lanes_count"]
        
        # Original should not be modified
        assert cfg["vehicles_count"] == 20
    
    print("✅ Mutation tests passed!")

if __name__ == "__main__":
    test_objectives()
    test_fitness()
    test_mutation()
    print("🎉 All tests passed!")