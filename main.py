from config.search_space import param_spec, base_cfg
from policies.pretrained_policy import load_pretrained_policy
from envs.highway_env_utils import make_env
from search.random_search import RandomSearch
from search.hill_climbing import HillClimbingSearch

def main():
    env_id = "highway-fast-v0"
    policy = load_pretrained_policy("agents/model")
    env, defaults = make_env(env_id)

    # Test Hill Climbing
    hc_search = HillClimbingSearch(env_id, base_cfg, param_spec, policy, defaults)
    crashes = hc_search.run_search(iterations=10, neighbors_per_iter=10, seed=11)

    # print(f"✅ Found {len(crashes)} crashes with Hill Climbing.")
    
    #Uncomment to also run random search for comparison
    # search = RandomSearch(env_id, base_cfg, param_spec, policy, defaults)
    # crashes_random = search.run_search(n_scenarios=50, seed=11)
    # print(f"✅ Found {len(crashes_random)} crashes with Random Search.")

if __name__ == "__main__":
    main()