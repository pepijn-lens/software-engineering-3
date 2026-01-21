import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.on_policy_algorithm")

from config.search_space import param_spec, base_cfg
from policies.pretrained_policy import load_pretrained_policy
from envs.highway_env_utils import make_env
from search.random_search import RandomSearch
from search.hill_climbing import HillClimbSearch

def main():
    env_id = "highway-fast-v0"
    policy = load_pretrained_policy("agents/model")
    env, defaults = make_env(env_id)

    # search = RandomSearch(env_id, base_cfg, param_spec, policy, defaults)
    search = HillClimbSearch(env_id, base_cfg, param_spec, policy, defaults)
    results = search.run_search(iterations=10, neighbors_per_iter=10)
    print(results)

if __name__ == "__main__":
    main()