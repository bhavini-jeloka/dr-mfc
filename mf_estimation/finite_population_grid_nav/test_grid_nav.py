import os
from pathlib import Path
import sys
import json
sys.path.append('../mf-env')

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import gym_mf_envs
from gym_mf_envs.wrappers import MultiDiscreteActionWrapper, FlattenObservationWrapper, AnyPopulationAllAgentDoneWrapper, PopulationRewardWrapper
from ..actor_network import PolicyNetwork

import numpy as np
from finite_population_estimation import Runner

def main(config):

    algo_name = config["algorithm"]
    env_name = config["env_name"]
    config["size"] = tuple(config["size"])

    for team in config["target_locations"]:
        if config["target_locations"][team] is not None:
            config["target_locations"][team] = [tuple(loc) for loc in config["target_locations"][team]]
            
    if config["obstacle_locations"] is not None:
        for team in config["obstacle_locations"]:
            if config["obstacle_locations"][team] is not None:
                config["obstacle_locations"][team] = [tuple(loc) for loc in config["obstacle_locations"][team]]

    # make environment
    env_fn = lambda: TimeLimit(AnyPopulationAllAgentDoneWrapper(PopulationRewardWrapper(FlattenObservationWrapper(
                 MultiDiscreteActionWrapper(gym.make(env_name, grid_size=config["size"],
                 num_population=config["num_population"], render_mode = config["render_mode"], num_agent_dict=config["num_agent_map"],
                 population_color_dict=config["population_color_list"],
                 identical_grid_world=config["identical_grid_world"],
                 penetrable_obstacles_dict=config["penetrable_obstacles"],
                 obstacle_locations_dict=config["obstacle_locations"], n_obstacles_dict=config["n_obstacles"],
                 random_init_obstacles_dict=config["random_obstacles"],
                 target_locations_dict=config["target_locations"], n_targets_dict=config["n_targets"], 
                 random_init_targets_dict=config["random_targets"],
                 action_map_dict=config["action_map"]))))), max_episode_steps=config["episode_length"])

    envs = gym.vector.AsyncVectorEnv([env_fn] * config["num_parallel_envs"])
    
    # run experiments
    runner = Runner(envs, config)
    runner.test()
    
if __name__ == "__main__":
    with open("config.json", "r") as jsonfile:
        config = json.load(jsonfile)
    main(config)