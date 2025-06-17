import os
from pathlib import Path
import sys
import json
import glob
import re
sys.path.append('../mf-env')

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import gym_mf_envs
from gym_mf_envs.wrappers import MultiDiscreteActionWrapper, FlattenObservationWrapper, AnyPopulationAllAgentDoneWrapper, PopulationRewardWrapper
from ..actor_network import PolicyNetwork

import numpy as np
from .finite_population_estimation_battlefield import Runner

def main(config):

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
    env =       TimeLimit(AnyPopulationAllAgentDoneWrapper(PopulationRewardWrapper(FlattenObservationWrapper(
                 MultiDiscreteActionWrapper(gym.make(env_name, grid_size=config["size"],
                 num_population=config["num_population"], num_agent_dict=config["num_agent_map"],
                 population_color_dict=config["population_color_list"],
                 identical_grid_world=config["identical_grid_world"],
                 penetrable_obstacles_dict=config["penetrable_obstacles"],
                 obstacle_locations_dict=config["obstacle_locations"], n_obstacles_dict=config["n_obstacles"],
                 random_init_obstacles_dict=config["random_obstacles"],
                 target_locations_dict=config["target_locations"], n_targets_dict=config["n_targets"], 
                 random_init_targets_dict=config["random_targets"],
                 action_map_dict=config["action_map"], oracle_kwargs=config["oracle_kwargs"]))))), max_episode_steps=config["episode_length"])

    
    # run experiments
    runner = Runner(env, config)
    runner.test()   # estimation technique in config: {"blue": "d-pc", "red": null}
    
if __name__ == "__main__":
    with open("configs/config_1.json", "r") as jsonfile:
            config = json.load(jsonfile)
    main(config)

    '''
    def numeric_key(path):
        match = re.search(r"config_(\d+)\.json", path)
        return int(match.group(1)) if match else -1

    config_files = sorted(glob.glob("configs/config_*.json"), key=numeric_key)

    for config_path in config_files:
        print(f"\nLoading config: {config_path}")
        with open(config_path, "r") as jsonfile:
            config = json.load(jsonfile)
        main(config)
    '''