import os
from pathlib import Path
import sys
import json
sys.path.append('../mf-env')

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import gym_mf_envs
from gym_mf_envs.wrappers import MultiDiscreteActionWrapper, FlattenObservationWrapper, AnyPopulationAllAgentDoneWrapper, PopulationRewardWrapper

import numpy as np
from .finite_population_estimation import Runner

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
                 num_population=config["num_population"], render_mode = config["render_mode"], num_agent_dict=config["num_agent_map"],
                 population_color_dict=config["population_color_list"],
                 identical_grid_world=config["identical_grid_world"],
                 penetrable_obstacles_dict=config["penetrable_obstacles"],
                 obstacle_locations_dict=config["obstacle_locations"], n_obstacles_dict=config["n_obstacles"],
                 random_init_obstacles_dict=config["random_obstacles"],
                 target_locations_dict=config["target_locations"], n_targets_dict=config["n_targets"], 
                 random_init_targets_dict=config["random_targets"],
                 action_map_dict=config["action_map"]))))), max_episode_steps=config["episode_length"])

    # Create a folder to store the files
    save_folder = "reset_initial_info"
    os.makedirs(save_folder, exist_ok=True)

    for reset_idx in range(100):
        obs, _ = env.reset()

        # Open a new file for this reset
        filename = os.path.join(save_folder, f"reset_{reset_idx}.txt")
        with open(filename, "w") as f:
            for population_id in env.agent_info:
                for agent_id in env.agent_info[population_id]:
                    agent_location = env.agent_info[population_id][agent_id]["current-location"]
                    agent_alive = env.agent_info[population_id][agent_id]["current-alive"]

                    # Write the info into the file
                    f.write(f"Population: {population_id}, Agent: {agent_id}, "
                            f"Location: {agent_location}, Alive: {agent_alive}\n")
        
if __name__ == "__main__":
    with open("config.json", "r") as jsonfile:
        config = json.load(jsonfile)
    main(config)