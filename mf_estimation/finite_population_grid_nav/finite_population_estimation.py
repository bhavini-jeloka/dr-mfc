# Use test file and for mf-mappo, use the network we have instead and restructure code
# Begin with no obstacles and basic nav under LCP and see videos and then add obstacles
# Add estimator as a separate plug-in module while cleaning code - maybe remove num env

import numpy as np
import torch
from ..actor_network import PolicyNetwork
from ..mean_field_estimation import MeanFieldEstimator
from ..utils import *
import importlib
import sys
from datetime import datetime
import time
from pathlib import Path
import copy
import matplotlib.pyplot as plt
import networkx as nx

sys.path.append('../mf-env')
import gym_mf_envs

class Runner():
    def __init__(self, env, config, algorithm_config=None, env_config=None):
        
        self.env_name = config['env_name']
        self.seed = config["seed"]
        
        self.max_ep_len = config["episode_length"]
        self.num_test_ep = config["num_test_episodes"] if "num_test_episodes" in config else None
        self._num_test_timesteps = self.num_test_ep*self.max_ep_len
        
        self.grid = config["size"]
        self.action_dim = config["action_dim"]
        self.num_states = self.grid[0] * self.grid[1]
        
        self.num_agent_list = list(config["num_agent_map"].values())
        self.team_list = list(config["num_agent_map"].keys())
        self.num_population = config["num_population"]
        self.total_num_agents = sum(self.num_agent_list)

        self.num_comm_rounds =  config["num_comm_rounds"]
        
        # Call environment 
        self.env = env

        if "render_mode" in config:
            self.render = 1
            self.frame_delay = 0

        self.model = PolicyNetwork(self.grid, state_dim_actor=(2, *self.grid), state_dim_critic=(1, *self.grid), action_dim=self.action_dim, policy_type=config["policy_type"])
        
        self.init_G_comms = get_adjacency_matrix(grid_size=self.grid[0])
        self.fixed_indices = {i: [i] for i in range(self.num_states)}
        self.estimator = MeanFieldEstimator(self.num_states, horizon_length=1, comms_graph=self.init_G_comms)
        
        # seed
        if self.seed:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
    
    def test(self):

        # track total training time
        test_running_reward = {team:0 for team in self.team_list}

        for ep in range(self.num_test_ep):

            state, _ = self.env.reset()
            ep_reward = {team:0 for team in self.team_list}

            fixed_values = get_fixed_values(self.fixed_indices, state["global-obs"].transpose(2, 0, 1).flatten())
            self.estimator.initialize_mean_field(self.fixed_indices, fixed_values)

            for t in range(self.max_ep_len):
                print('Timestep:', t)
            
                all_actions = np.zeros(self.total_num_agents, dtype=int)
                start_index = 0 

                for i in range(self.num_population):
                    mean_field = state["global-obs"].transpose(2, 0, 1).flatten()
                    fixed_values = get_fixed_values(self.fixed_indices, mean_field) 
                    self.estimator.initialize_comm_round(fixed_indices=self.fixed_indices, fixed_values=fixed_values)

                    new_graph = self.get_new_comms_graph(mean_field)
                    self.estimator.update_comms_graph(new_graph)

                    for _ in range(self.num_comm_rounds):
                        self.estimator.get_new_info()
                        self.estimator.get_projected_average_estimate(self.fixed_indices, fixed_values)
                        self.estimator.compute_estimate(copy=True)

                    mf_estimate = self.estimator.get_mf_estimate()
                    action = self.model.get_actions(state, mf_estimate, self.team_list[i], self.num_agent_list[i])
                    end_index = start_index + self.num_agent_list[i]
                    all_actions[start_index: end_index] = action
                    start_index = end_index

                state, reward, done, terminated,_ = self.env.step(all_actions.astype(int))
                
                for team, rew in reward.items():
                    ep_reward[team] += rew
                
                if done or terminated:
                    for team in self.team_list:
                        test_running_reward[team] += ep_reward[team]
                        print('Reward Team {}: {}'.format(team, round(ep_reward[team], 2)))
                    ep_reward = {team:0 for team in self.team_list}
                    break
            
        self.env.close()
        
        for team in self.team_list:
            avg_test_reward = test_running_reward[team] / self.num_test_ep
            print('average test reward team {}: {} '.format(team, str(avg_test_reward)))
    
    def get_new_comms_graph(self, mu):
        G_sub, active_nodes = self.reconstruct_connected_subgraph(mu)
        adj_matrix = np.zeros((self.num_states, self.num_states))

        # Fill in the subgraph structure at the correct indices
        for i, u in enumerate(active_nodes):
            for j, v in enumerate(active_nodes):
                adj_matrix[u, v] = G_sub[i, j]

        return adj_matrix
    
    def get_new_comms_graph_linearly(self, mu):
        # get new adjacency matrix based on graph and ensure connectedness - line graph at the moment s_i <-> s_{i+1}
        active_indices = [i for i, val in enumerate(self.mu) if val > 0]

        n_active = len(active_indices)
        if n_active <= 1:
            # Return a 0x0 or 1x1 matrix depending on if we have 0 or 1 active node
            return np.zeros((self.num_states, self.num_states), dtype=int)

        # Initialize adjacency matrix
        adj_matrix = np.zeros((self.num_states, self.num_states), dtype=int)

        # Connect nodes in a simple path: i <-> i+1
        for i in range(n_active - 1):
            a, b = active_indices[i], active_indices[i + 1]
            adj_matrix[a, b] = 1
            adj_matrix[b, a] = 1

        return adj_matrix
    
    def reconstruct_connected_subgraph(self, mu):
        # Step 1: Identify active nodes
        active_nodes = np.where(mu > 0)[0]
        
        if len(active_nodes) == 1:
            # Single node is trivially connected
            return np.array([[0]]), active_nodes

        # Step 2: Induce subgraph
        G_full = nx.from_numpy_array(self.init_G_comms)
        G_sub = G_full.subgraph(active_nodes).copy()

        # Step 3: Check if connected
        if nx.is_connected(G_sub):
            return nx.to_numpy_array(G_sub, nodelist=active_nodes), active_nodes

        # Step 4: Get components and connect them with fallback logic
        components = list(nx.connected_components(G_sub))

        new_edges = []
        added = set()

        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                min_edge = None

                # First try to find an edge in init_G_comms
                for u in components[i]:
                    for v in components[j]:
                        if self.init_G_comms[u, v] == 1 and (u, v) not in added and (v, u) not in added:
                            min_edge = (u, v)
                            break
                    if min_edge:
                        break

                # Fallback if no edge from original graph
                if not min_edge:
                    u = next(iter(components[i]))
                    v = next(iter(components[j]))
                    min_edge = (u, v)

                if min_edge and (min_edge not in added and (min_edge[1], min_edge[0]) not in added):
                    new_edges.append(min_edge)
                    added.add(min_edge)

        # Add new edges to G_sub
        G_sub.add_edges_from(new_edges)

        # Ensure graph is now connected
        assert nx.is_connected(G_sub), "Graph is still not connected after edge augmentation."

        return nx.to_numpy_array(G_sub, nodelist=active_nodes), active_nodes