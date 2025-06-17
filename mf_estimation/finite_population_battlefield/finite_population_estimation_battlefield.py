import numpy as np
import torch
from ..actor_network import PolicyNetwork
from ..mean_field_estimation import MeanFieldEstimator
from ..benchmark import BenchmarkEstimator
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
    #TODO: fix communication graph - no communication with dead agents and subgrid comms - initial estimate assumes most are alive only - subgrid comms should be position based
    def __init__(self, env, config, algorithm_config=None, env_config=None):
        
        self.env_name = config['env_name']
        self.seed = config["seed"]
        
        self.max_ep_len = config["episode_length"]
        self.num_test_ep = config["num_test_episodes"] if "num_test_episodes" in config else None
        self._num_test_timesteps = self.num_test_ep*self.max_ep_len
        
        self.grid = config["size"]
        self.action_dim = config["action_dim"]
        self.num_states = 2*self.grid[0] * self.grid[1]
        
        self.num_agent_list = list(config["num_agent_map"].values())
        self.team_list = list(config["num_agent_map"].keys())
        self.num_population = config["num_population"]
        self.total_num_agents = sum(self.num_agent_list)

        self.policy_type = list(config["policy_type"].values())

        self.num_comm_rounds =  config["num_comm_rounds"]
        self.estimation_module = list(config["estimation_module"].values())
        self.comms_graph_struct = list(config["comms_graph_struct"].values())
        
        # Call environment 
        self.env = env
        
        if "render_mode" in config:
            self.render = 1
            self.frame_delay = 0

        self.model = []
        self.estimator = []

        self.init_G_comms = [] 

        self.fixed_indices = {i: [i] for i in range(self.num_states)}

        for i in range(self.num_population):
            # Initialize model
            self.model.append(PolicyNetwork(
                self.grid,
                state_dim_actor=(5, *self.grid),
                state_dim_critic=(4, *self.grid),
                action_dim=self.action_dim,
                policy_type=self.policy_type[i]
            ))

            module_type = self.estimation_module[i]
            
            if module_type in ["none", None]:
                self.init_G_comms.append(None)
                self.estimator.append(None)
                continue  # Skip graph and estimator initialization
            
            graph_type = self.comms_graph_struct[i]
            self.init_G_comms.append(self.initialize_communication_graph(graph_type, self.num_states, self.grid[0]))
            self.estimator.append(self.initialize_estimator(module_type, self.num_states, self.init_G_comms[i]))

        # seed
        if self.seed:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)

    def initialize_communication_graph(self, graph_type, num_states, grid_size):
        if graph_type == "linear":#TODO: this should ideally take opponent states but for us it is the same so it is alright
            return get_linear_adjacency_matrix(num_states=num_states)
        else:
            return get_adjacency_matrix(grid_size=grid_size)

    def initialize_estimator(self, module_type, num_states, comms_graph):
        if module_type == "benchmark":
            return BenchmarkEstimator(num_states, horizon_length=1, comms_graph=comms_graph)
        elif module_type == "d-pc":
            return MeanFieldEstimator(num_states, horizon_length=1, comms_graph=comms_graph)
        elif module_type in ["none", None]:
            return None
        else:
            raise ValueError(f"Unknown estimation module: {module_type}")
    
    def test(self):

        # track total training time
        test_running_reward = {team:0 for team in self.team_list}

        for ep in range(self.num_test_ep):

            state, _ = self.env.reset() 
            ep_reward = {team:0 for team in self.team_list}
            global_obs_list = [state["global-obs"].transpose(2, 0, 1).flatten().copy()]

            for t in range(self.max_ep_len):
                print("Communication round", self.num_comm_rounds, "| Episode:", ep, "| Timestep:", t)
            
                all_actions = np.zeros(self.total_num_agents, dtype=int)
                start_index = 0 

                for i in range(self.num_population):

                    if self.team_list[i]=="blue":
                        mean_field_opp = state["global-obs"].transpose(2, 0, 1)[2:]
                    else:
                        mean_field_opp = state["global-obs"].transpose(2, 0, 1)[:2]

                    mean_field_opp = mean_field_opp.flatten() 
                    fixed_values = get_fixed_values(self.fixed_indices, mean_field_opp)

                    mf_estimate = self.estimate_mean_field(
                        team=i,
                        estimator=self.estimator[i],
                        estimation_type=self.estimation_module[i],
                        mean_field=mean_field_opp,
                        fixed_indices=self.fixed_indices,
                        fixed_values=fixed_values,
                        num_comm_rounds=self.num_comm_rounds,
                        graph_type = self.comms_graph_struct[i],
                        t=t
                    )

                    action = self.model[i].get_est_based_actions(state, self.team_list[i], self.num_agent_list[i], opp_mf_estimate=mf_estimate)
                    end_index = start_index + self.num_agent_list[i]
                    all_actions[start_index: end_index] = action
                    start_index = end_index

                state, reward, done, terminated,_ = self.env.step(all_actions.astype(int))
                global_obs_list.append(state["global-obs"].transpose(2, 0, 1).flatten().copy())
                
                for team, rew in reward.items():
                    ep_reward[team] += rew
                
                if done or terminated:
                    for team in self.team_list:
                        test_running_reward[team] += ep_reward[team]
                        print('Reward Team {}: {}'.format(team, round(ep_reward[team], 2)))
                    ep_reward = {team:0 for team in self.team_list}

                    est_module_str = "_".join([f"{team}-{name}" for team, name in self.estimation_module.items()])
                    save_dir = f"mean_field_trajectory/grid_{self.grid[0]}x{self.grid[1]}_comm_{self.num_comm_rounds}_{est_module_str}"
                    os.makedirs(save_dir, exist_ok=True)
                    filename = os.path.join(save_dir, f"ep_{ep}.npy")
                    np.save(filename, np.array(global_obs_list))
                    break
            
        self.env.close()
        
        for team in self.team_list:
            avg_test_reward = test_running_reward[team] / self.num_test_ep
            print('average test reward team {}: {} '.format(team, str(avg_test_reward)))

    def estimate_mean_field(self, team, estimator, estimation_type, mean_field, fixed_indices, fixed_values, num_comm_rounds, graph_type, t):
        if estimation_type == "d-pc":
            # Initialize
            estimator.initialize_mean_field(fixed_indices, fixed_values) if t == 0 \
                else estimator.initialize_comm_round(fixed_indices=fixed_indices, fixed_values=fixed_values)

            # Update graph and perform communication rounds
            estimator.update_comms_graph(self.get_new_comms_graph(team, mean_field, graph_type))
            for _ in range(num_comm_rounds):
                estimator.get_new_info()
                estimator.get_projected_average_estimate(fixed_indices, fixed_values)
                estimator.compute_estimate(copy=True)

        elif estimation_type == "benchmark":
            estimator.initialize_estimate(fixed_indices=fixed_indices, fixed_values=fixed_values)
            estimator.update_comms_graph(self.get_new_comms_graph(team, mean_field, graph_type))
            for _ in range(num_comm_rounds):
                estimator.get_new_info()
            estimator.compute_estimate()

        else:
            return None

        return estimator.get_mf_estimate()
    
    def get_new_comms_graph(self, team, mu, graph_type):

        adj_matrix = np.zeros((self.num_states, self.num_states), dtype=int)
        
        if graph_type=="linear":
            # get new adjacency matrix based on graph and ensure connectedness - line graph at the moment s_i <-> s_{i+1}
            active_indices = [i for i, val in enumerate(mu) if val > 0]

            n_active = len(active_indices)
            if n_active <= 1:
                # Return a 0x0 or 1x1 matrix depending on if we have 0 or 1 active node
                return np.zeros((self.num_states, self.num_states), dtype=int)

            # Connect nodes in a simple path: i <-> i+1
            for i in range(n_active - 1):
                a, b = active_indices[i], active_indices[i + 1]
                adj_matrix[a, b] = 1
                adj_matrix[b, a] = 1

        else:

            G_sub, active_nodes = self.reconstruct_connected_subgraph(team, mu)

            # Fill in the subgraph structure at the correct indices
            for i, u in enumerate(active_nodes):
                for j, v in enumerate(active_nodes):
                    adj_matrix[u, v] = G_sub[i, j]

        return adj_matrix
    
    def reconstruct_connected_subgraph(self, team, mu):
        # Step 1: Identify active nodes
        active_nodes = np.where(mu > 0)[0]
        
        if len(active_nodes) == 1:
            # Single node is trivially connected
            return np.array([[0]]), active_nodes

        # Step 2: Induce subgraph
        G_full = nx.from_numpy_array(self.init_G_comms[team])
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
                        if self.init_G_comms[team][u, v] == 1 and (u, v) not in added and (v, u) not in added:
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