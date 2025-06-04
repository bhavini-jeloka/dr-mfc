import torch
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

class LargeGridNavDynamicsEval():  # Under known fixed policy (included implicitly under transitions)
    def __init__(self, init_mean_field, num_states, num_actions, policy=None, init_G_comms=None):
        super().__init__()

        self.num_states = num_states
        self.grid = int(np.sqrt(self.num_states))
        self.num_actions = num_actions
        self.mu = init_mean_field
        self.policy = policy
        self.init_G_comms = init_G_comms

    def transition_dynamics(self, policy):
        transition_matrix = np.zeros((self.num_states, self.num_states))
        #TODO: fix
        
        return transition_matrix  # Shape: (N, num_states, num_states)
    
    def compute_reward(self):
        #TODO: fix
        return self.mu[0]

    def compute_next_mean_field(self, obs):
        policy = self.get_fixed_policy(obs)
        self.mu = self.mu@self.transition_dynamics(policy)

    def get_mf(self):
        return self.mu
    
    def get_new_comms_graph(self):
        G_sub, active_nodes = self.reconstruct_connected_subgraph()
        adj_matrix = np.zeros((self.num_states, self.num_states))

        # Fill in the subgraph structure at the correct indices
        for i, u in enumerate(active_nodes):
            for j, v in enumerate(active_nodes):
                adj_matrix[u, v] = G_sub[i, j]

        return adj_matrix
    
    def reconstruct_connected_subgraph(self):
        # Step 1: Identify active nodes
        active_nodes = np.where(self.mu > 0)[0]
        
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
    
    def get_fixed_policy(self, obs):
        #TODO: check this once
        act_dist = np.zeros((self.num_actions, self.num_states))

        # Create one-hot encoded local states (num_states x num_states)
        est_mf_local_states = np.eye(self.num_states).reshape(self.num_states, self.grid, self.grid)

        if isinstance(obs, dict):
            est_mf_global_states = np.array([obs[state].reshape(self.grid, self.grid) for state in range(self.num_states)])
        else:
            est_mf_global_states = np.tile(obs.reshape(1, self.grid, self.grid), (self.num_states, 1, 1))

        # Stack local and global state into (num_states, 2, grid, grid)
        concatenated_states = np.stack([est_mf_local_states, est_mf_global_states], axis=1)
        actions = self.policy.act(concatenated_states)  # Should return (num_states, num_actions)
        act_dist = actions.cpu().numpy().T  # Transpose to shape (num_actions, num_states)
        return act_dist
    