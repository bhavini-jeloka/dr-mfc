import torch
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

class GridNavDynamicsEval():  # Under known fixed policy (included implicitly under transitions)
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

        transition_matrix[0, 0] = policy[4][0] + (1-self.pen_prob(self.mu[0]))*policy[0][0] + policy[2][0] + policy[3][0]
        transition_matrix[1, 0] = policy[2][1]
        transition_matrix[2, 0] = 0
        transition_matrix[3, 0] = policy[3][3]
        transition_matrix[4, 0] = 0
        transition_matrix[5, 0] = 0
        transition_matrix[6, 0] = 0
        transition_matrix[7, 0] = 0
        transition_matrix[8, 0] = 0

        transition_matrix[0, 1] = self.pen_prob(self.mu[0])*policy[0][0]
        transition_matrix[1, 1] = policy[4][1] + policy[3][1] + (1-self.pen_prob(self.mu[1]))*policy[1][1]
        transition_matrix[2, 1] = self.pen_prob(self.mu[2])*policy[2][2]
        transition_matrix[3, 1] = 0
        transition_matrix[4, 1] = self.pen_prob(self.mu[4])*policy[3][4]
        transition_matrix[5, 1] = 0
        transition_matrix[6, 1] = 0
        transition_matrix[7, 1] = 0
        transition_matrix[8, 1] = 0

        transition_matrix[0, 2] = 0
        transition_matrix[1, 2] = policy[0][1]
        transition_matrix[2, 2] = policy[4][2] + (1-self.pen_prob(self.mu[2]))*policy[2][2] + policy[3][2] + policy[0][2]
        transition_matrix[3, 2] = 0
        transition_matrix[4, 2] = 0
        transition_matrix[5, 2] = policy[3][5]
        transition_matrix[6, 2] = 0
        transition_matrix[7, 2] = 0
        transition_matrix[8, 2] = 0

        transition_matrix[0, 3] = policy[1][0]
        transition_matrix[1, 3] = 0
        transition_matrix[2, 3] = 0
        transition_matrix[3, 3] = policy[4][3] + policy[2][3] + (1-self.pen_prob(self.mu[3]))*policy[0][3]
        transition_matrix[4, 3] = policy[2][4]
        transition_matrix[5, 3] = 0
        transition_matrix[6, 3] = policy[3][6]
        transition_matrix[7, 3] = 0
        transition_matrix[8, 3] = 0

        transition_matrix[0, 4] = 0
        transition_matrix[1, 4] = self.pen_prob(self.mu[1])*policy[1][1]
        transition_matrix[2, 4] = 0
        transition_matrix[3, 4] = self.pen_prob(self.mu[3])*policy[0][3]
        transition_matrix[4, 4] = policy[4][4] + (1-self.pen_prob(self.mu[4]))*policy[3][4] + (1-self.pen_prob(self.mu[4]))*policy[1][4]
        transition_matrix[5, 4] = self.pen_prob(self.mu[5])*policy[2][5]
        transition_matrix[6, 4] = 0
        transition_matrix[7, 4] = self.pen_prob(self.mu[7])*policy[3][7]
        transition_matrix[8, 4] = 0

        transition_matrix[0, 5] = 0
        transition_matrix[1, 5] = 0
        transition_matrix[2, 5] = policy[1][2]
        transition_matrix[3, 5] = 0
        transition_matrix[4, 5] = policy[0][4]
        transition_matrix[5, 5] = policy[4][5] + policy[0][5] + (1-self.pen_prob(self.mu[5]))*policy[2][5]
        transition_matrix[6, 5] = 0
        transition_matrix[7, 5] = 0
        transition_matrix[8, 5] = policy[3][8]

        transition_matrix[0, 6] = 0
        transition_matrix[1, 6] = 0
        transition_matrix[2, 6] = 0
        transition_matrix[3, 6] = policy[1][3]
        transition_matrix[4, 6] = 0
        transition_matrix[5, 6] = 0
        transition_matrix[6, 6] = policy[4][6] + (1-self.pen_prob(self.mu[6]))*policy[0][6] + policy[2][6] + policy[1][6]
        transition_matrix[7, 6] = policy[2][7]
        transition_matrix[8, 6] = 0

        transition_matrix[0, 7] = 0
        transition_matrix[1, 7] = 0
        transition_matrix[2, 7] = 0
        transition_matrix[3, 7] = 0
        transition_matrix[4, 7] = self.pen_prob(self.mu[4])*policy[1][4]
        transition_matrix[5, 7] = 0
        transition_matrix[6, 7] = self.pen_prob(self.mu[6])*policy[0][6]
        transition_matrix[7, 7] = policy[4][7] + policy[1][7] + (1-self.pen_prob(self.mu[7]))*policy[3][7]
        transition_matrix[8, 7] = self.pen_prob(self.mu[8])*policy[2][8]

        transition_matrix[0, 8] = 0
        transition_matrix[1, 8] = 0
        transition_matrix[2, 8] = 0
        transition_matrix[3, 8] = 0
        transition_matrix[4, 8] = 0
        transition_matrix[5, 8] = policy[1][5]
        transition_matrix[6, 8] = 0
        transition_matrix[7, 8] = policy[0][7]
        transition_matrix[8, 8] = policy[4][8] + (1-self.pen_prob(self.mu[8]))*policy[2][8] + policy[1][8] + policy[0][8]

        return transition_matrix  # Shape: (N, num_states, num_states)

    def pen_prob(self, x):
        return min(1, np.exp(10*(x-0.8)))
    
    def compute_reward(self):
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

        # Step 4: Make connected by adding edges from original G_comms
        # First, get the connected components
        components = list(nx.connected_components(G_sub))
        new_edges = []
        added = set()

        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                min_edge = None
                min_weight = float('inf')
                # Find the closest pair of nodes between components i and j
                for u in components[i]:
                    for v in components[j]:
                        if self.init_G_comms[u, v] == 1 and (u, v) not in added and (v, u) not in added:
                            # Prefer edges in original G_comms
                            min_edge = (u, v)
                            break
                    if min_edge:
                        break
                if min_edge:
                    new_edges.append(min_edge)
                    added.add(min_edge)

        # Add new edges to G_sub
        G_sub.add_edges_from(new_edges)

        # Now connected
        assert nx.is_connected(G_sub)
        return nx.to_numpy_array(G_sub, nodelist=active_nodes), active_nodes
    
    def get_fixed_policy(self, obs):
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
    