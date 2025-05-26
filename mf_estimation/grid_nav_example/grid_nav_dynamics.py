import torch
import matplotlib.pyplot as plt
import numpy as np

class GridNavDynamicsEval():  # Under known fixed policy (included implicitly under transitions)
    def __init__(self, init_mean_field, num_states, num_actions, policy=None):
        super().__init__()

        self.num_states = num_states
        self.grid = int(np.sqrt(self.num_states))
        self.num_actions = num_actions
        self.mu = init_mean_field
        self.policy = policy

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
        transition_matrix[5, 4] = policy[2][5]
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
        #TODO: fix this
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
    