import torch
import matplotlib.pyplot as plt
import numpy as np

class FiveStateDynamicsEval():  # Under known fixed policy (included implicitly under transitions)
    def __init__(self, init_mean_field, num_states, num_actions, policy=None):
        super().__init__()

        self.num_states = num_states
        self.num_actions = num_actions
        self.mu = init_mean_field

    def transition_dynamics(self, policy):
        transition_matrix = np.zeros((self.num_states, self.num_states))
        
        transition_matrix[0, 0] = self.mu[0]*policy[0][0] + policy[1][0]
        transition_matrix[0, 1] = (1-self.mu[0])*policy[0][0] + policy[2][0]
        
        transition_matrix[1, 0] = policy[1][1]
        transition_matrix[1, 1] = policy[0][1]
        transition_matrix[1, 2] = policy[2][1]
        
        transition_matrix[2, 1] = 0.5*(1-self.mu[2])*policy[0][2] + policy[1][2]
        transition_matrix[2, 2] = self.mu[2]*policy[0][2]
        transition_matrix[2, 3] = 0.5*(1-self.mu[2])*policy[0][2] + policy[2][2]
        
        transition_matrix[3, 2] = policy[1][3]
        transition_matrix[3, 3] = policy[0][3]
        transition_matrix[3, 4] = policy[2][3]
        
        transition_matrix[4, 3] = (1-self.mu[4])*policy[0][4] + policy[1][4] 
        transition_matrix[4, 4] = self.mu[4]*policy[0][4] + policy[2][4]

        return transition_matrix  # Shape: (N, num_states, num_states)

    def compute_reward(self):
        return self.mu[0] - self.mu[1]

    def compute_next_mean_field(self, obs):
        policy = self.get_fixed_policy(obs)
        self.mu = self.mu@self.transition_dynamics(policy)

    def get_mf(self):
        return self.mu
    
    def get_new_comms_graph(self):
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

        policy = np.zeros((self.num_actions, self.num_states))
        
        if type(obs)==dict:
            
            policy[0][0] = obs[0].flatten()[4] 
            policy[2][0] = 1 - policy[0][0]
            
            policy[1][1] = obs[1].flatten()[0] 
            policy[2][1] = obs[1].flatten()[2] 
            policy[0][1] = 1 - policy[1][1] - policy[2][1]
            
            policy[1][2] = obs[2].flatten()[1] 
            policy[2][2] = obs[2].flatten()[3] 
            policy[0][2] = 1 - policy[1][2] - policy[2][2]
            
            policy[1][3] = obs[3].flatten()[2] 
            policy[2][3] = obs[3].flatten()[4] 
            policy[0][3] = 1 - policy[1][3] - policy[2][3]
            
            policy[0][4] = obs[4].flatten()[0] 
            policy[1][4] = 1 - policy[0][4] 
            
        else:
            policy[0][0] = obs[4] 
            policy[2][0] = 1 - policy[0][0]
            
            policy[1][1] = obs[0] 
            policy[2][1] = obs[2] 
            policy[0][1] = 1 - policy[1][1] - policy[2][1]
            
            policy[1][2] = obs[1] 
            policy[2][2] = obs[3]
            policy[0][2] = 1 - policy[1][2] - policy[2][2]
             
            policy[1][3] = obs[2] 
            policy[2][3] = obs[4]
            policy[0][3] = 1 - policy[1][3] - policy[2][3]
             
            policy[0][4] = obs[0] 
            policy[1][4] = 1 - policy[0][4] 
            
        return policy
    