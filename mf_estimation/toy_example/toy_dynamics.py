import torch
import matplotlib.pyplot as plt
import numpy as np

class MeanFieldDynamicsEval():  # Under known fixed policy (included implicitly under transitions)
    def __init__(self, init_mean_field, num_states, num_actions, policy=None):
        super().__init__()

        self.num_states = num_states
        self.num_actions = num_actions
        self.mu = init_mean_field

    def transition_dynamics(self, policy):
        transition_matrix = np.zeros((self.num_states, self.num_states))
        transition_matrix[0, 0] = self.mu[1]*policy[0][1]
        transition_matrix[0, 1] = (1-self.mu[1])*policy[0][1]
        transition_matrix[0, 2] = policy[1][1]
        transition_matrix[1, 0] = policy[0][0]
        transition_matrix[1, 1] = 0
        transition_matrix[1, 2] = policy[0][1]
        transition_matrix[2, 0] = 1
        transition_matrix[2, 1] = 0
        transition_matrix[2, 2] = 0

        return transition_matrix  # Shape: (N, num_states, num_states)

    def compute_reward(self):
        return self.mu[0] - self.mu[1]

    def compute_next_mean_field(self, obs):
        policy = self.get_fixed_policy(obs)
        self.mu = self.mu@self.transition_dynamics(policy)

    def get_mf(self):
        return self.mu
    
    def get_fixed_policy(self, obs):

        policy = np.zeros((self.num_actions, self.num_states)) # pi[a][s] = pi(a|s)
        if type(obs)==dict:
            policy[0][0] = obs[0][2]
            policy[1][0] = 1 - policy[0][0] 
            policy[0][1] = obs[1][1]
            policy[1][1] = 1 - policy[0][1]
            policy[0][2] = 1
            policy[1][2] = 0
        else:
            policy[0][0] = obs[2]
            policy[1][0] = 1 - policy[0][0] 
            policy[0][1] = obs[1]
            policy[1][1] = 1 - policy[0][1]
            policy[0][2] = 1
            policy[1][2] = 0
        return policy
    