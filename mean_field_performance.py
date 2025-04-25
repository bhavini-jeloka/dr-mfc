import torch
import numpy as np

class DRMFC():
    def __init__(self, horizon_length=1, num_states=3):
        super().__init__()
        self.T = horizon_length
        self.num_states = num_states

    def transition_dynamics(self, pi_t, mu_t):
        """
        Compute the transition dynamics in a batched manner.
        - `mu_t`: (N, num_states) batch of distributions
        - `pi_t`: (N, 6) batch of policies
        - Returns: (N, num_states, num_states) transition matrices
        """
        N = mu_t.shape[0]  # Number of samples in the batch
        transition_matrix = torch.zeros((N, self.num_states, self.num_states), dtype=torch.float32)
        transition_matrix[:, 0, 0] = (1 - mu_t[:, 0]) * pi_t[0]
        transition_matrix[:, 0, 1] = mu_t[:, 0] * pi_t[0]
        transition_matrix[:, 0, 2] = pi_t[1]
        transition_matrix[:, 1, 1] = pi_t[2]
        transition_matrix[:, 1, 2] = pi_t[3]
        transition_matrix[:, 2, 1] = pi_t[4]
        transition_matrix[:, 2, 2] = pi_t[5]

        return transition_matrix  # Shape: (N, num_states, num_states)

    def reward_function(self, mu, t):
        """Compute reward function for batch input."""
        if t == 0:
            return torch.zeros(mu.shape[0], dtype=torch.float32)
        return -mu[:, 0] + mu[:, 1]  # Shape: (N,)

    def generate_mf_trajectory(self, mu_0, pi):
        """
        Compute mu_t over the time horizon in a batched manner.
        - `mu_0`: (N, num_states) batch of initial distributions
        - `pi`: (N, 6) batch of policies
        - Returns: List of (N, num_states) tensors for each time step
        """
        N = mu_0.shape[0]
        mu_t = mu_0  # Shape: (N, num_states)
        trajectory = [mu_t]  # Store all time-step distributions

        for t in range(self.T):
            transition_matrix = self.transition_dynamics(pi, mu_t)  # (N, num_states, num_states)
            mu_t_next = torch.bmm(mu_t.unsqueeze(1), transition_matrix).squeeze(1)  # (N, num_states)
            trajectory.append(mu_t_next)
            mu_t = mu_t_next  # Update

        return trajectory  # List of (N, num_states)

    def compute_cumulative_rewards(self, mu_0, pi):
        """
        Compute cumulative rewards for a batch of initial conditions.
        - `mu_0`: (N, num_states) batch of initial conditions
        - `pi`: (N, 6) batch of policies
        - Returns: (N,) tensor of cumulative rewards
        """
        trajectory = self.generate_mf_trajectory(mu_0, pi)  # List of (N, num_states)
        rewards = torch.zeros(mu_0.shape[0], dtype=torch.float32)  # (N,)

        for t, mu_t in enumerate(trajectory):
            rewards += self.reward_function(mu_t, t)  # Accumulate rewards across time steps

        return rewards  # (N,)
