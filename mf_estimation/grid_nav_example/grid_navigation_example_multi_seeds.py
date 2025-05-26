import torch
import matplotlib.pyplot as plt
import numpy as np
from .grid_nav_dynamics import GridNavDynamicsEval
from .actor_network import PolicyNetwork
from ..mean_field_estimation import MeanFieldEstimator
from ..utils import *

# Parameters
grid_size = 3
num_states = grid_size**2
num_actions = 5
num_timesteps = 10
num_seeds = 100
comm_rounds_list = [0, 1, 2, 5]

# Fixed policy
policy = PolicyNetwork(
    state_dim_actor=(2, grid_size, grid_size),
    state_dim_critic=(1, grid_size, grid_size),
    action_dim=num_actions,
    policy_type="non_lcp_policy"
)

# True mean-field
true_mean_field = np.array([0.043, 0.127, 0.212, 0.014, 0.092, 0.169, 0.026, 0.183, 0.134])

# Communication graph (static here)
G_comms = np.zeros((num_states, num_states))

# Fixed indices
fixed_indices = {i: [i] for i in range(num_states)}

# Plot setup
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

for idx, num_comm_rounds in enumerate(comm_rounds_list):
    rewards_actual_all_seeds = np.zeros((num_seeds, num_timesteps))
    rewards_desired_all_seeds = np.zeros((num_seeds, num_timesteps))

    for seed in range(num_seeds):
        mean_field = true_mean_field.copy()
        des_mean_field = true_mean_field.copy()

        estimator = MeanFieldEstimator(num_states, horizon_length=1, comms_graph=G_comms, seed=seed)
        dynamics = GridNavDynamicsEval(mean_field, num_states, num_actions, policy)
        desired_dynamics = GridNavDynamicsEval(des_mean_field, num_states, num_actions, policy)

        for t in range(num_timesteps):
            # Rewards
            rewards_actual_all_seeds[seed, t] = dynamics.compute_reward()
            rewards_desired_all_seeds[seed, t] = desired_dynamics.compute_reward()

            # Mean field estimation
            fixed_values = get_fixed_values(fixed_indices, mean_field)
            estimator.initialize_mean_field(fixed_indices, fixed_values)

            for _ in range(num_comm_rounds):
                estimator.get_new_info()
                estimator.get_projected_average_estimate(fixed_indices, fixed_values)
                estimator.compute_estimate(copy=True)

            # Update mean fields
            mf_estimate = estimator.get_mf_estimate()
            dynamics.compute_next_mean_field(obs=mf_estimate)
            mean_field = dynamics.get_mf()

            new_graph = dynamics.get_new_comms_graph()
            estimator.update_comms_graph(new_graph)

            desired_dynamics.compute_next_mean_field(des_mean_field)
            des_mean_field = desired_dynamics.get_mf()

    # Average across seeds
    avg_actual = rewards_actual_all_seeds.mean(axis=0)
    avg_desired = rewards_desired_all_seeds.mean(axis=0)
    error_percent = np.abs(np.sum(avg_actual - avg_desired)) / np.abs(np.sum(avg_desired))

    # Plot
    axs[idx].plot(avg_actual, label='Avg Actual')
    axs[idx].plot(avg_desired, label='Avg Desired', linestyle='dashed')
    axs[idx].set_title(f'Comm Rounds: {num_comm_rounds} | Error: {error_percent:.2%}')
    axs[idx].set_xlabel('Time')
    axs[idx].set_ylabel('Reward')
    axs[idx].legend()
    axs[idx].grid(True)

fig.suptitle("Mean-Field Estimation (Avg over Seeds) - LCP Policy", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("mean_field_comm_grid_nav_avg_seeds.png", dpi=300)
plt.show()
