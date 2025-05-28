import torch
import matplotlib.pyplot as plt
import numpy as np
from .five_state_dynamics import FiveStateDynamicsEval
from ..benchmark import BenchmarkEstimator
from ..utils import *

num_states = 5
num_actions = 3
num_timesteps = 10
num_seeds = 10
comm_rounds_list = [1, 2, 5, 10]

true_mean_field = np.array([0.02, 0.47, 0.02, 0.02, 0.47])

# Define communication graph
G_comms = np.zeros((num_states, num_states))
G_comms[0][1] = G_comms[1][0] = 1
G_comms[1][2] = G_comms[2][1] = 1
G_comms[2][3] = G_comms[3][2] = 1
G_comms[3][4] = G_comms[4][3] = 1

# Fixed indices
fixed_indices = {i: [i] for i in range(num_states)}

# Plot setup
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

for idx, num_comm_rounds in enumerate(comm_rounds_list):
    rewards_actual_all_seeds = np.zeros((num_seeds, num_timesteps))
    rewards_desired_all_seeds = np.zeros((num_seeds, num_timesteps))

    for seed in range(num_seeds):

        # Create a random generator with the current seed
        rng = np.random.default_rng(seed)
        true_mean_field = rng.dirichlet(np.ones(num_states))

        mean_field = true_mean_field.copy()
        des_mean_field = true_mean_field.copy()

        estimator = BenchmarkEstimator(num_states, horizon_length=1, comms_graph=G_comms, seed=seed)
        dynamics = FiveStateDynamicsEval(init_mean_field=mean_field, num_states=num_states, num_actions=num_actions)
        desired_dynamics = FiveStateDynamicsEval(init_mean_field=mean_field, num_states=num_states, num_actions=num_actions)

        for t in range(num_timesteps):
            
            # Rewards
            rewards_actual_all_seeds[seed, t] = dynamics.compute_reward()
            rewards_desired_all_seeds[seed, t] = desired_dynamics.compute_reward()

            # Mean field estimation
            fixed_values = get_fixed_values(fixed_indices, mean_field)
            estimator.initialize_estimate(fixed_indices, fixed_values)

            for _ in range(num_comm_rounds):
                estimator.get_new_info()
            estimator.compute_estimate()

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
    axs[idx].plot(avg_actual, label='Actual Reward')
    axs[idx].plot(avg_desired, label='Desired Reward', linestyle='dashed')
    axs[idx].set_title(f'Comm Rounds: {num_comm_rounds} | Error: {error_percent:.2%}')
    axs[idx].set_xlabel('Time')
    axs[idx].set_ylabel('Reward')
    axs[idx].legend()
    axs[idx].grid(True)
    
fig.suptitle("Mean-Field Estimation Performance for 5-States", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle
# Save the figure before displaying it
plt.savefig("mean_field_5_states_benchmark.png", dpi=300)
plt.show()