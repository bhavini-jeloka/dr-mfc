import torch
import matplotlib.pyplot as plt
import numpy as np
from .five_state_dynamics import FiveStateDynamicsEval
from ..mean_field_estimation import MeanFieldEstimator
from ..utils import *

num_states = 5
num_actions = 3
num_timesteps = 10
num_seeds = 10
comm_rounds_list = [0, 1, 5, 100]

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
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
axs = axs.flatten()
all_l1_errors = []

for idx, num_comm_rounds in enumerate(comm_rounds_list):
    rewards_actual_all_seeds = np.zeros((num_seeds, num_timesteps))
    rewards_desired_all_seeds = np.zeros((num_seeds, num_timesteps))
    l1_errors_all_seeds = np.zeros((num_seeds, num_timesteps))

    for seed in range(num_seeds):

        # Create a random generator with the current seed
        rng = np.random.default_rng(seed)
        true_mean_field = rng.dirichlet(np.ones(num_states))

        mean_field = true_mean_field.copy()
        des_mean_field = true_mean_field.copy()

        estimator = MeanFieldEstimator(num_states, horizon_length=1, comms_graph=G_comms, seed=seed)
        dynamics = FiveStateDynamicsEval(init_mean_field=mean_field, num_states=num_states, num_actions=num_actions)
        desired_dynamics = FiveStateDynamicsEval(init_mean_field=mean_field, num_states=num_states, num_actions=num_actions)

        for t in range(num_timesteps):
            
            l1_errors_all_seeds[seed, t] = np.sum(np.abs(mean_field - des_mean_field))
            
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
    
    avg_l1_error = l1_errors_all_seeds.mean(axis=0)
    all_l1_errors.append(avg_l1_error)

    # Plot
    cumsum_actual = np.cumsum(avg_actual)
    cumsum_desired = np.cumsum(avg_desired)
    axs[idx].plot(cumsum_actual, label='Actual Reward')
    axs[idx].plot(cumsum_desired, label='Desired Reward', linestyle='dashed')
    #axs[idx].plot(np.abs(cumsum_actual - cumsum_desired), label='|Cumulative Reward Diff|')
    axs[idx].set_title(f'Comm Rounds: {num_comm_rounds} | Error: {error_percent:.2%}')
    axs[idx].set_xlabel('Time')
    axs[idx].set_ylabel('Reward')
    axs[idx].legend()
    axs[idx].grid(True)
    
fig.suptitle("Mean-Field Estimation Performance for 5-States", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle
# Save the figure before displaying it
plt.savefig("mean_field_5_states.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 6))
for idx, l1_curve in enumerate(all_l1_errors):
    plt.plot(l1_curve, label=f'Comm Rounds: {comm_rounds_list[idx]}')

plt.title("Average L1 Error Between Estimated and Desired Mean Field")
plt.xlabel("Time")
plt.ylabel("L1 Norm Error")
plt.legend()
plt.grid(True)
plt.savefig("mean_field_l1_error.png", dpi=300)
plt.show()