import torch
import matplotlib.pyplot as plt
import numpy as np
from .toy_dynamics import MeanFieldDynamicsEval
from ..mean_field_estimation import MeanFieldEstimator
from ..utils import *

num_states = 3
num_actions = 2
num_timesteps = 50
num_agents = 500
true_mean_field = (1/num_agents)*np.array([200, 50, 250])

# Define communication graph
G_comms = np.zeros((num_states, num_states))
G_comms[0][1] = G_comms[1][0] = 1
G_comms[1][2] = G_comms[2][1] = 1

fixed_indices = {0: [0], 1: [1], 2: [2]}
comm_rounds_list = [1, 2, 5, 10]  # Values to test

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

for idx, num_comm_rounds in enumerate(comm_rounds_list):
    mean_field = true_mean_field.copy()
    des_mean_field = true_mean_field.copy()
    
    estimator = MeanFieldEstimator(num_states=num_states, horizon_length=1, comms_graph=G_comms, seed=4)
    dynamics = MeanFieldDynamicsEval(init_mean_field=mean_field, num_states=num_states, num_actions=num_actions)
    desired_dynamics = MeanFieldDynamicsEval(init_mean_field=mean_field, num_states=num_states, num_actions=num_actions)
    
    actual_reward = []
    desired_reward = []

    for t in range(num_timesteps):
        actual_reward.append(dynamics.compute_reward())
        desired_reward.append(desired_dynamics.compute_reward())

        fixed_values = {
            0: [mean_field[0]],
            1: [mean_field[1]],
            2: [mean_field[2]]
        }

        estimator.initialize_mean_field(fixed_indices=fixed_indices, fixed_values=fixed_values)

        for R in range(num_comm_rounds):
            estimator.get_new_info()
            estimator.get_projected_average_estimate(fixed_indices, fixed_values)
            estimator.compute_estimate(copy=True)

        mean_field_estimate = estimator.get_mf_estimate()
        dynamics.compute_next_mean_field(obs=mean_field_estimate)
        mean_field = dynamics.get_mf()

        desired_dynamics.compute_next_mean_field(des_mean_field)
        des_mean_field = desired_dynamics.get_mf()

    error_percent = abs(np.sum(np.array(actual_reward) - np.array(desired_reward))) / abs(np.sum(desired_reward))

    # Plot results
    axs[idx].plot(actual_reward, label='Actual Rewards')
    axs[idx].plot(desired_reward, label='Desired Rewards', linestyle='dashed')
    axs[idx].set_title(f'Communication Rounds: {num_comm_rounds} | Error: {error_percent:.2%}')
    axs[idx].set_xlabel('Time')
    axs[idx].set_ylabel('Reward')
    axs[idx].legend()
    axs[idx].grid(True)

fig.suptitle("Mean-Field Estimation Performance using Metropolis Weights", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle
# Save the figure before displaying it
plt.savefig("mean_field_communication_comparison_metropolis_weights.png", dpi=300)
plt.show()