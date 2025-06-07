import torch
import matplotlib.pyplot as plt
import numpy as np
from .five_state_dynamics import FiveStateDynamicsEval
from ..benchmark import BenchmarkEstimator
from ..utils import *

num_states = 5
num_actions = 3
num_timesteps = 100
num_seeds = 10
comm_rounds_list = [1, 2, 3, 4]

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
all_l1_errors = []

for idx, num_comm_rounds in enumerate(comm_rounds_list):
    rewards_actual_all_seeds = np.zeros((num_seeds, num_timesteps))
    rewards_desired_all_seeds = np.zeros((num_seeds, num_timesteps))
    l1_errors_all_seeds = np.zeros((num_seeds, num_timesteps))

    for seed in range(num_seeds):

        # Create a random generator with the current seed
        true_mean_field = get_or_create_mean_field(seed, num_states)

        mean_field = true_mean_field.copy()
        des_mean_field = true_mean_field.copy()

        estimator = BenchmarkEstimator(num_states, horizon_length=1, comms_graph=G_comms, seed=seed)
        dynamics = FiveStateDynamicsEval(init_mean_field=mean_field, num_states=num_states, num_actions=num_actions)
        desired_dynamics = FiveStateDynamicsEval(init_mean_field=mean_field, num_states=num_states, num_actions=num_actions)

        for t in range(num_timesteps):

            l1_errors_all_seeds[seed, t] = 0.5*np.sum(np.abs(mean_field - des_mean_field))
            
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

    np.save(f'rewards_actual_benchmark_all_seeds_{num_comm_rounds}.npy', rewards_actual_all_seeds)
    np.save(f'rewards_desired_all_seeds_{num_comm_rounds}.npy', rewards_desired_all_seeds)
    np.save(f'l1_errors_benchmark_all_seeds_{num_comm_rounds}.npy', l1_errors_all_seeds)
