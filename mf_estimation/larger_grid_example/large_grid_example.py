import torch
import matplotlib.pyplot as plt
import numpy as np
from .large_grid_dynamics import LargeGridNavDynamicsEval
from ..actor_network import PolicyNetwork
from ..mean_field_estimation import MeanFieldEstimator
from ..utils import *

# Parameters
grid_size = 9
num_states = grid_size**2
num_actions = 5
targets = [[3, 3], [3, 4], [3, 5], [4, 3], [4, 4], [4, 5], 
                    [5, 3], [5, 4], [5, 5]]
obstacles = [[2, 3], [2, 4], [2, 5], [3, 2], [3, 6], [4, 2], [4, 6],
                    [5, 2], [5, 6], [6, 3], [6, 4], [6, 5]]
num_timesteps = 250
num_seeds = 10
comm_rounds_list = [10, 20, 30, 40, 50, 60, 70, 80]

# Fixed policy
policy = PolicyNetwork(
    state_dim_actor=(2, grid_size, grid_size),
    state_dim_critic=(1, grid_size, grid_size),
    action_dim=num_actions,
    policy_type="lcp_policy_9x9"
)

# True mean-field
true_mean_field = np.array([0.043, 0.127, 0.212, 0.014, 0.092, 0.169, 0.026, 0.183, 0.134])

# Communication graph (static here)
# Define communication graph
G_comms = get_linear_adjacency_matrix(num_states=num_states)

init_G_comms = G_comms.copy()

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
        true_mean_field = get_or_create_mean_field(seed, num_states, filename="init_grid_mean_fields.csv")

        mean_field = true_mean_field.copy()
        des_mean_field = true_mean_field.copy()

        estimator = MeanFieldEstimator(num_states, horizon_length=1, comms_graph=G_comms, 
                                       seed=seed)
        
        dynamics = LargeGridNavDynamicsEval(mean_field, num_states, num_actions,targets=targets,
                                            obstacles=obstacles, policy=policy, 
                                            init_G_comms=init_G_comms)
        desired_dynamics = LargeGridNavDynamicsEval(des_mean_field, num_states, num_actions,
                                            targets=targets,obstacles=obstacles, policy=policy)
        
        fixed_values = get_fixed_values(fixed_indices, mean_field)
        estimator.initialize_mean_field(fixed_indices, fixed_values)
        
        for t in range(num_timesteps):
            print("D-PC", "| Communication round", num_comm_rounds, "| Seed:", seed, "| Timestep:", t)
            
            l1_errors_all_seeds[seed, t] = 0.5*np.sum(np.abs(mean_field - des_mean_field))
            
            # Rewards
            rewards_actual_all_seeds[seed, t] = dynamics.compute_reward()
            rewards_desired_all_seeds[seed, t] = desired_dynamics.compute_reward()

            # Mean field estimation
            fixed_values = get_fixed_values(fixed_indices, mean_field)
            estimator.initialize_comm_round(fixed_indices=fixed_indices, fixed_values=fixed_values)

            for _ in range(num_comm_rounds):
                estimator.get_new_info()
                estimator.get_projected_average_estimate(fixed_indices, fixed_values)
                estimator.compute_estimate(copy=True)

            # Update mean fields
            mf_estimate = estimator.get_mf_estimate()
            dynamics.compute_next_mean_field(obs=mf_estimate)
            mean_field = dynamics.get_mf()

            new_graph = dynamics.get_new_comms_graph_linearly()
            estimator.update_comms_graph(new_graph)

            desired_dynamics.compute_next_mean_field(des_mean_field)
            des_mean_field = desired_dynamics.get_mf()

    np.save(f'rewards_actual_dpc_all_seeds_{num_comm_rounds}.npy', rewards_actual_all_seeds)
    np.save(f'rewards_desired_all_seeds_{num_comm_rounds}.npy', rewards_desired_all_seeds)
    np.save(f'l1_errors_dpc_all_seeds_{num_comm_rounds}.npy', l1_errors_all_seeds)
