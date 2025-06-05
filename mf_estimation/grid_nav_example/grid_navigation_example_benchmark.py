import torch
import matplotlib.pyplot as plt
import numpy as np
from .grid_nav_dynamics import GridNavDynamicsEval
from ..actor_network import PolicyNetwork
from ..benchmark import BenchmarkEstimator
from ..utils import *

grid_size  = 3
num_states = grid_size**2
num_actions = 5
num_timesteps = 10

policy = PolicyNetwork(state_dim_actor=(2, grid_size, grid_size), state_dim_critic=(1, grid_size, grid_size), action_dim=num_actions, policy_type="lcp_policy_3x3")

true_mean_field = np.array([0.043, 0.127, 0.212, 0.014, 0.092, 0.169, 0.026, 0.183, 0.134]) #np.random.dirichlet(np.ones(9))

# Define communication graph
G_comms = np.array([[0, 1, 0, 1, 1, 0, 0, 0, 0],
                    [1, 0, 1, 1, 1, 1, 0, 0, 0], 
                    [0, 1, 0, 0, 1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 1, 0, 1, 1, 0],
                    [1, 1, 1, 1, 0, 1, 1, 1, 1],
                    [0, 1, 1, 0, 1, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1, 0, 0, 1, 0],
                    [0, 0, 0, 1, 1, 1, 1, 0, 1],
                    [0, 0, 0, 0, 1, 1, 0, 1, 0]])

init_G_comms = G_comms.copy()


fixed_indices = {0: [0], 1: [1], 2: [2], 3: [3], 4:[4], 5:[5], 6:[6], 7:[7], 8:[8]}
comm_rounds_list = [1, 2, 5, 10]  # Values to test

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

for idx, num_comm_rounds in enumerate(comm_rounds_list):
    mean_field = true_mean_field.copy()
    des_mean_field = true_mean_field.copy()
    
    estimator = BenchmarkEstimator(num_states=num_states, horizon_length=1, comms_graph=G_comms, seed=2)
    dynamics = GridNavDynamicsEval(init_mean_field=mean_field, num_states=num_states, num_actions=num_actions, policy=policy, init_G_comms=init_G_comms)
    desired_dynamics = GridNavDynamicsEval(init_mean_field=mean_field, num_states=num_states, num_actions=num_actions, policy=policy)
    
    actual_reward = []
    desired_reward = []

    for t in range(num_timesteps):
        
        print("timestep", t)
        actual_reward.append(dynamics.compute_reward())
        desired_reward.append(desired_dynamics.compute_reward())
        
        fixed_values = get_fixed_values(fixed_indices, mean_field)

        estimator.initialize_estimate(fixed_indices=fixed_indices, fixed_values=fixed_values)

        for R in range(num_comm_rounds):
            estimator.get_new_info()
        estimator.compute_estimate()

        mean_field_estimate = estimator.get_mf_estimate()
        dynamics.compute_next_mean_field(obs=mean_field_estimate)
        mean_field = dynamics.get_mf()
        
        new_comms_graph = dynamics.get_new_comms_graph()
        estimator.update_comms_graph(new_comms_graph)

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

fig.suptitle("Mean-Field Estimation Performance using LCP Policy", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
plt.savefig("mean_field_communication_grid_nav.png", dpi=300)
plt.show()