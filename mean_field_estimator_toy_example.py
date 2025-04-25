import torch
import matplotlib.pyplot as plt
import numpy as np
from mean_field_estimation import MeanFieldEstimator
from mean_field_dynamics import MeanFieldDynamicsEval

def plot_reward(actual_reward, desired_reward):
        plt.figure(figsize=(10, 6))

        plt.plot(actual_reward, label=f"Actual Reward")
        plt.plot(desired_reward, label=f"Desired Reward")

        plt.title("Accumulated Rewards")
        plt.xlabel("Timestep")
        plt.ylabel("Reward Value")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

num_states = 3
num_actions = 2
num_timesteps = 50
num_agents = 500
true_mean_field = (1/num_agents)*np.array([200, 50, 250])
mean_field = true_mean_field
des_mean_field = true_mean_field

# Define comms graph
G_comms = np.zeros((num_states, num_states))
G_comms[0][1] = 1
G_comms[1][0] = 1
G_comms[1][2] = 1
G_comms[2][1] = 1

num_particles = 5000
num_comm_rounds = 5

# Define init mean-field (can later define this based on the visualization graph)
fixed_indices = {0: [0], 1: [1], 2: [2]}

estimator = MeanFieldEstimator(num_states=num_states, horizon_length=1, num_particles=num_particles, 
                            comms_graph=G_comms, seed=4)
dynamics = MeanFieldDynamicsEval(init_mean_field=mean_field, num_states=num_states, num_actions=num_actions)
desired_dynamics = MeanFieldDynamicsEval(init_mean_field=mean_field, num_states=num_states, num_actions=num_actions)

actual_reward = []
desired_reward = []

for t in range(num_timesteps):

    actual_reward.append(dynamics.compute_reward())
    desired_reward.append(desired_dynamics.compute_reward())

    fixed_values = {0: [mean_field[0]], 
                    1: [mean_field[1]], 
                    2: [mean_field[2]]}
    
    estimator.sample_particles(fixed_indices=fixed_indices, fixed_values=fixed_values)

    for R in range(num_comm_rounds):
        estimator.compute_estimate()
        estimator.get_new_info()
        estimator.update_weights()

    estimator.compute_estimate()
    mean_field_estimate = estimator.get_mf_estimate()
    print("Estimated Mean-Field", mean_field_estimate)

    dynamics.compute_next_mean_field(obs=mean_field_estimate)
    mean_field = dynamics.get_mf()
    print("Actual Mean-Field", mean_field)
    

    desired_dynamics.compute_next_mean_field(true_mean_field)
    des_mean_field = desired_dynamics.get_mf()
    print("Desired Mean-Field", des_mean_field)

    #estimator.plot_estimates(true_mean_field)
    #estimator.plot_estimation_errors(true_mean_field)

plot_reward(actual_reward, desired_reward)

