import matplotlib.pyplot as plt
import numpy as np
import os

def pad_array(arr, target_length, num_states, pad_value=0.0): # Or np.nan
    current_length = arr.shape[0]
    if current_length >= target_length:
        return arr
    padding_shape = (target_length - current_length, num_states)
    padding = np.full(padding_shape, pad_value)
    return np.vstack((arr, padding))

def pos2index(grid, pos):
    return np.ravel_multi_index(pos, (grid[0], grid[1]))

comm_rounds_list = [1,2,3,4,5,6,7,8]
num_episodes = 10
num_timesteps = 501
grid = (9, 9)

targets = [[3, 3], [3, 4], [3, 5], [4, 3], [4, 4], [4, 5], 
                    [5, 3], [5, 4], [5, 5]]
target_indices = [pos2index(grid, arr) for arr in targets]

fig, axs = plt.subplots(2, 4, figsize=(12, 8), sharey=True)
axs = axs.flatten()
plt.subplots_adjust(hspace=0.4)

save_dir_des = f"mean_field_trajectory/grid_{grid[0]}x{grid[1]}_partial_obs_False_comm_1"

for idx, num_comm_rounds in enumerate(comm_rounds_list):


    save_dir_est = f"mean_field_trajectory/grid_{grid[0]}x{grid[1]}_partial_obs_True_comm_{num_comm_rounds}"
    save_dir_ben = f"mean_field_trajectory/grid_{grid[0]}x{grid[1]}_partial_obs_True_comm_{num_comm_rounds}_benchmark"

    l1_dpc = np.zeros((num_episodes, num_timesteps))
    l1_benchmark = np.zeros((num_episodes, num_timesteps))

    for ep in range(num_episodes):

        output_est = os.path.join(save_dir_est, f"ep_{ep}.npy")
        output_ben = os.path.join(save_dir_ben, f"ep_{ep}.npy")
        output_des = os.path.join(save_dir_des, f"ep_{ep}.npy")

        estimated_mf = np.load(output_est)
        benchmark_mf = np.load(output_ben)
        des_mf = np.load(output_des)

        # Find the maximum length among the arrays for this episode
        max_timesteps = max(estimated_mf.shape[0], benchmark_mf.shape[0], des_mf.shape[0])
        num_states = grid[0]*grid[1]

        estimated_mf_padded = pad_array(estimated_mf, max_timesteps, num_states, pad_value=0.0) # Or np.nan
        benchmark_mf_padded = pad_array(benchmark_mf, max_timesteps, num_states, pad_value=0.0) # Or np.nan
        des_mf_padded = pad_array(des_mf, max_timesteps, num_states, pad_value=0.0) # Or np.nan

        # Now perform your L1 error calculation on the padded arrays
        tv_error_dpc = 0.5*np.sum(np.abs(estimated_mf_padded - des_mf_padded), axis=1)
        tv_error_benchmark = 0.5*np.sum(np.abs(benchmark_mf_padded - des_mf_padded), axis=1)

        l1_dpc[ep, :] = tv_error_dpc
        l1_benchmark[ep, :] = tv_error_benchmark

    avg_l1_dpc = l1_dpc.mean(axis=0)
    avg_l1_benchmark = l1_benchmark.mean(axis=0)

    # Plot
    axs[idx].plot(avg_l1_dpc, label='D-PC (Ours)')
    axs[idx].plot(avg_l1_benchmark, label='Benchmark')
    axs[idx].set_title(fr'$R_{{\mathrm{{com}}}} = {num_comm_rounds}$', fontsize=14)
    axs[idx].set_xlabel('Time',fontsize=12)
    axs[idx].set_ylabel('Total Variation Error', fontsize=12)
    axs[idx].legend(fontsize=12)
    axs[idx].grid(True)

fig.suptitle("Total Variation Error $\mathrm{d}_{\mathrm{TV}}(\mu_t, \mu^A_t)$ vs Time", fontsize=18)
fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 
fig.savefig("finite_subgrid_comms_l1_error_vs_time_100.png", dpi=300)  


fig2, axs2 = plt.subplots(2, 4, figsize=(12, 8), sharey=True)
axs2 = axs2.flatten()
plt.subplots_adjust(hspace=0.4)

for idx, num_comm_rounds in enumerate(comm_rounds_list):

    save_dir_est = f"mean_field_trajectory/grid_{grid[0]}x{grid[1]}_partial_obs_True_comm_{num_comm_rounds}"
    save_dir_ben = f"mean_field_trajectory/grid_{grid[0]}x{grid[1]}_partial_obs_True_comm_{num_comm_rounds}_benchmark"

    reward_discrepancy_dpc = np.zeros((num_episodes, num_timesteps))
    reward_discrepancy_benchmark = np.zeros((num_episodes, num_timesteps))

    for ep in range(num_episodes):

        output_est = os.path.join(save_dir_est, f"ep_{ep}.npy")
        output_ben = os.path.join(save_dir_ben, f"ep_{ep}.npy")
        output_des = os.path.join(save_dir_des, f"ep_{ep}.npy")

        estimated_mf = np.load(output_est)
        benchmark_mf = np.load(output_ben)
        des_mf = np.load(output_des)

        # Find the maximum length among the arrays for this episode
        max_timesteps = max(estimated_mf.shape[0], benchmark_mf.shape[0], des_mf.shape[0])
        num_states = grid[0]*grid[1]

        estimated_mf_padded = pad_array(estimated_mf, max_timesteps, num_states, pad_value=0.0) # Or np.nan
        benchmark_mf_padded = pad_array(benchmark_mf, max_timesteps, num_states, pad_value=0.0) # Or np.nan
        des_mf_padded = pad_array(des_mf, max_timesteps, num_states, pad_value=0.0) # Or np.nan

        reward_dpc = np.cumsum(np.sum(estimated_mf_padded[:, target_indices], axis=1))
        reward_benchmark = np.cumsum(np.sum(benchmark_mf_padded[:, target_indices], axis=1))
        reward_des = np.cumsum(np.sum(des_mf_padded[:, target_indices], axis=1))

        reward_discrepancy_dpc[ep, :] = np.abs(reward_des-reward_dpc)
        reward_discrepancy_benchmark[ep, :] = np.abs(reward_des-reward_benchmark)

    avg_rew_dpc = reward_discrepancy_dpc.mean(axis=0)
    avg_rew_benchmark = reward_discrepancy_benchmark.mean(axis=0)

    # Plot
    axs2[idx].plot(avg_rew_dpc, label='D-PC (Ours)')
    axs2[idx].plot(avg_rew_benchmark, label='Benchmark')
    axs2[idx].set_title(fr'$R_{{\mathrm{{com}}}} = {num_comm_rounds}$',fontsize=14)
    axs2[idx].set_xlabel('Time',fontsize=12)
    axs2[idx].set_ylabel('Error in Cumulative Rewards',fontsize=12)
    axs2[idx].legend(fontsize=12)
    axs2[idx].grid(True)

fig2.suptitle("Comparing Cumulative Rewards $|\sum_{t=0}^{T} r^{\pi_t}(\mu_t) - \sum_{t=0}^{T} r^{\pi_t}(\mu^A_t)|$ for Grid-Navigation", fontsize=18)
fig2.tight_layout(rect=[0, 0.03, 1, 0.95]) 
fig2.savefig("finite_subgrid_comms_rewards_100.png", dpi=300)