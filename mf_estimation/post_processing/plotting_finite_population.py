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

comm_rounds_list = [10, 20, 30, 40, 50, 60, 70, 80]
num_episodes = 10
num_timesteps = 501
grid = (9, 9)

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
    axs[idx].set_title(f'Comm Rounds: {num_comm_rounds}')
    axs[idx].set_xlabel('Time')
    axs[idx].set_ylabel('Total Variation Error')
    axs[idx].legend()
    axs[idx].grid(True)

fig.suptitle("Total Variation Error Between Estimated and Desired Mean Field")
fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 
fig.savefig("finite_subgrid_comms_l1_error_vs_time.png", dpi=300)  