import matplotlib.pyplot as plt
import numpy as np
import os

comm_rounds_list = [1, 2, 3, 4, 5, 6, 7, 8]
num_episodes = 100
num_timesteps = 500
grid = (9, 9)

fig, axs = plt.subplots(2, 4, figsize=(12, 8), sharey=True)
axs = axs.flatten()
plt.subplots_adjust(hspace=0.4)

for idx, num_comm_rounds in enumerate(comm_rounds_list):


    save_dir_est = f"mean_field_trajectory/grid_{grid[0]}x{grid[1]}_partial_obs_True_comm_{num_comm_rounds}"
    save_dir_ben = f"mean_field_trajectory/grid_{grid[0]}x{grid[1]}_partial_obs_True_comm_{num_comm_rounds}_benchmark"
    save_dir_des = f"mean_field_trajectory/grid_{grid[0]}x{grid[1]}_partial_obs_False"

    l1_dpc = np.zeros((num_episodes, num_timesteps))
    l1_benchmark = np.zeros((num_episodes, num_timesteps))

    for ep in range(num_episodes):

        output_est = os.path.join(save_dir_est, f"ep_{ep}.npy")
        output_ben = os.path.join(save_dir_ben, f"ep_{ep}.npy")
        output_des = os.path.join(save_dir_des, f"ep_{ep}.npy")

        estimated_mf = np.load(output_est)
        benchmark_mf = np.load(output_ben)
        des_mf = np.load(output_des)

        print(estimated_mf.shape)
        print(output_ben.shape)
        print(des_mf.shape)

        # fill missing entries with last l1 value and make a marking

        # compute l1_error and store for each time step
        assert(1==0)

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