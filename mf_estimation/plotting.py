import matplotlib.pyplot as plt
import numpy as np

comm_rounds_list = [1, 2, 3, 4, 5, 6, 7, 8]

# Rewards metric
#fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
#axs = axs.flatten()

#plt.subplots_adjust(hspace=0.4)

# L1 metric
fig2, axs2 = plt.subplots(2, 4, figsize=(12, 8), sharey=True)
axs2 = axs2.flatten()
'''
for idx, num_comm_rounds in enumerate(comm_rounds_list):
    
    rewards_dpc = np.load(f'rewards_actual_dpc_all_seeds_{num_comm_rounds}.npy') 
    rewards_benchmark = np.load(f'rewards_actual_benchmark_all_seeds_{num_comm_rounds}.npy')
    rewards_desired = np.load(f'rewards_desired_all_seeds_{num_comm_rounds}.npy')

    # Average across seeds
    avg_dpc = rewards_dpc.mean(axis=0)
    avg_benchmark = rewards_benchmark.mean(axis=0)
    avg_desired = rewards_desired.mean(axis=0)

    error_dpc = np.abs(np.sum(avg_dpc - avg_desired)) / np.abs(np.sum(avg_desired))
    error_benchmark = np.abs(np.sum(avg_benchmark - avg_desired)) / np.abs(np.sum(avg_desired))

    # Plot
    cumsum_dpc = np.cumsum(avg_dpc)
    cumsum_benchmark = np.cumsum(avg_benchmark)
    cumsum_desired = np.cumsum(avg_desired)
    axs[idx].plot(cumsum_dpc, label='D-PC (Ours)')
    axs[idx].plot(cumsum_benchmark, label='Benchmark')
    axs[idx].plot(cumsum_desired, label='Desired Reward', linestyle='dashed')
    #axs[idx].plot(np.abs(cumsum_actual - cumsum_desired), label='|Cumulative Reward Diff|')
    axs[idx].set_title(f'Comm Rounds: {num_comm_rounds} | D-PC: {error_dpc:.2%} | Benchmark: {error_benchmark:.2%} ')
    axs[idx].set_xlabel('Time')
    axs[idx].set_ylabel('Reward')
    axs[idx].legend()
    axs[idx].grid(True)
    
fig.suptitle("Mean-Field Estimation Performance for 9x9 Grid-Navigation", fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 
fig.savefig("mean_field_large_grid_no_noise.png", dpi=300)
'''
for idx, num_comm_rounds in enumerate(comm_rounds_list):

    l1_dpc = np.load(f'l1_errors_dpc_all_seeds_{num_comm_rounds}.npy')
    l1_benchmark = np.load(f'l1_errors_benchmark_all_seeds_{num_comm_rounds}.npy')

    avg_l1_dpc = l1_dpc.mean(axis=0)
    avg_l1_benchmark = l1_benchmark.mean(axis=0)

    # Plot
    axs2[idx].plot(avg_l1_dpc, label='D-PC (Ours)')
    axs2[idx].plot(avg_l1_benchmark, label='Benchmark')
    axs2[idx].set_title(f'Comm Rounds: {num_comm_rounds}')
    axs2[idx].set_xlabel('Time')
    axs2[idx].set_ylabel('L1 Error')
    axs2[idx].legend()
    axs2[idx].grid(True)

fig2.suptitle("Average L1 Error Between Estimated and Desired Mean Field")
fig2.tight_layout(rect=[0, 0.03, 1, 0.95]) 
fig2.savefig("subgrid_comms_l1_error_vs_time.png", dpi=300)  