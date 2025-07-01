import matplotlib.pyplot as plt
import numpy as np

comm_rounds_list = [1, 2, 3, 4, 5, 6, 7, 8]

# Rewards metric
fig, axs = plt.subplots(2, 4, figsize=(12, 8), sharey=True)
axs = axs.flatten()

plt.subplots_adjust(hspace=0.4)

# L1 metric
fig2, axs2 = plt.subplots(2, 4, figsize=(12, 8), sharey=True)
axs2 = axs2.flatten()

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
    axs[idx].plot(np.abs(cumsum_dpc-cumsum_desired), label='D-PC (Ours)')
    axs[idx].plot(np.abs(cumsum_benchmark-cumsum_desired), label='Benchmark')
    #axs[idx].plot(cumsum_desired, label='Desired Reward', linestyle='dashed')
    #axs[idx].plot(np.abs(cumsum_actual - cumsum_desired), label='|Cumulative Reward Diff|')
    axs[idx].set_title(fr'$R_{{\mathrm{{com}}}} = {num_comm_rounds}$', fontsize=14)
    axs[idx].set_xlabel('Time', fontsize=12)
    axs[idx].set_ylabel('Error in Cumulative Rewards', fontsize=12)
    axs[idx].legend()
    axs[idx].grid(True)
    
fig.suptitle("Comparing Cumulative Rewards $|\sum_{t=0}^{T} r^{\pi_t}(\mu_t) - \sum_{t=0}^{T} r^{\pi_t}(\mu^A_t)|$ for Grid-Navigation", fontsize=18)
fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 
fig.savefig("subgrid_comms_rewards.png", dpi=300)

for idx, num_comm_rounds in enumerate(comm_rounds_list):

    l1_dpc = np.load(f'l1_errors_dpc_all_seeds_{num_comm_rounds}.npy')
    l1_benchmark = np.load(f'l1_errors_benchmark_all_seeds_{num_comm_rounds}.npy')

    avg_l1_dpc = l1_dpc.mean(axis=0)
    avg_l1_benchmark = l1_benchmark.mean(axis=0)

    # Plot
    axs2[idx].plot(avg_l1_dpc, label='D-PC (Ours)')
    axs2[idx].plot(avg_l1_benchmark, label='Benchmark')
    axs2[idx].set_title(fr'$R_{{\mathrm{{com}}}} = {num_comm_rounds}$', fontsize=14)
    axs2[idx].set_xlabel("Time", fontsize=12)
    axs2[idx].set_ylabel('Total Variation Error', fontsize=12)
    axs2[idx].legend(fontsize=12)
    axs2[idx].grid(True)

fig2.suptitle("Total Variation Error $\mathrm{d}_{\mathrm{TV}}(\mu_t, \mu^A_t)$ vs Time", fontsize=18)
fig2.tight_layout(rect=[0, 0.03, 1, 0.95]) 
fig2.savefig("subgrid_comms_l1_error_vs_time.png", dpi=300)  
plt.show()