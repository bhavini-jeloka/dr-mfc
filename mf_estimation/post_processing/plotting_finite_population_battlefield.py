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

def pad_with_last(arr, target_len):
    return np.pad(arr, (0, target_len - len(arr)), mode='edge')

comm_rounds_list = [5, 10, 15, 20, 25, 30]
num_episodes = 25
num_timesteps = 21
grid = (4, 4)
num_states = 2*grid[0]*grid[1]

fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharey=True)
axs = axs.flatten()
plt.subplots_adjust(hspace=0.4)


for idx, num_comm_rounds in enumerate(comm_rounds_list):
    #TODO: generalize this
    save_dir_est = f"rewards/grid_{grid[0]}x{grid[1]}_comm_{num_comm_rounds}_blue-benchmark_red-d-pc"
    save_dir_ben = f"rewards/grid_{grid[0]}x{grid[1]}_comm_{num_comm_rounds}_blue-benchmark_red-benchmark"

    cum_rew_dpc = []
    cum_rew_benchmark = []

    for ep in range(num_episodes):

        print("Communication round:", num_comm_rounds, "Epsiode:", ep)

        output_est = os.path.join(save_dir_est, f"ep_{ep}.npy")
        output_ben = os.path.join(save_dir_ben, f"ep_{ep}.npy")

        estimated_rew = np.load(output_est)
        benchmark_rew = np.load(output_ben)

        cum_rew_dpc.append(np.cumsum(estimated_rew))
        cum_rew_benchmark.append(np.cumsum(benchmark_rew))

    max_len = max(max(len(arr) for arr in cum_rew_dpc),
              max(len(arr) for arr in cum_rew_benchmark))

    cum_rew_dpc = np.array([pad_with_last(arr, max_len) for arr in cum_rew_dpc])
    cum_rew_benchmark = np.array([pad_with_last(arr, max_len) for arr in cum_rew_benchmark])

    avg_cum_rew_dpc = cum_rew_dpc.mean(axis=0)
    avg_cum_rew_benchmark = cum_rew_benchmark.mean(axis=0)

    # Plot
    axs[idx].plot(avg_cum_rew_dpc, label='D-PC (Ours)')
    axs[idx].plot(avg_cum_rew_benchmark, label='Benchmark')
    axs[idx].set_title(f'Comm Rounds: {num_comm_rounds}')
    axs[idx].set_xlabel('Time')
    axs[idx].set_ylabel('Cumulative Rewards')
    axs[idx].legend()
    axs[idx].grid(True)

fig.suptitle("Cumulative Rewards by the Blue Team (Blue: Benchmark)")
fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 
plt.show()
fig.savefig("battlefield_blue_benchmark.png", dpi=300)  