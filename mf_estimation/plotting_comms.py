import matplotlib.pyplot as plt
import numpy as np

comm_rounds_list = list(range(1, 11))  # [1, 2, ..., 10]
time_steps = [50, 100, 150, 200, 250, 300]

fig, axs = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
axs = axs.flatten()

for idx, time_step in enumerate(time_steps):
    l1_errors_at_t_dpc = []
    l1_errors_at_t_benchmark = []
    valid_comm_rounds = []

    for num_comm_rounds in comm_rounds_list:
        try:
            l1_dpc = np.load(f'l1_errors_dpc_all_seeds_{num_comm_rounds}.npy')  # shape: (seeds, time)
            l1_benchmark = np.load(f'l1_errors_benchmark_all_seeds_{num_comm_rounds}.npy')

            if time_step < l1_dpc.shape[1]:
                avg_l1_dpc = l1_dpc.mean(axis=0)[time_step]
                avg_l1_benchmark = l1_benchmark.mean(axis=0)[time_step]

                l1_errors_at_t_dpc.append(avg_l1_dpc)
                l1_errors_at_t_benchmark.append(avg_l1_benchmark)
                valid_comm_rounds.append(num_comm_rounds)
        except FileNotFoundError:
            print(f"Missing file for comm_rounds={num_comm_rounds}, skipping.")

    axs[idx].plot(valid_comm_rounds, l1_errors_at_t_dpc, marker='o', label='D-PC (Ours)')
    axs[idx].plot(valid_comm_rounds, l1_errors_at_t_benchmark, marker='s', label='Benchmark')
    axs[idx].set_title(f'L1 Error at Time Step {time_step}')
    axs[idx].set_xlabel('Communication Rounds')
    axs[idx].set_ylabel('L1 Error')
    axs[idx].grid(True)
    axs[idx].legend()

fig.suptitle("L1 Error vs Communication Rounds at Different Time Steps", fontsize=18)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig("subgrid_l1_error_vs_comm_rounds_no_noise.png", dpi=300)
plt.show()