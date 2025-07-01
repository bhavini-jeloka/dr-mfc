import matplotlib.pyplot as plt
import numpy as np

comm_rounds_list = [1, 2, 3, 4, 5, 6, 7, 8]
time_steps = [100, 200, 300, 400, 500, 600, 700, 800, 900, -1]

fig, axs = plt.subplots(2, 5, figsize=(18, 10), sharey=True)
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

    if time_step == -1:
        time_step=1000
    axs[idx].plot(valid_comm_rounds, l1_errors_at_t_dpc, marker='o', label='D-PC (Ours)')
    axs[idx].plot(valid_comm_rounds, l1_errors_at_t_benchmark, marker='s', label='Benchmark')
    axs[idx].set_title(fr'$t = {time_step}$', fontsize=14)
    axs[idx].set_xlabel(fr'$R_{{\mathrm{{com}}}}$',fontsize=12)
    axs[idx].set_ylabel('Total Variation Error',fontsize=12)
    axs[idx].grid(True)
    axs[idx].legend(fontsize=12)

fig.suptitle("Total Variation Error $\mathrm{d}_{\mathrm{TV}}(\mu_t, \mu^A_t)$ vs $R_{{\mathrm{{com}}}}$ at Different Time Steps", fontsize=18)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig("subgrid_l1_error_vs_comm_rounds.png", dpi=300)
plt.show()

summed_avg_l1_dpc = []
summed_avg_l1_benchmark = []
valid_comm_rounds = []

for num_comm_rounds in comm_rounds_list:
    try:
        l1_dpc = np.load(f'l1_errors_dpc_all_seeds_{num_comm_rounds}.npy')         # shape: (seeds, time)
        l1_benchmark = np.load(f'l1_errors_benchmark_all_seeds_{num_comm_rounds}.npy')

        avg_over_seeds_dpc = l1_dpc.mean(axis=0)        # shape: (time,)
        avg_over_seeds_benchmark = l1_benchmark.mean(axis=0)

        sum_dpc = avg_over_seeds_dpc.sum()
        sum_benchmark = avg_over_seeds_benchmark.sum()

        summed_avg_l1_dpc.append(sum_dpc)
        summed_avg_l1_benchmark.append(sum_benchmark)
        valid_comm_rounds.append(num_comm_rounds)
    except FileNotFoundError:
        print(f"Missing file for comm_rounds={num_comm_rounds}, skipping.")

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(valid_comm_rounds, summed_avg_l1_dpc, marker='o', label='D-PC (Ours)')
plt.plot(valid_comm_rounds, summed_avg_l1_benchmark, marker='s', label='Benchmark')
plt.xlabel(fr'$R_{{\mathrm{{com}}}}$', fontsize=12)
plt.ylabel("Cumulative Total Variation", fontsize=12)
plt.title("Cumulative Total Variation Error $\sum_{t=0}^T\mathrm{d}_{\mathrm{TV}}(\mu_t, \mu^A_t)$ vs $R_{{\mathrm{{com}}}}$", fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("subgrid_total_l1_error_vs_comm_rounds.png", dpi=300)
plt.show()