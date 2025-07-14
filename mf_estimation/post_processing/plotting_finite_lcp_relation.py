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
num_episodes = 25
num_timesteps = 101
grid = (9, 9)
lcp_coeff = [0.000, 0.005, 0.01]

targets = [[3, 3], [3, 4], [3, 5], [4, 3], [4, 4], [4, 5], 
                    [5, 3], [5, 4], [5, 5]]
target_indices = [pos2index(grid, arr) for arr in targets]

fig, axs = plt.subplots(2, 4, figsize=(12, 8), sharey=True)
axs = axs.flatten()
plt.subplots_adjust(hspace=0.4)

# loop over communication rounds _and_ coefficients
for idx, num_comm_rounds in enumerate(comm_rounds_list):
    for coeff in lcp_coeff:

        save_dir_des = f"9x9-nav-varying-LCP-coeff/Coeff_{coeff}/mean_field_trajectory/grid_{grid[0]}x{grid[1]}_partial_obs_False_comm_1"
        save_dir_est = f"9x9-nav-varying-LCP-coeff/Coeff_{coeff}/mean_field_trajectory/grid_{grid[0]}x{grid[1]}_partial_obs_True_comm_{num_comm_rounds}"

        l1_dpc = np.zeros((num_episodes, num_timesteps))
        for ep in range(num_episodes):
            est = np.load(os.path.join(save_dir_est, f"ep_{ep}.npy"))
            des = np.load(os.path.join(save_dir_des, f"ep_{ep}.npy"))

            # pad to same length
            #max_t = max(est.shape[0], des.shape[0])
            S = grid[0]*grid[1]
            est_p = pad_array(est, num_timesteps, S)
            des_p = pad_array(des, num_timesteps, S)

            l1_dpc[ep] = 0.5 * np.sum(np.abs(est_p - des_p), axis=1)

        avg_l1 = l1_dpc.mean(axis=0)
        axs[idx].plot(avg_l1, label=f"$\\lambda={coeff}$")   # use coeff in label

    # after all coeffs for this subplot, add titles & legend
    axs[idx].set_title(fr'$R_{{\mathrm{{com}}}} = {num_comm_rounds}$', fontsize=14)
    axs[idx].set_xlabel('Time', fontsize=12)
    axs[idx].set_ylabel('Total Variation Error', fontsize=12)
    axs[idx].grid(True)
    axs[idx].legend(fontsize=10, loc='upper right')

fig.suptitle(
    "Total Variation Error $d_{TV}(\\mu_t, \\mu^A_t)$ vs Time", 
    fontsize=18
)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig("finite_subgrid_comms_lcp_l1_error_vs_time_1000.png", dpi=300) 


fig2, axs2 = plt.subplots(1, 1, figsize=(12, 8), sharey=True)

for coeff in lcp_coeff:

    summed_avg_l1_dpc = []
    summed_avg_l1_benchmark = []
    valid_comm_rounds = []

    for num_comm_rounds in comm_rounds_list:

        save_dir_des = f"9x9-nav-varying-LCP-coeff/Coeff_{coeff}/mean_field_trajectory/grid_{grid[0]}x{grid[1]}_partial_obs_False_comm_1"
        save_dir_est = f"9x9-nav-varying-LCP-coeff/Coeff_{coeff}/mean_field_trajectory/grid_{grid[0]}x{grid[1]}_partial_obs_True_comm_{num_comm_rounds}"

        l1_dpc = np.zeros((num_episodes, num_timesteps))

        for ep in range(num_episodes):
            est = np.load(os.path.join(save_dir_est, f"ep_{ep}.npy"))
            des = np.load(os.path.join(save_dir_des, f"ep_{ep}.npy"))

            # pad to same length
            #max_t = max(est.shape[0], des.shape[0])
            S = grid[0]*grid[1]
            est_p = pad_array(est, num_timesteps, S)
            des_p = pad_array(des, num_timesteps, S)

            l1_dpc[ep] = 0.5 * np.sum(np.abs(est_p - des_p), axis=1)

        sum_per_seed_dpc = l1_dpc.sum(axis=1) 
        mean_sum_dpc = sum_per_seed_dpc.mean()
        summed_avg_l1_dpc.append(mean_sum_dpc)
        valid_comm_rounds.append(num_comm_rounds)

    axs2.plot(valid_comm_rounds, summed_avg_l1_dpc, label=f"$\\lambda={coeff}$")   # use coeff in label

    # after all coeffs for this subplot, add titles & legend
    axs2.set_title("Cumulative Total Variation Error $\sum_{t=0}^T\mathrm{d}_{\mathrm{TV}}(\mu_t, \mu^A_t)$ vs $R_{{\mathrm{{com}}}}$", fontsize=24)
    axs2.set_xlabel(fr'$R_{{\mathrm{{com}}}}$', fontsize=24)
    axs2.set_ylabel("Cumulative Total Variation", fontsize=24)
    axs2.grid(True)
    axs2.legend(fontsize=24, loc='upper right')

fig2.tight_layout()
fig2.savefig("finite_subgrid_comms_lcp_total_l1_error_vs_time_1000.png", dpi=300) 