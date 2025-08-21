import matplotlib.pyplot as plt
import numpy as np
import os

def get_l1_error(config_dir, ep, num_timesteps, num_states, ground_truth_blue, ground_truth_red):
    path = os.path.join(config_dir, f"ep_{ep}.npy")
    if os.path.exists(path):
        mf_blue = np.load(path)[:, :num_states]
        mf_red = np.load(path)[:, num_states:]

        traj_len = min(mf_blue.shape[0], ground_truth_blue.shape[0])

        l1_blue = 0.5 * np.sum(np.abs(mf_blue[:traj_len, :] - ground_truth_blue[:traj_len, :]), axis=1)
        l1_red = 0.5 * np.sum(np.abs(mf_red[:traj_len, :] - ground_truth_red[:traj_len, :]), axis=1)
        return l1_blue, l1_red
    else:
        raise FileNotFoundError(f"Missing file: {path}")

comm_rounds_list = [10, 20, 30, 40, 50, 60]
num_episodes = 25
num_timesteps = 25
grid = (8, 8)
num_states = 2 * grid[0] * grid[1]

# All (label, directory_suffix) pairs
configurations = [
    ("D-PC", "blue-None_red-d-pc"),
    ("Benchmark", "blue-None_red-benchmark"),
]

base_dir = "mf"
ground_truth_suffix = "blue-None_red-None"
num_configs = len(configurations)

# --- One figure with 2x3 subplots ---
fig, axs = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
fig.subplots_adjust(right=0.8, hspace=0.4)
axs = axs.flatten()

for idx, num_comm_rounds in enumerate(comm_rounds_list):

    ground_truth_dir = f"{base_dir}/grid_{grid[0]}x{grid[1]}_comm_{num_comm_rounds}_{ground_truth_suffix}"
    
    all_l1_errors_blue = {label: [] for label, _ in configurations}
    all_l1_errors_red = {label: [] for label, _ in configurations}

    for ep in range(num_episodes):
        print(f"Processing R_com = {num_comm_rounds}, Episode={ep}")
        gt_mf = np.load(os.path.join(ground_truth_dir, f"ep_{ep}.npy"))
        gt_mf_blue, gt_mf_red = gt_mf[:, :num_states], gt_mf[:, num_states:]

        for label, suffix in configurations:
            config_dir = f"{base_dir}/grid_{grid[0]}x{grid[1]}_comm_{num_comm_rounds}_{suffix}"
            l1_blue, l1_red = get_l1_error(config_dir, ep, num_timesteps, num_states, gt_mf_blue, gt_mf_red)

            padded_blue = np.full(num_timesteps, np.nan)
            padded_red = np.full(num_timesteps, np.nan)
            padded_blue[:len(l1_blue)] = l1_blue
            padded_red[:len(l1_red)] = l1_red

            all_l1_errors_blue[label].append(padded_blue)
            all_l1_errors_red[label].append(padded_red)

    ax = axs[idx]
    for label, _ in configurations:
        arr_blue = np.vstack(all_l1_errors_blue[label])
        arr_red = np.vstack(all_l1_errors_red[label])

        avg_l1_blue = np.nanmean(arr_blue, axis=0)
        avg_l1_red = np.nanmean(arr_red, axis=0)

        # --- plotting with style mapping ---
        color_map = {"Blue": "blue", "Red": "red"}
        style_map = {"D-PC": "-", "Benchmark": ":"}

        # Blue team
        ax.plot(avg_l1_blue, 
                color="blue", 
                linestyle=style_map[label], 
                label=f"Blue {label}")

        # Red team
        ax.plot(avg_l1_red, 
                color="red", 
                linestyle=style_map[label], 
                label=f"Red {label}")

    ax.set_title(fr"$R_{{\mathrm{{com}}}} = {num_comm_rounds}$", fontsize=14)
    ax.set_xlabel("Time", fontsize=12)
    if idx % 3 == 0:
        ax.set_ylabel("Total Variation Error", fontsize=12)
    ax.grid(True)

# Create one legend for all
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=12, loc="center right", frameon=False)

fig.suptitle("Comparing Total Variation Error (Blue vs Red)", fontsize=18)
plt.tight_layout(rect=[0, 0, 0.85, 0.95])
plt.savefig(f"finite_subgrid_battlefield_l1_error_{grid[0]}x{grid[1]}.png", dpi=300)
plt.show()
