import matplotlib.pyplot as plt
import numpy as np
import os
    
def get_l1_error(config_dir, ep, num_timesteps, num_states, ground_truth_blue, ground_truth_red):
    path = os.path.join(config_dir, f"ep_{ep}.npy")
    if os.path.exists(path):
        mf_blue = np.load(path)[:, :num_states]
        mf_red = np.load(path)[:, num_states:]

        target_len = ground_truth_blue.shape[0]

        mf_blue = pad_array(mf_blue, num_timesteps, num_states)
        mf_red = pad_array(mf_red, num_timesteps, num_states)

        l1_blue =  0.5*np.sum(np.abs(mf_blue - ground_truth_blue), axis=1)
        l1_red =  0.5*np.sum(np.abs(mf_red - ground_truth_red), axis=1)
        return l1_blue, l1_red
    else:
        raise FileNotFoundError(f"Missing file: {path}")
    
def pad_array(arr, target_length, num_states, pad_value=0.0): # Or np.nan
    current_length = arr.shape[0]
    if current_length >= target_length:
        return arr
    padding_shape = (target_length - current_length, num_states)
    padding = np.full(padding_shape, pad_value)
    return np.vstack((arr, padding))

comm_rounds_list = [10, 20, 30, 40, 50, 60]
num_episodes = 25
num_timesteps = 20
grid = (8, 8)
num_states = 2*grid[0]*grid[1]

# All (label, directory_suffix) pairs
configurations = [
    ("D-PC (B) v Benchmark (R)", "blue-d-pc_red-benchmark"),
    ("D-PC (B) v D-PC (R)", "blue-d-pc_red-d-pc"),
    ("D-PC (B) v None (R)", "blue-d-pc_red-None"),
    ("Benchmark (B) v Benchmark (R)", "blue-benchmark_red-benchmark"),
    ("Benchmark (B) v D-PC (R)", "blue-benchmark_red-d-pc"),
    ("Benchmark (B) v None (R)", "blue-benchmark_red-None"),
    ("None (B) v Benchmark (R)", "blue-None_red-benchmark"),
    ("None (B) v D-PC (R)", "blue-None_red-d-pc")
]

base_dir = "mf"
ground_truth_suffix = "blue-None_red-None"
num_configs = len(configurations)

fig_blue, axs_blue = plt.subplots(2, 3, figsize=(12, 8), sharey=True)
fig_blue.subplots_adjust(right=0.75)  # still leave room for legend
fig_blue.subplots_adjust(hspace=0.4)
axs_blue = axs_blue.flatten()

fig_red, axs_red = plt.subplots(2, 3, figsize=(12, 8), sharey=True)
fig_red.subplots_adjust(right=0.75)  # still leave room for legend
fig_red.subplots_adjust(hspace=0.4)
axs_red = axs_red.flatten()

for idx, num_comm_rounds in enumerate(comm_rounds_list):
    print(f"Processing R_com = {num_comm_rounds}")

    ground_truth_dir = f"{base_dir}/grid_{grid[0]}x{grid[1]}_comm_{num_comm_rounds}_{ground_truth_suffix}"
    
    # Initialize cumulative errors for each config
    all_l1_errors_blue = {label: np.zeros((num_episodes, num_timesteps)) for label, _ in configurations}
    all_l1_errors_red = {label: np.zeros((num_episodes, num_timesteps)) for label, _ in configurations}

    for ep in range(num_episodes):
        
        print(f"  Episode {ep}")

        gt_mf_blue = np.load(os.path.join(ground_truth_dir, f"ep_{ep}.npy"))[:, :num_states]
        gt_mf_red = np.load(os.path.join(ground_truth_dir, f"ep_{ep}.npy"))[:, num_states:]

        gt_mf_blue = pad_array(gt_mf_blue, num_timesteps, num_states)
        gt_mf_red = pad_array(gt_mf_red, num_timesteps, num_states)

        for label, suffix in configurations:
            config_dir = f"{base_dir}/grid_{grid[0]}x{grid[1]}_comm_{num_comm_rounds}_{suffix}"
            l1_blue, l1_red = get_l1_error(config_dir, ep, num_timesteps, num_states, gt_mf_blue, gt_mf_red)
            
            all_l1_errors_blue[label][ep, :] = l1_blue
            all_l1_errors_red[label][ep, :] = l1_red

    # Plot
    for label, _ in configurations:
        avg_l1_blue = all_l1_errors_blue[label].mean(axis=0)
        avg_l1_red = all_l1_errors_red[label].mean(axis=0)
        axs_blue[idx].plot(avg_l1_blue, label=label)
        axs_red[idx].plot(avg_l1_red, label=label)

    axs_blue[idx].set_title(fr'$R_{{\mathrm{{com}}}} = {num_comm_rounds}$', fontsize=14)
    axs_blue[idx].set_xlabel('Time', fontsize=12)
    axs_blue[idx].set_ylabel('Total Variation Error', fontsize=12)
    axs_blue[idx].grid(True)

    axs_red[idx].set_title(fr'$R_{{\mathrm{{com}}}} = {num_comm_rounds}$', fontsize=14)
    axs_red[idx].set_xlabel('Time', fontsize=12)
    axs_red[idx].set_ylabel('Total Variation Error', fontsize=12)
    axs_red[idx].grid(True)

handles, labels = axs_blue[0].get_legend_handles_labels()
fig_blue.legend(handles, labels, loc='center right', fontsize=12)
fig_red.legend(handles, labels, loc='center right', fontsize=12)

fig_blue.suptitle(r"Comparing Total Variation Error for Team Blue", fontsize=18)
fig_red.suptitle(r"Comparing Total Variation Error for Team Red", fontsize=18)
fig_blue.savefig("finite_subgrid_battlefield_l1_blue_100.png", dpi=300)
fig_red.savefig("finite_subgrid_battlefield_l1_red_100.png", dpi=300)

plt.show()
