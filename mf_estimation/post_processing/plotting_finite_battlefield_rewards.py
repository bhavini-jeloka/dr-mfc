import matplotlib.pyplot as plt
import numpy as np
import os

def get_reward_discrepancy(config_dir, ep, ground_truth):
    path = os.path.join(config_dir, f"ep_{ep}.npy")
    if os.path.exists(path):
        rewards = np.load(path)
        target_len = len(ground_truth)
        if len(rewards) < target_len:
            rewards = np.pad(rewards, (0, target_len - len(rewards)), mode='constant', constant_values=0.0)
        else:
            rewards = rewards[:target_len]
        estimate = np.cumsum(rewards[:target_len])
        discrepancy = np.abs(estimate - ground_truth)
        return discrepancy
    else:
        raise FileNotFoundError(f"Missing file: {path}")

comm_rounds_list = [5, 10, 15, 20, 25, 30]
num_episodes = 25
num_timesteps = 21
grid = (4, 4)

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

base_dir = "rewards"
ground_truth_suffix = "blue-None_red-None"
num_configs = len(configurations)

fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharey=True)
fig.subplots_adjust(right=0.75)  # still leave room for legend
plt.subplots_adjust(hspace=0.4)
axs = axs.flatten()

for idx, num_comm_rounds in enumerate(comm_rounds_list):
    print(f"Processing R_com = {num_comm_rounds}")

    ground_truth_dir = f"{base_dir}/grid_{grid[0]}x{grid[1]}_comm_{num_comm_rounds}_{ground_truth_suffix}"
    
    # Initialize cumulative errors for each config
    all_discrepancies = {label: np.zeros((num_episodes, num_timesteps)) for label, _ in configurations}

    for ep in range(num_episodes):
        
        print(f"  Episode {ep}")

        gt_rewards = np.load(os.path.join(ground_truth_dir, f"ep_{ep}.npy"))

        if len(gt_rewards) < num_timesteps:
            gt_rewards = np.pad(gt_rewards, (0, num_timesteps - len(gt_rewards)), mode='constant', constant_values=0.0)
        else:
            gt_rewards = gt_rewards[:num_timesteps]

        ground_truth_output = np.cumsum(gt_rewards)

        for label, suffix in configurations:
            config_dir = f"{base_dir}/grid_{grid[0]}x{grid[1]}_comm_{num_comm_rounds}_{suffix}"
            discrepancy = get_reward_discrepancy(config_dir, ep, ground_truth_output)
            
            if len(discrepancy) < num_timesteps:
                discrepancy = np.pad(discrepancy, (0, num_timesteps - len(discrepancy)), mode='edge')
            else:
                discrepancy = discrepancy[:num_timesteps]

            all_discrepancies[label][ep, :] = discrepancy

    # Plot
    for label, _ in configurations:
        avg_discrepancy = all_discrepancies[label].mean(axis=0)
        axs[idx].plot(avg_discrepancy, label=label)

    axs[idx].set_title(fr'$R_{{\mathrm{{com}}}} = {num_comm_rounds}$', fontsize=14)
    axs[idx].set_xlabel('Time', fontsize=12)
    axs[idx].set_ylabel('Error in Cumulative Rewards', fontsize=12)
    #axs[idx].legend(fontsize=10)
    axs[idx].grid(True)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', fontsize=12)
fig.suptitle(r"Comparing Cumulative Rewards for Battlefield", fontsize=18)
#fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig("finite_subgrid_battlefield_rewards_100.png", dpi=300)
plt.show()