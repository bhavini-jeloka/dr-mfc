import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

comm_rounds_list = [5, 10, 15, 20, 25, 30]
total_comm_rounds = len(comm_rounds_list)
num_episodes = 25
grid = (4, 4)

# All (label, directory_suffix) pairs
configurations = [
    ("D-PC (B) v Benchmark (R)", "blue-d-pc_red-benchmark"),
    ("D-PC (B) v D-PC (R)", "blue-d-pc_red-d-pc"),
    ("D-PC (B) v Full-Info (R)", "blue-d-pc_red-None"),
    ("Benchmark (B) v Benchmark (R)", "blue-benchmark_red-benchmark"),
    ("Benchmark (B) v D-PC (R)", "blue-benchmark_red-d-pc"),
    ("Benchmark (B) v Full-Info (R)", "blue-benchmark_red-None"),
    ("Full-Info (B) v Benchmark (R)", "blue-None_red-benchmark"),
    ("Full-Info (B) v D-PC (R)", "blue-None_red-d-pc"),
    ("Full-Info (B) v Full-Info (R)", "blue-None_red-None")
]

base_dir = "rewards"
avg_rewards = {label: np.zeros(total_comm_rounds) for label, _ in configurations}

# Collect avg_rewards (data collection part remains the same)
for idx, comm_rounds in enumerate(comm_rounds_list):
    all_rewards = {label: np.zeros(num_episodes) for label, _ in configurations}
    for ep in range(num_episodes):
        for label, suffix in configurations:
            config_dir = f"{base_dir}/grid_{grid[0]}x{grid[1]}_comm_{comm_rounds}_{suffix}"
            path = os.path.join(config_dir, f"ep_{ep}.npy")
            if os.path.exists(path):
                rewards = np.load(path)
            else:
                raise FileNotFoundError(f"Missing file: {path}")
            all_rewards[label][ep] = np.sum(rewards)
    for label, _ in configurations:
        avg_rewards[label][idx] = all_rewards[label].mean()

# New Plotting Logic (Heat Maps)
blue_agents = ["D-PC", "Full-Info"]
red_agents = ["Full-Info", "D-PC"]
gt_label = "Full-Info (B) v Full-Info (R)"

fig, axs = plt.subplots(2, 3, figsize=(18, 12))
axs = axs.flatten()

# Create a single color bar axis
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

for idx, comm_rounds in enumerate(comm_rounds_list):
    reward_matrix = np.zeros((len(blue_agents), len(red_agents)))
    for b_idx, b_agent in enumerate(blue_agents):
        for r_idx, r_agent in enumerate(red_agents):
            label = f"{b_agent} (B) v {r_agent} (R)"
            if label in avg_rewards:
                error = np.abs(avg_rewards[label][idx] - avg_rewards[gt_label][idx])*100 / (avg_rewards[gt_label][idx])
                reward_matrix[b_idx, r_idx] = error
    
    # Plot the heatmap
    cbar = True if idx == 0 else False
    cbar_ax_use = cbar_ax if idx == 0 else None
    
    sns.heatmap(
        reward_matrix,
        ax=axs[idx],
        annot=True,
        fmt=".2f",
        cmap="YlOrBr",
        vmin=0,
        vmax=3,
        xticklabels=red_agents,
        yticklabels=blue_agents,
        cbar=cbar,
        cbar_ax=cbar_ax_use,
        cbar_kws={'label': 'Relative Percentage Error'} if cbar else None
    )

    # Set custom title with LaTeX
    axs[idx].set_title(fr'$R_{{\mathrm{{com}}}} = {comm_rounds}$', fontsize=14)
    
    axs[idx].tick_params(axis='y', labelsize=14)
    for ticklabel in axs[idx].get_yticklabels():
        ticklabel.set_color('blue')
        
    axs[idx].tick_params(axis='x', labelsize=14)
    for ticklabel in axs[idx].get_xticklabels():
        ticklabel.set_color('red')

# Set the font size of the color bar label after the heatmaps are drawn
cbar_ax.yaxis.label.set_fontsize(16)

fig.suptitle("Average Test Rewards Heat Map (by Communication Rounds)", fontsize=20)
plt.tight_layout(rect=[0, 0, 0.9, 0.96])
plt.savefig(f"finite_subgrid_battlefield_rewards_heatmap_{grid[0]}x{grid[1]}.png", dpi=300)
plt.show()