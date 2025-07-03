import matplotlib.pyplot as plt
import numpy as np
import os

comm_rounds_list = [5, 10, 15, 20, 25, 30]
total_comm_rounds = len(comm_rounds_list)
num_episodes = 25
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
    ("None (B) v D-PC (R)", "blue-None_red-d-pc"),
    ("None (B) v None (R)", "blue-None_red-None")
]

base_dir = "rewards"
avg_rewards = {label: np.zeros(total_comm_rounds) for label, _ in configurations}

# Collect avg_rewards
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

# Plotting
fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
axs = axs.flatten()

labels = [label for label, _ in configurations]
x = np.arange(len(labels))  # bar positions

# Define bar colors
highlight_label = "None (B) v None (R)"
bar_colors = ['magenta' if label == highlight_label else 'green' for label in labels]

for idx, ax in enumerate(axs):
    comm_round = comm_rounds_list[idx]
    heights = [avg_rewards[label][idx] for label in labels]

    bars = ax.bar(x, heights, color=bar_colors)

    ax.set_title(fr'$R_{{\mathrm{{com}}}} = {comm_round}$')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    ax.set_xticks([])  # Hide default x-tick labels

    # Add config names on the bars (horizontally)
    for i, (bar, label) in enumerate(zip(bars, labels)):
        bar_center = bar.get_x() + bar.get_width() / 2

        # Configuration name (on the bar)
        ax.text(
            bar_center,
            1,  # base of the bar
            label,
            ha='center',
            va='bottom',
            fontsize=10,
            rotation=90
        )

        # Average reward value (on top of bar)
        ax.text(
            bar_center,
            bar.get_height(),
            f"{bar.get_height():.1f}",
            ha='center',
            va='bottom',
            fontsize=10,
            rotation=0
        )

fig.suptitle("Average Test Rewards over 25 Episodes", fontsize=18)
fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 
fig.savefig(f"finite_subgrid_battlefield_test_rewards_{grid[0]}x{grid[1]}.png", dpi=300)  
plt.show()
