import numpy as np
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import torch
import os
import csv
from pathlib import Path

def project_rows_onto_simplex(theta, num_states, num_actions):
    """Projects each row of the given tensor onto the probability simplex."""
    theta = theta.view(num_states, num_actions)  # Reshape (6,) -> (3,2)
    theta = torch.nn.functional.softmax(theta, dim=1)  # Ensure row sum = 1
    return theta.view(-1)  # Flatten back to (6,)


def project_onto_l1_ball(x, radius=1.0):
    """ Projects x onto the L1 ball of given radius. """
    if np.linalg.norm(x, 1) <= radius:
        return x
    u = np.abs(x)
    if u.sum() <= radius:
        return x
    s = np.sort(u)[::-1]
    cssv = np.cumsum(s)
    rho = np.where(s > (cssv - radius) / (np.arange(len(s)) + 1))[0][-1]
    theta = (cssv[rho] - radius) / (rho + 1.0)
    return np.sign(x) * np.maximum(u - theta, 0)


def sample_l1_ball(n, d, x_c, R=1.0):
    # Step 1: Sample from an L1 ball centered at the origin
    exp_samples = np.random.exponential(scale=1.0, size=(n, d))  # Positive values
    l1_norms = exp_samples.sum(axis=1, keepdims=True)  # Sum to get L1 norm
    u = (exp_samples / l1_norms) * R  # Normalize to sum to R

    signs = np.random.choice([-1, 1], size=(n, d))  # Assign random signs
    perturbations = u * signs  # L1 perturbation

    # Step 2: Shift by x_c
    x = x_c + perturbations

    # Step 3: Project into the simplex if necessary
    x = np.clip(x, 0, 1)  # Ensure non-negativity
    row_sums = x.sum(axis=1, keepdims=True)
    x /= row_sums  # Normalize back to the simplex if sum ≠ 1

    return torch.tensor(x, dtype=torch.float32)


def plot_l1_ball_with_samples(x_c, R, samples):
    """
    Plots the L1 ball around a center point `x_c` in 2D or 3D and overlays sampled points.
    """
    x_c = np.array(x_c)
    d = len(x_c)
    assert d in [2, 3], "Visualization is only supported for 2D or 3D."
    
    fig = plt.figure(figsize=(8, 6))
    
    if d == 2:
        ax = fig.add_subplot(111)
        
        # L1 Ball boundary (diamond shape)
        corners = np.array([
            [x_c[0] + R, x_c[1]],  # Right
            [x_c[0], x_c[1] + R],  # Top
            [x_c[0] - R, x_c[1]],  # Left
            [x_c[0], x_c[1] - R],  # Bottom
            [x_c[0] + R, x_c[1]]   # Close the loop
        ])
        
        ax.plot(corners[:, 0], corners[:, 1], 'r-', label="L1 Ball Boundary")
        
        # Plot samples
        ax.scatter(samples[:, 0], samples[:, 1], c='blue', marker='o', label="Sampled Points")
        ax.scatter(x_c[0], x_c[1], c='black', marker='x', s=100, label="Center $x_c$")
        
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_title("L1 Ball with Sampled Points (2D)")
        ax.legend()
        ax.grid(True)
    
    elif d == 3:
        ax = fig.add_subplot(111, projection='3d')

        # L1 Ball boundary (octahedron shape)
        corners = np.array(list(product([-R, R], repeat=3))) + x_c  # 8 corners of the octahedron
        
        for i, start in enumerate(corners):
            for j, end in enumerate(corners):
                if np.sum(np.abs(start - end)) == 2 * R:  # Only connect L1 neighbors
                    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'r-')

        # Plot samples
        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c='blue', marker='o', label="Sampled Points")
        ax.scatter(x_c[0], x_c[1], x_c[2], c='black', marker='x', s=100, label="Center $x_c$")
        
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("X3")
        ax.set_title("L1 Ball with Sampled Points (3D)")
        ax.legend()
    
    plt.show()


def plot_performance_comparison(test_sample_indices, robust_performance,  nominal_pi_performance):
    # Create scatter plot
    plt.figure(figsize=(8, 5))
    plt.scatter(test_sample_indices, robust_performance, label="Robust Policy", color="blue", alpha=0.7)
    plt.scatter(test_sample_indices, nominal_pi_performance, label="Nominal Policy", color="red", alpha=0.7)

    # Labels and title
    plt.xlabel("Test Sample Number")
    plt.ylabel("Performance")
    plt.title("Comparison of Robust and Nominal Policy Performance")
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()


def plot_estimates(estimate_history, num_states, true_mean_field=None):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Plot each state's mean field trajectory
        for state in range(num_states):
            history = np.array(estimate_history[state])  # shape: (T, 3)
            ax.scatter(
                history[:, 0], history[:, 1], history[:, 2],
                label=f"Estimate: State {state}"
            )

        # Plot the true mean-field as a point
        if true_mean_field is not None:
            ax.scatter(
                true_mean_field[0],
                true_mean_field[1],
                true_mean_field[2],
                color='black',
                s=100,
                marker='X',
                label='True Mean Field'
            )

        # --- Draw the 3D simplex triangle ---
        vertices = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        triangle = Poly3DCollection(
            [vertices],
            color='lightgray',
            alpha=0.2
        )
        ax.add_collection3d(triangle)

        # Draw the edges of the simplex
        for i in range(3):
            for j in range(i + 1, 3):
                ax.plot(
                    [vertices[i, 0], vertices[j, 0]],
                    [vertices[i, 1], vertices[j, 1]],
                    [vertices[i, 2], vertices[j, 2]],
                    color='gray',
                    linewidth=1
                )

        # Set axes labels and limits
        ax.set_title("3D Mean-Field Estimates with Simplex")
        ax.set_xlabel("Component 0")
        ax.set_ylabel("Component 1")
        ax.set_zlabel("Component 2")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.legend()
        plt.tight_layout()
        plt.show()


def plot_estimation_errors(estimate_history, true_mean_field, num_states):
    plt.figure(figsize=(10, 6))

    for state in range(num_states):
        history = np.array(estimate_history[state])  # shape: (T, 3)
        history = np.vstack(history)
        
        # Compute L2 norm error between estimate and true mean field at each timestep
        errors = np.linalg.norm(history - true_mean_field, axis=1)  # shape: (T,)

        plt.plot(errors, label=f"State {state} Error")

    plt.title("Normed Error vs True Mean-Field Over Time")
    plt.xlabel("Communication Round #")
    plt.ylabel("L2 Norm Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_reward(actual_reward, desired_reward):
    plt.figure(figsize=(12, 10))

    # First subplot: actual and desired rewards
    plt.subplot(2, 1, 1)
    plt.plot(actual_reward, label="Actual Reward", color='blue', linestyle='-', linewidth=2.5)
    plt.plot(desired_reward, label="Desired Reward", color='orange', linestyle='--', linewidth=2.5)
    plt.title("Accumulated Rewards", fontsize=16)
    plt.xlabel("Timestep", fontsize=14)
    plt.ylabel("Reward Value", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # Second subplot: absolute difference
    plt.subplot(2, 1, 2)
    diff = np.abs(np.array(actual_reward) - np.array(desired_reward))
    plt.plot(diff, label="Absolute Difference", color='green', linestyle='-', linewidth=2.5)
    plt.title("Absolute Difference Between Rewards", fontsize=16)
    plt.xlabel("Timestep", fontsize=14)
    plt.ylabel("Absolute Difference", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.show()
    
def get_latest_model_dir(policy_type):
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    run_dir = base_dir / policy_type
    return run_dir

def get_fixed_values(fixed_indices, mean_field):
    return {
        key: [mean_field[i] for i in indices]
        for key, indices in fixed_indices.items()
    }

def get_or_create_mean_field(seed, num_states, filename="initial_mean_fields.csv"):
    # If file exists, try to load the mean field for the given seed
    if os.path.exists(filename):
        print("file exists")
        with open(filename, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if int(row[0]) == seed:
                    return np.array([float(x) for x in row[1:]])

    # If seed not found, generate new mean field and append it to the file
    rng = np.random.default_rng(seed)
    true_mean_field = rng.dirichlet(np.ones(num_states))

    # Create and/or append to the file
    with open(filename, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([seed] + true_mean_field.tolist())

    return true_mean_field