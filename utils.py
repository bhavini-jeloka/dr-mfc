import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import torch

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


# Sampling from the L1 ball using Dirichlet
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

    Args:
        x_c (array): Center of the L1 ball (must be 2D or 3D for visualization).
        R (float): Radius of the L1 ball.
        samples (array): Sampled points from the L1 ball.
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