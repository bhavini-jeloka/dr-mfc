import torch
import numpy as np
from utils import project_onto_l1_ball, sample_l1_ball, project_rows_onto_simplex, plot_performance_comparison
from mean_field_performance import DRMFC

# Parameters for DR-MFC
num_states = 3
horizon_length = 1

# Parameters for optimization
num_states = 3  # Dimension of IC
num_actions = 2
pi_dim =  num_states*num_actions
N = 100  # MC samples
lambda_ = 0.5  # Balancing weight
eta_theta = 0.01  # Learning rate for theta
eta_x = 0.1  # Learning rate for x minimization
x_c = np.array([1, 0, 0])
R = 0.1

# Initialize theta
theta = torch.randn(pi_dim, requires_grad=True)
theta = project_rows_onto_simplex(theta, num_states, num_actions)

# Initialize objective class
dr_mfc_objective = DRMFC(horizon_length, num_states)

# Function f(x, theta)
def f(x, theta):
    theta = project_rows_onto_simplex(theta, num_states, num_actions)
    return dr_mfc_objective.compute_cumulative_rewards(x, theta)

# Optimization loop
for step in range(100):
    x_samples = sample_l1_ball(N, num_states, x_c)  # MC samples

    # Expectation term
    expectation_term = f(x_samples, theta).mean()

    #TODO: fix derivative component
    # Min optimization for x
    x_min = x_samples.clone().detach().requires_grad_(True)
    for _ in range(10):  # Inner PGD steps
        loss_x = f(x_min, theta)
        loss_x.backward()
        with torch.no_grad():
            x_min -= eta_x * x_min.grad  # Gradient step
            x_min = torch.tensor(project_onto_l1_ball(x_min.detach().numpy()))  # Projection
        x_min.grad.zero_()

    min_term = f(x_min, theta).min()

    # Compute final loss
    S = lambda_ * expectation_term + (1 - lambda_) * min_term

    # Gradient ascent for theta
    S.backward()
    with torch.no_grad():
        theta += eta_theta * theta.grad
        theta = project_rows_onto_simplex(theta, num_states, num_actions)
        theta.grad.zero_()

    if step % 10 == 0:
        print(f"Step {step}, S(θ): {S.item()}")

# Testing performance of robust policy
x_test_samples = sample_l1_ball(N, num_states, x_c)
robust_performance = f(x_test_samples, theta).detach().numpy()
pi_nominal = [1, 0, 0, 1, 1, 0]
nomial_pi_performance = f(x_test_samples, torch.tensor(pi_nominal, dtype=torch.float32)).detach().numpy()
test_sample_indices = np.arange(len(x_test_samples))
plot_performance_comparison(test_sample_indices, robust_performance, nomial_pi_performance)