import numpy as np
from .utils import *
import copy

class BenchmarkEstimator():
    def __init__(self, num_states=3, horizon_length=1, num_particles=1000, 
                 comms_graph=None, fixed_indices=None, fixed_values=None, seed=None):
        super().__init__()

        if seed is not None:
            np.random.seed(seed)

        self.num_states = num_states
        self.T = horizon_length

        self.G_comms = self.compute_metropolis_weights(comms_graph) 

        self.mean_field_estimate = {}
        self.mean_field_estimate_copy = {}

        self.state_info = {i: [] for i in range(self.num_states)}
        self.estimate_history = {state: [] for state in range(self.num_states)}

        self.noise_std = 0.0

    def initialize_estimate(self, fixed_indices, fixed_values):
        self.mean_field_estimate = {}
        for state in range(self.num_states):
            est = np.full(self.num_states, np.nan)
            est[fixed_indices[state]] = fixed_values[state]
            self.mean_field_estimate[state] = est

    '''
    def get_new_info(self):
        updated_estimates = {state: self.mean_field_estimate[state].copy()
                            for state in range(self.num_states)}
        for state in range(self.num_states):
            for nbr in range(self.num_states):
                if self.G_comms[state][nbr]:
                    to_update = ~np.isnan(self.mean_field_estimate[nbr])
                    updated_estimates[state][to_update] = self.mean_field_estimate[nbr][to_update]
        self.mean_field_estimate = updated_estimates
    '''
    
    def get_new_info(self):
        updated_estimates = {state: self.mean_field_estimate[state].copy()
                            for state in range(self.num_states)}

        for state in range(self.num_states):
            # Accumulators for averaging noisy values
            sum_estimates = np.zeros_like(self.mean_field_estimate[state])
            count_estimates = np.zeros_like(self.mean_field_estimate[state])

            for nbr in range(self.num_states):
                if self.G_comms[state][nbr]:
                    neighbor_estimate = self.mean_field_estimate[nbr]
                    to_update = ~np.isnan(neighbor_estimate)

                    # Add Gaussian noise to neighbor's estimate (communication noise)
                    noise = np.random.normal(loc=0.0, scale=self.noise_std, size=neighbor_estimate.shape) if nbr != state else 0
                    noisy_estimate = neighbor_estimate + noise

                    # Accumulate values and count for averaging
                    sum_estimates[to_update] += noisy_estimate[to_update]
                    count_estimates[to_update] += 1

            # Average over all neighbors that provided information
            to_update_final = count_estimates > 0
            updated_estimates[state][to_update_final] = sum_estimates[to_update_final] / count_estimates[to_update_final]

        self.mean_field_estimate = updated_estimates

    '''
    def get_new_info(self, mean_field_self=None):
        updated_estimates = {state: self.mean_field_estimate[state].copy()
                            for state in range(self.num_states)}

        for state in range(self.num_states):
            # Accumulators for averaging noisy values
            sum_estimates = np.zeros_like(self.mean_field_estimate[state])
            count_estimates = np.zeros_like(self.mean_field_estimate[state])

            for nbr in range(self.num_states):
                if self.G_comms[state][nbr] and ((mean_field_self[state]> 0 and mean_field_self[nbr]>0) or state==nbr):
                    neighbor_estimate = self.mean_field_estimate[nbr]
                    to_update = ~np.isnan(neighbor_estimate)

                    # Add Gaussian noise to neighbor's estimate (communication noise)
                    noise = np.random.normal(loc=0.0, scale=self.noise_std, size=neighbor_estimate.shape) if nbr != state else 0
                    noisy_estimate = neighbor_estimate + noise

                    # Accumulate values and count for averaging
                    sum_estimates[to_update] += noisy_estimate[to_update]
                    count_estimates[to_update] += 1

            # Average over all neighbors that provided information
            to_update_final = count_estimates > 0
            updated_estimates[state][to_update_final] = sum_estimates[to_update_final] / count_estimates[to_update_final]

        self.mean_field_estimate = updated_estimates
    '''
    

    def compute_estimate(self):
        for state in range(self.num_states):
            est = self.mean_field_estimate[state]
            nan_mask = np.isnan(est)
            num_nans = np.sum(nan_mask)
            if num_nans > 0:
                known_sum = np.nansum(est)
                remaining_mass = 1.0 - known_sum
                est[nan_mask] = remaining_mass / num_nans
            else:
                est = est/np.sum(est) 
            self.estimate_history[state].append(est.copy())

    def get_mf_estimate(self):
        return self.mean_field_estimate

    def compute_metropolis_weights(self, A):
        # Degree of each node
        degrees = A.sum(axis=1)
        
        # Initialize the Metropolis weight matrix
        n = A.shape[0]
        W = np.zeros_like(A, dtype=float)
        
        # Compute the Metropolis weights
        for i in range(n):
            for j in range(i, n):  # Iterate over the upper triangle to ensure symmetry
                if A[i, j] == 1:  # There is an edge between node i and j
                    W[i, j] = 1 / (1 + max(degrees[i], degrees[j]))
                    W[j, i] = W[i, j]  # Ensure symmetry
            # Set the diagonal element
            W[i, i] = 1 - np.sum(W[i, :])  # Normalize the row

        return W
    
    def update_comms_graph(self, A):
        self.G_comms = self.compute_metropolis_weights(A) 

if __name__ == "__main__":
    num_states = 4
    num_comm_rounds = 2
    num_agents = 500
    true_mean_field = (1/num_agents)*np.array([100, 50, 250, 100])

    # Define comms graph
    G_comms = np.zeros((num_states, num_states))
    G_comms[0][1] = 1
    G_comms[1][0] = 1
    G_comms[1][2] = 1
    G_comms[2][1] = 1
    G_comms[2][3] = 1
    G_comms[3][2] = 1

    # Define init mean-field (can later define this based on the visualization graph)
    
    fixed_indices = {0: [0], 1: [1], 2: [2], 3:[3]}
    fixed_values = {0: [true_mean_field[0]], 
                    1: [true_mean_field[1]], 
                    2: [true_mean_field[2]], 
                    3: [true_mean_field[3]]}
    
    '''
    fixed_indices = {0: [0, 3], 1: [1, 2], 2: [1, 2], 3:[0, 3]}
    fixed_values = {0: [true_mean_field[0], true_mean_field[3]], 
                    1: [true_mean_field[1], true_mean_field[2]], 
                    2: [true_mean_field[1], true_mean_field[2]], 
                    3: [true_mean_field[0], true_mean_field[3]]}
    '''
    
    '''

    fixed_indices = {0: [0], 1: [1], 2: [2]}
    fixed_values = {0: [true_mean_field[0]], 
                    1: [true_mean_field[1]], 
                    2: [true_mean_field[2]]}
    '''

    estimator = BenchmarkEstimator(num_states=num_states, horizon_length=1, comms_graph=G_comms, seed=42)

    #estimator.sample_particles(fixed_indices=fixed_indices, fixed_values=fixed_values)
    estimator.initialize_estimate(fixed_indices=fixed_indices, fixed_values=fixed_values)

    for R in range(num_comm_rounds):
        print('Round', R+1)
        estimator.get_new_info()
    
    estimator.compute_estimate()

    #estimator.compute_estimate()
    #estimator.plot_estimates(true_mean_field)
    #plot_estimation_errors(estimator.estimate_history, true_mean_field, num_states)