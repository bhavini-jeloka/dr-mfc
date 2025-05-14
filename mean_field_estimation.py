import numpy as np
from utils import *

class MeanFieldEstimator():
    def __init__(self, num_states=3, horizon_length=1, num_particles=1000, 
                 comms_graph=None, fixed_indices=None, fixed_values=None, seed=None):
        super().__init__()

        if seed is not None:
            np.random.seed(seed)

        self.num_states = num_states
        self.T = horizon_length

        self.num_particles = num_particles
        self.G_comms = self.compute_metropolis_weights(comms_graph) 

        self.mean_field_estimate = {}
        self.mean_field_estimate_copy = {}

        self.state_info = {i: [] for i in range(self.num_states)}
        self.estimate_history = {state: [] for state in range(self.num_states)}


    def sample_particles(self, fixed_indices, fixed_values):
        self.weights =  {i: np.ones(self.num_particles)/self.num_particles for i in range(self.num_states)}
        self.particles = np.zeros((self.num_states, self.num_particles, self.num_states))
        for state in range(self.num_states):
                self.particles[state] = self._sample_constrained_dirichlet(
                                        fixed_indices[state], fixed_values[state], D=self.num_particles) 
    
    def _likelihood(self, y, x_i): 
        diff = y - x_i
        exponent = -0.5 * np.dot(diff, diff)  # equivalent to -0.5 * ||y - x_i||^2
        return np.exp(exponent) # Gaussian prior


    def _sample_constrained_dirichlet(self, fixed_indices, fixed_values, alpha=1.0, D=1):
        fixed_indices = np.array(fixed_indices)
        fixed_values = np.array(fixed_values)
        
        assert len(fixed_indices) == len(fixed_values), "Mismatch in fixed inputs"
        assert np.all(fixed_values >= 0), "Fixed values must be non-negative"
        assert np.sum(fixed_values) <= 1, "Fixed values must sum to ≤ 1"

        remaining_mass = 1.0 - np.sum(fixed_values)
        all_indices = np.arange(self.num_states)

        free_indices = np.setdiff1d(all_indices, fixed_indices)
        num_free = len(free_indices)

        # Sample D Dirichlet vectors for free indices
        dirichlet_samples = np.random.dirichlet([alpha] * num_free, size=D)
        scaled_samples = dirichlet_samples * remaining_mass  # shape (D, num_free)

        # Construct full D samples
        full_samples = np.zeros((D, self.num_states))

        # Fill in fixed values
        full_samples[:, fixed_indices] = fixed_values

        # Fill in sampled values for free indices
        full_samples[:, free_indices] = scaled_samples

        return full_samples  # shape: (D, num_states)
    
    def initialize_mean_field(self, fixed_indices, fixed_values):
        for state in range(self.num_states):
            self.mean_field_estimate[state] = self._sample_constrained_dirichlet(
                                        fixed_indices[state], fixed_values[state], D=1)
            self.estimate_history[state].append(self.mean_field_estimate[state].copy())

    def get_new_info(self):
        for i in range(self.num_states):
            self.state_info[i] = []
            for j in range(self.num_states):
                if self.G_comms[i][j]:
                    self.state_info[i].append(self.mean_field_estimate[j].flatten())

    def get_projected_average_estimate(self, fixed_indices, fixed_values):
        for state in range(self.num_states):
            # Start with the initial mean field estimate
            estimated_mf = self.compute_weighted_average(state)

            # Project the result onto the simplex with fixed indices
            self.mean_field_estimate_copy[state] = self.project_to_partial_simplex(
                estimated_mf, fixed_indices[state], fixed_values[state]
            )

    def compute_weighted_average(self, state):
        weights = self.G_comms[state]             
        mask = weights != 0
        masked_weights = weights[mask]
        #masked_weights /= masked_weights.sum()

        vectors = np.array(self.state_info[state])  

        avg = masked_weights @ vectors             
        return avg.reshape(1, -1)      

    def update_weights(self):
        for state in range(self.num_states):
            for i in range(self.num_particles):
                for info in self.state_info[state]:
                    self.weights[state][i] *= self._likelihood(info, self.particles[state][i])
            self.weights[state] /= np.sum(self.weights[state])

    def compute_estimate(self, copy=False): 
        for state in range(self.num_states):
            if copy: 
                self.mean_field_estimate[state] = self.mean_field_estimate_copy[state]
            else:
                # Reshape weights to (D, 1) so broadcasting works with (D, num_states)
                weighted_particles = self.weights[state][:, None] * self.particles[state]
                # Sum across the D particles
                self.mean_field_estimate[state] = np.sum(weighted_particles, axis=0)
            self.estimate_history[state].append(self.mean_field_estimate[state].copy())

    def get_mf_estimate(self):
        return self.mean_field_estimate
    
    def project_to_partial_simplex(self, z, fixed_indices, x_fixed_values):
        z = np.squeeze(z)
        n = len(z)
        x = np.zeros(n)
        x[fixed_indices] = x_fixed_values

        # Identify free indices
        all_indices = set(range(n))
        free_indices = sorted(list(all_indices - set(fixed_indices)))
        
        z_free = z[free_indices]
        rhs = 1.0 - np.sum(x_fixed_values)

        # Project z_free onto simplex of mass = rhs
        x_free = self.project_onto_simplex(z_free, rhs)
        x[free_indices] = x_free
        return x

    def project_onto_simplex(self, v, z=1.0):
        """Projects vector v onto the simplex {x : x >= 0, sum x = z}"""
        v = np.asarray(v)
        n = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        
        rho = np.nonzero(u * np.arange(1, n+1) > (cssv - z))[0][-1]
        theta = (cssv[rho] - z) / (rho + 1)
        return np.maximum(v - theta, 0)

    
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
    
    def doubly_stochastic_comms_graph(self, adj_matrix):
        """
        Computes M = I - (1/d) * L where L = D - A is the unnormalized Laplacian,
        and d = max degree. Returns a doubly stochastic matrix M.
        """
        assert (adj_matrix == adj_matrix.T).all(), "Adjacency matrix must be symmetric (undirected graph)"
        
        degrees = np.sum(adj_matrix, axis=1)
        max_degree = np.max(degrees)
        d = max_degree

        # Degree matrix
        D = np.diag(degrees)

        # Laplacian
        L = D - adj_matrix

        # Compute M = I - (1/d) * L
        I = np.eye(adj_matrix.shape[0])
        M = I - (1/d) * L

        return M


if __name__ == "__main__":
    num_states = 4
    num_comm_rounds = 100
    num_particles = 5000
    num_agents = 500
    true_mean_field = (1/num_agents)*np.array([100, 50, 350, 0])

    # Define comms graph
    G_comms = np.zeros((num_states, num_states))
    G_comms[0][1] = 1
    G_comms[1][0] = 1
    G_comms[1][2] = 1
    G_comms[2][1] = 1
    #G_comms[2][3] = 1
    #G_comms[3][2] = 1

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

    estimator = MeanFieldEstimator(num_states=num_states, horizon_length=1, num_particles=num_particles, 
                                comms_graph=G_comms, seed=42)

    #estimator.sample_particles(fixed_indices=fixed_indices, fixed_values=fixed_values)
    estimator.initialize_mean_field(fixed_indices=fixed_indices, fixed_values=fixed_values)

    for R in range(num_comm_rounds):
        print('Round', R)
        estimator.get_new_info()
        estimator.get_projected_average_estimate(fixed_indices, fixed_values)
        estimator.compute_estimate(copy=True)
        #estimator.update_weights()

    #estimator.compute_estimate()
    #estimator.plot_estimates(true_mean_field)
    plot_estimation_errors(estimator.estimate_history, true_mean_field, num_states)