import numpy as np
import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB
import torch
from .utils import *

class MeanFieldEstimator():
    def __init__(self, num_states=3, horizon_length=1, num_particles=1000, 
                 comms_graph=None, alpha_low=1, alpha_high=1, seed=None):
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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.noise_std = 0.0

        self._gurobi_model = None
        self._gurobi_x = None
        self._gurobi_rhs_constr = None

        self.alpha_vec = np.full(self.num_states, alpha_high)
        self.alpha_vec[:self.num_states // 2] = alpha_low

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


    def _sample_constrained_dirichlet(self, fixed_indices, fixed_values, D=1):
        fixed_indices = np.array(fixed_indices)
        fixed_values = np.array(fixed_values)
        
        assert len(fixed_indices) == len(fixed_values), "Mismatch in fixed inputs"
        assert np.all(fixed_values >= 0), "Fixed values must be non-negative"
        assert np.sum(fixed_values) <= 1, "Fixed values must sum to ≤ 1"

        remaining_mass = 1.0 - np.sum(fixed_values)
        all_indices = np.arange(self.num_states)

        free_indices = np.setdiff1d(all_indices, fixed_indices)

        # Use only the alphas corresponding to free indices
        dirichlet_samples = np.random.dirichlet(self.alpha_vec[free_indices], size=D)
        scaled_samples = dirichlet_samples * remaining_mass  # shape (D, len(free_indices))

        # Construct full D samples
        full_samples = np.zeros((D, self.num_states))
        full_samples[:, fixed_indices] = fixed_values
        full_samples[:, free_indices] = scaled_samples

        return full_samples  # shape: (D, num_states)
    
    def initialize_mean_field(self, fixed_indices, fixed_values):
        for state in range(self.num_states):
            self.mean_field_estimate[state] = self._sample_constrained_dirichlet(
                                        fixed_indices[state], fixed_values[state], D=1)
            self.estimate_history[state].append(self.mean_field_estimate[state].copy())

    def initialize_comm_round(self, fixed_indices, fixed_values):
        comm_round_init = {}
        for state in range(self.num_states):
            comm_round_init[state] = self.project_to_partial_simplex(
                self.mean_field_estimate[state], fixed_indices[state], fixed_values[state]
            )
        self.mean_field_estimate = comm_round_init

    '''
    def get_new_info(self):
        for i in range(self.num_states):
            self.state_info[i] = []
            for j in range(self.num_states):
                if self.G_comms[i][j]:
                    self.state_info[i].append(self.mean_field_estimate[j].flatten())
    '''
    '''
    def get_new_info(self):
        for i in range(self.num_states):
            self.state_info[i] = []
            for j in range(self.num_states):
                if self.G_comms[i][j]:
                    estimate = self.mean_field_estimate[j].flatten()
                    
                    # Add Gaussian noise of appropriate size
                    noise = np.random.normal(loc=0.0, scale=self.noise_std, size=estimate.shape) if j != i else 0
                    noisy_estimate = estimate + noise
                    
                    self.state_info[i].append(noisy_estimate)

    '''
    def get_new_info(self, mean_field_self=None):
        for i in range(self.num_states):
            self.state_info[i] = []
            for j in range(self.num_states):
                if self.G_comms[i][j] and ((mean_field_self[i]>0 and mean_field_self[j]>0) or (i==j)):
                    estimate = self.mean_field_estimate[j].flatten()
                    
                    # Add Gaussian noise of appropriate size
                    noise = np.random.normal(loc=0.0, scale=self.noise_std, size=estimate.shape) if j != i else 0
                    noisy_estimate = estimate + noise
                    
                    self.state_info[i].append(noisy_estimate)
    
    

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

        vectors = masked_weights = 1/vectors.shape[0]*np.ones(vectors.shape[0]) # np.array(self.state_info[state]) #masked_weights = 1/vectors.shape[0]*np.ones(vectors.shape[0])

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
    
    def project_to_partial_simplex(self, z, fixed_indices, x_fixed_values, tol=1e-6):
        z = np.squeeze(z)
        n = len(z)
        x = np.zeros(n)
        x[fixed_indices] = x_fixed_values

        # Identify free indices
        all_indices = set(range(n))
        free_indices = sorted(list(all_indices - set(fixed_indices)))
        
        z_free = z[free_indices]
        rhs = 1.0 - np.sum(x_fixed_values)

        # If the remaining mass is negligible, assign zeros
        if rhs < tol:
            x[free_indices] = 0.0
        else:
            # Project z_free onto simplex of mass = rhs
            x_free = self.project_l2_onto_simplex(z_free, rhs)
            x[free_indices] = x_free

        return x

    def build_gurobi_l1_projection_model(self, n):
        """
        Build and cache a reusable Gurobi model for L1 projection.
        This should be called once with a fixed `n` (length of z).
        """
        model = gp.Model()
        model.setParam('OutputFlag', 0)  # Changed to 1 to enable Gurobi output for debugging

        x = model.addMVar(n, lb=0, name="x")
        t = model.addMVar(n, lb=0, name="t")

        model.setObjective(0.5 * t.sum(), GRB.MINIMIZE)
        
        abs_constr_pos = model.addConstr(t - x >= 0.0, name="abs_constr_pos")
        abs_constr_neg = model.addConstr(t + x >= 0.0, name="abs_constr_neg")

        # Sum constraint for x: sum(x_i) == rhs
        rhs_constr = model.addConstr(x.sum() == 1.0, name="sum_constraint")

        model.update() # Apply changes to the model

        # Store Gurobi model components for later reuse
        self._gurobi_model = model
        self._gurobi_x = x
        self._gurobi_rhs_constr = rhs_constr
        self._gurobi_abs_constr_pos = abs_constr_pos
        self._gurobi_abs_constr_neg = abs_constr_neg

    def gurobi_l1_projection_reuse(self, z_target, rhs, tol=1e-9):
        """
        Use prebuilt Gurobi model to solve min 0.5*||x - z||_1 s.t. sum x = rhs, x >= 0
        """
        z_target = np.squeeze(z_target) # Ensure z_target is a 1D array
        n = len(z_target)

        # Rebuild model if it's not initialized or if 'n' has changed
        if self._gurobi_model is None or self._gurobi_x.shape[0] != n:
            self.build_gurobi_l1_projection_model(n)

        # Update the z parameter
        self._gurobi_abs_constr_pos.setAttr('RHS', -z_target)
        self._gurobi_abs_constr_neg.setAttr('RHS', z_target)

        # Update the right-hand side of the sum constraint
        self._gurobi_rhs_constr.rhs = rhs

        self._gurobi_model.update()
        self._gurobi_model.reset()
        self._gurobi_model.optimize()
        
        return self._gurobi_x.X # Return the optimal solution for x
        
    
    '''

    def project_onto_simplex(self, z, rhs, tol=1e-6):
        """
        Projects z onto the simplex {x >= 0, sum x = rhs}, minimizing L1 distance.
        That is: argmin_x 0.5 * ||x - z||_1 subject to sum x = rhs, x >= 0
        """
        z = np.squeeze(z)
        n = len(z)
        x = cp.Variable(n)
        t = cp.Variable(n)

        objective = cp.Minimize(0.5 * cp.sum(t))
        constraints = [
            t >= x - z,
            t >= z - x,
            cp.sum(x) == rhs,
            x >= 0,
            t >= 0
        ]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        if prob.status != cp.OPTIMAL:
            raise ValueError("Projection failed or problem is infeasible.")

        return x.value

    def gurobi_l1_projection(self, z, rhs):
        """
        Solves: min_x 0.5 * ||x - z||_1 s.t. sum x = rhs, x >= 0
        using Gurobi with matrix constraint batching.
        """
        z = np.squeeze(z)
        n = len(z)

        model = gp.Model()
        model.setParam('OutputFlag', 0)  # Silent

        # Create variables x and t (concatenated)
        x = model.addMVar(n, lb=0, name="x")
        t = model.addMVar(n, lb=0, name="t")

        # Objective: 0.5 * sum(t)
        model.setObjective(0.5 * t.sum(), GRB.MINIMIZE)

        # Constraint: t >= x - z, t >= z - x
        model.addConstr(t >= x - z)
        model.addConstr(t >= z - x)

        # Simplex constraint: sum x == rhs
        model.addConstr(x.sum() == rhs)

        model.optimize()

        if model.Status != GRB.OPTIMAL:
            raise ValueError("Gurobi projection failed.")

        return x.X  # already a NumPy array
    
    def l1_projection_mirror_descent(self, z, rhs, num_iters=100, lr=0.01):
        z = torch.tensor(z, dtype=torch.float32, device=self.device)
        n = z.shape[0]

        rhs = torch.tensor(rhs, dtype=z.dtype, device=self.device)

        # Initialize x with uniform simplex projection
        x = torch.full_like(z, rhs / n, device=self.device)

        for _ in range(num_iters):
            grad = 0.5 * torch.sign(x - z)
            x = x - lr * grad
            x = torch.clamp(x, min=0)
            x = x - (x.sum() - rhs) / n
            x = torch.clamp(x, min=0)

        return x.cpu().numpy()
    '''

    
    def project_l2_onto_simplex(self, v, z=1.0):
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
    
    def update_comms_graph(self, A):
        self.G_comms = self.compute_metropolis_weights(A) 

if __name__ == "__main__":
    num_states = 4
    num_comm_rounds = 100
    num_agents = 500
    true_mean_field = np.array([0.2, 0.1, 0.7, 0]) 

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

    estimator = MeanFieldEstimator(num_states=num_states, horizon_length=1, comms_graph=G_comms, seed=42)

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