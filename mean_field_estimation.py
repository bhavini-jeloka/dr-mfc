from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import numpy as np


class MeanFieldEstimator():
    def __init__(self, num_states=3, horizon_length=1, num_particles=1000, 
                 comms_graph=None, fixed_indices=None, fixed_values=None, seed=None):
        super().__init__()

        if seed is not None:
            np.random.seed(seed)

        self.num_states = num_states
        self.T = horizon_length

        self.num_particles = num_particles
        self.G_comms = comms_graph

        self.mean_field_estimate = {}

        self.fixed_indices = fixed_indices
        self.fixed_values = fixed_values

        self.state_info = {i: [] for i in range(self.num_states)}
        self.weights =  {i: np.ones(self.num_particles)/self.num_particles for i in range(self.num_states)}
        self.estimate_history = {state: [] for state in range(self.num_states)}


    def sample_particles(self):
        self.particles = np.zeros((self.num_states, self.num_particles, self.num_states))
        for state in range(self.num_states):
                self.particles[state] = self._sample_constrained_dirichlet(
                                        fixed_indices=self.fixed_indices[state], 
                                        fixed_values=self.fixed_values[state], D=self.num_particles) 
    
    def _likelihood(self, y, x_i):
        """
        Computes the likelihood of observation y given particle x_i
        under a unit-variance Gaussian assumption (N(x_i, I)).

        Args:
            y (np.ndarray): observation vector
            x_i (np.ndarray): particle vector

        Returns:
            float: unnormalized likelihood value
        """
        diff = y - x_i
        exponent = -0.5 * np.dot(diff, diff)  # equivalent to -0.5 * ||y - x_i||^2
        return np.exp(exponent)


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

    def get_new_info(self):
        for i in range(self.num_states):
            self.state_info[i] = []
            for j in range(self.num_states):
                if self.G_comms[i][j]:  # only based on one-hop neighbor
                    self.state_info[i].append(self.mean_field_estimate[j])

    def update_weights(self):
        for state in range(self.num_states):
            for i in range(self.num_particles):
                for info in self.state_info[state]:
                    self.weights[state][i] *= self._likelihood(info, self.particles[state][i])
            # Step 3: Normalize
            self.weights[state] /= np.sum(self.weights[state])
        '''

        for state in range(self.num_states):
            # Likelihoods for all particles (D x num_infos)
            likelihoods = np.array([
                [self._likelihood(info, particle) for info in self.state_info[state]]
                for particle in self.particles[state]
            ])  # shape: (D, num_infos)
            
            self.weights[state] = np.prod(likelihoods, axis=1)
            self.weights[state] /= np.sum(self.weights[state])
            '''

    def compute_estimate(self): 
        for state in range(self.num_states):
            # Reshape weights to (D, 1) so broadcasting works with (D, num_states)
            weighted_particles = self.weights[state][:, None] * self.particles[state]
            # Sum across the D particles
            self.mean_field_estimate[state] = np.sum(weighted_particles, axis=0)
            self.estimate_history[state].append(self.mean_field_estimate[state].copy())

    def plot_estimates(self, true_mean_field=None):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Plot each state's mean field trajectory
        for state in range(self.num_states):
            history = np.array(self.estimate_history[state])  # shape: (T, 3)
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

    def plot_estimation_errors(self, true_mean_field):
        plt.figure(figsize=(10, 6))

        for state in range(self.num_states):
            history = np.array(self.estimate_history[state])  # shape: (T, 3)
            
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

num_states = 4
num_comm_rounds = 5
num_particles = 5000
num_agents = 500
true_mean_field = (1/num_agents)*np.array([100, 50, 100, 250])

# Define comms graph
G_comms = np.zeros((num_states, num_states))
G_comms[0][1] = 1
G_comms[1][0] = 1
G_comms[1][2] = 1
G_comms[2][1] = 1
G_comms[2][3] = 1
G_comms[3][2] = 1

# Define init mean-field (can later define this based on the visualization graph)
fixed_indices = {0: [0, 3], 1: [1, 2], 2: [1, 2], 3:[0, 3]}
fixed_values = {0: [true_mean_field[0], true_mean_field[3]], 
                1: [true_mean_field[1], true_mean_field[2]], 
                2: [true_mean_field[1], true_mean_field[2]], 
                3: [true_mean_field[0], true_mean_field[3]]}

estimator = MeanFieldEstimator(num_states=num_states, horizon_length=1, num_particles=num_particles, 
                            comms_graph=G_comms, 
                            fixed_indices=fixed_indices, 
                            fixed_values=fixed_values, seed=42)

estimator.sample_particles()

for R in range(num_comm_rounds):
    print(R)
    estimator.compute_estimate()
    estimator.get_new_info()
    estimator.update_weights()

estimator.compute_estimate()
#estimator.plot_estimates(true_mean_field)
estimator.plot_estimation_errors(true_mean_field)