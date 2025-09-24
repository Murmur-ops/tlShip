"""
Metropolis-Hastings weight computation for consensus algorithms

Generates doubly-stochastic weight matrices that guarantee convergence
to the average consensus while adapting to network topology.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class MetropolisConfig:
    """Configuration for Metropolis weight computation"""
    self_weight_min: float = 0.1  # Minimum self-weight to ensure stability
    neighbor_weight_max: float = 0.4  # Maximum weight for any neighbor
    epsilon: float = 1e-10  # Numerical tolerance
    verbose: bool = False


class MetropolisWeights:
    """
    Compute Metropolis-Hastings weights for consensus

    Creates optimal weights that:
    1. Are doubly stochastic (rows and columns sum to 1)
    2. Respect network topology (zero for non-neighbors)
    3. Maximize convergence rate
    4. Are symmetric for undirected graphs
    """

    def __init__(self, config: MetropolisConfig = None):
        self.config = config or MetropolisConfig()

    def compute_weights(self, adjacency_matrix: np.ndarray) -> np.ndarray:
        """
        Compute Metropolis weights from adjacency matrix

        Args:
            adjacency_matrix: Binary matrix where A[i,j] = 1 if nodes i,j are connected

        Returns:
            Weight matrix W for consensus updates: x^{k+1} = W * x^k
        """
        n = len(adjacency_matrix)
        W = np.zeros((n, n))

        # Compute node degrees
        degrees = np.sum(adjacency_matrix, axis=1)

        # Compute Metropolis weights
        for i in range(n):
            for j in range(n):
                if i == j:
                    # Self-weight: computed last
                    continue
                elif adjacency_matrix[i, j] > 0:
                    # Neighbor weight: 1 / (1 + max(degree_i, degree_j))
                    W[i, j] = 1.0 / (1 + max(degrees[i], degrees[j]))
                else:
                    # Non-neighbor: zero weight
                    W[i, j] = 0.0

            # Self-weight: ensures row sum = 1
            W[i, i] = 1.0 - np.sum(W[i, :])

            # Ensure self-weight is not too small
            if W[i, i] < self.config.self_weight_min:
                # Scale down neighbor weights
                scale = (1 - self.config.self_weight_min) / (1 - W[i, i] + self.config.epsilon)
                for j in range(n):
                    if i != j:
                        W[i, j] *= scale
                W[i, i] = self.config.self_weight_min

        # Verify doubly stochastic
        if self.config.verbose:
            self._verify_doubly_stochastic(W)

        return W

    def compute_laplacian_weights(self, adjacency_matrix: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian-based consensus weights

        Uses graph Laplacian for faster convergence on regular graphs

        Args:
            adjacency_matrix: Binary adjacency matrix

        Returns:
            Weight matrix based on graph Laplacian
        """
        n = len(adjacency_matrix)

        # Compute graph Laplacian: L = D - A
        degrees = np.sum(adjacency_matrix, axis=1)
        D = np.diag(degrees)
        L = D - adjacency_matrix

        # Find optimal step size (related to eigenvalues of L)
        # For safety, use conservative estimate
        max_degree = np.max(degrees)
        if max_degree > 0:
            epsilon = 1.0 / (2 * max_degree)  # Guarantees convergence
        else:
            epsilon = 0.5

        # Weight matrix: W = I - ε*L
        W = np.eye(n) - epsilon * L

        # Ensure weights are in valid range
        W = np.clip(W, 0, 1)

        # Normalize to ensure row stochastic
        row_sums = np.sum(W, axis=1, keepdims=True)
        W = W / (row_sums + self.config.epsilon)

        return W

    def compute_max_degree_weights(self, adjacency_matrix: np.ndarray) -> np.ndarray:
        """
        Simple max-degree weighting scheme

        Fast to compute, good for dense networks

        Args:
            adjacency_matrix: Binary adjacency matrix

        Returns:
            Weight matrix using max-degree weighting
        """
        n = len(adjacency_matrix)
        degrees = np.sum(adjacency_matrix, axis=1)
        max_degree = np.max(degrees)

        if max_degree == 0:
            # No connections, return identity
            return np.eye(n)

        # Weight for neighbors
        neighbor_weight = 1.0 / (max_degree + 1)

        W = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    # Self-loop weight
                    W[i, i] = 1.0 - degrees[i] * neighbor_weight
                elif adjacency_matrix[i, j] > 0:
                    # Neighbor weight
                    W[i, j] = neighbor_weight

        return W

    def compute_adaptive_weights(self,
                                adjacency_matrix: np.ndarray,
                                state_differences: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Adaptive weights that consider current state differences

        Assigns higher weights to neighbors with larger state differences
        to accelerate consensus

        Args:
            adjacency_matrix: Binary adjacency matrix
            state_differences: Optional matrix of state differences between nodes

        Returns:
            Adaptive weight matrix
        """
        # Start with Metropolis weights
        W = self.compute_weights(adjacency_matrix)

        if state_differences is not None:
            n = len(adjacency_matrix)

            # Compute importance of each edge based on state difference
            importance = np.abs(state_differences)
            importance = importance / (np.max(importance) + self.config.epsilon)

            # Adapt weights based on importance
            for i in range(n):
                neighbors = np.where(adjacency_matrix[i, :] > 0)[0]
                if len(neighbors) == 0:
                    continue

                # Redistribute weights based on importance
                total_weight = np.sum(W[i, neighbors])
                importances = importance[i, neighbors]

                if np.sum(importances) > 0:
                    # Weighted redistribution
                    new_weights = total_weight * importances / np.sum(importances)
                    W[i, neighbors] = new_weights

            # Recompute self-weights
            for i in range(n):
                W[i, i] = 1.0 - np.sum(W[i, :]) + W[i, i]

        return W

    def _verify_doubly_stochastic(self, W: np.ndarray):
        """Verify that weight matrix is doubly stochastic"""
        n = len(W)

        # Check row sums
        row_sums = np.sum(W, axis=1)
        row_error = np.max(np.abs(row_sums - 1))

        # Check column sums
        col_sums = np.sum(W, axis=0)
        col_error = np.max(np.abs(col_sums - 1))

        if self.config.verbose:
            print(f"Weight matrix verification:")
            print(f"  Max row sum error: {row_error:.2e}")
            print(f"  Max col sum error: {col_error:.2e}")

        if row_error > 1e-6:
            print(f"WARNING: Weight matrix not row stochastic (error: {row_error:.2e})")

        if col_error > 1e-6 and np.allclose(W, W.T):
            print(f"WARNING: Weight matrix not column stochastic (error: {col_error:.2e})")

    def compute_convergence_rate(self, W: np.ndarray) -> float:
        """
        Compute convergence rate of consensus with given weights

        Rate is determined by second-largest eigenvalue magnitude

        Args:
            W: Weight matrix

        Returns:
            Convergence rate (smaller is faster)
        """
        eigenvalues = np.linalg.eigvals(W)

        # Sort by magnitude
        eigenvalues = np.sort(np.abs(eigenvalues))[::-1]

        # Second largest eigenvalue determines convergence
        if len(eigenvalues) > 1:
            return float(eigenvalues[1])
        else:
            return 0.0

    def optimize_weights(self,
                        adjacency_matrix: np.ndarray,
                        max_iterations: int = 100) -> np.ndarray:
        """
        Optimize weights to minimize convergence time

        Uses gradient descent on the second eigenvalue

        Args:
            adjacency_matrix: Binary adjacency matrix
            max_iterations: Maximum optimization iterations

        Returns:
            Optimized weight matrix
        """
        # Start with Metropolis weights
        W = self.compute_weights(adjacency_matrix)
        best_W = W.copy()
        best_rate = self.compute_convergence_rate(W)

        n = len(adjacency_matrix)
        learning_rate = 0.01

        for iteration in range(max_iterations):
            # Compute gradient numerically
            gradient = np.zeros_like(W)
            epsilon = 1e-6

            for i in range(n):
                for j in range(n):
                    if adjacency_matrix[i, j] > 0 and i != j:
                        # Perturb weight
                        W_plus = W.copy()
                        W_plus[i, j] += epsilon
                        W_plus[i, i] -= epsilon  # Maintain row sum

                        # Compute change in convergence rate
                        rate_plus = self.compute_convergence_rate(W_plus)
                        gradient[i, j] = (rate_plus - best_rate) / epsilon

            # Update weights
            W = W - learning_rate * gradient

            # Project back to feasible set
            W = np.maximum(W, 0)  # Non-negative

            # Restore row sums
            for i in range(n):
                row_sum = np.sum(W[i, :]) - W[i, i]
                W[i, i] = 1.0 - row_sum

            # Check improvement
            rate = self.compute_convergence_rate(W)
            if rate < best_rate:
                best_rate = rate
                best_W = W.copy()

            # Decay learning rate
            learning_rate *= 0.99

            if self.config.verbose and iteration % 20 == 0:
                print(f"  Iteration {iteration}: rate = {rate:.4f}")

        return best_W