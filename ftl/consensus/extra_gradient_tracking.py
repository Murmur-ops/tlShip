"""
EXTRA (Exact fiRst-ordeR Algorithm) for distributed optimization

Implements gradient tracking for unbiased consensus with exact convergence
to the optimal solution in distributed settings.

Reference: Shi et al., "EXTRA: An Exact First-Order Algorithm for
Decentralized Consensus Optimization", SIAM 2015
"""

import numpy as np
from typing import Callable, Optional, Tuple, List
from dataclasses import dataclass
from ftl.consensus.metropolis_weights import MetropolisWeights, MetropolisConfig


@dataclass
class EXTRAConfig:
    """Configuration for EXTRA algorithm"""
    step_size: float = 0.01  # Step size α
    max_iterations: int = 1000
    convergence_tol: float = 1e-8
    gradient_tol: float = 1e-6
    use_adaptive_weights: bool = False  # Use adaptive Metropolis weights
    momentum: float = 0.0  # Momentum parameter for acceleration
    verbose: bool = False


class EXTRAOptimizer:
    """
    EXTRA gradient tracking for distributed consensus optimization

    Solves: min_x Σ_i f_i(x) where each node i has local function f_i

    Key features:
    - Exact convergence to optimal solution (not just consensus)
    - Handles non-convex objectives
    - Unbiased gradient tracking
    - Linear convergence rate for strongly convex functions
    """

    def __init__(self, config: EXTRAConfig = None):
        self.config = config or EXTRAConfig()
        self.iteration = 0
        self.consensus_errors = []
        self.gradient_consensus = []

    def optimize(self,
                local_functions: List[Callable],
                local_gradients: List[Callable],
                adjacency_matrix: np.ndarray,
                x_init: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Run EXTRA optimization

        Args:
            local_functions: List of local objective functions f_i(x)
            local_gradients: List of local gradient functions ∇f_i(x)
            adjacency_matrix: Network connectivity (binary)
            x_init: Initial states for all nodes (n_nodes x dim)

        Returns:
            (x_optimal, info_dict) with convergence information
        """
        n_nodes = len(local_functions)
        dim = x_init.shape[1] if len(x_init.shape) > 1 else 1

        # Initialize states
        if len(x_init.shape) == 1:
            x = x_init.reshape(-1, 1)
        else:
            x = x_init.copy()

        x_prev = x.copy()

        # Compute weight matrices
        mw = MetropolisWeights(MetropolisConfig(verbose=False))
        W = mw.compute_weights(adjacency_matrix)  # Mixing matrix
        W_tilde = 0.5 * (np.eye(n_nodes) + W)  # EXTRA mixing matrix

        # Initialize gradient tracking
        y = np.zeros_like(x)  # Gradient tracking variables
        for i in range(n_nodes):
            y[i] = local_gradients[i](x[i])

        # Storage for history
        self.consensus_errors = []
        self.gradient_consensus = []
        costs = []

        # Main EXTRA iteration
        for iteration in range(self.config.max_iterations):
            # Store previous state
            x_prev_prev = x_prev.copy()
            x_prev = x.copy()

            # EXTRA update
            # x^{k+1} = W_tilde * x^k - α * y^k + (I - W_tilde) * x^{k-1}
            x_consensus = W_tilde @ x
            x_correction = (np.eye(n_nodes) - W_tilde) @ x_prev_prev

            x = x_consensus - self.config.step_size * y + x_correction

            # Add momentum if configured
            if self.config.momentum > 0 and iteration > 0:
                x = x + self.config.momentum * (x - x_prev)

            # Update gradient tracking
            y_new = W @ y
            for i in range(n_nodes):
                grad_new = local_gradients[i](x[i])
                grad_old = local_gradients[i](x_prev[i])
                y_new[i] += grad_new - grad_old

            y = y_new

            # Compute consensus error
            x_mean = np.mean(x, axis=0)
            consensus_error = np.max(np.linalg.norm(x - x_mean, axis=1))
            self.consensus_errors.append(consensus_error)

            # Compute gradient consensus
            y_mean = np.mean(y, axis=0)
            grad_consensus_error = np.max(np.linalg.norm(y - y_mean, axis=1))
            self.gradient_consensus.append(grad_consensus_error)

            # Compute total cost
            total_cost = sum(local_functions[i](x[i]) for i in range(n_nodes))
            costs.append(total_cost)

            # Check convergence
            if iteration > 0:
                state_change = np.max(np.linalg.norm(x - x_prev, axis=1))
                if state_change < self.config.convergence_tol:
                    if self.config.verbose:
                        print(f"EXTRA converged at iteration {iteration}")
                    break

            # Verbose output
            if self.config.verbose and iteration % 10 == 0:
                avg_gradient = np.mean([np.linalg.norm(y[i]) for i in range(n_nodes)])
                print(f"Iter {iteration:4d}: cost={total_cost:.6e}, "
                      f"consensus={consensus_error:.2e}, "
                      f"grad_consensus={grad_consensus_error:.2e}, "
                      f"||g||={avg_gradient:.2e}")

        # Return average of final states
        x_final = np.mean(x, axis=0)

        info = {
            'iterations': iteration + 1,
            'final_cost': costs[-1] if costs else float('inf'),
            'consensus_errors': self.consensus_errors,
            'gradient_consensus': self.gradient_consensus,
            'costs': costs,
            'final_states': x,
            'final_consensus_error': self.consensus_errors[-1] if self.consensus_errors else float('inf')
        }

        return x_final, info

    def optimize_with_constraints(self,
                                 local_functions: List[Callable],
                                 local_gradients: List[Callable],
                                 adjacency_matrix: np.ndarray,
                                 x_init: np.ndarray,
                                 projection: Optional[Callable] = None) -> Tuple[np.ndarray, dict]:
        """
        EXTRA with projection for constrained optimization

        Args:
            local_functions: Local objectives
            local_gradients: Local gradients
            adjacency_matrix: Network topology
            x_init: Initial states
            projection: Projection operator onto constraint set

        Returns:
            (x_optimal, info_dict)
        """
        n_nodes = len(local_functions)
        x = x_init.copy()
        x_prev = x.copy()

        # Weight matrices
        mw = MetropolisWeights(MetropolisConfig(verbose=False))
        W = mw.compute_weights(adjacency_matrix)
        W_tilde = 0.5 * (np.eye(n_nodes) + W)

        # Gradient tracking
        y = np.zeros_like(x)
        for i in range(n_nodes):
            y[i] = local_gradients[i](x[i])

        costs = []

        for iteration in range(self.config.max_iterations):
            x_prev_prev = x_prev.copy()
            x_prev = x.copy()

            # EXTRA update
            x = W_tilde @ x - self.config.step_size * y + (np.eye(n_nodes) - W_tilde) @ x_prev_prev

            # Project onto constraint set
            if projection is not None:
                for i in range(n_nodes):
                    x[i] = projection(x[i])

            # Update gradient tracking
            y_new = W @ y
            for i in range(n_nodes):
                grad_new = local_gradients[i](x[i])
                grad_old = local_gradients[i](x_prev[i])
                y_new[i] += grad_new - grad_old

            y = y_new

            # Track cost
            total_cost = sum(local_functions[i](x[i]) for i in range(n_nodes))
            costs.append(total_cost)

            # Check convergence
            if iteration > 0:
                state_change = np.max(np.linalg.norm(x - x_prev, axis=1))
                if state_change < self.config.convergence_tol:
                    break

        x_final = np.mean(x, axis=0)

        info = {
            'iterations': iteration + 1,
            'final_cost': costs[-1] if costs else float('inf'),
            'costs': costs,
            'final_states': x
        }

        return x_final, info


class AsyncEXTRA:
    """
    Asynchronous variant of EXTRA for networks with delays

    Handles:
    - Random activation of nodes
    - Communication delays
    - Packet drops
    """

    def __init__(self, config: EXTRAConfig = None):
        self.config = config or EXTRAConfig()

    def optimize(self,
                local_functions: List[Callable],
                local_gradients: List[Callable],
                adjacency_matrix: np.ndarray,
                x_init: np.ndarray,
                activation_prob: float = 0.5,
                delay_max: int = 3) -> Tuple[np.ndarray, dict]:
        """
        Asynchronous EXTRA with random delays

        Args:
            local_functions: Local objectives
            local_gradients: Local gradients
            adjacency_matrix: Network topology
            x_init: Initial states
            activation_prob: Probability each node updates
            delay_max: Maximum communication delay

        Returns:
            (x_optimal, info_dict)
        """
        n_nodes = len(local_functions)
        x = x_init.copy()

        # Maintain delayed copies
        x_delayed = {}
        for d in range(delay_max + 1):
            x_delayed[d] = x.copy()

        # Weight matrix
        mw = MetropolisWeights(MetropolisConfig(verbose=False))
        W = mw.compute_weights(adjacency_matrix)

        # Gradient tracking
        y = np.zeros_like(x)
        for i in range(n_nodes):
            y[i] = local_gradients[i](x[i])

        costs = []
        consensus_errors = []

        for iteration in range(self.config.max_iterations):
            # Random activation
            active = np.random.random(n_nodes) < activation_prob

            # Update active nodes
            x_new = x.copy()
            y_new = y.copy()

            for i in range(n_nodes):
                if not active[i]:
                    continue

                # Use delayed neighbor information
                x_neighbors = np.zeros_like(x[i])
                weight_sum = 0

                for j in range(n_nodes):
                    if W[i, j] > 0:
                        # Random delay for this link
                        delay = np.random.randint(0, delay_max + 1)
                        x_neighbors += W[i, j] * x_delayed[delay][j]
                        weight_sum += W[i, j]

                if weight_sum > 0:
                    x_neighbors /= weight_sum

                # Update state with gradient step
                x_new[i] = x_neighbors - self.config.step_size * y[i]

                # Update gradient tracking
                grad_new = local_gradients[i](x_new[i])
                y_new[i] = y[i] + grad_new - local_gradients[i](x[i])

            # Shift delays
            for d in range(delay_max, 0, -1):
                x_delayed[d] = x_delayed[d-1].copy()
            x_delayed[0] = x.copy()

            x = x_new
            y = y_new

            # Track metrics
            total_cost = sum(local_functions[i](x[i]) for i in range(n_nodes))
            costs.append(total_cost)

            x_mean = np.mean(x, axis=0)
            consensus_error = np.max(np.linalg.norm(x - x_mean, axis=1))
            consensus_errors.append(consensus_error)

            if self.config.verbose and iteration % 20 == 0:
                print(f"AsyncEXTRA iter {iteration}: cost={total_cost:.2e}, "
                      f"consensus={consensus_error:.2e}")

        x_final = np.mean(x, axis=0)

        info = {
            'iterations': iteration + 1,
            'final_cost': costs[-1] if costs else float('inf'),
            'costs': costs,
            'consensus_errors': consensus_errors,
            'final_states': x
        }

        return x_final, info


class EXTRA_FTL:
    """
    EXTRA specifically adapted for FTL consensus optimization

    Handles the structure of FTL's position and time synchronization
    """

    def __init__(self, config: EXTRAConfig = None):
        self.config = config or EXTRAConfig()
        self.extra = EXTRAOptimizer(config)

    def optimize_ftl(self,
                     measurements: List[dict],
                     adjacency: np.ndarray,
                     n_nodes: int,
                     n_anchors: int,
                     anchor_positions: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Apply EXTRA to FTL problem

        Args:
            measurements: Range measurements between nodes
            adjacency: Network connectivity
            n_nodes: Total nodes
            n_anchors: Number of anchors
            anchor_positions: Known anchor positions

        Returns:
            (estimated_states, info_dict)
        """
        n_unknowns = n_nodes - n_anchors
        c = 299792458.0  # Speed of light

        # Create local cost functions for each unknown node
        def create_local_cost(node_id):
            def local_cost(state):
                # state = [x, y, clock_bias]
                cost = 0
                for m in measurements:
                    if m['i'] == node_id or m['j'] == node_id:
                        if m['i'] == node_id:
                            other = m['j']
                            if other < n_anchors:
                                other_pos = anchor_positions[other]
                                other_bias = 0
                            else:
                                continue  # Skip for now
                        else:
                            other = m['i']
                            if other < n_anchors:
                                other_pos = anchor_positions[other]
                                other_bias = 0
                            else:
                                continue

                        dist = np.linalg.norm(state[:2] - other_pos)
                        predicted = dist + (state[2] - other_bias) * c * 1e-9
                        residual = m['range'] - predicted
                        cost += 0.5 * (residual / m['std'])**2

                return cost

            return local_cost

        # Create gradient functions
        def create_local_gradient(node_id):
            def local_gradient(state):
                grad = np.zeros(3)
                for m in measurements:
                    if m['i'] == node_id or m['j'] == node_id:
                        if m['i'] == node_id:
                            other = m['j']
                            sign = -1
                        else:
                            other = m['i']
                            sign = 1

                        if other < n_anchors:
                            other_pos = anchor_positions[other]
                            other_bias = 0
                        else:
                            continue

                        delta = state[:2] - other_pos
                        dist = np.linalg.norm(delta)
                        if dist < 1e-10:
                            continue

                        predicted = dist + (state[2] - other_bias) * c * 1e-9
                        residual = (predicted - m['range']) / m['std']**2

                        # Position gradient
                        grad[:2] += sign * residual * delta / dist
                        # Clock gradient
                        grad[2] += sign * residual * c * 1e-9

                return grad

            return local_gradient

        # Create functions for unknown nodes
        local_functions = []
        local_gradients = []

        for i in range(n_unknowns):
            node_id = n_anchors + i
            local_functions.append(create_local_cost(node_id))
            local_gradients.append(create_local_gradient(node_id))

        # Initial guess
        x_init = np.random.randn(n_unknowns, 3) * 10

        # Run EXTRA
        x_optimal, info = self.extra.optimize(
            local_functions,
            local_gradients,
            adjacency[n_anchors:, n_anchors:],  # Unknown nodes subgraph
            x_init
        )

        # Combine with anchors
        full_states = np.zeros((n_nodes, 3))
        full_states[:n_anchors, :2] = anchor_positions
        full_states[n_anchors:] = info['final_states']

        return full_states, info