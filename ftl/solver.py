"""
Factor Graph Solver
Levenberg-Marquardt optimization for joint [x, y, b, d, f] estimation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .factors import Factor, ToAFactor, TDOAFactor, TWRFactor, CFOFactor, PriorFactor
from .robust import RobustConfig, RobustKernel


@dataclass
class Node:
    """Graph node representing a device"""
    id: int
    state: np.ndarray  # [x, y, bias, drift, cfo]
    is_anchor: bool = False
    fixed_dims: List[int] = field(default_factory=list)  # Which dimensions are fixed


@dataclass
class OptimizationResult:
    """Result of graph optimization"""
    estimates: Dict[int, np.ndarray]  # Final state estimates
    converged: bool
    iterations: int
    initial_cost: float
    final_cost: float
    residuals: np.ndarray
    weights: np.ndarray


class FactorGraph:
    """
    Factor graph for joint position, time, and frequency estimation
    """

    def __init__(self, robust_config: Optional[RobustConfig] = None):
        self.nodes: Dict[int, Node] = {}
        self.factors: List[Factor] = []
        self.robust_config = robust_config or RobustConfig()
        self.robust_kernel = RobustKernel(self.robust_config)

    def add_node(
        self,
        node_id: int,
        initial_estimate: np.ndarray,
        is_anchor: bool = False
    ):
        """Add a node to the graph"""
        node = Node(
            id=node_id,
            state=initial_estimate.copy(),
            is_anchor=is_anchor
        )

        if is_anchor:
            # Fix position and clock for anchors
            node.fixed_dims = [0, 1, 2, 3, 4]

        self.nodes[node_id] = node

    def add_toa_factor(
        self,
        i: int,
        j: int,
        measurement: float,
        variance: float
    ):
        """Add a ToA measurement factor"""
        factor = ToAFactor(i, j, measurement, variance)
        self.factors.append(factor)

    def add_tdoa_factor(
        self,
        i: int,
        j: int,
        k: int,
        measurement: float,
        variance: float
    ):
        """Add a TDOA measurement factor"""
        factor = TDOAFactor(i, j, k, measurement, variance)
        self.factors.append(factor)

    def add_twr_factor(
        self,
        i: int,
        j: int,
        measurement: float,
        variance: float
    ):
        """Add a TWR measurement factor"""
        factor = TWRFactor(i, j, measurement, variance)
        self.factors.append(factor)

    def add_cfo_factor(
        self,
        i: int,
        j: int,
        measurement: float,
        variance: float
    ):
        """Add a CFO measurement factor"""
        factor = CFOFactor(i, j, measurement, variance)
        self.factors.append(factor)

    def add_prior_factor(
        self,
        node_id: int,
        prior_mean: np.ndarray,
        prior_covariance: np.ndarray
    ):
        """Add a prior/constraint factor"""
        factor = PriorFactor(node_id, prior_mean, prior_covariance)
        self.factors.append(factor)

    def _compute_residuals(self) -> Tuple[np.ndarray, List[float]]:
        """Compute all residuals and standard deviations"""
        residuals = []
        stds = []

        for factor in self.factors:
            if isinstance(factor, ToAFactor):
                xi = self.nodes[factor.i].state
                xj = self.nodes[factor.j].state
                r = factor.residual(xi, xj)

            elif isinstance(factor, TDOAFactor):
                xi = self.nodes[factor.i].state
                xj = self.nodes[factor.j].state
                xk = self.nodes[factor.k].state
                r = factor.residual(xi, xj, xk)

            elif isinstance(factor, TWRFactor):
                xi = self.nodes[factor.i].state
                xj = self.nodes[factor.j].state
                r = factor.residual(xi, xj)

            elif isinstance(factor, CFOFactor):
                xi = self.nodes[factor.i].state
                xj = self.nodes[factor.j].state
                r = factor.residual(xi, xj)

            elif isinstance(factor, PriorFactor):
                x = self.nodes[factor.node_id].state
                r = np.linalg.norm(factor.residual(x))

            else:
                continue

            residuals.append(r)
            stds.append(np.sqrt(factor.variance))

        return np.array(residuals), stds

    def _compute_cost(self, weights: Optional[np.ndarray] = None) -> float:
        """Compute total cost (weighted sum of squared residuals)"""
        residuals, stds = self._compute_residuals()

        if weights is None:
            # Compute weights with same bounding as in optimization
            max_weight = 1e10
            weights = np.ones(len(residuals))
            for i, std in enumerate(stds):
                weight = 1.0 / (std * std) if std > 0 else max_weight
                weights[i] = min(weight, max_weight)

        # Weighted least squares cost
        cost = 0.0
        for r, w in zip(residuals, weights):
            if self.robust_config.use_huber:
                # Note: Huber cost needs to be updated for proper weighting
                cost += w * r**2  # Simplified for now
            else:
                cost += 0.5 * w * r**2

        return cost

    def _build_jacobian(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the full Jacobian matrix and residual vector

        Returns:
            (J, r): Jacobian matrix and residual vector
        """
        # Count total unknowns
        n_unknowns = 0
        node_to_idx = {}
        idx_to_node = {}

        for node_id, node in self.nodes.items():
            if not node.is_anchor:
                node_to_idx[node_id] = n_unknowns
                idx_to_node[n_unknowns] = node_id
                n_unknowns += 5  # [x, y, b, d, f]

        # Count measurements
        n_measurements = len(self.factors)

        # Initialize Jacobian and residual vector
        J = np.zeros((n_measurements, n_unknowns))
        r = np.zeros(n_measurements)

        # Fill Jacobian and residuals
        for m, factor in enumerate(self.factors):
            if isinstance(factor, ToAFactor):
                xi = self.nodes[factor.i].state
                xj = self.nodes[factor.j].state

                r[m] = factor.residual(xi, xj)
                Ji, Jj = factor.jacobian(xi, xj)

                # Place in global Jacobian
                if factor.i in node_to_idx:
                    idx = node_to_idx[factor.i]
                    J[m, idx:idx+5] = Ji

                if factor.j in node_to_idx:
                    idx = node_to_idx[factor.j]
                    J[m, idx:idx+5] = Jj

            elif isinstance(factor, TWRFactor):
                xi = self.nodes[factor.i].state
                xj = self.nodes[factor.j].state

                r[m] = factor.residual(xi, xj)
                Ji, Jj = factor.jacobian(xi, xj)

                if factor.i in node_to_idx:
                    idx = node_to_idx[factor.i]
                    J[m, idx:idx+5] = Ji

                if factor.j in node_to_idx:
                    idx = node_to_idx[factor.j]
                    J[m, idx:idx+5] = Jj

            elif isinstance(factor, CFOFactor):
                xi = self.nodes[factor.i].state
                xj = self.nodes[factor.j].state

                r[m] = factor.residual(xi, xj)
                Ji, Jj = factor.jacobian(xi, xj)

                if factor.i in node_to_idx:
                    idx = node_to_idx[factor.i]
                    J[m, idx:idx+5] = Ji

                if factor.j in node_to_idx:
                    idx = node_to_idx[factor.j]
                    J[m, idx:idx+5] = Jj

            # Add other factor types as needed

        return J, r, node_to_idx, idx_to_node

    def optimize(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        lambda_init: float = 1e-3,
        verbose: bool = False
    ) -> OptimizationResult:
        """
        Optimize the factor graph using Levenberg-Marquardt

        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            lambda_init: Initial damping parameter
            verbose: Print progress

        Returns:
            OptimizationResult with final estimates
        """
        # Initialize
        lambda_lm = lambda_init
        prev_cost = self._compute_cost()
        initial_cost = prev_cost
        converged = False

        if verbose:
            print(f"Initial cost: {initial_cost:.6f}")

        for iteration in range(max_iterations):
            # Build Jacobian and residuals
            J, r, node_to_idx, idx_to_node = self._build_jacobian()

            # Compute weights from variances with bounding
            residuals, stds = self._compute_residuals()
            weights = np.ones(len(residuals))

            # Compute weights as 1/variance with maximum bound
            max_weight = 1e10  # Maximum weight to prevent numerical issues
            for i, std in enumerate(stds):
                # Weight = 1/variance = 1/std²
                weight = 1.0 / (std * std) if std > 0 else max_weight
                weights[i] = min(weight, max_weight)

            # Apply robust kernel if needed
            if self.robust_config.use_huber or self.robust_config.use_dcs:
                for i, (res, std) in enumerate(zip(residuals, stds)):
                    # Multiply by robust weight (0-1 scale factor)
                    weights[i] *= self.robust_kernel.weight(res, std)

            # Apply weights
            W = np.diag(np.sqrt(weights))  # Use sqrt for proper scaling
            J_weighted = W @ J
            r_weighted = W @ r

            # Levenberg-Marquardt update
            H = J_weighted.T @ J_weighted
            g = J_weighted.T @ r_weighted

            # Add damping with minimum regularization for unconstrained variables
            diag_H = np.diag(H)
            regularization = lambda_lm * diag_H
            # Add minimum regularization to prevent singular matrix
            min_regularization = 1e-6  # Increased for better numerical stability
            regularization = np.maximum(regularization, min_regularization)
            H_damped = H + np.diag(regularization)

            try:
                # Solve for update
                delta = np.linalg.solve(H_damped, g)
            except np.linalg.LinAlgError:
                # Singular matrix, increase damping
                lambda_lm *= 10
                continue

            # Apply update to non-anchor nodes
            for node_id, node in self.nodes.items():
                if not node.is_anchor and node_id in node_to_idx:
                    idx = node_to_idx[node_id]
                    node.state -= delta[idx:idx+5]

            # Compute new cost
            new_cost = self._compute_cost(weights)

            # Check if update improved cost
            if new_cost < prev_cost:
                # Accept update, decrease damping
                lambda_lm *= 0.5
                cost_decrease = prev_cost - new_cost

                if verbose and iteration % 10 == 0:
                    print(f"Iteration {iteration}: cost = {new_cost:.6f}, "
                          f"λ = {lambda_lm:.2e}")

                # Check convergence using relative change
                relative_decrease = cost_decrease / (prev_cost + 1e-20)
                if relative_decrease < tolerance:
                    converged = True
                    break

                prev_cost = new_cost

            else:
                # Reject update, increase damping
                for node_id, node in self.nodes.items():
                    if not node.is_anchor and node_id in node_to_idx:
                        idx = node_to_idx[node_id]
                        node.state += delta[idx:idx+5]  # Undo update

                lambda_lm *= 10

        # Final residuals and weights
        final_residuals, _ = self._compute_residuals()
        final_cost = self._compute_cost(weights)

        if verbose:
            print(f"Final cost: {final_cost:.6f}")
            print(f"Converged: {converged} in {iteration+1} iterations")

        # Collect estimates
        estimates = {node_id: node.state.copy()
                    for node_id, node in self.nodes.items()}

        return OptimizationResult(
            estimates=estimates,
            converged=converged,
            iterations=iteration + 1,
            initial_cost=initial_cost,
            final_cost=final_cost,
            residuals=final_residuals,
            weights=weights
        )


if __name__ == "__main__":
    # Test solver with simple example
    print("Testing Factor Graph Solver...")
    print("=" * 50)

    graph = FactorGraph()

    # Add anchors at known positions
    graph.add_node(0, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
    graph.add_node(1, np.array([10.0, 0.0, 0.0, 0.0, 0.0]), is_anchor=True)
    graph.add_node(2, np.array([5.0, 8.66, 0.0, 0.0, 0.0]), is_anchor=True)

    # Add unknown node (true position: [5, 3])
    graph.add_node(3, np.array([6.0, 4.0, 0.0, 0.0, 0.0]), is_anchor=False)

    # Add TWR measurements
    true_pos = np.array([5.0, 3.0])
    d0 = np.linalg.norm(true_pos - np.array([0.0, 0.0]))
    d1 = np.linalg.norm(true_pos - np.array([10.0, 0.0]))
    d2 = np.linalg.norm(true_pos - np.array([5.0, 8.66]))

    graph.add_twr_factor(0, 3, measurement=d0, variance=0.01)
    graph.add_twr_factor(1, 3, measurement=d1, variance=0.01)
    graph.add_twr_factor(2, 3, measurement=d2, variance=0.01)

    # Optimize
    result = graph.optimize(verbose=True)

    # Check result
    estimated_pos = result.estimates[3][:2]
    error = np.linalg.norm(estimated_pos - true_pos)

    print(f"\nTrue position: {true_pos}")
    print(f"Estimated position: {estimated_pos}")
    print(f"Position error: {error:.3f} m")