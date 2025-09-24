"""
Square-root information solver with proper scaling
Works in whitened (normalized) space to avoid numerical issues
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from ftl.factors_scaled import ToAFactorMeters, TDOAFactorMeters, ClockPriorFactor


@dataclass
class ScaledNode:
    """Node with scaled state"""
    id: int
    state: np.ndarray  # [x_m, y_m, bias_ns, drift_ppb, cfo_ppm]
    is_anchor: bool

    def copy(self):
        return ScaledNode(self.id, self.state.copy(), self.is_anchor)


@dataclass
class OptimizationConfig:
    """Configuration for optimization"""
    max_iterations: int = 100
    gradient_tol: float = 1e-6      # Gradient infinity norm threshold
    step_tol: float = 1e-8          # Step relative threshold
    cost_tol: float = 1e-9          # Cost relative change threshold
    lambda_init: float = 1e-4       # Initial Levenberg-Marquardt damping
    lambda_scale_up: float = 10.0   # Scale factor when step rejected
    lambda_scale_down: float = 0.1  # Scale factor when step accepted
    state_scale: np.ndarray = None  # Diagonal scaling for state variables

    def get_default_state_scale(self):
        """Default state scaling [m, m, ns, ppb, ppm]"""
        if self.state_scale is None:
            return np.array([1.0, 1.0, 1.0, 0.1, 0.1])
        return self.state_scale


@dataclass
class OptimizationResult:
    """Result from optimization"""
    converged: bool
    iterations: int
    final_cost: float
    gradient_norm: float
    step_norm: float
    estimates: Dict[int, np.ndarray]
    convergence_reason: str


class SquareRootSolver:
    """
    Square-root information solver for factor graphs
    Uses whitened formulation to avoid numerical issues
    """

    def __init__(self, config: OptimizationConfig = None):
        """
        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        self.nodes: Dict[int, ScaledNode] = {}
        self.factors: List = []

    def add_node(self, node_id: int, initial_state: np.ndarray, is_anchor: bool = False):
        """Add a node to the graph"""
        self.nodes[node_id] = ScaledNode(node_id, initial_state.copy(), is_anchor)

    def add_toa_factor(self, i: int, j: int, range_meas_m: float, range_var_m2: float):
        """Add ToA factor with measurement in meters"""
        factor = ToAFactorMeters(i, j, range_meas_m, range_var_m2)
        self.factors.append(factor)

    def add_tdoa_factor(self, i: int, j: int, k: int, tdoa_range_m: float, range_var_m2: float):
        """Add TDOA factor with measurement in meters"""
        factor = TDOAFactorMeters(i, j, k, tdoa_range_m, range_var_m2)
        self.factors.append(factor)

    def add_clock_prior(self, node_id: int, bias_ns: float, drift_ppb: float,
                        bias_var_ns2: float, drift_var_ppb2: float):
        """Add prior on clock parameters"""
        factor = ClockPriorFactor(node_id, bias_ns, drift_ppb, bias_var_ns2, drift_var_ppb2)
        self.factors.append(factor)

    def _build_whitened_system(self) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
        """
        Build whitened Jacobian and residual vector

        Returns:
            J_wh: Whitened Jacobian
            r_wh: Whitened residual vector
            node_to_idx: Map from node_id to column index
            idx_to_node: Map from column index to node_id
        """
        # Map nodes to indices (skip anchors)
        node_to_idx = {}
        idx_to_node = {}
        col_idx = 0

        for node_id, node in self.nodes.items():
            if not node.is_anchor:
                node_to_idx[node_id] = col_idx
                for i in range(5):  # 5 state variables
                    idx_to_node[col_idx + i] = (node_id, i)
                col_idx += 5

        n_cols = col_idx

        # Count rows
        n_rows = 0
        for factor in self.factors:
            if isinstance(factor, (ToAFactorMeters, TDOAFactorMeters)):
                n_rows += 1
            elif isinstance(factor, ClockPriorFactor):
                n_rows += 2  # Clock prior has 2 components

        # Build whitened system
        J_wh = np.zeros((n_rows, n_cols))
        r_wh = np.zeros(n_rows)

        row_idx = 0
        for factor in self.factors:
            if isinstance(factor, ToAFactorMeters):
                # Get states
                xi = self.nodes[factor.i].state
                xj = self.nodes[factor.j].state

                # Compute whitened residual and Jacobian
                r, Ji, Jj = factor.whitened_residual_and_jacobian(xi, xj)

                r_wh[row_idx] = r

                # Place Jacobians
                if factor.i in node_to_idx:
                    col = node_to_idx[factor.i]
                    J_wh[row_idx, col:col+5] = Ji

                if factor.j in node_to_idx:
                    col = node_to_idx[factor.j]
                    J_wh[row_idx, col:col+5] = Jj

                row_idx += 1

            elif isinstance(factor, TDOAFactorMeters):
                # Get states
                xi = self.nodes[factor.i].state
                xj = self.nodes[factor.j].state
                xk = self.nodes[factor.k].state

                # Compute residual and Jacobian
                r = factor.residual(xi, xj, xk)
                Ji, Jj, Jk = factor.jacobian(xi, xj, xk)

                # Whiten
                r_wh[row_idx] = factor.whiten(r)

                if factor.i in node_to_idx:
                    col = node_to_idx[factor.i]
                    J_wh[row_idx, col:col+5] = factor.whiten_jacobian(Ji)

                if factor.j in node_to_idx:
                    col = node_to_idx[factor.j]
                    J_wh[row_idx, col:col+5] = factor.whiten_jacobian(Jj)

                if factor.k in node_to_idx:
                    col = node_to_idx[factor.k]
                    J_wh[row_idx, col:col+5] = factor.whiten_jacobian(Jk)

                row_idx += 1

            elif isinstance(factor, ClockPriorFactor):
                # Get state
                x = self.nodes[factor.node_id].state

                # Compute whitened residual and Jacobian (2D)
                r = factor.whitened_residual(x)
                J = factor.whitened_jacobian(x)

                # Place in system
                r_wh[row_idx:row_idx+2] = r

                if factor.node_id in node_to_idx:
                    col = node_to_idx[factor.node_id]
                    J_wh[row_idx:row_idx+2, col:col+5] = J

                row_idx += 2

        return J_wh, r_wh, node_to_idx, idx_to_node

    def _apply_state_scaling(self, J_wh: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply state scaling to improve conditioning

        Returns:
            J_scaled: Scaled Jacobian
            S_x: Scaling matrix
        """
        S_x = self.config.get_default_state_scale()

        # Build full scaling matrix
        n_cols = J_wh.shape[1]
        S_full = np.ones(n_cols)

        for i in range(0, n_cols, 5):
            S_full[i:i+5] = S_x

        S_mat = np.diag(S_full)
        J_scaled = J_wh @ S_mat

        return J_scaled, S_mat

    def _compute_cost(self, r_wh: np.ndarray) -> float:
        """Compute cost from whitened residuals"""
        return 0.5 * np.dot(r_wh, r_wh)

    def _check_convergence(self, J_wh: np.ndarray, r_wh: np.ndarray,
                          delta: np.ndarray, cost: float, prev_cost: float,
                          iteration: int) -> Tuple[bool, str]:
        """
        Check convergence using multiple criteria

        Returns:
            (converged, reason)
        """
        # Gradient test (scale-free)
        gradient = J_wh.T @ r_wh
        grad_norm = np.linalg.norm(gradient, ord=np.inf)
        if grad_norm < self.config.gradient_tol:
            return True, f"Gradient converged ({grad_norm:.2e} < {self.config.gradient_tol:.2e})"

        # Step test (relative) - only if delta is not empty
        if delta.size > 0:
            step_norm = np.linalg.norm(delta, ord=np.inf)
            state_norm = 1.0  # Already normalized in whitened space
            if step_norm < self.config.step_tol * (1 + state_norm):
                return True, f"Step converged ({step_norm:.2e} < {self.config.step_tol:.2e})"

        # Cost test (relative change)
        if iteration > 0 and prev_cost > 0:
            cost_change = abs(cost - prev_cost) / prev_cost
            if cost_change < self.config.cost_tol:
                return True, f"Cost converged (rel change {cost_change:.2e} < {self.config.cost_tol:.2e})"

        return False, "Not converged"

    def optimize(self, verbose: bool = False) -> OptimizationResult:
        """
        Optimize the factor graph using Levenberg-Marquardt

        Returns:
            Optimization result
        """
        # Initialize
        lambda_lm = self.config.lambda_init
        prev_cost = float('inf')
        converged = False
        reason = "Max iterations reached"
        delta_unscaled = np.zeros(0)  # Initialize for first iteration

        # Store initial states
        initial_states = {nid: node.state.copy() for nid, node in self.nodes.items()}

        for iteration in range(self.config.max_iterations):
            # Build whitened system
            J_wh, r_wh, node_to_idx, idx_to_node = self._build_whitened_system()

            # Apply state scaling
            J_scaled, S_mat = self._apply_state_scaling(J_wh)

            # Current cost
            cost = self._compute_cost(r_wh)

            if iteration == 0 and verbose:
                print(f"Initial cost: {cost:.6f}")

            # Check convergence (skip on first iteration)
            if iteration > 0:
                converged, reason = self._check_convergence(
                    J_wh, r_wh, delta_unscaled, cost, prev_cost, iteration
                )
                if converged:
                    break

            # Levenberg-Marquardt step
            H = J_scaled.T @ J_scaled
            g = J_scaled.T @ r_wh

            # Add damping with minimum regularization for unobserved variables
            diag_H = np.diag(H)
            # Add small regularization where diagonal is zero (unobserved variables)
            min_diag = 1e-6
            diag_regularized = np.where(diag_H < min_diag, min_diag, diag_H)
            H_damped = H + lambda_lm * np.diag(diag_regularized)

            # Solve for update (in scaled space)
            try:
                delta_scaled = np.linalg.solve(H_damped, g)
            except np.linalg.LinAlgError:
                if verbose:
                    print(f"Failed to solve at iteration {iteration}")
                lambda_lm *= self.config.lambda_scale_up
                continue

            # Unscale the update
            delta_unscaled = S_mat @ delta_scaled

            # Apply update to nodes
            for node_id, node in self.nodes.items():
                if not node.is_anchor and node_id in node_to_idx:
                    idx = node_to_idx[node_id]
                    node.state -= delta_unscaled[idx:idx+5]

            # Compute new cost
            J_wh_new, r_wh_new, _, _ = self._build_whitened_system()
            new_cost = self._compute_cost(r_wh_new)

            # Gain ratio (how well did linear model predict decrease?)
            # Use the damped Hessian for correct predicted decrease
            predicted_decrease = np.dot(g, delta_scaled) - 0.5 * np.dot(delta_scaled, H_damped @ delta_scaled)
            actual_decrease = cost - new_cost

            if predicted_decrease > 0:
                gain_ratio = actual_decrease / predicted_decrease
            else:
                gain_ratio = 0

            if verbose and iteration % 10 == 0:
                grad_norm = np.linalg.norm(J_wh.T @ r_wh, ord=np.inf)
                print(f"Iter {iteration}: cost={cost:.6f}, grad={grad_norm:.2e}, "
                      f"λ={lambda_lm:.2e}, gain={gain_ratio:.2f}")

            # Accept or reject step based on gain ratio
            if gain_ratio > 0.25:  # Good step
                # Accept (already applied)
                lambda_lm *= self.config.lambda_scale_down
                lambda_lm = max(lambda_lm, 1e-10)
                prev_cost = cost
            else:
                # Reject - restore previous state
                for node_id, node in self.nodes.items():
                    if not node.is_anchor and node_id in node_to_idx:
                        idx = node_to_idx[node_id]
                        node.state += delta_unscaled[idx:idx+5]

                lambda_lm *= self.config.lambda_scale_up
                lambda_lm = min(lambda_lm, 1e10)

        # Final system for reporting
        J_wh_final, r_wh_final, _, _ = self._build_whitened_system()
        final_cost = self._compute_cost(r_wh_final)
        final_grad_norm = np.linalg.norm(J_wh_final.T @ r_wh_final, ord=np.inf)

        # Extract estimates
        estimates = {node_id: node.state.copy() for node_id, node in self.nodes.items()}

        if verbose:
            print(f"Optimization complete: {reason}")
            print(f"Final cost: {final_cost:.6f}, gradient: {final_grad_norm:.2e}")

        return OptimizationResult(
            converged=converged,
            iterations=iteration + 1,
            final_cost=final_cost,
            gradient_norm=final_grad_norm,
            step_norm=np.linalg.norm(delta_unscaled, ord=np.inf) if delta_unscaled.size > 0 else 0,
            estimates=estimates,
            convergence_reason=reason
        )