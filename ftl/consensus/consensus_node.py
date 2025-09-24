"""
Consensus node implementation
Each node maintains its own state and exchanges information with neighbors
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time

from ..factors_scaled import ToAFactorMeters, TDOAFactorMeters, ClockPriorFactor
from .message_types import StateMessage, ConvergenceStatus


@dataclass
class ConsensusNodeConfig:
    """Configuration for a consensus node"""
    node_id: int
    is_anchor: bool = False
    consensus_gain: float = 1.0  # μ in the algorithm
    step_size: float = 0.5  # α for gradient updates
    gradient_tol: float = 1e-6
    step_tol: float = 1e-8
    max_stale_time: float = 1.0  # Max age for neighbor states
    damping_lambda: float = 1e-4  # Levenberg-Marquardt damping
    state_scale: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0, 0.1, 0.1]))


class ConsensusNode:
    """
    Individual node in the consensus network
    Maintains local state and performs distributed optimization
    """

    def __init__(self, config: ConsensusNodeConfig, initial_state: np.ndarray):
        """
        Initialize consensus node

        Args:
            config: Node configuration
            initial_state: Initial state estimate [x, y, bias, drift, cfo]
        """
        self.config = config
        self.state = initial_state.copy()
        self.previous_state = initial_state.copy()

        # Neighbor management
        self.neighbors: Dict[int, int] = {}  # neighbor_id -> last_update_time
        self.neighbor_states: Dict[int, np.ndarray] = {}
        self.neighbor_timestamps: Dict[int, float] = {}

        # Local measurements/factors
        self.local_factors: List = []

        # Optimization state
        self.iteration = 0
        self.converged = False
        self.gradient_norm = float('inf')
        self.step_norm = float('inf')
        self.cost = float('inf')
        self.convergence_count = 0  # Consecutive iterations meeting criteria

        # History for analysis
        self.state_history: List[np.ndarray] = [initial_state.copy()]
        self.cost_history: List[float] = []
        self.gradient_history: List[float] = []

    def add_neighbor(self, neighbor_id: int):
        """Add a neighbor to this node's neighbor list"""
        if neighbor_id not in self.neighbors:
            self.neighbors[neighbor_id] = 0
            self.neighbor_states[neighbor_id] = None
            self.neighbor_timestamps[neighbor_id] = 0.0

    def remove_neighbor(self, neighbor_id: int):
        """Remove a neighbor from the list"""
        if neighbor_id in self.neighbors:
            del self.neighbors[neighbor_id]
            if neighbor_id in self.neighbor_states:
                del self.neighbor_states[neighbor_id]
                del self.neighbor_timestamps[neighbor_id]

    def add_measurement(self, factor):
        """Add a measurement factor (ToA, TDOA, etc.)"""
        self.local_factors.append(factor)

    def receive_state(self, msg: StateMessage):
        """
        Receive and store neighbor state

        Args:
            msg: State message from neighbor
        """
        if msg.node_id not in self.neighbors:
            return  # Ignore messages from non-neighbors

        # Check if message is too old
        if msg.age() > self.config.max_stale_time:
            return

        # Update neighbor state
        self.neighbor_states[msg.node_id] = msg.state.copy()
        self.neighbor_timestamps[msg.node_id] = msg.timestamp
        self.neighbors[msg.node_id] = self.iteration

    def build_local_system(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build local linearized system from measurements

        Returns:
            H: Hessian matrix (normal equations)
            g: Gradient vector
        """
        n_vars = 5  # State dimension

        # Initialize system
        H = np.zeros((n_vars, n_vars))
        g = np.zeros(n_vars)

        # Process each factor
        for factor in self.local_factors:
            if isinstance(factor, ToAFactorMeters):
                # Get involved node states
                if factor.i == self.config.node_id:
                    xi = self.state
                    xj = self._get_node_state(factor.j)
                elif factor.j == self.config.node_id:
                    xi = self._get_node_state(factor.i)
                    xj = self.state
                else:
                    continue  # Factor doesn't involve this node

                if xi is None or xj is None:
                    continue  # Skip if we don't have the state

                # Compute whitened residual and Jacobian
                r_wh, Ji_wh, Jj_wh = factor.whitened_residual_and_jacobian(xi, xj)

                # Add to normal equations (only our part)
                if factor.i == self.config.node_id:
                    H += np.outer(Ji_wh, Ji_wh)  # For scalar residual, J^T @ J equals outer(J, J)
                    g += Ji_wh * r_wh  # Gradient contribution
                else:
                    H += np.outer(Jj_wh, Jj_wh)  # For scalar residual, J^T @ J equals outer(J, J)
                    g += Jj_wh * r_wh  # Gradient contribution

            elif isinstance(factor, ClockPriorFactor):
                if factor.node_id != self.config.node_id:
                    continue

                # Add clock prior
                r_wh = factor.whitened_residual(self.state)
                J_wh = factor.whitened_jacobian(self.state)

                H += J_wh.T @ J_wh
                g += J_wh.T @ r_wh

        return H, g

    def add_consensus_penalty(self, H: np.ndarray, g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add consensus penalty term to the optimization

        Args:
            H: Current Hessian
            g: Current gradient

        Returns:
            Modified H and g with consensus terms
        """
        # Get valid neighbor states
        valid_neighbors = self._get_valid_neighbors()

        if len(valid_neighbors) == 0:
            return H, g  # No neighbors to consensus with

        # Consensus penalty: μ/2 * Σ ||x_i - x_j||²
        n_neighbors = len(valid_neighbors)
        mu = self.config.consensus_gain

        # Add to Hessian
        H += mu * n_neighbors * np.eye(5)

        # Add to gradient
        consensus_term = np.zeros(5)
        for neighbor_id, neighbor_state in valid_neighbors.items():
            consensus_term += mu * (neighbor_state - self.state)

        g -= consensus_term

        return H, g

    def compute_step(self, H: np.ndarray, g: np.ndarray) -> np.ndarray:
        """
        Compute optimization step using damped Gauss-Newton

        Args:
            H: Hessian matrix
            g: Gradient vector

        Returns:
            Step vector
        """
        # Apply state scaling
        S = np.diag(self.config.state_scale)
        H_scaled = S.T @ H @ S
        g_scaled = S.T @ g

        # Levenberg-Marquardt damping
        lambda_lm = self.config.damping_lambda
        diag_H = np.diag(H_scaled)

        # Regularize small/zero diagonal elements
        min_diag = 1e-6
        diag_regularized = np.where(diag_H < min_diag, min_diag, diag_H)
        H_damped = H_scaled + lambda_lm * np.diag(diag_regularized)

        try:
            # Solve for scaled step
            delta_scaled = np.linalg.solve(H_damped, g_scaled)

            # Unscale
            delta = S @ delta_scaled

            return delta
        except np.linalg.LinAlgError:
            # If solve fails, use gradient descent fallback
            return self.config.step_size * g / (np.linalg.norm(g) + 1e-10)

    def update_state(self):
        """
        Perform one consensus iteration
        Updates state based on local measurements and neighbor consensus
        """
        if self.config.is_anchor:
            # Anchors don't update their state
            self.converged = True
            self.gradient_norm = 0.0
            self.step_norm = 0.0
            return

        # Save previous state
        self.previous_state = self.state.copy()

        # Build local system
        H, g = self.build_local_system()

        # Add consensus penalty
        H, g = self.add_consensus_penalty(H, g)

        # Compute step
        delta = self.compute_step(H, g)

        # Update state
        # We solve H*delta = g where g = J^T*r (gradient of ||r||^2)
        # For descent, we move opposite to gradient: x_new = x - alpha*delta
        self.state = self.state - self.config.step_size * delta

        # Update convergence metrics
        self.gradient_norm = np.linalg.norm(g, ord=np.inf)
        self.step_norm = np.linalg.norm(delta, ord=np.inf)

        # Compute cost (optional, for monitoring)
        self.cost = self._compute_cost()

        # Check convergence
        self._check_convergence()

        # Update history
        self.state_history.append(self.state.copy())
        self.cost_history.append(self.cost)
        self.gradient_history.append(self.gradient_norm)

        self.iteration += 1

    def _compute_cost(self) -> float:
        """Compute local cost function value"""
        cost = 0.0

        for factor in self.local_factors:
            if isinstance(factor, ToAFactorMeters):
                if factor.i == self.config.node_id:
                    xi = self.state
                    xj = self._get_node_state(factor.j)
                elif factor.j == self.config.node_id:
                    xi = self._get_node_state(factor.i)
                    xj = self.state
                else:
                    continue

                if xi is None or xj is None:
                    continue

                r = factor.residual(xi, xj)
                cost += 0.5 * (r ** 2) / factor.variance

        return cost

    def _check_convergence(self):
        """Check if node has converged"""
        grad_converged = self.gradient_norm < self.config.gradient_tol
        step_converged = self.step_norm < self.config.step_tol

        if grad_converged and step_converged:
            self.convergence_count += 1
            if self.convergence_count >= 3:  # Require 3 consecutive iterations
                self.converged = True
        else:
            self.convergence_count = 0
            self.converged = False

    def _get_node_state(self, node_id: int) -> Optional[np.ndarray]:
        """Get state of a node (self or neighbor)"""
        if node_id == self.config.node_id:
            return self.state
        elif node_id in self.neighbor_states:
            return self.neighbor_states[node_id]
        return None

    def _get_valid_neighbors(self) -> Dict[int, np.ndarray]:
        """Get neighbors with valid (non-stale) states"""
        valid = {}
        current_time = time.time()

        for neighbor_id, state in self.neighbor_states.items():
            if state is None:
                continue

            timestamp = self.neighbor_timestamps.get(neighbor_id, 0)
            age = current_time - timestamp

            if age < self.config.max_stale_time:
                valid[neighbor_id] = state

        return valid

    def get_state_message(self) -> StateMessage:
        """Create state message for broadcasting"""
        return StateMessage(
            node_id=self.config.node_id,
            state=self.state.copy(),
            iteration=self.iteration,
            timestamp=time.time()
        )

    def get_convergence_status(self) -> ConvergenceStatus:
        """Get current convergence status"""
        return ConvergenceStatus(
            node_id=self.config.node_id,
            iteration=self.iteration,
            converged=self.converged,
            gradient_norm=self.gradient_norm,
            step_norm=self.step_norm,
            cost=self.cost
        )

    def reset(self, initial_state: Optional[np.ndarray] = None):
        """Reset node to initial conditions"""
        if initial_state is not None:
            self.state = initial_state.copy()
        else:
            self.state = self.state_history[0].copy()

        self.previous_state = self.state.copy()
        self.iteration = 0
        self.converged = False
        self.gradient_norm = float('inf')
        self.step_norm = float('inf')
        self.cost = float('inf')
        self.convergence_count = 0
        self.state_history = [self.state.copy()]
        self.cost_history = []
        self.gradient_history = []

        # Clear neighbor states
        for neighbor_id in self.neighbor_states:
            self.neighbor_states[neighbor_id] = None
            self.neighbor_timestamps[neighbor_id] = 0.0