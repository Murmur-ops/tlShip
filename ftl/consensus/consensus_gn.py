"""
Consensus-Gauss-Newton distributed optimization algorithm
Coordinates multiple nodes to achieve network-wide consensus while respecting local measurements
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import time
import logging

from .consensus_node import ConsensusNode, ConsensusNodeConfig
from .message_types import StateMessage, ConvergenceStatus, NetworkMessage, MessageType
from ..factors_scaled import ToAFactorMeters, TDOAFactorMeters, ClockPriorFactor


@dataclass
class ConsensusGNConfig:
    """Configuration for Consensus-Gauss-Newton algorithm"""
    max_iterations: int = 20
    consensus_gain: float = 1.0  # μ parameter
    step_size: float = 0.5  # α parameter
    gradient_tol: float = 1e-6
    step_tol: float = 1e-8
    synchronous: bool = True  # If True, wait for all nodes each iteration
    timeout: float = 1.0  # Timeout for synchronous mode
    min_neighbors: int = 1  # Minimum neighbors for non-anchor nodes
    require_global_convergence: bool = True  # All nodes must converge
    verbose: bool = False


class ConsensusGaussNewton:
    """
    Consensus-Gauss-Newton algorithm implementation
    Manages distributed optimization across network of nodes
    """

    def __init__(self, config: ConsensusGNConfig = None):
        """
        Initialize Consensus-GN algorithm

        Args:
            config: Algorithm configuration
        """
        self.config = config or ConsensusGNConfig()
        self.nodes: Dict[int, ConsensusNode] = {}
        self.edges: Set[Tuple[int, int]] = set()  # Network topology
        self.measurements: List = []  # All measurements in network
        self.iteration = 0
        self.converged = False

        # Logging
        self.logger = logging.getLogger(__name__)
        if self.config.verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        # Performance tracking
        self.convergence_history: List[Dict] = []
        self.message_count = 0
        self.total_bytes = 0

    def add_node(self, node_id: int, initial_state: np.ndarray, is_anchor: bool = False):
        """
        Add a node to the network

        Args:
            node_id: Unique node identifier
            initial_state: Initial state estimate [x, y, bias, drift, cfo]
            is_anchor: Whether this is an anchor node
        """
        config = ConsensusNodeConfig(
            node_id=node_id,
            is_anchor=is_anchor,
            consensus_gain=self.config.consensus_gain,
            step_size=self.config.step_size,
            gradient_tol=self.config.gradient_tol,
            step_tol=self.config.step_tol
        )

        node = ConsensusNode(config, initial_state)
        self.nodes[node_id] = node

        self.logger.debug(f"Added {'anchor' if is_anchor else 'unknown'} node {node_id} at {initial_state[:2]}")

    def add_edge(self, node_i: int, node_j: int):
        """
        Add communication edge between two nodes

        Args:
            node_i: First node ID
            node_j: Second node ID
        """
        if node_i not in self.nodes or node_j not in self.nodes:
            raise ValueError(f"Both nodes must exist before adding edge")

        # Add bidirectional edge
        self.edges.add((min(node_i, node_j), max(node_i, node_j)))

        # Update neighbor lists
        self.nodes[node_i].add_neighbor(node_j)
        self.nodes[node_j].add_neighbor(node_i)

        self.logger.debug(f"Added edge between nodes {node_i} and {node_j}")

    def add_measurement(self, factor):
        """
        Add a measurement to the network

        Args:
            factor: Measurement factor (ToA, TDOA, etc.)
        """
        self.measurements.append(factor)

        # Add to relevant nodes
        if isinstance(factor, ToAFactorMeters):
            if factor.i in self.nodes:
                self.nodes[factor.i].add_measurement(factor)
            if factor.j in self.nodes:
                self.nodes[factor.j].add_measurement(factor)

        elif isinstance(factor, TDOAFactorMeters):
            if factor.i in self.nodes:
                self.nodes[factor.i].add_measurement(factor)
            if factor.j in self.nodes:
                self.nodes[factor.j].add_measurement(factor)
            if factor.k in self.nodes:
                self.nodes[factor.k].add_measurement(factor)

        elif isinstance(factor, ClockPriorFactor):
            if factor.node_id in self.nodes:
                self.nodes[factor.node_id].add_measurement(factor)

    def validate_network(self) -> Tuple[bool, List[str]]:
        """
        Validate network connectivity and configuration

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []

        # Check if network is connected
        if not self._is_connected():
            issues.append("Network is not fully connected")

        # Check each non-anchor has enough neighbors
        for node_id, node in self.nodes.items():
            if not node.config.is_anchor:
                n_neighbors = len(node.neighbors)
                if n_neighbors < self.config.min_neighbors:
                    issues.append(f"Node {node_id} has only {n_neighbors} neighbors (min: {self.config.min_neighbors})")

        # Check if we have any anchors
        anchor_count = sum(1 for n in self.nodes.values() if n.config.is_anchor)
        if anchor_count == 0:
            issues.append("No anchor nodes in network")

        # Check if we have measurements
        if len(self.measurements) == 0:
            issues.append("No measurements in network")

        return len(issues) == 0, issues

    def _is_connected(self) -> bool:
        """Check if network graph is connected using DFS"""
        if len(self.nodes) == 0:
            return True

        # Start from any node
        start_node = next(iter(self.nodes.keys()))
        visited = set()
        stack = [start_node]

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)

            # Add neighbors
            if node in self.nodes:
                for neighbor in self.nodes[node].neighbors:
                    if neighbor not in visited:
                        stack.append(neighbor)

        return len(visited) == len(self.nodes)

    def _exchange_states(self):
        """
        Simulate state exchange between neighbors
        All nodes broadcast their states and receive neighbor states
        """
        # Collect all state messages
        messages = {}
        for node_id, node in self.nodes.items():
            messages[node_id] = node.get_state_message()
            self.message_count += 1
            self.total_bytes += len(messages[node_id].serialize())

        # Distribute messages to neighbors
        for node_id, node in self.nodes.items():
            for neighbor_id in node.neighbors:
                if neighbor_id in messages:
                    node.receive_state(messages[neighbor_id])

    def _check_global_convergence(self) -> Tuple[bool, Dict]:
        """
        Check if all nodes have converged

        Returns:
            (converged, statistics)
        """
        converged_nodes = []
        max_gradient = 0.0
        max_step = 0.0
        total_cost = 0.0

        for node_id, node in self.nodes.items():
            if not node.config.is_anchor:  # Anchors always "converged"
                if node.converged:
                    converged_nodes.append(node_id)
                max_gradient = max(max_gradient, node.gradient_norm)
                max_step = max(max_step, node.step_norm)
                total_cost += node.cost

        n_unknown = sum(1 for n in self.nodes.values() if not n.config.is_anchor)
        convergence_ratio = len(converged_nodes) / max(n_unknown, 1)

        all_converged = convergence_ratio == 1.0 if self.config.require_global_convergence else convergence_ratio > 0.8

        stats = {
            'converged_nodes': len(converged_nodes),
            'total_unknown_nodes': n_unknown,
            'convergence_ratio': convergence_ratio,
            'max_gradient_norm': max_gradient,
            'max_step_norm': max_step,
            'total_cost': total_cost,
            'all_converged': all_converged
        }

        return all_converged, stats

    def optimize(self) -> Dict:
        """
        Run Consensus-Gauss-Newton optimization

        Returns:
            Dictionary with optimization results
        """
        # Validate network
        is_valid, issues = self.validate_network()
        if not is_valid:
            self.logger.error(f"Network validation failed: {issues}")
            return {
                'success': False,
                'converged': False,
                'errors': issues,
                'iterations': 0
            }

        self.logger.info(f"Starting Consensus-GN with {len(self.nodes)} nodes, {len(self.edges)} edges")

        start_time = time.time()

        for iteration in range(self.config.max_iterations):
            self.iteration = iteration

            # Exchange states between neighbors
            self._exchange_states()

            # Each node performs local update
            for node in self.nodes.values():
                node.update_state()

            # Check convergence
            converged, stats = self._check_global_convergence()
            stats['iteration'] = iteration
            self.convergence_history.append(stats)

            if self.config.verbose or iteration % 10 == 0:
                self.logger.info(
                    f"Iteration {iteration}: "
                    f"converged {stats['converged_nodes']}/{stats['total_unknown_nodes']}, "
                    f"max_grad={stats['max_gradient_norm']:.2e}, "
                    f"cost={stats['total_cost']:.3f}"
                )

            if converged:
                self.converged = True
                self.logger.info(f"Converged after {iteration + 1} iterations")
                break

        optimization_time = time.time() - start_time

        # Collect final states
        final_states = {}
        for node_id, node in self.nodes.items():
            final_states[node_id] = node.state.copy()

        # Compute final statistics
        position_errors = self._compute_position_errors() if hasattr(self, 'true_positions') else {}

        results = {
            'success': True,
            'converged': self.converged,
            'iterations': self.iteration + 1,
            'time': optimization_time,
            'final_states': final_states,
            'convergence_history': self.convergence_history,
            'message_count': self.message_count,
            'total_bytes': self.total_bytes,
            'position_errors': position_errors
        }

        return results

    def _compute_position_errors(self) -> Dict:
        """Compute position errors if true positions are known"""
        if not hasattr(self, 'true_positions'):
            return {}

        errors = {}
        for node_id, node in self.nodes.items():
            if node_id in self.true_positions:
                true_pos = self.true_positions[node_id]
                est_pos = node.state[:2]
                errors[node_id] = np.linalg.norm(est_pos - true_pos)

        if errors:
            return {
                'node_errors': errors,
                'mean': np.mean(list(errors.values())),
                'std': np.std(list(errors.values())),
                'max': np.max(list(errors.values())),
                'rmse': np.sqrt(np.mean([e**2 for e in errors.values()]))
            }
        return {}

    def set_true_positions(self, positions: Dict[int, np.ndarray]):
        """
        Set true positions for error computation

        Args:
            positions: Dict mapping node_id to true position [x, y]
        """
        self.true_positions = positions

    def get_node_states(self) -> Dict[int, np.ndarray]:
        """Get current states of all nodes"""
        return {node_id: node.state.copy() for node_id, node in self.nodes.items()}

    def get_convergence_status(self) -> Dict[int, ConvergenceStatus]:
        """Get convergence status of all nodes"""
        return {node_id: node.get_convergence_status() for node_id, node in self.nodes.items()}

    def reset(self):
        """Reset algorithm to initial conditions"""
        self.iteration = 0
        self.converged = False
        self.convergence_history = []
        self.message_count = 0
        self.total_bytes = 0

        for node in self.nodes.values():
            node.reset()

    def get_network_statistics(self) -> Dict:
        """Get network topology statistics"""
        n_nodes = len(self.nodes)
        n_anchors = sum(1 for n in self.nodes.values() if n.config.is_anchor)
        n_edges = len(self.edges)

        # Degree distribution
        degrees = [len(n.neighbors) for n in self.nodes.values()]

        return {
            'n_nodes': n_nodes,
            'n_anchors': n_anchors,
            'n_unknowns': n_nodes - n_anchors,
            'n_edges': n_edges,
            'n_measurements': len(self.measurements),
            'avg_degree': np.mean(degrees) if degrees else 0,
            'min_degree': min(degrees) if degrees else 0,
            'max_degree': max(degrees) if degrees else 0,
            'is_connected': self._is_connected()
        }