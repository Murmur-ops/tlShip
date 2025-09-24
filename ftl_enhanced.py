#!/usr/bin/env python3
"""
Enhanced FTL with adaptive Levenberg-Marquardt and line search
Tests integration of optimization improvements
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ftl.optimization.adaptive_lm import AdaptiveLM, AdaptiveLMConfig
from ftl.optimization.line_search import LineSearch, LineSearchConfig
from dataclasses import dataclass
from typing import Tuple, Dict, List


@dataclass
class EnhancedFTLConfig:
    """Configuration for enhanced FTL system"""
    # Network
    n_nodes: int = 8
    n_anchors: int = 3
    area_size: float = 50.0  # Size of square area in meters
    random_seed: int = None  # Random seed for reproducibility

    # Optimization method
    use_adaptive_lm: bool = True
    use_line_search: bool = True

    # LM parameters
    lm_initial_lambda: float = 1e-3
    lm_lambda_increase: float = 10.0
    lm_lambda_decrease: float = 10.0

    # Line search parameters
    ls_method: str = "armijo"  # "armijo", "wolfe", "strong_wolfe"
    ls_c1: float = 1e-4
    ls_c2: float = 0.9

    # Basic optimization (fallback)
    basic_step_size: float = 0.5
    basic_damping: float = 1e-6

    # Convergence
    max_iterations: int = 100
    gradient_tol: float = 1e-10
    cost_tol: float = 1e-12

    # Measurement
    measurement_std: float = 0.01  # 1cm

    verbose: bool = True


class EnhancedFTL:
    """FTL with adaptive optimization"""

    def __init__(self, config: EnhancedFTLConfig = None):
        self.config = config or EnhancedFTLConfig()

        # Initialize optimizers
        if self.config.use_adaptive_lm:
            lm_config = AdaptiveLMConfig(
                initial_lambda=self.config.lm_initial_lambda,
                lambda_increase_factor=self.config.lm_lambda_increase,
                lambda_decrease_factor=self.config.lm_lambda_decrease,
                gradient_tol=1e-6,  # More reasonable tolerance
                step_tol=1e-8,
                verbose=False
            )
            self.lm_optimizer = AdaptiveLM(lm_config)

        if self.config.use_line_search:
            ls_config = LineSearchConfig(
                method=self.config.ls_method,
                c1=self.config.ls_c1,
                c2=self.config.ls_c2,
                verbose=False
            )
            self.line_search = LineSearch(ls_config)

        # Setup network
        self._setup_network()

        # Tracking
        self.cost_history = []
        self.position_rmse_history = []
        self.time_rmse_history = []
        self.lambda_history = []
        self.alpha_history = []
        self.state_history = []  # Track full state at each iteration

    def _setup_network(self):
        """Create network topology"""
        area_size = 20.0

        # Generate positions
        self.true_positions = np.zeros((self.config.n_nodes, 2))

        # Anchors
        self.true_positions[0] = [0, 0]
        self.true_positions[1] = [area_size, 0]
        self.true_positions[2] = [area_size/2, area_size]

        # Unknowns in grid
        n_unknowns = self.config.n_nodes - self.config.n_anchors
        idx = self.config.n_anchors
        for i in range(n_unknowns):
            angle = 2 * np.pi * i / n_unknowns
            radius = area_size * 0.3
            self.true_positions[idx] = [
                area_size/2 + radius * np.cos(angle),
                area_size/2 + radius * np.sin(angle)
            ]
            idx += 1

        # Initialize states with error
        np.random.seed(42)
        self.states = np.zeros((self.config.n_nodes, 3))  # [x, y, clock_bias]

        for i in range(self.config.n_nodes):
            if i < self.config.n_anchors:
                # Anchors: perfect position
                self.states[i, :2] = self.true_positions[i]
                self.states[i, 2] = 0
            else:
                # Unknowns: add error
                self.states[i, :2] = self.true_positions[i] + np.random.normal(0, 3, 2)
                self.states[i, 2] = np.random.normal(0, 20)  # 20ns error

        # Create measurements
        self._generate_measurements()

    def _generate_measurements(self):
        """Generate distance measurements"""
        self.measurements = []

        for i in range(self.config.n_nodes):
            for j in range(i+1, self.config.n_nodes):
                true_dist = np.linalg.norm(self.true_positions[i] - self.true_positions[j])

                # Add measurement noise (or zero for testing)
                noise = 0  # Zero noise for convergence testing
                measured_dist = true_dist + noise

                self.measurements.append({
                    'i': i,
                    'j': j,
                    'range': measured_dist,
                    'std': self.config.measurement_std
                })

    def compute_cost(self, states_flat):
        """Compute total cost function"""
        # Reshape states
        states = states_flat.reshape(-1, 3)

        # Only update unknowns
        test_states = self.states.copy()
        for i in range(self.config.n_anchors, self.config.n_nodes):
            idx = i - self.config.n_anchors
            test_states[i] = states[idx]

        cost = 0
        c = 299792458.0

        for meas in self.measurements:
            i, j = meas['i'], meas['j']

            # Positions and clock biases
            pi = test_states[i, :2]
            pj = test_states[j, :2]
            bi = test_states[i, 2]
            bj = test_states[j, 2]

            # Predicted range with clock
            geom_dist = np.linalg.norm(pi - pj)
            clock_contrib = (bj - bi) * c * 1e-9
            predicted = geom_dist + clock_contrib

            # Residual
            residual = meas['range'] - predicted

            # Weighted squared error
            cost += 0.5 * (residual / meas['std'])**2

        return cost

    def compute_gradient_hessian(self, states_flat):
        """Compute gradient and Hessian"""
        n_unknowns = self.config.n_nodes - self.config.n_anchors
        n_vars = n_unknowns * 3

        # Reshape states
        states = states_flat.reshape(-1, 3)

        # Update full state array
        test_states = self.states.copy()
        for i in range(self.config.n_anchors, self.config.n_nodes):
            idx = i - self.config.n_anchors
            test_states[i] = states[idx]

        H = np.zeros((n_vars, n_vars))
        g = np.zeros(n_vars)

        c = 299792458.0

        for meas in self.measurements:
            i, j = meas['i'], meas['j']

            # Skip if both are anchors
            if i < self.config.n_anchors and j < self.config.n_anchors:
                continue

            # Get positions and clock biases
            pi = test_states[i, :2]
            pj = test_states[j, :2]
            bi = test_states[i, 2]
            bj = test_states[j, 2]

            # Geometric distance
            delta = pj - pi
            dist = np.linalg.norm(delta)

            if dist < 1e-10:
                continue

            u = delta / dist

            # Predicted and residual
            clock_contrib = (bj - bi) * c * 1e-9
            predicted = dist + clock_contrib
            residual = meas['range'] - predicted

            # Weight by measurement std
            weight = 1.0 / meas['std']**2

            # Build Jacobians for unknowns only
            if i >= self.config.n_anchors:
                idx = (i - self.config.n_anchors) * 3
                Ji = np.zeros(n_vars)
                Ji[idx:idx+2] = -u
                Ji[idx+2] = -c * 1e-9

                H += weight * np.outer(Ji, Ji)
                g -= weight * Ji * residual  # Correct gradient (negative)

            if j >= self.config.n_anchors:
                idx = (j - self.config.n_anchors) * 3
                Jj = np.zeros(n_vars)
                Jj[idx:idx+2] = u
                Jj[idx+2] = c * 1e-9

                H += weight * np.outer(Jj, Jj)
                g -= weight * Jj * residual  # Correct gradient (negative)

        return H, g

    def step_basic(self):
        """Basic gradient descent step (fallback)"""
        # Get states for unknowns only
        n_unknowns = self.config.n_nodes - self.config.n_anchors
        unknown_states = np.zeros((n_unknowns, 3))
        for i in range(n_unknowns):
            unknown_states[i] = self.states[self.config.n_anchors + i]

        states_flat = unknown_states.flatten()

        # Compute gradient and Hessian
        H, g = self.compute_gradient_hessian(states_flat)

        # Add damping
        H += self.config.basic_damping * np.eye(len(H))

        # Solve for step
        try:
            delta = np.linalg.solve(H, g)
        except:
            delta = g / (np.linalg.norm(g) + 1e-10)

        # Apply step
        states_flat += self.config.basic_step_size * delta

        # Update states
        unknown_states = states_flat.reshape(-1, 3)
        for i in range(n_unknowns):
            self.states[self.config.n_anchors + i] = unknown_states[i]

    def step_with_lm(self):
        """Step with adaptive Levenberg-Marquardt"""
        # Get states for unknowns
        n_unknowns = self.config.n_nodes - self.config.n_anchors
        unknown_states = np.zeros((n_unknowns, 3))
        for i in range(n_unknowns):
            unknown_states[i] = self.states[self.config.n_anchors + i]

        states_flat = unknown_states.flatten()

        # Define functions for LM
        def cost_fn(x):
            return self.compute_cost(x)

        def grad_fn(x):
            _, g = self.compute_gradient_hessian(x)
            return g  # Direct gradient

        def hess_fn(x):
            H, _ = self.compute_gradient_hessian(x)
            return H

        # Take LM step
        states_new, cost_new, converged = self.lm_optimizer.step(
            states_flat, hess_fn(states_flat), grad_fn(states_flat), cost_fn
        )

        # Update states
        unknown_states = states_new.reshape(-1, 3)
        for i in range(n_unknowns):
            self.states[self.config.n_anchors + i] = unknown_states[i]

        # Track lambda
        self.lambda_history.append(self.lm_optimizer.lambda_current)

        return converged

    def step_with_line_search(self):
        """Step with line search"""
        # Get states for unknowns
        n_unknowns = self.config.n_nodes - self.config.n_anchors
        unknown_states = np.zeros((n_unknowns, 3))
        for i in range(n_unknowns):
            unknown_states[i] = self.states[self.config.n_anchors + i]

        states_flat = unknown_states.flatten()

        # Compute gradient and Hessian
        H, g = self.compute_gradient_hessian(states_flat)

        # Add small damping for stability
        H += 1e-9 * np.eye(len(H))

        # Solve for search direction (Gauss-Newton direction)
        # For Gauss-Newton: p = H^{-1}*g
        try:
            p = np.linalg.solve(H, g)
        except:
            p = g  # Use gradient for descent

        # Line search
        def f(x):
            return self.compute_cost(x)

        def grad_f(x):
            _, g = self.compute_gradient_hessian(x)
            return g  # Direct gradient

        alpha, n_evals = self.line_search.search(f, grad_f, states_flat, p)

        # Apply step
        states_flat += alpha * p

        # Update states
        unknown_states = states_flat.reshape(-1, 3)
        for i in range(n_unknowns):
            self.states[self.config.n_anchors + i] = unknown_states[i]

        # Track alpha
        self.alpha_history.append(alpha)

    def step_combined(self):
        """Combined LM + line search"""
        # Get states for unknowns
        n_unknowns = self.config.n_nodes - self.config.n_anchors
        unknown_states = np.zeros((n_unknowns, 3))
        for i in range(n_unknowns):
            unknown_states[i] = self.states[self.config.n_anchors + i]

        states_flat = unknown_states.flatten()

        # Compute gradient and Hessian
        H, g = self.compute_gradient_hessian(states_flat)

        # Debug output
        grad_norm = np.linalg.norm(g)
        if grad_norm > 1e10:
            print(f"WARNING: Huge gradient norm: {grad_norm:.2e}")
            print(f"  g range: [{np.min(g):.2e}, {np.max(g):.2e}]")
            print(f"  H diag range: [{np.min(np.diag(H)):.2e}, {np.max(np.diag(H)):.2e}]")

        # Get damped step from LM
        delta = self.lm_optimizer.compute_damped_step(H, g, self.lm_optimizer.lambda_current)

        # Use line search to find optimal step size
        def f(x):
            return self.compute_cost(x)

        def grad_f(x):
            _, g = self.compute_gradient_hessian(x)
            return g  # Direct gradient

        # Line search along LM direction
        alpha, _ = self.line_search.search(f, grad_f, states_flat, delta)

        # Evaluate step
        current_cost = f(states_flat)
        new_cost = f(states_flat + alpha * delta)
        accept = new_cost < current_cost

        # Update LM damping based on success
        gain_ratio = (current_cost - new_cost) / (abs(current_cost) + 1e-10) if accept else 0
        self.lm_optimizer.update_damping(gain_ratio, accept)

        # Apply step if accepted
        if accept:
            states_flat += alpha * delta
            unknown_states = states_flat.reshape(-1, 3)
            for i in range(n_unknowns):
                self.states[self.config.n_anchors + i] = unknown_states[i]

        # Track parameters
        self.lambda_history.append(self.lm_optimizer.lambda_current)
        self.alpha_history.append(alpha if accept else 0)

        # Check convergence (g is -gradient)
        converged = np.linalg.norm(g) < self.config.gradient_tol
        return converged

    def compute_errors(self):
        """Compute current errors"""
        pos_errors = []
        time_errors = []

        for i in range(self.config.n_anchors, self.config.n_nodes):
            pos_error = np.linalg.norm(self.states[i, :2] - self.true_positions[i])
            pos_errors.append(pos_error)
            time_errors.append(abs(self.states[i, 2]))

        pos_rmse = np.sqrt(np.mean(np.array(pos_errors)**2))
        time_rmse = np.sqrt(np.mean(np.array(time_errors)**2))

        return pos_rmse, time_rmse

    def run(self):
        """Run optimization"""
        if self.config.verbose:
            print("="*60)
            print("Enhanced FTL Optimization")
            print(f"Method: ", end="")
            if self.config.use_adaptive_lm and self.config.use_line_search:
                print("Adaptive LM + Line Search")
            elif self.config.use_adaptive_lm:
                print("Adaptive LM")
            elif self.config.use_line_search:
                print("Line Search")
            else:
                print("Basic Gradient Descent")
            print("="*60)

        # Initial errors
        pos_rmse, time_rmse = self.compute_errors()
        self.position_rmse_history.append(pos_rmse)
        self.time_rmse_history.append(time_rmse)
        self.state_history.append(self.states.copy())

        if self.config.verbose:
            print(f"Initial: pos RMSE = {pos_rmse:.3f}m, time RMSE = {time_rmse:.3f}ns")

        # Run optimization
        for iteration in range(self.config.max_iterations):
            # Choose optimization method
            if self.config.use_adaptive_lm and self.config.use_line_search:
                converged = self.step_combined()
            elif self.config.use_adaptive_lm:
                converged = self.step_with_lm()
            elif self.config.use_line_search:
                self.step_with_line_search()
                converged = False
            else:
                self.step_basic()
                converged = False

            # Compute errors
            pos_rmse, time_rmse = self.compute_errors()
            self.position_rmse_history.append(pos_rmse)
            self.time_rmse_history.append(time_rmse)
            self.state_history.append(self.states.copy())

            # Compute cost
            n_unknowns = self.config.n_nodes - self.config.n_anchors
            unknown_states = np.zeros((n_unknowns, 3))
            for i in range(n_unknowns):
                unknown_states[i] = self.states[self.config.n_anchors + i]
            cost = self.compute_cost(unknown_states.flatten())
            self.cost_history.append(cost)

            # Print progress
            if self.config.verbose and (iteration % 10 == 0 or converged):
                print(f"Iter {iteration:3d}: pos RMSE = {pos_rmse:.3e}m, "
                      f"time RMSE = {time_rmse:.3e}ns, cost = {cost:.3e}")

            # Check convergence
            if converged or pos_rmse < 1e-12 and time_rmse < 1e-12:
                if self.config.verbose:
                    print(f"Converged at iteration {iteration}")
                break

        if self.config.verbose:
            print(f"\nFinal: pos RMSE = {pos_rmse:.3e}m, time RMSE = {time_rmse:.3e}ns")
            print("="*60)


def compare_methods():
    """Compare different optimization methods"""

    methods = [
        ("Basic", False, False),
        ("Line Search", False, True),
        ("Adaptive LM", True, False),
        ("LM + Line Search", True, True)
    ]

    results = {}

    for name, use_lm, use_ls in methods:
        print(f"\nTesting {name}...")

        config = EnhancedFTLConfig(
            use_adaptive_lm=use_lm,
            use_line_search=use_ls,
            max_iterations=50,
            verbose=False
        )

        ftl = EnhancedFTL(config)
        ftl.run()

        results[name] = {
            'pos_rmse': ftl.position_rmse_history,
            'time_rmse': ftl.time_rmse_history,
            'cost': ftl.cost_history,
            'iterations': len(ftl.cost_history)
        }

        print(f"  Final pos RMSE: {ftl.position_rmse_history[-1]:.3e}m")
        print(f"  Iterations: {len(ftl.cost_history)}")

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Position RMSE
    ax = axes[0, 0]
    for name in results:
        ax.semilogy(results[name]['pos_rmse'], label=name, linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Position RMSE (m)')
    ax.set_title('Position Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Time RMSE
    ax = axes[0, 1]
    for name in results:
        time_rmse = [max(t, 1e-15) for t in results[name]['time_rmse']]
        ax.semilogy(time_rmse, label=name, linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Time RMSE (ns)')
    ax.set_title('Time Synchronization')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cost function
    ax = axes[1, 0]
    for name in results:
        if results[name]['cost']:
            ax.semilogy(results[name]['cost'], label=name, linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    ax.set_title('Cost Function')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Convergence speed comparison
    ax = axes[1, 1]
    names = list(results.keys())
    iterations = [results[name]['iterations'] for name in names]
    final_errors = [results[name]['pos_rmse'][-1] for name in names]

    x = np.arange(len(names))
    ax.bar(x, iterations, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Iterations to Converge')
    ax.set_title('Convergence Speed')
    ax.grid(True, alpha=0.3, axis='y')

    # Add final error as text
    for i, (iters, error) in enumerate(zip(iterations, final_errors)):
        ax.text(i, iters + 1, f'{error:.1e}m', ha='center', fontsize=9)

    plt.suptitle('Optimization Method Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('optimization_comparison.png', dpi=150)
    plt.show()

    return results


if __name__ == "__main__":
    # Test individual enhanced system
    print("\n" + "="*60)
    print("Testing Enhanced FTL with LM + Line Search")
    print("="*60)

    config = EnhancedFTLConfig(
        use_adaptive_lm=True,
        use_line_search=True,
        verbose=True
    )

    ftl = EnhancedFTL(config)
    ftl.run()

    # Compare all methods
    print("\n" + "="*60)
    print("Comparing Optimization Methods")
    print("="*60)

    results = compare_methods()

    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    for name in results:
        r = results[name]
        print(f"{name:15s}: {r['iterations']:2d} iters, "
              f"final pos RMSE = {r['pos_rmse'][-1]:.2e}m")

    print("\nPlots saved to optimization_comparison.png")