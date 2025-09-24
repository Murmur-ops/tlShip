"""
Adaptive Levenberg-Marquardt optimization with automatic damping adjustment
"""

import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class AdaptiveLMConfig:
    """Configuration for adaptive Levenberg-Marquardt"""
    initial_lambda: float = 1e-4  # Initial damping parameter
    lambda_increase_factor: float = 10.0  # Factor to increase λ on failure
    lambda_decrease_factor: float = 10.0  # Factor to decrease λ on success
    min_lambda: float = 1e-12  # Minimum damping (near Newton)
    max_lambda: float = 1e8  # Maximum damping (near gradient descent)
    min_improvement: float = 1e-4  # Minimum relative cost improvement
    gradient_tol: float = 1e-8  # Gradient norm tolerance
    step_tol: float = 1e-10  # Step norm tolerance
    max_iterations: int = 100
    verbose: bool = False


class AdaptiveLM:
    """
    Adaptive Levenberg-Marquardt optimizer

    Automatically adjusts damping parameter based on cost reduction:
    - If step reduces cost significantly: decrease λ (trust Newton more)
    - If step fails or barely reduces cost: increase λ (trust gradient more)
    """

    def __init__(self, config: AdaptiveLMConfig = None):
        self.config = config or AdaptiveLMConfig()
        self.lambda_current = self.config.initial_lambda
        self.cost_history = []
        self.lambda_history = []
        self.success_history = []

    def compute_damped_step(self, H: np.ndarray, g: np.ndarray, lambda_val: float) -> np.ndarray:
        """
        Compute step with Levenberg-Marquardt damping

        Solves: (H + λI) δ = -g

        Args:
            H: Hessian matrix (or J^T J approximation)
            g: Gradient vector
            lambda_val: Damping parameter

        Returns:
            Step vector δ
        """
        n = len(g)

        # Add damping to diagonal
        # Use max(diag(H), 1) for scaling as in Marquardt's modification
        diag_H = np.diag(H)
        scaling = np.maximum(diag_H, np.ones(n))
        H_damped = H + lambda_val * np.diag(scaling)

        try:
            # Solve for step
            # Standard formulation: solve (H + λI)δ = -g
            delta = np.linalg.solve(H_damped, -g)
            return delta
        except np.linalg.LinAlgError:
            # If solve fails, use gradient descent fallback
            if self.config.verbose:
                print(f"  LM solve failed with λ={lambda_val:.2e}, using gradient")
            return -g / (np.linalg.norm(g) + 1e-10)

    def evaluate_step(self,
                     x_current: np.ndarray,
                     delta: np.ndarray,
                     cost_fn: Callable,
                     current_cost: float) -> Tuple[float, float, bool]:
        """
        Evaluate if step improves cost

        Args:
            x_current: Current state
            delta: Proposed step
            cost_fn: Function to evaluate cost
            current_cost: Current cost value

        Returns:
            (new_cost, gain_ratio, accept):
                new_cost: Cost at new point
                gain_ratio: Actual/predicted improvement ratio
                accept: Whether to accept step
        """
        # Evaluate cost at new point
        x_new = x_current + delta
        new_cost = cost_fn(x_new)

        # Calculate actual reduction
        actual_reduction = current_cost - new_cost

        # Accept if there's any improvement
        if actual_reduction > 0:
            # Calculate gain ratio (how well quadratic model predicted improvement)
            relative_reduction = actual_reduction / (abs(current_cost) + 1e-10)
            gain_ratio = relative_reduction

            # Accept if improvement is sufficient
            accept = relative_reduction > self.config.min_improvement or new_cost < current_cost
        else:
            gain_ratio = 0.0
            accept = False

        return new_cost, gain_ratio, accept

    def update_damping(self, gain_ratio: float, accept: bool):
        """
        Update damping parameter based on step success

        Args:
            gain_ratio: Actual/predicted improvement ratio
            accept: Whether step was accepted
        """
        if accept:
            # Good step - decrease damping (trust Newton more)
            if gain_ratio > 0.75:  # Very good prediction
                self.lambda_current /= self.config.lambda_decrease_factor
            elif gain_ratio > 0.25:  # Reasonable prediction
                self.lambda_current /= 2.0
            # Else keep current λ

            # Enforce minimum
            self.lambda_current = max(self.lambda_current, self.config.min_lambda)

            if self.config.verbose:
                print(f"  Step accepted, λ decreased to {self.lambda_current:.2e}")
        else:
            # Bad step - increase damping (trust gradient more)
            self.lambda_current *= self.config.lambda_increase_factor

            # Enforce maximum
            self.lambda_current = min(self.lambda_current, self.config.max_lambda)

            if self.config.verbose:
                print(f"  Step rejected, λ increased to {self.lambda_current:.2e}")

    def step(self,
             x_current: np.ndarray,
             H: np.ndarray,
             g: np.ndarray,
             cost_fn: Callable) -> Tuple[np.ndarray, float, bool]:
        """
        Perform one adaptive LM step

        Args:
            x_current: Current state
            H: Hessian (or J^T J)
            g: Gradient
            cost_fn: Function to evaluate cost

        Returns:
            (x_new, new_cost, converged)
        """
        # Evaluate current cost
        current_cost = cost_fn(x_current)

        # Try step with current damping
        delta = self.compute_damped_step(H, g, self.lambda_current)
        new_cost, gain_ratio, accept = self.evaluate_step(x_current, delta, cost_fn, current_cost)

        # If rejected, try with increased damping
        attempts = 0
        while not accept and attempts < 10:
            self.lambda_current *= self.config.lambda_increase_factor
            self.lambda_current = min(self.lambda_current, self.config.max_lambda)

            delta = self.compute_damped_step(H, g, self.lambda_current)
            new_cost, gain_ratio, accept = self.evaluate_step(x_current, delta, cost_fn, current_cost)
            attempts += 1

        # Update damping for next iteration
        self.update_damping(gain_ratio, accept)

        # Update history
        self.cost_history.append(new_cost if accept else current_cost)
        self.lambda_history.append(self.lambda_current)
        self.success_history.append(accept)

        # Check convergence
        grad_norm = np.linalg.norm(g)
        step_norm = np.linalg.norm(delta) if accept else 0.0

        converged = (grad_norm < self.config.gradient_tol or
                    step_norm < self.config.step_tol or
                    abs(current_cost - new_cost) < 1e-12)

        if accept:
            return x_current + delta, new_cost, converged
        else:
            return x_current, current_cost, converged

    def optimize(self,
                x_init: np.ndarray,
                gradient_fn: Callable,
                hessian_fn: Callable,
                cost_fn: Callable) -> Tuple[np.ndarray, dict]:
        """
        Run full optimization

        Args:
            x_init: Initial state
            gradient_fn: Function to compute gradient
            hessian_fn: Function to compute Hessian
            cost_fn: Function to compute cost

        Returns:
            (x_optimal, info_dict)
        """
        x = x_init.copy()

        for iteration in range(self.config.max_iterations):
            # Compute gradient and Hessian
            g = gradient_fn(x)
            H = hessian_fn(x)

            # Take step
            x_new, cost, converged = self.step(x, H, g, cost_fn)

            if self.config.verbose and iteration % 10 == 0:
                print(f"Iter {iteration}: cost={cost:.6e}, λ={self.lambda_current:.2e}, "
                      f"||g||={np.linalg.norm(g):.2e}")

            if converged:
                if self.config.verbose:
                    print(f"Converged at iteration {iteration}")
                break

            x = x_new

        info = {
            'iterations': iteration + 1,
            'final_cost': self.cost_history[-1] if self.cost_history else float('inf'),
            'final_lambda': self.lambda_current,
            'cost_history': self.cost_history,
            'lambda_history': self.lambda_history,
            'success_rate': sum(self.success_history) / len(self.success_history) if self.success_history else 0
        }

        return x, info