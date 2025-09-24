"""
Line search algorithms for robust step size selection
"""

import numpy as np
from typing import Tuple, Callable, Optional
from dataclasses import dataclass


@dataclass
class LineSearchConfig:
    """Configuration for line search algorithms"""
    method: str = "armijo"  # "armijo", "wolfe", "strong_wolfe"
    max_iterations: int = 20
    c1: float = 1e-4  # Armijo constant (sufficient decrease)
    c2: float = 0.9   # Wolfe curvature constant
    alpha_init: float = 1.0  # Initial step size
    alpha_min: float = 1e-10  # Minimum step size
    alpha_max: float = 10.0  # Maximum step size
    rho: float = 0.5  # Backtracking factor
    verbose: bool = False


class LineSearch:
    """
    Line search algorithms for finding optimal step size

    Supports:
    - Armijo backtracking
    - Wolfe conditions
    - Strong Wolfe conditions
    """

    def __init__(self, config: LineSearchConfig = None):
        self.config = config or LineSearchConfig()

    def armijo_backtracking(self,
                           f: Callable,
                           grad_f: Callable,
                           x: np.ndarray,
                           p: np.ndarray,
                           f0: Optional[float] = None,
                           grad0: Optional[np.ndarray] = None) -> Tuple[float, int]:
        """
        Armijo backtracking line search

        Finds α such that:
            f(x + α*p) ≤ f(x) + c1*α*(∇f(x)^T * p)

        Args:
            f: Objective function
            grad_f: Gradient function
            x: Current point
            p: Search direction
            f0: f(x) if already computed
            grad0: ∇f(x) if already computed

        Returns:
            (alpha, n_evals): Step size and number of function evaluations
        """
        # Compute initial values if not provided
        if f0 is None:
            f0 = f(x)
        if grad0 is None:
            grad0 = grad_f(x)

        # Check that p is a descent direction
        descent = np.dot(grad0, p)
        if descent > 0:
            if self.config.verbose:
                print(f"Warning: Not a descent direction (∇f·p = {descent:.2e} > 0)")
            return 0.0, 1

        alpha = self.config.alpha_init
        c1 = self.config.c1
        n_evals = 0

        for _ in range(self.config.max_iterations):
            # Evaluate function at new point
            f_new = f(x + alpha * p)
            n_evals += 1

            # Check Armijo condition
            if f_new <= f0 + c1 * alpha * descent:
                return alpha, n_evals

            # Backtrack
            alpha *= self.config.rho

            if alpha < self.config.alpha_min:
                if self.config.verbose:
                    print(f"Line search: alpha too small ({alpha:.2e})")
                alpha = self.config.alpha_min  # Enforce minimum
                break

        return alpha, n_evals

    def wolfe_search(self,
                    f: Callable,
                    grad_f: Callable,
                    x: np.ndarray,
                    p: np.ndarray,
                    f0: Optional[float] = None,
                    grad0: Optional[np.ndarray] = None) -> Tuple[float, int]:
        """
        Line search satisfying Wolfe conditions

        Finds α such that:
            1. f(x + α*p) ≤ f(x) + c1*α*(∇f(x)^T * p)  [Armijo]
            2. ∇f(x + α*p)^T * p ≥ c2*(∇f(x)^T * p)    [Curvature]

        Args:
            f: Objective function
            grad_f: Gradient function
            x: Current point
            p: Search direction
            f0: f(x) if already computed
            grad0: ∇f(x) if already computed

        Returns:
            (alpha, n_evals): Step size and number of evaluations
        """
        if f0 is None:
            f0 = f(x)
        if grad0 is None:
            grad0 = grad_f(x)

        descent = np.dot(grad0, p)
        if descent > 0:
            return 0.0, 1

        c1 = self.config.c1
        c2 = self.config.c2
        alpha_lo = 0.0
        alpha_hi = self.config.alpha_max
        alpha = self.config.alpha_init
        n_evals = 0

        for _ in range(self.config.max_iterations):
            x_new = x + alpha * p
            f_new = f(x_new)
            n_evals += 1

            # Check Armijo condition
            if f_new > f0 + c1 * alpha * descent:
                # Alpha too large, zoom between alpha_lo and alpha
                alpha_hi = alpha
                alpha = 0.5 * (alpha_lo + alpha_hi)
            else:
                # Armijo satisfied, check curvature
                grad_new = grad_f(x_new)
                n_evals += 1  # Count gradient evaluation

                new_descent = np.dot(grad_new, p)

                if new_descent < c2 * descent:
                    # Need larger alpha
                    alpha_lo = alpha
                    if alpha_hi < self.config.alpha_max:
                        alpha = 0.5 * (alpha_lo + alpha_hi)
                    else:
                        alpha = 2.0 * alpha
                else:
                    # Both conditions satisfied
                    return alpha, n_evals

            if abs(alpha_hi - alpha_lo) < self.config.alpha_min:
                break

        return alpha, n_evals

    def strong_wolfe_search(self,
                           f: Callable,
                           grad_f: Callable,
                           x: np.ndarray,
                           p: np.ndarray,
                           f0: Optional[float] = None,
                           grad0: Optional[np.ndarray] = None) -> Tuple[float, int]:
        """
        Line search satisfying strong Wolfe conditions

        Finds α such that:
            1. f(x + α*p) ≤ f(x) + c1*α*(∇f(x)^T * p)    [Armijo]
            2. |∇f(x + α*p)^T * p| ≤ c2*|∇f(x)^T * p|   [Strong curvature]

        Returns:
            (alpha, n_evals): Step size and number of evaluations
        """
        if f0 is None:
            f0 = f(x)
        if grad0 is None:
            grad0 = grad_f(x)

        descent = np.dot(grad0, p)
        if descent > 0:
            return 0.0, 1

        c1 = self.config.c1
        c2 = self.config.c2
        alpha = self.config.alpha_init
        n_evals = 0

        # Bracketing phase
        alpha_prev = 0.0
        f_prev = f0
        grad_prev = grad0
        first = True

        for i in range(self.config.max_iterations):
            x_new = x + alpha * p
            f_new = f(x_new)
            n_evals += 1

            if f_new > f0 + c1 * alpha * descent or (not first and f_new >= f_prev):
                # Found bracket [alpha_prev, alpha]
                return self._zoom(f, grad_f, x, p, alpha_prev, alpha,
                                f0, grad0, descent, n_evals)

            grad_new = grad_f(x_new)
            n_evals += 1

            new_descent = np.dot(grad_new, p)

            if abs(new_descent) <= -c2 * descent:
                # Strong Wolfe conditions satisfied
                return alpha, n_evals

            if new_descent >= 0:
                # Found bracket [alpha, alpha_prev]
                return self._zoom(f, grad_f, x, p, alpha, alpha_prev,
                                f0, grad0, descent, n_evals)

            # Continue bracketing
            alpha_prev = alpha
            f_prev = f_new
            grad_prev = grad_new
            alpha = min(2 * alpha, self.config.alpha_max)
            first = False

        return alpha, n_evals

    def _zoom(self, f, grad_f, x, p, alpha_lo, alpha_hi,
             f0, grad0, descent, n_evals) -> Tuple[float, int]:
        """
        Zoom phase for strong Wolfe line search

        Refines bracket [alpha_lo, alpha_hi] to find point satisfying strong Wolfe
        """
        c1 = self.config.c1
        c2 = self.config.c2

        for _ in range(10):  # Max zoom iterations
            # Bisection (could use cubic interpolation for efficiency)
            alpha = 0.5 * (alpha_lo + alpha_hi)

            x_new = x + alpha * p
            f_new = f(x_new)
            n_evals += 1

            f_lo = f(x + alpha_lo * p)

            if f_new > f0 + c1 * alpha * descent or f_new >= f_lo:
                alpha_hi = alpha
            else:
                grad_new = grad_f(x_new)
                n_evals += 1
                new_descent = np.dot(grad_new, p)

                if abs(new_descent) <= -c2 * descent:
                    return alpha, n_evals

                if new_descent * (alpha_hi - alpha_lo) >= 0:
                    alpha_hi = alpha_lo

                alpha_lo = alpha

            if abs(alpha_hi - alpha_lo) < self.config.alpha_min:
                break

        return alpha, n_evals

    def search(self, *args, **kwargs) -> Tuple[float, int]:
        """
        Perform line search using configured method

        Returns:
            (alpha, n_evals): Optimal step size and function evaluations
        """
        if self.config.method == "armijo":
            return self.armijo_backtracking(*args, **kwargs)
        elif self.config.method == "wolfe":
            return self.wolfe_search(*args, **kwargs)
        elif self.config.method == "strong_wolfe":
            return self.strong_wolfe_search(*args, **kwargs)
        else:
            raise ValueError(f"Unknown line search method: {self.config.method}")


class BacktrackingLineSearch:
    """
    Simple backtracking line search for quick use
    """

    @staticmethod
    def search(f: Callable,
              x: np.ndarray,
              p: np.ndarray,
              grad_dot_p: float,
              alpha_init: float = 1.0,
              rho: float = 0.5,
              c: float = 1e-4,
              max_iter: int = 20) -> float:
        """
        Simple backtracking line search

        Args:
            f: Objective function
            x: Current point
            p: Search direction
            grad_dot_p: Gradient dot product with direction
            alpha_init: Initial step size
            rho: Backtracking factor
            c: Armijo constant
            max_iter: Maximum iterations

        Returns:
            Optimal step size
        """
        alpha = alpha_init
        f0 = f(x)

        for _ in range(max_iter):
            if f(x + alpha * p) <= f0 + c * alpha * grad_dot_p:
                return alpha
            alpha *= rho

        return alpha