"""
Factor Graph Components
Individual factors for ToA, TDOA, TWR, and CFO measurements
"""

import numpy as np
from typing import Tuple, Optional
from abc import ABC, abstractmethod


class Factor(ABC):
    """Base class for all factors"""

    def __init__(self, variance: float):
        self.variance = variance
        self.information = 1.0 / variance  # Information matrix (inverse covariance)

    @abstractmethod
    def residual(self, *states) -> float:
        """Compute residual given node states"""
        pass

    @abstractmethod
    def jacobian(self, *states) -> Tuple:
        """Compute Jacobian of residual w.r.t. states"""
        pass

    def error(self, *states) -> float:
        """Compute weighted squared error"""
        r = self.residual(*states)
        return 0.5 * r**2 * self.information


class ToAFactor(Factor):
    """
    Time of Arrival factor
    Measurement model: τ_ij = ||p_i - p_j||/c + b_j - b_i + ε
    """

    def __init__(self, i: int, j: int, measurement: float, variance: float):
        super().__init__(variance)
        self.i = i  # Node i index
        self.j = j  # Node j index
        self.measurement = measurement
        self.c = 3e8  # Speed of light

    def residual(self, xi: np.ndarray, xj: np.ndarray) -> float:
        """
        Compute residual for ToA measurement

        Args:
            xi: State of node i [x, y, bias, drift, cfo]
            xj: State of node j [x, y, bias, drift, cfo]

        Returns:
            Residual (measurement - prediction)
        """
        # Extract positions
        pi = xi[:2]
        pj = xj[:2]

        # Extract clock biases
        bi = xi[2]
        bj = xj[2]

        # Predicted ToA
        distance = np.linalg.norm(pi - pj)
        predicted = distance / self.c + bj - bi

        # Residual
        return self.measurement - predicted

    def jacobian(self, xi: np.ndarray, xj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Jacobian of residual w.r.t. states

        Returns:
            (J_xi, J_xj): Jacobians w.r.t. node i and j states
        """
        # Extract positions
        pi = xi[:2]
        pj = xj[:2]

        # Distance and unit vector
        diff = pi - pj
        distance = np.linalg.norm(diff)

        if distance < 1e-10:
            # Avoid division by zero
            unit_vec = np.zeros(2)
        else:
            unit_vec = diff / distance

        # Jacobian w.r.t. xi
        J_xi = np.zeros(5)
        J_xi[0] = -unit_vec[0] / self.c  # ∂r/∂x_i
        J_xi[1] = -unit_vec[1] / self.c  # ∂r/∂y_i
        J_xi[2] = 1.0  # ∂r/∂b_i (negative because -bi in residual)
        # drift and cfo don't affect ToA directly

        # Jacobian w.r.t. xj
        J_xj = np.zeros(5)
        J_xj[0] = unit_vec[0] / self.c  # ∂r/∂x_j
        J_xj[1] = unit_vec[1] / self.c  # ∂r/∂y_j
        J_xj[2] = -1.0  # ∂r/∂b_j
        # drift and cfo don't affect ToA directly

        return J_xi, J_xj


class TDOAFactor(Factor):
    """
    Time Difference of Arrival factor (anchor-referenced)
    Measurement model: τ_ij,k = (||p_j - p_k|| - ||p_i - p_k||)/c + b_j - b_i + ε
    """

    def __init__(self, i: int, j: int, k: int, measurement: float, variance: float):
        super().__init__(variance)
        self.i = i  # Node i
        self.j = j  # Node j
        self.k = k  # Reference anchor k
        self.measurement = measurement
        self.c = 3e8

    def residual(self, xi: np.ndarray, xj: np.ndarray, xk: np.ndarray) -> float:
        """Compute TDOA residual"""
        pi, pj, pk = xi[:2], xj[:2], xk[:2]
        bi, bj = xi[2], xj[2]

        # TDOA prediction
        dik = np.linalg.norm(pi - pk)
        djk = np.linalg.norm(pj - pk)
        predicted = (djk - dik) / self.c + bj - bi

        return self.measurement - predicted

    def jacobian(self, xi: np.ndarray, xj: np.ndarray, xk: np.ndarray) -> Tuple:
        """Compute Jacobians for TDOA"""
        pi, pj, pk = xi[:2], xj[:2], xk[:2]

        # Distance vectors
        dik_vec = pi - pk
        djk_vec = pj - pk
        dik = np.linalg.norm(dik_vec)
        djk = np.linalg.norm(djk_vec)

        # Unit vectors
        uik = dik_vec / (dik + 1e-10)
        ujk = djk_vec / (djk + 1e-10)

        # Jacobians
        J_xi = np.zeros(5)
        J_xi[:2] = uik / self.c
        J_xi[2] = 1.0

        J_xj = np.zeros(5)
        J_xj[:2] = -ujk / self.c
        J_xj[2] = -1.0

        J_xk = np.zeros(5)
        J_xk[:2] = (-uik + ujk) / self.c
        # Anchor has no clock error

        return J_xi, J_xj, J_xk


class TWRFactor(Factor):
    """
    Two-Way Ranging factor (bias cancels out)
    Measurement model: ρ_ij = ||p_i - p_j|| + ε
    """

    def __init__(self, i: int, j: int, measurement: float, variance: float):
        super().__init__(variance)
        self.i = i
        self.j = j
        self.measurement = measurement

    def residual(self, xi: np.ndarray, xj: np.ndarray) -> float:
        """Compute TWR residual (no bias effect)"""
        pi = xi[:2]
        pj = xj[:2]
        predicted = np.linalg.norm(pi - pj)
        return self.measurement - predicted

    def jacobian(self, xi: np.ndarray, xj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Jacobians for TWR"""
        pi = xi[:2]
        pj = xj[:2]

        diff = pi - pj
        distance = np.linalg.norm(diff)
        unit_vec = diff / (distance + 1e-10)

        # Only position affects TWR
        J_xi = np.zeros(5)
        J_xi[:2] = -unit_vec

        J_xj = np.zeros(5)
        J_xj[:2] = unit_vec

        return J_xi, J_xj


class CFOFactor(Factor):
    """
    Carrier Frequency Offset factor
    Measurement model: Δf_ij = f_j - f_i + ε
    """

    def __init__(self, i: int, j: int, measurement: float, variance: float):
        super().__init__(variance)
        self.i = i
        self.j = j
        self.measurement = measurement

    def residual(self, xi: np.ndarray, xj: np.ndarray) -> float:
        """Compute CFO residual"""
        fi = xi[4]  # CFO is 5th element
        fj = xj[4]
        predicted = fj - fi
        return self.measurement - predicted

    def jacobian(self, xi: np.ndarray, xj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Jacobians for CFO"""
        J_xi = np.zeros(5)
        J_xi[4] = 1.0  # ∂r/∂f_i

        J_xj = np.zeros(5)
        J_xj[4] = -1.0  # ∂r/∂f_j

        return J_xi, J_xj


class PriorFactor(Factor):
    """
    Prior factor for fixing or constraining node states
    Used for anchors or initial estimates
    """

    def __init__(self, node_id: int, prior_mean: np.ndarray, prior_covariance: np.ndarray):
        # Use determinant for scalar variance (simplified)
        variance = np.linalg.det(prior_covariance)**(1/len(prior_mean))
        super().__init__(variance)
        self.node_id = node_id
        self.prior_mean = prior_mean
        self.prior_info = np.linalg.inv(prior_covariance)

    def residual(self, x: np.ndarray) -> np.ndarray:
        """Compute prior residual"""
        return x - self.prior_mean

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian is identity for prior"""
        return np.eye(len(x))

    def error(self, x: np.ndarray) -> float:
        """Compute Mahalanobis distance"""
        r = self.residual(x)
        return 0.5 * r.T @ self.prior_info @ r