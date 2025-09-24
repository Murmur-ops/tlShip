"""
Scaled factor classes for numerically stable FTL optimization
All measurements and states use physically sensible units:
- Positions: meters
- Range residuals: meters
- Clock bias: nanoseconds
- Clock drift: ppb (parts per billion)
- CFO: ppm (parts per million)
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class ScaledState:
    """
    State vector with proper units
    """
    x_m: float          # x position in meters
    y_m: float          # y position in meters
    bias_ns: float      # clock bias in nanoseconds
    drift_ppb: float    # clock drift in ppb
    cfo_ppm: float      # carrier frequency offset in ppm

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x, y, b, d, f]"""
        return np.array([self.x_m, self.y_m, self.bias_ns, self.drift_ppb, self.cfo_ppm])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'ScaledState':
        """Create from numpy array"""
        return cls(arr[0], arr[1], arr[2], arr[3], arr[4])


class ScaledFactor:
    """Base class for scaled factors with proper SRIF whitening"""

    def __init__(self, variance: float):
        """
        Args:
            variance: Measurement variance in appropriate units²
        """
        self.variance = variance
        self.information = 1.0 / variance if variance > 0 else 1e10
        self.std = np.sqrt(variance) if variance > 0 else 1e-5  # Avoid division by zero

        # Square root of information matrix (L such that L^T L = Information)
        # For scalar case, this is just 1/std
        # This implements proper Square Root Information Form (SRIF)
        self.sqrt_information = 1.0 / self.std if self.std > 0 else np.sqrt(1e10)

    def whiten(self, residual: float) -> float:
        """
        Whiten residual by square root of information matrix (SRIF)

        In SRIF, we transform: r_whitened = L * r
        where L^T L = Σ^(-1) (information matrix)

        Args:
            residual: Raw residual

        Returns:
            Whitened residual (should be ~N(0,1) if model is correct)
        """
        return residual * self.sqrt_information

    def whiten_jacobian(self, jacobian: np.ndarray) -> np.ndarray:
        """
        Whiten Jacobian by square root of information matrix (SRIF)

        In SRIF, we transform: J_whitened = L * J

        Args:
            jacobian: Raw Jacobian

        Returns:
            Whitened Jacobian
        """
        return jacobian * self.sqrt_information


class ToAFactorMeters(ScaledFactor):
    """
    Time of Arrival factor with residuals in meters

    Measurement model:
        ρ_ij = ||p_i - p_j|| + c*(b_j - b_i)/1e9 + ε
    where positions are in meters and bias is in nanoseconds
    """

    def __init__(self, i: int, j: int, range_meas_m: float, range_var_m2: float):
        """
        Args:
            i: Node i index
            j: Node j index
            range_meas_m: Measured range in meters (c * tau_measured)
            range_var_m2: Range variance in meters²
        """
        super().__init__(range_var_m2)
        self.i = i
        self.j = j
        self.range_meas_m = range_meas_m
        self.c = 299792458.0  # Speed of light in m/s

    def residual(self, xi: np.ndarray, xj: np.ndarray, delta_t: float = 1.0) -> float:
        """
        Compute residual in meters

        Args:
            xi: State of node i [x_m, y_m, bias_ns, drift_ppb, cfo_ppm]
            xj: State of node j [x_m, y_m, bias_ns, drift_ppb, cfo_ppm]
            delta_t: Time elapsed since reference epoch (seconds)

        Returns:
            Residual in meters (measurement - prediction)
        """
        # Extract positions (meters)
        pi = xi[:2]
        pj = xj[:2]

        # Extract clock parameters
        bi_ns = xi[2]  # bias in nanoseconds
        bj_ns = xj[2]
        di_ppb = xi[3]  # drift in ppb (parts per billion)
        dj_ppb = xj[3]

        # Predicted range (meters)
        geometric_range = np.linalg.norm(pi - pj)

        # Clock bias contribution in meters
        # (bj - bi) nanoseconds * c meters/second * 1e-9 seconds/nanosecond
        clock_contribution = (bj_ns - bi_ns) * self.c * 1e-9

        # Clock drift contribution in meters
        # drift in ppb * time in seconds * c meters/second * 1e-9
        drift_contribution = (dj_ppb - di_ppb) * delta_t * self.c * 1e-9

        predicted_range = geometric_range + clock_contribution + drift_contribution

        # Residual in meters
        return self.range_meas_m - predicted_range

    def jacobian(self, xi: np.ndarray, xj: np.ndarray, delta_t: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Jacobian of residual w.r.t. states

        Args:
            xi: State of node i [x_m, y_m, bias_ns, drift_ppb, cfo_ppm]
            xj: State of node j [x_m, y_m, bias_ns, drift_ppb, cfo_ppm]
            delta_t: Time elapsed since reference epoch (seconds)

        Returns:
            (J_xi, J_xj): Jacobians w.r.t. node i and j states
        """
        # Extract positions
        pi = xi[:2]
        pj = xj[:2]

        # Compute unit direction vector
        delta = pj - pi
        distance = np.linalg.norm(delta)

        if distance > 1e-10:
            u = delta / distance  # Unit vector from i to j
        else:
            u = np.array([0.0, 0.0])

        # Jacobian w.r.t. node i state [x_m, y_m, bias_ns, drift_ppb, cfo_ppm]
        J_xi = np.zeros(5)
        J_xi[0] = u[0]                    # ∂r/∂xi in meters/meter
        J_xi[1] = u[1]                    # ∂r/∂yi in meters/meter
        J_xi[2] = self.c * 1e-9          # ∂r/∂bi in meters/nanosecond
        J_xi[3] = self.c * delta_t * 1e-9  # ∂r/∂di in meters/ppb (NEW: drift term)
        J_xi[4] = 0.0                    # ∂r/∂fi (CFO doesn't affect ToA directly)

        # Jacobian w.r.t. node j state
        J_xj = np.zeros(5)
        J_xj[0] = -u[0]                   # ∂r/∂xj
        J_xj[1] = -u[1]                   # ∂r/∂yj
        J_xj[2] = -self.c * 1e-9         # ∂r/∂bj
        J_xj[3] = -self.c * delta_t * 1e-9 # ∂r/∂dj (NEW: drift term)
        J_xj[4] = 0.0                    # ∂r/∂fj

        return J_xi, J_xj

    def whitened_residual_and_jacobian(self, xi: np.ndarray, xj: np.ndarray, delta_t: float = 1.0) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute whitened (normalized) residual and Jacobian

        Args:
            xi: State of node i
            xj: State of node j
            delta_t: Time elapsed since reference epoch (seconds)

        Returns:
            (r_whitened, J_xi_whitened, J_xj_whitened)
        """
        r = self.residual(xi, xj, delta_t)
        J_xi, J_xj = self.jacobian(xi, xj, delta_t)

        # Whiten by dividing by std
        r_wh = self.whiten(r)
        J_xi_wh = self.whiten_jacobian(J_xi)
        J_xj_wh = self.whiten_jacobian(J_xj)

        return r_wh, J_xi_wh, J_xj_wh


class TDOAFactorMeters(ScaledFactor):
    """
    Time Difference of Arrival factor with residuals in meters

    Measurement model:
        ρ_ijk = (||p_i - p_j|| - ||p_i - p_k||) + c*(b_j - b_k)/1e9 + ε
    """

    def __init__(self, i: int, j: int, k: int, tdoa_range_m: float, range_var_m2: float):
        """
        Args:
            i: Node i index (mobile)
            j: Node j index (anchor 1)
            k: Node k index (anchor 2)
            tdoa_range_m: Measured TDOA * c in meters
            range_var_m2: Range variance in meters²
        """
        super().__init__(range_var_m2)
        self.i = i
        self.j = j
        self.k = k
        self.tdoa_range_m = tdoa_range_m
        self.c = 299792458.0

    def residual(self, xi: np.ndarray, xj: np.ndarray, xk: np.ndarray) -> float:
        """
        Compute TDOA residual in meters
        """
        # Positions
        pi = xi[:2]
        pj = xj[:2]
        pk = xk[:2]

        # Clock biases (nanoseconds)
        bi_ns = xi[2]
        bj_ns = xj[2]
        bk_ns = xk[2]

        # Geometric ranges
        dij = np.linalg.norm(pi - pj)
        dik = np.linalg.norm(pi - pk)

        # TDOA range prediction
        geometric_tdoa = dij - dik
        clock_tdoa = (bj_ns - bk_ns) * self.c * 1e-9
        predicted = geometric_tdoa + clock_tdoa

        return self.tdoa_range_m - predicted

    def jacobian(self, xi: np.ndarray, xj: np.ndarray, xk: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Jacobian for TDOA
        """
        # Positions
        pi = xi[:2]
        pj = xj[:2]
        pk = xk[:2]

        # Unit vectors
        delta_ij = pj - pi
        delta_ik = pk - pi
        dij = np.linalg.norm(delta_ij)
        dik = np.linalg.norm(delta_ik)

        if dij > 1e-10:
            uij = delta_ij / dij
        else:
            uij = np.array([0.0, 0.0])

        if dik > 1e-10:
            uik = delta_ik / dik
        else:
            uik = np.array([0.0, 0.0])

        # Jacobian w.r.t. node i
        J_xi = np.zeros(5)
        J_xi[0] = uij[0] - uik[0]
        J_xi[1] = uij[1] - uik[1]
        J_xi[2] = 0.0  # Mobile node bias cancels in TDOA

        # Jacobian w.r.t. node j
        J_xj = np.zeros(5)
        J_xj[0] = -uij[0]
        J_xj[1] = -uij[1]
        J_xj[2] = -self.c * 1e-9

        # Jacobian w.r.t. node k
        J_xk = np.zeros(5)
        J_xk[0] = uik[0]
        J_xk[1] = uik[1]
        J_xk[2] = self.c * 1e-9

        return J_xi, J_xj, J_xk


class ClockPriorFactor(ScaledFactor):
    """
    Prior factor for clock parameters with proper SRIF implementation
    """

    def __init__(self, node_id: int, bias_ns: float, drift_ppb: float,
                 bias_var_ns2: float, drift_var_ppb2: float):
        """
        Args:
            node_id: Node index
            bias_ns: Prior mean for bias (nanoseconds)
            drift_ppb: Prior mean for drift (ppb)
            bias_var_ns2: Bias variance (ns²)
            drift_var_ppb2: Drift variance (ppb²)
        """
        # Store as covariance matrix
        self.cov = np.diag([bias_var_ns2, drift_var_ppb2])
        super().__init__(1.0)  # Dummy variance, we use cov matrix

        self.node_id = node_id
        self.prior_bias = bias_ns
        self.prior_drift = drift_ppb

        # Compute square root of information matrix using Cholesky decomposition
        # This is the proper SRIF form: L such that L^T L = Σ^(-1)
        info_matrix = np.linalg.inv(self.cov)
        self.L_info = np.linalg.cholesky(info_matrix)

    def residual(self, x: np.ndarray) -> np.ndarray:
        """
        Compute residual for clock prior

        Args:
            x: State [x_m, y_m, bias_ns, drift_ppb, cfo_ppm]

        Returns:
            2D residual [bias_error, drift_error]
        """
        bias_error = x[2] - self.prior_bias
        drift_error = x[3] - self.prior_drift
        return np.array([bias_error, drift_error])

    def whitened_residual(self, x: np.ndarray) -> np.ndarray:
        """
        Compute whitened residual
        """
        r = self.residual(x)
        return self.L_info @ r

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian w.r.t. state

        Returns:
            2x5 Jacobian matrix
        """
        J = np.zeros((2, 5))
        J[0, 2] = 1.0  # ∂(bias_residual)/∂bias = 1
        J[1, 3] = 1.0  # ∂(drift_residual)/∂drift = 1
        return J

    def whitened_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Compute whitened Jacobian
        """
        J = self.jacobian(x)
        return self.L_info @ J