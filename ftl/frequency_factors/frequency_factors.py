"""
Factor classes for frequency synchronization in FTL

Implements measurement factors that account for clock frequency offset (drift)
in addition to time offset (bias).
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class FrequencyConfig:
    """Configuration for frequency synchronization"""
    c: float = 299792458.0  # Speed of light (m/s)
    frequency_prior_ppb: float = 10.0  # Expected frequency offset (parts per billion)
    phase_wavelength: float = 0.19  # Carrier wavelength for phase measurements (m)
    doppler_carrier_freq: float = 1.575e9  # GPS L1 frequency (Hz)


class RangeFrequencyFactor:
    """
    Range measurement factor with frequency offset compensation

    State vector: [x, y, tau, delta_f]
    - x, y: position (meters)
    - tau: clock bias (nanoseconds)
    - delta_f: frequency offset (ppb - parts per billion)
    """

    def __init__(self, measured_range: float, timestamp: float, sigma: float = 0.01):
        """
        Initialize range factor with frequency

        Args:
            measured_range: Measured pseudorange (meters)
            timestamp: Time since reference epoch (seconds)
            sigma: Measurement standard deviation (meters)
        """
        self.range = measured_range
        self.timestamp = timestamp
        self.sigma = sigma
        self.c = 299792458.0

    def error(self, state_i: np.ndarray, state_j: np.ndarray) -> float:
        """
        Compute measurement residual

        Args:
            state_i: State of node i [x, y, tau, delta_f]
            state_j: State of node j [x, y, tau, delta_f]

        Returns:
            Normalized residual
        """
        # Extract state components
        pos_i = state_i[:2]
        pos_j = state_j[:2]
        tau_i = state_i[2] if len(state_i) > 2 else 0.0
        tau_j = state_j[2] if len(state_j) > 2 else 0.0

        # Frequency offsets (in ppb)
        df_i = state_i[3] if len(state_i) > 3 else 0.0
        df_j = state_j[3] if len(state_j) > 3 else 0.0

        # Geometric range
        dist = np.linalg.norm(pos_j - pos_i)

        # Clock contribution with frequency drift
        # tau is in nanoseconds, df is in ppb
        # Clock drift accumulation: df * timestamp gives additional ns
        time_offset = (tau_j - tau_i) + (df_j - df_i) * self.timestamp
        clock_contrib = time_offset * self.c * 1e-9  # Convert ns to meters

        # Total predicted range
        predicted = dist + clock_contrib

        # Residual
        return (predicted - self.range) / self.sigma

    def jacobian(self, state_i: np.ndarray, state_j: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Jacobian of error with respect to states

        Returns:
            (jacobian_i, jacobian_j): Gradients with respect to each state
        """
        # Extract positions
        pos_i = state_i[:2]
        pos_j = state_j[:2]

        # Geometric range and direction
        delta = pos_j - pos_i
        dist = np.linalg.norm(delta)

        if dist < 1e-10:
            # Avoid division by zero
            direction = np.zeros(2)
        else:
            direction = delta / dist

        # Jacobian for node i (4 components)
        J_i = np.zeros(4)
        J_i[:2] = -direction / self.sigma  # Position gradient
        J_i[2] = -self.c * 1e-9 / self.sigma  # Clock bias gradient
        J_i[3] = -self.c * 1e-9 * self.timestamp / self.sigma  # Frequency gradient

        # Jacobian for node j
        J_j = np.zeros(4)
        J_j[:2] = direction / self.sigma
        J_j[2] = self.c * 1e-9 / self.sigma
        J_j[3] = self.c * 1e-9 * self.timestamp / self.sigma

        return J_i, J_j


class FrequencyPrior:
    """
    Prior constraint on frequency offset

    Encodes expectation that clock frequency is close to nominal
    """

    def __init__(self, nominal_freq_ppb: float = 0.0, sigma_ppb: float = 10.0):
        """
        Initialize frequency prior

        Args:
            nominal_freq_ppb: Expected frequency offset (ppb)
            sigma_ppb: Standard deviation of frequency offset (ppb)
        """
        self.nominal = nominal_freq_ppb
        self.sigma = sigma_ppb

    def error(self, state: np.ndarray) -> float:
        """
        Compute prior error

        Args:
            state: Node state [x, y, tau, delta_f]

        Returns:
            Normalized error
        """
        if len(state) < 4:
            return 0.0  # No frequency component

        df = state[3]
        return (df - self.nominal) / self.sigma

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of prior

        Returns:
            Gradient with respect to state
        """
        J = np.zeros(4)
        J[3] = 1.0 / self.sigma  # Only frequency component
        return J


class DopplerFactor:
    """
    Doppler shift measurement factor

    Measures frequency shift due to relative motion and clock drift
    """

    def __init__(self, doppler_shift: float, carrier_freq: float, sigma_hz: float = 1.0):
        """
        Initialize Doppler factor

        Args:
            doppler_shift: Measured Doppler shift (Hz)
            carrier_freq: Carrier frequency (Hz)
            sigma_hz: Measurement uncertainty (Hz)
        """
        self.doppler = doppler_shift
        self.f0 = carrier_freq
        self.sigma = sigma_hz
        self.c = 299792458.0

    def error(self, state_i: np.ndarray, state_j: np.ndarray,
              vel_i: Optional[np.ndarray] = None, vel_j: Optional[np.ndarray] = None) -> float:
        """
        Compute Doppler residual

        Args:
            state_i, state_j: Node states [x, y, tau, delta_f]
            vel_i, vel_j: Optional velocity vectors (m/s)

        Returns:
            Normalized residual
        """
        # Extract positions
        pos_i = state_i[:2]
        pos_j = state_j[:2]

        # Line-of-sight vector
        los = pos_j - pos_i
        dist = np.linalg.norm(los)
        if dist > 0:
            los = los / dist
        else:
            return 0.0

        # Velocity contribution (if provided)
        doppler_vel = 0.0
        if vel_i is not None and vel_j is not None:
            v_rel = vel_j - vel_i
            doppler_vel = np.dot(v_rel, los) * self.f0 / self.c

        # Frequency offset contribution
        df_i = state_i[3] if len(state_i) > 3 else 0.0
        df_j = state_j[3] if len(state_j) > 3 else 0.0
        doppler_freq = (df_j - df_i) * 1e-9 * self.f0  # Convert ppb to Hz

        # Total predicted Doppler
        predicted = doppler_vel + doppler_freq

        return (predicted - self.doppler) / self.sigma


class CarrierPhaseFactor:
    """
    Carrier phase measurement factor

    High-precision measurement using carrier phase observations
    """

    def __init__(self, phase_cycles: float, wavelength: float, timestamp: float, sigma_cycles: float = 0.01):
        """
        Initialize carrier phase factor

        Args:
            phase_cycles: Measured phase difference (cycles)
            wavelength: Carrier wavelength (meters)
            timestamp: Time since reference (seconds)
            sigma_cycles: Phase measurement uncertainty (cycles)
        """
        self.phase = phase_cycles
        self.wavelength = wavelength
        self.timestamp = timestamp
        self.sigma = sigma_cycles
        self.c = 299792458.0

    def error(self, state_i: np.ndarray, state_j: np.ndarray, ambiguity: Optional[int] = None) -> float:
        """
        Compute phase residual

        Args:
            state_i, state_j: Node states [x, y, tau, delta_f]
            ambiguity: Optional integer ambiguity

        Returns:
            Normalized residual
        """
        # Extract state components
        pos_i = state_i[:2]
        pos_j = state_j[:2]
        tau_i = state_i[2] if len(state_i) > 2 else 0.0
        tau_j = state_j[2] if len(state_j) > 2 else 0.0
        df_i = state_i[3] if len(state_i) > 3 else 0.0
        df_j = state_j[3] if len(state_j) > 3 else 0.0

        # Geometric phase (in cycles)
        dist = np.linalg.norm(pos_j - pos_i)
        geom_phase = dist / self.wavelength

        # Clock phase contribution
        time_offset = (tau_j - tau_i) * 1e-9  # Convert ns to seconds
        clock_phase = time_offset * self.c / self.wavelength

        # Frequency phase contribution (integrated drift)
        freq_phase = 0.5 * (df_j - df_i) * 1e-9 * self.timestamp**2 * self.c / self.wavelength

        # Total predicted phase
        predicted = geom_phase + clock_phase + freq_phase

        # Add ambiguity if provided
        if ambiguity is not None:
            predicted += ambiguity

        # Phase difference: measured - predicted
        phase_diff = self.phase - predicted

        # Wrap to [-0.5, 0.5] cycles
        phase_diff = (phase_diff + 0.5) % 1.0 - 0.5

        return phase_diff / self.sigma


class MultiEpochFactor:
    """
    Factor combining measurements from multiple epochs for better frequency observability
    """

    def __init__(self, measurements: list, timestamps: list, sigma: float = 0.01):
        """
        Initialize multi-epoch factor

        Args:
            measurements: List of range measurements
            timestamps: List of corresponding timestamps
            sigma: Measurement uncertainty
        """
        self.measurements = np.array(measurements)
        self.timestamps = np.array(timestamps)
        self.sigma = sigma
        self.c = 299792458.0
        self.n_epochs = len(measurements)

    def error_vector(self, state_i: np.ndarray, state_j: np.ndarray) -> np.ndarray:
        """
        Compute residual vector for all epochs

        Returns:
            Vector of normalized residuals
        """
        errors = np.zeros(self.n_epochs)

        for k, (meas, t) in enumerate(zip(self.measurements, self.timestamps)):
            # Create temporary factor for this epoch
            factor = RangeFrequencyFactor(meas, t, self.sigma)
            errors[k] = factor.error(state_i, state_j)

        return errors

    def total_error(self, state_i: np.ndarray, state_j: np.ndarray) -> float:
        """
        Compute total squared error

        Returns:
            Sum of squared residuals
        """
        errors = self.error_vector(state_i, state_j)
        return np.sum(errors**2)