"""
Measurement Covariance Estimation
Connects CRLB, SNR, and NLOS features to factor graph edge weights
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .signal import compute_rms_bandwidth
from .rx_frontend import (
    toa_crlb, cov_from_crlb,
    extract_correlation_features,
    covariance_from_features,
    classify_propagation
)


@dataclass
class MeasurementCovariance:
    """Covariance information for a measurement edge"""
    toa_variance: float
    snr_linear: float
    is_los: bool
    nlos_confidence: float
    rms_bandwidth_hz: float
    features: Dict[str, float]

    @property
    def toa_std(self) -> float:
        """Standard deviation in seconds"""
        return np.sqrt(self.toa_variance)

    @property
    def range_std_m(self) -> float:
        """Range standard deviation in meters"""
        return self.toa_std * 3e8


def compute_measurement_covariance(
    correlation: np.ndarray,
    template: np.ndarray,
    sample_rate: float,
    peak_idx: Optional[int] = None,
    min_variance: float = 1e-20,
    nlos_inflation_factor: float = 2.0,
    use_feature_scaling: bool = True
) -> MeasurementCovariance:
    """
    Compute measurement covariance from signal correlation

    This is the key function that bridges physical layer characteristics
    to factor graph optimization weights.

    Args:
        correlation: Cross-correlation output
        template: Template signal used for correlation
        sample_rate: Sample rate in Hz
        peak_idx: Peak index (auto-detect if None)
        min_variance: Minimum variance floor
        nlos_inflation_factor: NLOS variance inflation
        use_feature_scaling: Use correlation features for scaling

    Returns:
        MeasurementCovariance with all relevant information
    """
    # Find peak if not provided
    if peak_idx is None:
        peak_idx = np.argmax(np.abs(correlation))

    # Extract correlation features
    features = extract_correlation_features(correlation, peak_idx)

    # Classify propagation
    classification = classify_propagation(correlation)
    is_los = classification['type'] == 'LOS'
    nlos_confidence = classification['confidence']

    # Estimate SNR from correlation peak
    peak_power = np.abs(correlation[peak_idx])**2

    # Noise floor estimation (use samples away from peak)
    noise_region = np.concatenate([
        correlation[:max(0, peak_idx-100)],
        correlation[min(len(correlation), peak_idx+100):]
    ])

    if len(noise_region) > 0:
        noise_power = np.mean(np.abs(noise_region)**2)
        snr_linear = peak_power / noise_power if noise_power > 0 else 1000
    else:
        snr_linear = 100  # Default 20 dB

    # Compute RMS bandwidth
    beta_rms = compute_rms_bandwidth(template, sample_rate)

    # Base CRLB variance
    crlb_var = toa_crlb(snr_linear, beta_rms)

    # Apply NLOS inflation if needed
    if not is_los:
        base_var = cov_from_crlb(
            snr_linear, beta_rms,
            is_los=False,
            nlos_factor=nlos_inflation_factor,
            min_variance=min_variance
        )
    else:
        base_var = cov_from_crlb(
            snr_linear, beta_rms,
            is_los=True,
            min_variance=min_variance
        )

    # Apply feature-based scaling if requested
    if use_feature_scaling:
        final_var = covariance_from_features(base_var, features)
    else:
        final_var = base_var

    return MeasurementCovariance(
        toa_variance=final_var,
        snr_linear=snr_linear,
        is_los=is_los,
        nlos_confidence=nlos_confidence,
        rms_bandwidth_hz=beta_rms,
        features=features
    )


def estimate_toa_variance(
    snr_db: float,
    bandwidth_hz: float,
    is_los: bool = True,
    min_variance: float = 1e-20,
    nlos_factor: float = 2.0
) -> float:
    """
    Simple interface for estimating ToA variance

    Args:
        snr_db: SNR in dB
        bandwidth_hz: Signal bandwidth in Hz
        is_los: Whether path is line-of-sight
        min_variance: Minimum variance floor
        nlos_factor: NLOS inflation factor

    Returns:
        ToA variance in seconds squared
    """
    snr_linear = 10**(snr_db / 10)
    beta_rms = bandwidth_hz / np.sqrt(3)  # Approximate for flat spectrum

    return cov_from_crlb(
        snr_linear, beta_rms,
        is_los=is_los,
        min_variance=min_variance,
        nlos_factor=nlos_factor
    )


def scale_variance_by_distance(
    base_variance: float,
    distance_m: float,
    reference_distance_m: float = 10.0,
    path_loss_exponent: float = 2.0
) -> float:
    """
    Scale variance based on distance (accounting for path loss)

    SNR degrades with distance, so variance increases.

    Args:
        base_variance: Variance at reference distance
        distance_m: Actual distance
        reference_distance_m: Reference distance
        path_loss_exponent: Path loss exponent (2 for free space)

    Returns:
        Scaled variance
    """
    if distance_m <= 0 or reference_distance_m <= 0:
        return base_variance

    # Path loss increases variance (lower SNR)
    path_loss_factor = (distance_m / reference_distance_m) ** path_loss_exponent

    return base_variance * path_loss_factor


def compute_tdoa_variance(
    toa_var_1: float,
    toa_var_2: float,
    correlation: float = 0.0
) -> float:
    """
    Compute TDOA variance from two ToA variances

    TDOA = ToA_2 - ToA_1
    Var(TDOA) = Var(ToA_1) + Var(ToA_2) - 2*Cov(ToA_1, ToA_2)

    Args:
        toa_var_1: Variance of first ToA
        toa_var_2: Variance of second ToA
        correlation: Correlation between ToA errors (usually 0)

    Returns:
        TDOA variance
    """
    return toa_var_1 + toa_var_2 - 2 * correlation * np.sqrt(toa_var_1 * toa_var_2)


def compute_twr_variance(
    toa_var_forward: float,
    toa_var_reverse: float
) -> float:
    """
    Compute TWR variance from forward and reverse ToA variances

    TWR = (ToA_forward + ToA_reverse) / 2
    Var(TWR) = (Var(ToA_f) + Var(ToA_r)) / 4

    Args:
        toa_var_forward: Forward path ToA variance
        toa_var_reverse: Reverse path ToA variance

    Returns:
        TWR variance
    """
    return (toa_var_forward + toa_var_reverse) / 4.0


@dataclass
class EdgeWeight:
    """
    Edge weight information for factor graph
    """
    variance: float
    weight: float  # 1/variance
    confidence: float  # 0-1 confidence in measurement
    measurement_type: str  # 'ToA', 'TDOA', 'TWR', etc.

    @classmethod
    def from_covariance(
        cls,
        covariance: MeasurementCovariance,
        measurement_type: str = 'ToA'
    ):
        """Create edge weight from measurement covariance"""
        variance = covariance.toa_variance
        weight = 1.0 / variance if variance > 0 else 1e20

        # Confidence based on LOS probability and SNR
        snr_factor = min(1.0, covariance.snr_linear / 100)  # Normalized by 20dB
        los_factor = 1.0 if covariance.is_los else 0.5
        confidence = snr_factor * los_factor

        return cls(
            variance=variance,
            weight=weight,
            confidence=confidence,
            measurement_type=measurement_type
        )


if __name__ == "__main__":
    # Test covariance estimation
    print("Testing Measurement Covariance...")
    print("=" * 50)

    # Test simple variance estimation
    print("\nSimple ToA Variance Estimation:")
    for snr_db in [10, 20, 30]:
        for is_los in [True, False]:
            var = estimate_toa_variance(snr_db, 500e6, is_los)
            std_cm = np.sqrt(var) * 3e8 * 100
            los_str = "LOS" if is_los else "NLOS"
            print(f"  SNR={snr_db}dB, {los_str}: σ={std_cm:.2f} cm")

    # Test distance scaling
    print("\nDistance Scaling:")
    base_var = estimate_toa_variance(20, 500e6, True)
    for dist in [1, 10, 100]:
        scaled_var = scale_variance_by_distance(base_var, dist, 10.0)
        std_cm = np.sqrt(scaled_var) * 3e8 * 100
        print(f"  {dist}m: σ={std_cm:.2f} cm")

    # Test TDOA/TWR variance
    print("\nTDOA and TWR Variance:")
    toa_var = 1e-18  # 1 attosecond squared
    tdoa_var = compute_tdoa_variance(toa_var, toa_var)
    twr_var = compute_twr_variance(toa_var, toa_var)

    print(f"  ToA variance: {toa_var:.2e} s²")
    print(f"  TDOA variance: {tdoa_var:.2e} s² (√2 larger)")
    print(f"  TWR variance: {twr_var:.2e} s² (2x smaller)")

    print("\nTest completed successfully!")