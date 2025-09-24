"""
Robust Kernels for Outlier Handling
Huber, DCS (Dynamic Covariance Scaling), and switchable constraints
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class RobustConfig:
    """Configuration for robust optimization"""
    use_huber: bool = True
    huber_delta: float = 1.0
    use_dcs: bool = False
    dcs_phi: float = 1.0
    use_switchable: bool = False
    switch_threshold: float = 3.0


def huber_weight(residual: float, delta: float = 1.0) -> float:
    """
    Huber robust kernel weight function

    For |r| <= δ: weight = 1 (quadratic)
    For |r| > δ: weight = δ/|r| (linear)

    Args:
        residual: Residual value
        delta: Threshold between quadratic and linear regions

    Returns:
        Weight in [0, 1]
    """
    abs_r = abs(residual)

    if abs_r <= delta:
        return 1.0
    else:
        return delta / abs_r


def huber_cost(residual: float, delta: float = 1.0) -> float:
    """
    Huber robust cost function

    For |r| <= δ: cost = 0.5 * r²
    For |r| > δ: cost = δ * (|r| - 0.5 * δ)

    Args:
        residual: Residual value
        delta: Threshold

    Returns:
        Robust cost
    """
    abs_r = abs(residual)

    if abs_r <= delta:
        return 0.5 * residual**2
    else:
        return delta * (abs_r - 0.5 * delta)


def dcs_weight(residual: float, sigma: float, phi: float = 1.0) -> float:
    """
    Dynamic Covariance Scaling (DCS) weight

    Based on Agarwal et al., "Robust Map Optimization" ICRA 2013

    Weight = Φ / (Φ + r²/σ²)

    Args:
        residual: Residual value
        sigma: Standard deviation
        phi: Scaling parameter (1.0 = standard)

    Returns:
        Weight in [0, 1]
    """
    normalized_r2 = (residual / sigma)**2
    return phi / (phi + normalized_r2)


def dcs_scale_covariance(residual: float, sigma: float, phi: float = 1.0) -> float:
    """
    Compute scaled variance for DCS

    New variance = σ² * (Φ + r²/σ²) / Φ

    Args:
        residual: Residual value
        sigma: Original standard deviation
        phi: Scaling parameter

    Returns:
        Scaled variance
    """
    normalized_r2 = (residual / sigma)**2
    scale = (phi + normalized_r2) / phi
    return sigma**2 * scale


def switchable_weight(
    residual: float,
    sigma: float,
    switch_var: float,
    prior_weight: float = 100.0
) -> float:
    """
    Switchable constraints weight

    Uses a latent switching variable to turn constraints on/off

    Args:
        residual: Residual value
        sigma: Standard deviation
        switch_var: Switching variable (0 = off, 1 = on)
        prior_weight: Prior on switch being on

    Returns:
        Effective weight
    """
    # Sigmoid activation of switch variable
    switch_on = 1.0 / (1.0 + np.exp(-switch_var))

    # Base weight from residual magnitude
    base_weight = np.exp(-(residual / sigma)**2 / 2)

    # Combined weight
    return switch_on * base_weight


class RobustKernel:
    """
    Unified robust kernel interface
    """

    def __init__(self, config: RobustConfig):
        self.config = config

    def weight(self, residual: float, sigma: float) -> float:
        """
        Compute robust weight for a residual

        Args:
            residual: Residual value
            sigma: Standard deviation

        Returns:
            Weight to apply
        """
        weight = 1.0

        if self.config.use_huber:
            # Apply Huber weighting
            weight *= huber_weight(
                residual,
                delta=self.config.huber_delta * sigma
            )

        if self.config.use_dcs:
            # Apply DCS weighting
            weight *= dcs_weight(
                residual,
                sigma,
                phi=self.config.dcs_phi
            )

        return weight

    def cost(self, residual: float, sigma: float) -> float:
        """
        Compute robust cost

        Args:
            residual: Residual value
            sigma: Standard deviation

        Returns:
            Robust cost
        """
        if self.config.use_huber:
            return huber_cost(
                residual,
                delta=self.config.huber_delta * sigma
            )
        else:
            # Standard least squares
            return 0.5 * (residual / sigma)**2


def identify_outliers(
    residuals: np.ndarray,
    sigmas: np.ndarray,
    method: str = 'mad',
    threshold: float = 3.0
) -> np.ndarray:
    """
    Identify outlier measurements

    Args:
        residuals: Array of residuals
        sigmas: Array of standard deviations
        method: 'mad' (median absolute deviation) or 'zscore'
        threshold: Threshold for outlier detection

    Returns:
        Boolean array (True = outlier)
    """
    # Normalize residuals
    normalized = residuals / sigmas

    if method == 'mad':
        # Median Absolute Deviation
        median = np.median(normalized)
        mad = np.median(np.abs(normalized - median))
        # Scale MAD to estimate std (1.4826 for normal distribution)
        mad_std = 1.4826 * mad
        is_outlier = np.abs(normalized - median) > threshold * mad_std

    elif method == 'zscore':
        # Standard z-score
        is_outlier = np.abs(normalized) > threshold

    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

    return is_outlier


def compute_robust_statistics(
    residuals: np.ndarray,
    weights: np.ndarray
) -> dict:
    """
    Compute robust statistics for weighted residuals

    Args:
        residuals: Array of residuals
        weights: Array of weights

    Returns:
        Dictionary of statistics
    """
    # Weighted mean
    weighted_mean = np.sum(weights * residuals) / np.sum(weights)

    # Weighted variance
    weighted_var = np.sum(weights * (residuals - weighted_mean)**2) / np.sum(weights)
    weighted_std = np.sqrt(weighted_var)

    # Effective sample size
    eff_n = np.sum(weights)**2 / np.sum(weights**2)

    # Percentage of downweighted measurements
    downweight_pct = 100 * np.sum(weights < 0.9) / len(weights)

    return {
        'mean': weighted_mean,
        'std': weighted_std,
        'variance': weighted_var,
        'effective_n': eff_n,
        'downweight_pct': downweight_pct,
        'min_weight': np.min(weights),
        'max_weight': np.max(weights)
    }


if __name__ == "__main__":
    # Test robust kernels
    print("Testing Robust Kernels...")
    print("=" * 50)

    residuals = np.array([0.1, 0.2, 0.15, 5.0, 0.12, -4.0])  # Two outliers
    sigma = 0.2

    print("\nResiduals:", residuals)
    print(f"Sigma: {sigma}")

    # Test Huber weights
    print("\nHuber weights (δ=1σ):")
    for r in residuals:
        w = huber_weight(r, delta=sigma)
        print(f"  r={r:5.2f} → w={w:.3f}")

    # Test DCS weights
    print("\nDCS weights (Φ=1):")
    for r in residuals:
        w = dcs_weight(r, sigma, phi=1.0)
        print(f"  r={r:5.2f} → w={w:.3f}")

    # Test outlier detection
    is_outlier = identify_outliers(residuals, np.full_like(residuals, sigma))
    print("\nOutliers detected:", np.where(is_outlier)[0])