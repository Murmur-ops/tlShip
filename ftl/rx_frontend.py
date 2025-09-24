"""
Receiver Front-End Processing
ToA detection, CFO estimation, NLOS classification, and CRLB-based covariance
"""

import numpy as np
from typing import Dict, Optional, Tuple
from scipy import signal as scipy_signal
from scipy.optimize import curve_fit


def matched_filter(
    received_signal: np.ndarray,
    template: np.ndarray,
    mode: str = 'same'
) -> np.ndarray:
    """
    Perform matched filtering for ToA estimation

    Args:
        received_signal: Received signal
        template: Template waveform
        mode: Correlation mode ('same', 'valid', 'full')

    Returns:
        Correlation output
    """
    # Normalize template
    template_norm = template / np.sqrt(np.sum(np.abs(template)**2))

    # Matched filter is correlation with time-reversed conjugate
    correlation = scipy_signal.correlate(received_signal,
                                        np.conj(template_norm[::-1]),
                                        mode=mode)

    return correlation


def detect_toa(
    correlation: np.ndarray,
    sample_rate: float,
    mode: str = 'peak',
    threshold: float = 0.5,
    enable_subsample: bool = True
) -> Dict:
    """
    Detect Time of Arrival from correlation output

    Args:
        correlation: Matched filter output
        sample_rate: Sampling rate
        mode: Detection mode ('peak' or 'leading_edge')
        threshold: Detection threshold (relative to peak)
        enable_subsample: Enable sub-sample refinement

    Returns:
        Dictionary with ToA estimate and metrics
    """
    correlation_mag = np.abs(correlation)

    # Find peak
    peak_idx = np.argmax(correlation_mag)
    peak_value = correlation_mag[peak_idx]

    # Estimate noise floor (use early samples)
    noise_samples = correlation_mag[:int(len(correlation_mag)*0.1)]
    noise_floor = np.median(noise_samples)
    noise_std = np.std(noise_samples)

    # SNR estimation
    signal_power = peak_value**2
    noise_power = noise_std**2
    snr_linear = signal_power / noise_power if noise_power > 0 else float('inf')

    if mode == 'peak':
        # Peak detection
        toa_idx = peak_idx
    elif mode == 'leading_edge':
        # Find first crossing above threshold (for NLOS mitigation)
        above_threshold = np.where(correlation_mag > threshold * peak_value)[0]

        if len(above_threshold) > 0:
            toa_idx = above_threshold[0]
        else:
            toa_idx = peak_idx
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Sub-sample refinement
    if enable_subsample and toa_idx > 0 and toa_idx < len(correlation) - 1:
        # Parabolic interpolation
        y1 = correlation_mag[toa_idx - 1]
        y2 = correlation_mag[toa_idx]
        y3 = correlation_mag[toa_idx + 1]

        if y2 > y1 and y2 > y3:  # Valid peak
            # Parabolic fit: y = ax^2 + bx + c
            # Peak at x = -b/(2a)
            denom = 2*y2 - y1 - y3
            if abs(denom) > 1e-10:
                delta = 0.5 * (y3 - y1) / denom
                toa_idx_refined = toa_idx + delta
            else:
                toa_idx_refined = float(toa_idx)
        else:
            toa_idx_refined = float(toa_idx)
    else:
        toa_idx_refined = float(toa_idx)

    # Convert to time
    toa_seconds = toa_idx_refined / sample_rate

    return {
        'toa': toa_seconds,
        'toa_samples': toa_idx_refined,
        'peak_value': peak_value,
        'snr': snr_linear,
        'snr_db': 10*np.log10(snr_linear) if snr_linear > 0 else -np.inf,
        'noise_floor': noise_floor,
        'noise_std': noise_std
    }


def estimate_cfo(
    blocks: list,
    block_separation_s: float,
    method: str = 'ml'
) -> float:
    """
    Estimate CFO from repeated signal blocks

    Uses phase difference between repeated blocks to estimate frequency offset.
    Based on maximum likelihood estimation.

    Args:
        blocks: List of repeated signal blocks
        block_separation_s: Time separation between blocks
        method: Estimation method ('ml' for maximum likelihood)

    Returns:
        Estimated CFO in Hz
    """
    if len(blocks) < 2:
        return 0.0

    # Accumulate phase differences
    phase_diffs = []

    for i in range(len(blocks) - 1):
        # Correlate consecutive blocks
        block1 = blocks[i]
        block2 = blocks[i + 1]

        # Ensure same length
        min_len = min(len(block1), len(block2))
        block1 = block1[:min_len]
        block2 = block2[:min_len]

        # Compute correlation (ML estimator)
        correlation = np.sum(np.conj(block1) * block2)

        # Extract phase difference
        phase_diff = np.angle(correlation)
        phase_diffs.append(phase_diff)

    # Average phase differences
    avg_phase_diff = np.mean(phase_diffs)

    # Convert to frequency
    # Phase accumulated over block_separation_s: Δφ = 2π * CFO * T
    cfo_hz = avg_phase_diff / (2 * np.pi * block_separation_s)

    return cfo_hz


def toa_crlb(
    snr_linear: float,
    bandwidth_hz: float
) -> float:
    """
    Calculate Cramér-Rao Lower Bound for ToA estimation

    For a deterministic signal in AWGN:
    var(τ) ≥ 1 / (8π² * β² * SNR)

    where β² is the mean-square bandwidth

    Args:
        snr_linear: Linear SNR (not dB)
        bandwidth_hz: Signal bandwidth in Hz

    Returns:
        CRLB variance in seconds²
    """
    # For rectangular spectrum, RMS bandwidth ≈ BW / sqrt(12)
    # But for practical signals, use effective bandwidth
    beta_rms = bandwidth_hz / np.sqrt(3)

    # CRLB formula
    variance = 1.0 / (8 * np.pi**2 * beta_rms**2 * snr_linear)

    return variance


def cov_from_crlb(
    snr_linear: float,
    beta_rms_hz: float,
    is_los: bool = True,
    nlos_factor: float = 2.0,
    min_variance: float = 1e-12  # 1 ns² - prevents numerical issues
) -> float:
    """
    Compute edge covariance from CRLB and channel conditions

    Args:
        snr_linear: Linear SNR from matched filter
        beta_rms_hz: RMS bandwidth of actual signal
        is_los: Whether channel is LOS
        nlos_factor: Variance inflation factor for NLOS
        min_variance: Minimum variance floor

    Returns:
        Variance for factor graph edge
    """
    # Base CRLB
    crlb_var = toa_crlb(snr_linear, beta_rms_hz)

    # Apply NLOS inflation
    if not is_los:
        crlb_var *= nlos_factor

    # Apply minimum floor to prevent numerical issues
    variance = max(crlb_var, min_variance)

    return variance


def extract_correlation_features(
    correlation: np.ndarray,
    peak_idx: int,
    window: int = 100
) -> Dict:
    """
    Extract features from correlation function for LOS/NLOS classification

    Args:
        correlation: Correlation function
        peak_idx: Index of main peak
        window: Window size around peak

    Returns:
        Dictionary of features
    """
    correlation_mag = np.abs(correlation)

    # Window around peak
    start = max(0, peak_idx - window)
    end = min(len(correlation), peak_idx + window)
    windowed = correlation_mag[start:end]

    # Peak value
    peak_value = correlation_mag[peak_idx]

    # RMS width
    indices = np.arange(start, end)
    normalized = windowed / np.sum(windowed)
    mean_idx = np.sum(indices * normalized)
    variance = np.sum((indices - mean_idx)**2 * normalized)
    rms_width = np.sqrt(variance)

    # Peak to sidelobe ratio
    # Remove peak region
    sidelobe_region = correlation_mag.copy()
    sidelobe_region[max(0, peak_idx-5):min(len(correlation), peak_idx+5)] = 0
    max_sidelobe = np.max(sidelobe_region)
    peak_to_sidelobe = peak_value / max_sidelobe if max_sidelobe > 0 else 100

    # Multipath ratio (energy in tail vs peak)
    tail_start = min(peak_idx + 10, len(correlation_mag) - 1)
    tail_energy = np.sum(correlation_mag[tail_start:]**2)
    peak_energy = np.sum(windowed**2)
    multipath_ratio = tail_energy / peak_energy if peak_energy > 0 else 0

    # Excess delay (center of mass after peak)
    if tail_start < len(correlation_mag):
        tail = correlation_mag[tail_start:]
        if np.sum(tail) > 0:
            tail_indices = np.arange(len(tail))
            excess_delay = np.sum(tail_indices * tail) / np.sum(tail)
        else:
            excess_delay = 0
    else:
        excess_delay = 0

    # Additional features for enhanced NLOS detection
    # Leading edge width (rise time)
    peak_10_percent = 0.1 * peak_value
    peak_90_percent = 0.9 * peak_value
    rise_indices = np.where((correlation_mag[:peak_idx] > peak_10_percent) &
                           (correlation_mag[:peak_idx] < peak_90_percent))[0]
    lead_width = len(rise_indices) if len(rise_indices) > 0 else 1

    # Rise slope (sharpness)
    if len(rise_indices) > 1:
        rise_slope = (peak_90_percent - peak_10_percent) / len(rise_indices)
    else:
        rise_slope = peak_value  # Very sharp

    # Early-late energy ratio
    early_window = 20
    if peak_idx > early_window:
        early_energy = np.sum(correlation_mag[peak_idx-early_window:peak_idx]**2)
        late_energy = np.sum(correlation_mag[peak_idx:peak_idx+early_window]**2)
        early_late_ratio = early_energy / late_energy if late_energy > 0 else 10
    else:
        early_late_ratio = 1.0

    # Kurtosis (peakedness)
    if len(windowed) > 3:
        mean = np.mean(windowed)
        std = np.std(windowed)
        if std > 0:
            kurtosis = np.mean(((windowed - mean) / std)**4) - 3
        else:
            kurtosis = 0
    else:
        kurtosis = 0

    return {
        'rms_width': rms_width,
        'peak_to_sidelobe_ratio': peak_to_sidelobe,
        'multipath_ratio': multipath_ratio,
        'excess_delay': excess_delay,
        'lead_width': lead_width,
        'rise_slope': rise_slope,
        'early_late_ratio': early_late_ratio,
        'kurtosis': kurtosis
    }


def classify_propagation(
    correlation: np.ndarray,
    peak_threshold: float = 5.0
) -> Dict:
    """
    Classify propagation as LOS or NLOS based on correlation shape

    Args:
        correlation: Correlation function
        peak_threshold: Threshold for peak detection

    Returns:
        Classification result with confidence
    """
    correlation_mag = np.abs(correlation)

    # Find main peak
    peak_idx = np.argmax(correlation_mag)

    # Extract features
    features = extract_correlation_features(correlation, peak_idx)

    # Classification logic based on features
    score_los = 0
    score_nlos = 0

    # Sharp peak indicates LOS
    if features['rms_width'] < 3:  # Stricter threshold for LOS
        score_los += 2
    elif features['rms_width'] > 8:
        score_nlos += 2

    # High peak-to-sidelobe ratio indicates LOS
    if features['peak_to_sidelobe_ratio'] > 3:
        score_los += 1
    else:
        score_nlos += 1

    # Low multipath ratio indicates LOS
    if features['multipath_ratio'] < 0.2:  # Stricter threshold
        score_los += 2
    else:
        score_nlos += 2

    # Small excess delay indicates LOS
    if features['excess_delay'] < 5:  # Much stricter for LOS
        score_los += 1
    else:
        score_nlos += 1

    # New feature-based scoring
    if features['lead_width'] < 10:
        score_los += 1
    else:
        score_nlos += 1

    if features['kurtosis'] > 0:  # Peaked distribution
        score_los += 1
    else:
        score_nlos += 1

    # Determine classification
    total_score = score_los + score_nlos
    if total_score > 0:
        confidence_los = score_los / total_score
        confidence_nlos = score_nlos / total_score
    else:
        confidence_los = 0.5
        confidence_nlos = 0.5

    if score_los > score_nlos:
        prop_type = 'LOS'
        confidence = confidence_los
    else:
        prop_type = 'NLOS'
        confidence = confidence_nlos

    return {
        'type': prop_type,
        'confidence': confidence,
        'score_los': score_los,
        'score_nlos': score_nlos,
        'features': features
    }


def covariance_from_features(
    base_variance: float,
    features: Dict,
    max_inflation: float = 4.0
) -> float:
    """
    Scale variance based on correlation shape features

    Args:
        base_variance: CRLB-based variance
        features: Correlation shape features
        max_inflation: Maximum inflation factor

    Returns:
        Scaled variance
    """
    inflation_factor = 1.0

    # Fat leading edge indicates NLOS
    if features['lead_width'] > 15:
        inflation_factor *= 2.0
    elif features['lead_width'] > 10:
        inflation_factor *= 1.5

    # Low kurtosis (spread peak) indicates multipath
    if features['kurtosis'] < -0.5:
        inflation_factor *= 1.5
    elif features['kurtosis'] < 0:
        inflation_factor *= 1.2

    # High multipath ratio
    if features['multipath_ratio'] > 0.3:
        inflation_factor *= 1.5
    elif features['multipath_ratio'] > 0.2:
        inflation_factor *= 1.2

    # Low peak-to-sidelobe ratio
    if features['peak_to_sidelobe_ratio'] < 2:
        inflation_factor *= 1.3

    # Cap inflation
    inflation_factor = min(inflation_factor, max_inflation)

    return base_variance * inflation_factor


def estimate_sco(
    correlation: np.ndarray,
    nominal_peak_width: int,
    sample_rate: float
) -> float:
    """
    Estimate sample clock offset from correlation peak width

    SCO causes correlation peak broadening

    Args:
        correlation: Matched filter output
        nominal_peak_width: Expected peak width without SCO
        sample_rate: Sampling rate

    Returns:
        Estimated SCO in ppm
    """
    correlation_mag = np.abs(correlation)
    peak_idx = np.argmax(correlation_mag)
    peak_value = correlation_mag[peak_idx]

    # Find 3dB width
    threshold_3db = peak_value / np.sqrt(2)
    above_threshold = np.where(correlation_mag > threshold_3db)[0]

    if len(above_threshold) > 0:
        actual_width = above_threshold[-1] - above_threshold[0] + 1
    else:
        actual_width = 1

    # Width increase due to SCO
    width_ratio = actual_width / nominal_peak_width

    # Approximate SCO (simplified model)
    # Width scales approximately linearly with SCO for small errors
    sco_ppm = (width_ratio - 1) * 1e6

    return sco_ppm


if __name__ == "__main__":
    # Test receiver processing
    print("Testing Receiver Front-End...")
    print("=" * 50)

    # Create test signal
    from .signal import SignalConfig, gen_hrp_burst, compute_rms_bandwidth

    cfg = SignalConfig()
    template = gen_hrp_burst(cfg, n_repeats=1)

    # Add noise
    snr_db = 20
    snr_linear = 10**(snr_db/10)
    signal_power = np.mean(np.abs(template)**2)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(template)) +
                                      1j * np.random.randn(len(template)))
    received = template + noise

    # Test matched filter
    print("\nMatched Filter:")
    correlation = matched_filter(received, template)
    print(f"  Correlation length: {len(correlation)}")

    # Test ToA detection
    print("\nToA Detection:")
    toa_result = detect_toa(correlation, cfg.sample_rate, enable_subsample=True)
    print(f"  ToA: {toa_result['toa']*1e9:.2f} ns")
    print(f"  SNR: {toa_result['snr_db']:.1f} dB")
    print(f"  Peak value: {toa_result['peak_value']:.3f}")

    # Test CRLB calculation
    print("\nCRLB Calculation:")
    beta_rms = compute_rms_bandwidth(template, cfg.sample_rate)
    crlb_var = toa_crlb(snr_linear, beta_rms)
    crlb_std = np.sqrt(crlb_var)
    print(f"  RMS bandwidth: {beta_rms/1e6:.1f} MHz")
    print(f"  CRLB σ(ToA): {crlb_std*1e12:.1f} ps")
    print(f"  CRLB σ(range): {crlb_std*3e8*100:.2f} cm")

    # Test covariance from CRLB
    print("\nCovariance from CRLB:")
    cov_los = cov_from_crlb(snr_linear, beta_rms, is_los=True)
    cov_nlos = cov_from_crlb(snr_linear, beta_rms, is_los=False)
    print(f"  LOS variance: {cov_los:.3e} s²")
    print(f"  NLOS variance: {cov_nlos:.3e} s²")
    print(f"  NLOS/LOS ratio: {cov_nlos/cov_los:.1f}")

    # Test feature extraction
    print("\nCorrelation Features:")
    features = extract_correlation_features(correlation, np.argmax(np.abs(correlation)))
    print(f"  RMS width: {features['rms_width']:.1f}")
    print(f"  Peak/sidelobe: {features['peak_to_sidelobe_ratio']:.1f}")
    print(f"  Lead width: {features['lead_width']}")
    print(f"  Kurtosis: {features['kurtosis']:.2f}")

    # Test classification
    print("\nLOS/NLOS Classification:")
    classification = classify_propagation(correlation)
    print(f"  Type: {classification['type']}")
    print(f"  Confidence: {classification['confidence']*100:.1f}%")