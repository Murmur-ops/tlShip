"""
Signal Generation Module
IEEE 802.15.4z HRP-UWB and Zadoff-Chu waveform generation with RMS bandwidth calculation
"""

import numpy as np
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from scipy import signal as scipy_signal


@dataclass
class SignalConfig:
    """Configuration for signal generation"""
    carrier_freq: float = 6.5e9  # Hz
    bandwidth: float = 499.2e6  # Hz
    sample_rate: float = 1e9  # Hz
    burst_duration: float = 1e-6  # seconds
    prf: float = 124.8e6  # Pulse repetition frequency (Hz)
    sequence_length: int = 127
    cyclic_prefix_length: int = 32
    pilot_power_db: float = -10  # Pilot power relative to signal


def gen_ternary_sequence(length: int, density: float = 0.5, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate ternary preamble sequence {-1, 0, +1}

    Args:
        length: Sequence length
        density: Fraction of non-zero elements
        seed: Random seed

    Returns:
        Ternary sequence array
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate sparse ternary
    seq = np.zeros(length)
    n_nonzero = int(length * density)
    nonzero_idx = np.random.choice(length, n_nonzero, replace=False)
    seq[nonzero_idx] = np.random.choice([-1, 1], n_nonzero)

    return seq


def gen_hrp_burst(
    cfg: SignalConfig,
    n_repeats: Optional[int] = None
) -> np.ndarray:
    """
    Generate IEEE 802.15.4z HRP-UWB burst

    High Rate Pulse UWB with ternary preamble

    Args:
        cfg: Signal configuration
        n_repeats: Number of preamble repetitions (auto if None)

    Returns:
        Complex baseband signal
    """
    # Calculate samples
    n_samples = int(cfg.burst_duration * cfg.sample_rate)

    # PRF period in samples
    prf_period_samples = int(cfg.sample_rate / cfg.prf)

    if n_repeats is None:
        n_repeats = n_samples // (cfg.sequence_length * prf_period_samples)
        n_repeats = max(1, n_repeats)

    # Generate ternary preamble
    preamble = gen_ternary_sequence(cfg.sequence_length, density=0.5)

    # Create pulse train
    signal_out = np.zeros(n_samples, dtype=complex)

    for rep in range(n_repeats):
        for i, symbol in enumerate(preamble):
            if symbol != 0:
                idx = rep * cfg.sequence_length * prf_period_samples + i * prf_period_samples
                if idx < n_samples:
                    # Generate UWB pulse (Gaussian derivative)
                    pulse = gen_uwb_pulse(cfg.sample_rate, cfg.bandwidth)
                    pulse_len = len(pulse)

                    # Place pulse
                    end_idx = min(idx + pulse_len, n_samples)
                    signal_out[idx:end_idx] = symbol * pulse[:end_idx-idx]

    return signal_out


def gen_uwb_pulse(sample_rate: float, bandwidth: float) -> np.ndarray:
    """
    Generate UWB pulse (Gaussian monocycle)

    Args:
        sample_rate: Sampling frequency
        bandwidth: Pulse bandwidth

    Returns:
        UWB pulse waveform
    """
    # Pulse duration inversely proportional to bandwidth
    sigma = 1.0 / (2 * np.pi * bandwidth / 4)
    pulse_duration = 10 * sigma

    t = np.arange(-pulse_duration/2, pulse_duration/2, 1/sample_rate)

    # Gaussian monocycle (first derivative of Gaussian)
    pulse = -t / sigma**2 * np.exp(-t**2 / (2*sigma**2))

    # Normalize
    pulse = pulse / np.max(np.abs(pulse))

    return pulse


def gen_zc_burst(
    cfg: SignalConfig,
    n_repeats: Optional[int] = None
) -> np.ndarray:
    """
    Generate Zadoff-Chu CAZAC sequence burst

    Constant Amplitude Zero Autocorrelation Cyclic sequence

    Args:
        cfg: Signal configuration
        n_repeats: Number of sequence repetitions

    Returns:
        Complex baseband signal
    """
    # Calculate samples
    n_samples = int(cfg.burst_duration * cfg.sample_rate)

    # Generate ZC sequence
    N = cfg.sequence_length
    u = find_coprime(N)  # Root index

    n = np.arange(N)
    if N % 2 == 0:
        zc_seq = np.exp(-1j * np.pi * u * n * (n + 1) / N)
    else:
        zc_seq = np.exp(-1j * np.pi * u * n * n / N)

    # Add cyclic prefix
    cp = zc_seq[-cfg.cyclic_prefix_length:]
    zc_with_cp = np.concatenate([cp, zc_seq])

    # Determine number of repeats
    seq_len = len(zc_with_cp)
    if n_repeats is None:
        n_repeats = n_samples // seq_len
        n_repeats = max(1, n_repeats)

    # Create burst
    signal_out = np.zeros(n_samples, dtype=complex)

    for rep in range(n_repeats):
        start_idx = rep * seq_len
        end_idx = min(start_idx + seq_len, n_samples)
        if start_idx < n_samples:
            signal_out[start_idx:end_idx] = zc_with_cp[:end_idx-start_idx]

    return signal_out


def find_coprime(N: int) -> int:
    """
    Find coprime root index for Zadoff-Chu sequence

    Args:
        N: Sequence length

    Returns:
        Coprime root index
    """
    # Common good choices
    if N == 127:
        return 25
    elif N == 139:
        return 50

    # Find first coprime
    for u in range(1, N):
        if np.gcd(u, N) == 1:
            return u

    return 1


def add_pilot_tones(
    signal: np.ndarray,
    sample_rate: float,
    pilot_freqs: list,
    pilot_power_db: float = -10
) -> np.ndarray:
    """
    Add pilot tones for CFO estimation

    Args:
        signal: Input signal
        sample_rate: Sampling rate
        pilot_freqs: List of pilot frequencies
        pilot_power_db: Pilot power relative to signal (dB)

    Returns:
        Signal with pilots added
    """
    signal_out = signal.copy()
    t = np.arange(len(signal)) / sample_rate

    # Calculate pilot amplitude
    signal_power = np.mean(np.abs(signal)**2)
    if signal_power == 0:
        pilot_power_linear = 10**(pilot_power_db / 10)
    else:
        pilot_power_linear = 10**(pilot_power_db / 10) * signal_power
    pilot_amplitude = np.sqrt(pilot_power_linear)

    # Add pilots
    for freq in pilot_freqs:
        pilot = pilot_amplitude * np.exp(1j * 2 * np.pi * freq * t)
        signal_out += pilot

    return signal_out


def gen_rrc_pulse(
    span: int,
    sps: int,
    beta: float
) -> np.ndarray:
    """
    Generate root-raised cosine pulse

    Args:
        span: Filter span in symbols
        sps: Samples per symbol
        beta: Roll-off factor (0 to 1)

    Returns:
        RRC pulse shape
    """
    N = span * sps
    t = np.arange(-N/2, N/2 + 1) / sps

    # Handle special cases
    pulse = np.zeros_like(t)

    # t = 0
    idx_zero = np.where(t == 0)[0]
    if len(idx_zero) > 0:
        pulse[idx_zero] = (1 + beta*(4/np.pi - 1))

    # t = ±1/(4β)
    idx_special = np.where(np.abs(t) == 1/(4*beta))[0]
    if len(idx_special) > 0:
        pulse[idx_special] = (beta/np.sqrt(2)) * ((1 + 2/np.pi) * np.sin(np.pi/(4*beta)) +
                                                  (1 - 2/np.pi) * np.cos(np.pi/(4*beta)))

    # General case
    idx_general = np.where((t != 0) & (np.abs(t) != 1/(4*beta)))[0]
    t_g = t[idx_general]
    pulse[idx_general] = (np.sin(np.pi*t_g*(1-beta)) + 4*beta*t_g*np.cos(np.pi*t_g*(1+beta))) / \
                        (np.pi*t_g*(1 - (4*beta*t_g)**2))

    # Normalize
    pulse = pulse / np.sqrt(np.sum(pulse**2))

    return pulse


def apply_lowpass_filter(
    signal: np.ndarray,
    sample_rate: float,
    cutoff_freq: float,
    order: int = 5
) -> np.ndarray:
    """
    Apply lowpass filter to signal

    Args:
        signal: Input signal
        sample_rate: Sampling rate
        cutoff_freq: Cutoff frequency
        order: Filter order

    Returns:
        Filtered signal
    """
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_freq / nyquist

    b, a = scipy_signal.butter(order, normalized_cutoff, btype='low')
    filtered = scipy_signal.filtfilt(b, a, signal)

    return filtered


def compute_rms_bandwidth(
    signal: np.ndarray,
    sample_rate: float
) -> float:
    """
    Compute RMS bandwidth of signal for CRLB calculation

    β_rms = sqrt(∫f²|S(f)|²df / ∫|S(f)|²df)

    Args:
        signal: Time-domain signal
        sample_rate: Sampling rate in Hz

    Returns:
        RMS bandwidth in Hz
    """
    # Compute PSD
    freqs, psd = scipy_signal.periodogram(signal, sample_rate, scaling='density')

    # Remove DC and negative frequencies
    positive_idx = freqs > 0
    freqs = freqs[positive_idx]
    psd = psd[positive_idx]

    # Numerical integration
    df = freqs[1] - freqs[0]

    # Denominator: ∫|S(f)|²df (total power)
    total_power = np.sum(psd) * df

    if total_power == 0:
        # Fallback to nominal bandwidth / sqrt(3) for flat spectrum
        return sample_rate / (2 * np.sqrt(3))

    # Numerator: ∫f²|S(f)|²df
    f2_weighted_power = np.sum(freqs**2 * psd) * df

    # RMS bandwidth
    beta_rms = np.sqrt(f2_weighted_power / total_power)

    return beta_rms


def compute_signal_snr(
    signal: np.ndarray,
    noise: np.ndarray
) -> float:
    """
    Compute SNR from signal and noise samples

    Args:
        signal: Signal samples
        noise: Noise samples

    Returns:
        Linear SNR (not dB)
    """
    signal_power = np.mean(np.abs(signal)**2)
    noise_power = np.mean(np.abs(noise)**2)

    if noise_power == 0:
        return float('inf')

    return signal_power / noise_power


if __name__ == "__main__":
    # Test signal generation
    print("Testing Signal Generation...")
    print("=" * 50)

    cfg = SignalConfig()

    # Test HRP-UWB
    print("\nHRP-UWB Burst:")
    hrp = gen_hrp_burst(cfg, n_repeats=3)
    print(f"  Length: {len(hrp)} samples")
    print(f"  Duration: {len(hrp)/cfg.sample_rate*1e6:.1f} μs")
    print(f"  Peak amplitude: {np.max(np.abs(hrp)):.3f}")

    # Compute RMS bandwidth
    beta_rms = compute_rms_bandwidth(hrp, cfg.sample_rate)
    print(f"  RMS bandwidth: {beta_rms/1e6:.1f} MHz")
    print(f"  Nominal BW: {cfg.bandwidth/1e6:.1f} MHz")
    print(f"  Ratio: {beta_rms/cfg.bandwidth:.2f}")

    # Test Zadoff-Chu
    print("\nZadoff-Chu Burst:")
    zc = gen_zc_burst(cfg, n_repeats=3)
    print(f"  Length: {len(zc)} samples")
    print(f"  Duration: {len(zc)/cfg.sample_rate*1e6:.1f} μs")
    print(f"  Peak amplitude: {np.max(np.abs(zc)):.3f}")

    # Check CAZAC property
    autocorr = np.abs(np.correlate(zc[:cfg.sequence_length],
                                   zc[:cfg.sequence_length], 'full'))
    peak_idx = cfg.sequence_length - 1
    peak_val = autocorr[peak_idx]
    sidelobe_max = np.max(np.concatenate([autocorr[:peak_idx-1],
                                         autocorr[peak_idx+2:]]))
    print(f"  CAZAC ratio: {peak_val/sidelobe_max:.1f}")

    # Test pilot tones
    print("\nPilot Tone Addition:")
    signal_with_pilots = add_pilot_tones(hrp, cfg.sample_rate,
                                        [1e6, -1e6], -10)
    print(f"  Original power: {np.mean(np.abs(hrp)**2):.3e}")
    print(f"  With pilots power: {np.mean(np.abs(signal_with_pilots)**2):.3e}")