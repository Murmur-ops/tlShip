"""
Saleh-Valenzuela Channel Model
Wideband multipath channel with cluster structure for UWB
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from scipy import signal as scipy_signal


@dataclass
class ChannelConfig:
    """Configuration for Saleh-Valenzuela channel model"""

    # Carrier and bandwidth
    carrier_freq_ghz: float = 6.5  # UWB center frequency
    bandwidth_mhz: float = 499.2    # Channel bandwidth

    # Environment preset
    environment: str = 'indoor_office'

    # Path loss parameters
    path_loss_exponent: float = field(default=None)
    reference_distance_m: float = 1.0
    shadowing_std_db: float = field(default=None)

    # S-V cluster parameters
    cluster_arrival_rate: float = field(default=None)  # 1/ns (Lambda)
    ray_arrival_rate: float = field(default=None)      # 1/ns (lambda)
    cluster_decay_factor: float = field(default=None)  # ns (Gamma)
    ray_decay_factor: float = field(default=None)      # ns (gamma)

    # Rician K-factor for LOS
    k_factor_db: float = field(default=None)

    # NLOS excess delay
    nlos_excess_delay_mean_ns: float = field(default=None)
    nlos_excess_delay_std_ns: float = field(default=None)

    # Number of clusters and rays
    n_clusters: int = 4
    n_rays_per_cluster: int = 10

    def __post_init__(self):
        """Set environment-specific defaults"""
        # Environment presets based on IEEE 802.15.4a channel models
        presets = {
            'indoor_office': {
                'path_loss_exponent': 1.8,
                'shadowing_std_db': 3.0,
                'cluster_arrival_rate': 0.0233,  # 1/ns
                'ray_arrival_rate': 0.4,         # 1/ns
                'cluster_decay_factor': 7.0,     # ns
                'ray_decay_factor': 4.0,          # ns
                'k_factor_db': 10.0,
                'nlos_excess_delay_mean_ns': 10.0,
                'nlos_excess_delay_std_ns': 5.0
            },
            'indoor_industrial': {
                'path_loss_exponent': 2.0,
                'shadowing_std_db': 6.0,
                'cluster_arrival_rate': 0.0667,
                'ray_arrival_rate': 0.5,
                'cluster_decay_factor': 14.0,
                'ray_decay_factor': 6.0,
                'k_factor_db': 6.0,
                'nlos_excess_delay_mean_ns': 25.0,
                'nlos_excess_delay_std_ns': 10.0
            },
            'urban_nlos': {
                'path_loss_exponent': 3.5,
                'shadowing_std_db': 8.0,
                'cluster_arrival_rate': 0.1,
                'ray_arrival_rate': 0.8,
                'cluster_decay_factor': 20.0,
                'ray_decay_factor': 8.0,
                'k_factor_db': 0.0,  # Pure NLOS
                'nlos_excess_delay_mean_ns': 50.0,
                'nlos_excess_delay_std_ns': 20.0
            },
            'outdoor_los': {
                'path_loss_exponent': 2.0,
                'shadowing_std_db': 4.0,
                'cluster_arrival_rate': 0.05,
                'ray_arrival_rate': 0.3,
                'cluster_decay_factor': 10.0,
                'ray_decay_factor': 5.0,
                'k_factor_db': 15.0,
                'nlos_excess_delay_mean_ns': 5.0,
                'nlos_excess_delay_std_ns': 2.0
            }
        }

        # Apply preset if available
        if self.environment in presets:
            preset = presets[self.environment]
            for key, value in preset.items():
                if getattr(self, key) is None:
                    setattr(self, key, value)

        # Set remaining defaults if not set
        if self.path_loss_exponent is None:
            self.path_loss_exponent = 2.0
        if self.shadowing_std_db is None:
            self.shadowing_std_db = 4.0
        if self.cluster_arrival_rate is None:
            self.cluster_arrival_rate = 0.05
        if self.ray_arrival_rate is None:
            self.ray_arrival_rate = 0.5
        if self.cluster_decay_factor is None:
            self.cluster_decay_factor = 10.0
        if self.ray_decay_factor is None:
            self.ray_decay_factor = 5.0
        if self.k_factor_db is None:
            self.k_factor_db = 6.0
        if self.nlos_excess_delay_mean_ns is None:
            self.nlos_excess_delay_mean_ns = 20.0
        if self.nlos_excess_delay_std_ns is None:
            self.nlos_excess_delay_std_ns = 10.0


class SalehValenzuelaChannel:
    """
    Saleh-Valenzuela clustered channel model for UWB

    Reference: IEEE 802.15.4a channel model
    """

    def __init__(self, config: ChannelConfig):
        self.config = config
        self.wavelength_m = 3e8 / (config.carrier_freq_ghz * 1e9)

    def generate_channel_realization(
        self,
        distance_m: float,
        is_los: bool = True,
        seed: Optional[int] = None
    ) -> Dict:
        """
        Generate a channel realization using S-V model

        Args:
            distance_m: TX-RX distance in meters
            is_los: Whether LOS path exists
            seed: Random seed for reproducibility

        Returns:
            Dictionary with channel impulse response
        """
        if seed is not None:
            np.random.seed(seed)

        # Initialize lists for taps
        delays = []
        gains = []

        # Generate cluster arrivals (Poisson process)
        cluster_delays = []
        t_cluster = 0

        for _ in range(self.config.n_clusters):
            # Exponential inter-arrival times
            if len(cluster_delays) > 0:
                dt = np.random.exponential(1.0 / self.config.cluster_arrival_rate)
                t_cluster += dt
            cluster_delays.append(t_cluster)

        # Generate rays within each cluster
        for i, t_c in enumerate(cluster_delays):
            t_ray = t_c

            for j in range(self.config.n_rays_per_cluster):
                # Ray arrival time
                if j > 0:
                    dt = np.random.exponential(1.0 / self.config.ray_arrival_rate)
                    t_ray += dt

                delays.append(t_ray)

                # Ray amplitude with dual exponential decay
                cluster_decay = np.exp(-t_c / self.config.cluster_decay_factor)
                ray_decay = np.exp(-(t_ray - t_c) / self.config.ray_decay_factor)

                # Random phase
                phase = np.random.uniform(0, 2 * np.pi)

                # Complex gain
                amplitude = cluster_decay * ray_decay
                gain = amplitude * np.exp(1j * phase)

                # Add Rayleigh fading
                rayleigh = np.sqrt(0.5) * (np.random.randn() + 1j * np.random.randn())
                gain = gain * rayleigh

                gains.append(gain)

        # Convert to arrays
        delays = np.array(delays)
        gains = np.array(gains)

        # Sort by delay
        idx = np.argsort(delays)
        delays = delays[idx]
        gains = gains[idx]

        # Apply LOS component if present
        k_factor = 0
        if is_los and self.config.k_factor_db > 0:
            k_factor = 10**(self.config.k_factor_db / 10)

            # Scale NLOS components
            nlos_power = 1.0 / (1 + k_factor)
            gains = gains * np.sqrt(nlos_power)

            # Add strong LOS component at first tap
            los_power = k_factor / (1 + k_factor)
            gains[0] = np.sqrt(los_power) + gains[0]

        # Add NLOS excess delay
        excess_delay = 0
        if not is_los:
            excess_delay = abs(np.random.normal(
                self.config.nlos_excess_delay_mean_ns,
                self.config.nlos_excess_delay_std_ns
            ))
            delays = delays + excess_delay

        # Normalize total power
        total_power = np.sum(np.abs(gains)**2)
        if total_power > 0:
            gains = gains / np.sqrt(total_power)

        return {
            'taps': gains,
            'delays_ns': delays,
            'tap_gains': gains,
            'k_factor': k_factor,
            'excess_delay_ns': excess_delay,
            'n_taps': len(gains),
            'rms_delay_spread_ns': self._compute_rms_delay_spread(delays, gains)
        }

    def _compute_rms_delay_spread(self, delays: np.ndarray, gains: np.ndarray) -> float:
        """Compute RMS delay spread"""
        powers = np.abs(gains)**2
        if np.sum(powers) == 0:
            return 0

        # Normalize powers
        powers = powers / np.sum(powers)

        # Mean delay
        mean_delay = np.sum(delays * powers)

        # RMS delay spread
        rms_spread = np.sqrt(np.sum((delays - mean_delay)**2 * powers))

        return rms_spread

    def apply_channel(
        self,
        signal: np.ndarray,
        channel: Dict,
        sample_rate: float
    ) -> np.ndarray:
        """
        Apply channel impulse response to signal

        Args:
            signal: Input signal
            channel: Channel realization from generate_channel_realization
            sample_rate: Sample rate in Hz

        Returns:
            Signal after multipath channel
        """
        output = np.zeros_like(signal)

        # Apply each tap
        for delay_ns, gain in zip(channel['delays_ns'], channel['tap_gains']):
            # Convert delay to samples
            delay_samples = int(delay_ns * 1e-9 * sample_rate)

            if delay_samples < len(signal):
                # Apply tap with delay and gain
                output[delay_samples:] += signal[:-delay_samples] * gain if delay_samples > 0 else signal * gain

        return output


def compute_path_loss(
    distance_m: float,
    frequency_ghz: float,
    model: str = 'logdistance',
    path_loss_exponent: float = 2.0,
    reference_distance_m: float = 1.0
) -> float:
    """
    Compute path loss in dB

    Args:
        distance_m: Distance in meters
        frequency_ghz: Frequency in GHz
        model: Path loss model ('freespace', 'logdistance', 'tworay')
        path_loss_exponent: Path loss exponent for log-distance model
        reference_distance_m: Reference distance

    Returns:
        Path loss in dB
    """
    if distance_m < reference_distance_m:
        distance_m = reference_distance_m

    wavelength_m = 3e8 / (frequency_ghz * 1e9)

    if model == 'freespace':
        # Friis free space path loss
        path_loss_db = 20 * np.log10(4 * np.pi * distance_m / wavelength_m)

    elif model == 'logdistance':
        # Log-distance model
        # PL(d) = PL(d0) + 10*n*log10(d/d0)
        pl_ref = 20 * np.log10(4 * np.pi * reference_distance_m / wavelength_m)
        path_loss_db = pl_ref + 10 * path_loss_exponent * np.log10(distance_m / reference_distance_m)

    elif model == 'tworay':
        # Two-ray ground reflection (for outdoor)
        # Simplified: assumes antenna heights
        h_tx = 10.0  # meters
        h_rx = 1.5   # meters
        d_break = 4 * h_tx * h_rx / wavelength_m

        if distance_m < d_break:
            # Use free space below breakpoint
            path_loss_db = 20 * np.log10(4 * np.pi * distance_m / wavelength_m)
        else:
            # Two-ray model
            path_loss_db = 40 * np.log10(distance_m) - 10 * np.log10(h_tx**2 * h_rx**2)

    else:
        raise ValueError(f"Unknown path loss model: {model}")

    return path_loss_db


def apply_sample_clock_offset(
    signal: np.ndarray,
    sco_ppm: float,
    sample_rate: float
) -> np.ndarray:
    """
    Apply sample clock offset to signal (models ADC sample rate error)

    SCO causes the receiver to sample at a slightly different rate than
    the transmitter, leading to accumulated timing error proportional to
    the signal duration.

    Args:
        signal: Input signal
        sco_ppm: Sample clock offset in parts per million
        sample_rate: Nominal sample rate in Hz

    Returns:
        Signal with SCO applied (resampled)
    """
    if abs(sco_ppm) < 0.01:  # Less than 0.01 ppm, negligible
        return signal

    # Calculate actual vs nominal sample rates
    actual_rate_ratio = 1 + sco_ppm / 1e6

    # Resample signal to simulate sampling at wrong rate
    # Use polyphase resampling for accuracy
    if actual_rate_ratio > 1:
        # Receiver sampling faster than transmitter
        # Signal appears stretched in time
        p = 10000  # Numerator
        q = int(p / actual_rate_ratio)  # Denominator
    else:
        # Receiver sampling slower
        # Signal appears compressed in time
        q = 10000  # Denominator
        p = int(q * actual_rate_ratio)  # Numerator

    # Ensure p and q are coprime for best resampling
    from math import gcd
    common = gcd(p, q)
    p = p // common
    q = q // common

    # Apply resampling
    resampled = scipy_signal.resample_poly(signal, p, q)

    # Trim or pad to original length
    if len(resampled) > len(signal):
        resampled = resampled[:len(signal)]
    elif len(resampled) < len(signal):
        # Pad with zeros
        padding = len(signal) - len(resampled)
        resampled = np.pad(resampled, (0, padding), mode='constant')

    return resampled


def add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add AWGN to achieve specified SNR

    Args:
        signal: Input signal
        snr_db: Desired SNR in dB

    Returns:
        Signal with noise added
    """
    # Calculate signal power
    signal_power = np.mean(np.abs(signal)**2)

    if signal_power == 0:
        return signal

    # Calculate noise power for desired SNR
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear

    # Generate complex AWGN
    noise_std = np.sqrt(noise_power / 2)  # Divide by 2 for complex noise
    noise = noise_std * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))

    return signal + noise


def propagate_signal(
    signal: np.ndarray,
    channel: Dict,
    sample_rate: float,
    snr_db: float = 20.0,
    cfo_hz: float = 0.0,
    clock_bias_s: float = 0.0,
    clock_drift_ppm: float = 0.0,
    sco_ppm: float = 0.0
) -> Dict:
    """
    Propagate signal through channel with all impairments

    Args:
        signal: Input signal
        channel: Channel realization
        sample_rate: Sample rate in Hz
        snr_db: SNR in dB
        cfo_hz: Carrier frequency offset in Hz
        clock_bias_s: Clock bias in seconds
        clock_drift_ppm: Clock drift in ppm
        sco_ppm: Sample clock offset in ppm

    Returns:
        Dictionary with propagated signal and metadata
    """
    # Apply multipath channel
    sv = SalehValenzuelaChannel(ChannelConfig())
    signal_multipath = sv.apply_channel(signal, channel, sample_rate)

    # Apply CFO
    if cfo_hz != 0:
        t = np.arange(len(signal_multipath)) / sample_rate
        cfo_phasor = np.exp(1j * 2 * np.pi * cfo_hz * t)
        signal_multipath = signal_multipath * cfo_phasor

    # Apply sample clock offset (SCO)
    if abs(sco_ppm) > 0.01:
        signal_multipath = apply_sample_clock_offset(signal_multipath, sco_ppm, sample_rate)

    # Add AWGN
    signal_noisy = add_awgn(signal_multipath, snr_db)

    # Calculate true ToA (includes bias and first path delay)
    true_toa = clock_bias_s
    if len(channel['delays_ns']) > 0:
        true_toa += channel['delays_ns'][0] * 1e-9

    return {
        'signal': signal_noisy,
        'true_toa': true_toa,
        'snr_actual': snr_db,
        'cfo_actual': cfo_hz,
        'sco_actual': sco_ppm,
        'multipath_profile': channel
    }


if __name__ == "__main__":
    # Test channel generation
    print("Testing Saleh-Valenzuela Channel Model...")
    print("=" * 50)

    config = ChannelConfig(environment='indoor_office')
    sv = SalehValenzuelaChannel(config)

    # Generate LOS channel
    print("\nLOS Channel (10m):")
    channel_los = sv.generate_channel_realization(distance_m=10.0, is_los=True)
    print(f"  Number of taps: {channel_los['n_taps']}")
    print(f"  K-factor: {channel_los['k_factor']:.1f}")
    print(f"  RMS delay spread: {channel_los['rms_delay_spread_ns']:.1f} ns")
    print(f"  First 5 tap delays: {channel_los['delays_ns'][:5]}")
    print(f"  First 5 tap powers: {np.abs(channel_los['tap_gains'][:5])**2}")

    # Generate NLOS channel
    print("\nNLOS Channel (20m):")
    channel_nlos = sv.generate_channel_realization(distance_m=20.0, is_los=False)
    print(f"  Number of taps: {channel_nlos['n_taps']}")
    print(f"  Excess delay: {channel_nlos['excess_delay_ns']:.1f} ns")
    print(f"  RMS delay spread: {channel_nlos['rms_delay_spread_ns']:.1f} ns")

    # Test path loss
    print("\nPath Loss (Free Space, 6.5 GHz):")
    for d in [1, 10, 100]:
        pl = compute_path_loss(d, 6.5, model='freespace')
        print(f"  {d}m: {pl:.1f} dB")

    print("\nTest completed successfully!")