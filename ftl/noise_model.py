"""
Comprehensive Noise Model for Time-Localization System
Provides configurable, realistic noise sources for UWB ranging measurements
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional, List
from enum import Enum


class NoisePreset(Enum):
    """Predefined noise configurations"""
    IDEAL = "ideal"          # No noise (original behavior)
    CLEAN = "clean"          # Minimal noise, ideal lab conditions
    REALISTIC = "realistic"  # Typical UWB deployment
    HARSH = "harsh"          # Challenging environment


@dataclass
class ThermalNoiseConfig:
    """Thermal/AWGN noise configuration"""
    enabled: bool = True
    snr_db: float = 20.0        # Signal-to-noise ratio in dB
    bandwidth_mhz: float = 500.0  # System bandwidth
    noise_figure_db: float = 6.0 # Receiver noise figure


@dataclass
class DistanceNoiseConfig:
    """Distance-dependent noise configuration"""
    enabled: bool = True
    coefficient: float = 0.001   # std = coefficient * sqrt(distance)
    min_std_m: float = 0.001    # Minimum standard deviation (1mm)


@dataclass
class QuantizationNoiseConfig:
    """ADC quantization noise"""
    enabled: bool = True
    adc_bits: int = 12          # ADC resolution
    full_scale_range_m: float = 100.0  # ADC full scale range


@dataclass
class ClockNoiseConfig:
    """Clock jitter and timing noise"""
    enabled: bool = True
    allan_deviation_ps: float = 100.0  # Allan deviation in picoseconds
    integration_time_s: float = 1.0    # Integration time


@dataclass
class MultipathConfig:
    """Multipath/NLOS configuration"""
    enabled: bool = True
    nlos_probability: float = 0.1      # Probability of NLOS
    bias_range_m: Tuple[float, float] = (0.0, 0.5)  # Positive bias range
    excess_std_factor: float = 2.0     # Additional std in NLOS


@dataclass
class ShadowingConfig:
    """Large-scale fading/shadowing"""
    enabled: bool = True
    std_db: float = 3.0         # Log-normal shadowing standard deviation
    correlation_distance_m: float = 10.0  # Decorrelation distance


@dataclass
class SmallScaleFadingConfig:
    """Small-scale fading (Rayleigh/Rician)"""
    enabled: bool = False       # Disabled by default (for static scenarios)
    k_factor_db: float = 6.0    # Rician K-factor (LOS power / scattered power)


@dataclass
class FrequencyOffsetConfig:
    """Crystal frequency offset"""
    enabled: bool = True
    max_offset_ppb: float = 10.0  # Parts per billion
    temperature_coefficient_ppb_per_c: float = 0.5


@dataclass
class PhaseNoiseConfig:
    """Oscillator phase noise"""
    enabled: bool = True
    phase_noise_dbc_per_hz: float = -80.0  # At 1kHz offset
    corner_frequency_hz: float = 1000.0


@dataclass
class AntennaDelayConfig:
    """Antenna delay calibration errors"""
    enabled: bool = True
    calibration_error_std_ps: float = 50.0  # Standard deviation in ps
    temperature_coefficient_ps_per_c: float = 2.0


@dataclass
class TemperatureDriftConfig:
    """Temperature-induced drift"""
    enabled: bool = False       # Disabled by default
    temperature_range_c: Tuple[float, float] = (-10.0, 50.0)
    drift_rate_c_per_hour: float = 2.0


@dataclass
class NodeMotionConfig:
    """Node motion effects"""
    enabled: bool = False       # Disabled for static scenarios
    velocity_std_mps: float = 0.1  # Standard deviation of velocity
    max_velocity_mps: float = 1.0   # Maximum velocity


@dataclass
class DopplerConfig:
    """Doppler frequency shift"""
    enabled: bool = False       # Disabled for static scenarios
    max_velocity_mps: float = 1.0
    carrier_freq_ghz: float = 6.5  # UWB center frequency


@dataclass
class InterferenceConfig:
    """RF interference"""
    enabled: bool = False       # Disabled by default
    sinr_db: float = 15.0      # Signal-to-interference-plus-noise ratio
    burst_probability: float = 0.01  # Probability of interference burst


@dataclass
class NoiseConfig:
    """Master noise configuration"""
    # Master control
    enable_noise: bool = True
    preset: Optional[NoisePreset] = NoisePreset.REALISTIC
    random_seed: Optional[int] = None  # For reproducibility

    # Measurement noise sources
    thermal: ThermalNoiseConfig = field(default_factory=ThermalNoiseConfig)
    distance_dependent: DistanceNoiseConfig = field(default_factory=DistanceNoiseConfig)
    quantization: QuantizationNoiseConfig = field(default_factory=QuantizationNoiseConfig)
    clock_jitter: ClockNoiseConfig = field(default_factory=ClockNoiseConfig)

    # Channel effects
    multipath: MultipathConfig = field(default_factory=MultipathConfig)
    shadowing: ShadowingConfig = field(default_factory=ShadowingConfig)
    small_scale_fading: SmallScaleFadingConfig = field(default_factory=SmallScaleFadingConfig)

    # Hardware imperfections
    frequency_offset: FrequencyOffsetConfig = field(default_factory=FrequencyOffsetConfig)
    phase_noise: PhaseNoiseConfig = field(default_factory=PhaseNoiseConfig)
    antenna_delay: AntennaDelayConfig = field(default_factory=AntennaDelayConfig)
    temperature_drift: TemperatureDriftConfig = field(default_factory=TemperatureDriftConfig)

    # Environmental factors
    node_motion: NodeMotionConfig = field(default_factory=NodeMotionConfig)
    doppler: DopplerConfig = field(default_factory=DopplerConfig)
    interference: InterferenceConfig = field(default_factory=InterferenceConfig)

    def __post_init__(self):
        """Apply preset configurations"""
        if self.preset:
            self._apply_preset(self.preset)

    def _apply_preset(self, preset: NoisePreset):
        """Apply predefined noise configuration"""
        if preset == NoisePreset.IDEAL:
            # Disable all noise
            self.enable_noise = False
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if hasattr(attr, 'enabled'):
                    attr.enabled = False

        elif preset == NoisePreset.CLEAN:
            # Minimal noise - lab conditions
            self.thermal.snr_db = 30.0
            self.thermal.noise_figure_db = 3.0
            self.distance_dependent.coefficient = 0.0005
            self.quantization.adc_bits = 14
            self.clock_jitter.allan_deviation_ps = 10.0
            self.multipath.enabled = False
            self.shadowing.std_db = 1.0
            self.frequency_offset.max_offset_ppb = 2.0
            self.phase_noise.phase_noise_dbc_per_hz = -90.0
            self.antenna_delay.calibration_error_std_ps = 10.0

        elif preset == NoisePreset.REALISTIC:
            # Typical UWB deployment
            self.thermal.snr_db = 20.0
            self.thermal.noise_figure_db = 6.0
            self.distance_dependent.coefficient = 0.001
            self.quantization.adc_bits = 12
            self.clock_jitter.allan_deviation_ps = 100.0
            self.multipath.nlos_probability = 0.1
            self.shadowing.std_db = 3.0
            self.frequency_offset.max_offset_ppb = 10.0
            self.phase_noise.phase_noise_dbc_per_hz = -80.0
            self.antenna_delay.calibration_error_std_ps = 50.0

        elif preset == NoisePreset.HARSH:
            # Challenging environment
            self.thermal.snr_db = 10.0
            self.thermal.noise_figure_db = 10.0
            self.distance_dependent.coefficient = 0.005
            self.quantization.adc_bits = 10
            self.clock_jitter.allan_deviation_ps = 500.0
            self.multipath.nlos_probability = 0.3
            self.multipath.bias_range_m = (0.0, 2.0)
            self.multipath.excess_std_factor = 5.0
            self.shadowing.std_db = 8.0
            self.small_scale_fading.enabled = True
            self.frequency_offset.max_offset_ppb = 50.0
            self.phase_noise.phase_noise_dbc_per_hz = -60.0
            self.antenna_delay.calibration_error_std_ps = 200.0
            self.node_motion.enabled = True
            self.interference.enabled = True


class NoiseGenerator:
    """Generate realistic noise for ranging measurements"""

    def __init__(self, config: NoiseConfig):
        """Initialize noise generator

        Args:
            config: Noise configuration
        """
        self.config = config
        self.rng = np.random.RandomState(config.random_seed)
        self.c = 299792458.0  # Speed of light in m/s

        # Cache for correlated noise
        self._shadowing_cache: Dict[Tuple[int, int], float] = {}
        self._frequency_offset_cache: Dict[int, float] = {}
        self._antenna_delay_cache: Dict[int, float] = {}
        self._temperature_cache: Dict[int, float] = {}

        # Initialize per-node parameters
        self._initialize_node_parameters()

    def _initialize_node_parameters(self):
        """Initialize random parameters for each node"""
        if self.config.frequency_offset.enabled:
            # Each node has a random frequency offset
            for i in range(100):  # Support up to 100 nodes
                offset_ppb = self.rng.uniform(
                    -self.config.frequency_offset.max_offset_ppb,
                    self.config.frequency_offset.max_offset_ppb
                )
                self._frequency_offset_cache[i] = offset_ppb * 1e-9

        if self.config.antenna_delay.enabled:
            # Each node has a calibration error
            for i in range(100):
                error_ps = self.rng.normal(0, self.config.antenna_delay.calibration_error_std_ps)
                self._antenna_delay_cache[i] = error_ps * 1e-12  # Convert to seconds

        if self.config.temperature_drift.enabled:
            # Each node has a current temperature
            for i in range(100):
                temp = self.rng.uniform(*self.config.temperature_drift.temperature_range_c)
                self._temperature_cache[i] = temp

    def add_measurement_noise(self,
                            true_range: float,
                            node_i: int,
                            node_j: int,
                            timestamp: float = 0.0,
                            position_i: Optional[np.ndarray] = None,
                            position_j: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """Apply all enabled noise sources to a ranging measurement

        Args:
            true_range: True distance between nodes (m)
            node_i: First node index
            node_j: Second node index
            timestamp: Measurement timestamp (s)
            position_i: Position of node i (optional, for motion)
            position_j: Position of node j (optional, for motion)

        Returns:
            Tuple of (noisy_range, measurement_std)
        """
        if not self.config.enable_noise:
            return true_range, self.config.distance_dependent.min_std_m

        noisy_range = true_range
        total_variance = 0.0

        # 1. Thermal noise
        if self.config.thermal.enabled:
            var = self._compute_thermal_variance(true_range)
            total_variance += var

        # 2. Distance-dependent noise
        if self.config.distance_dependent.enabled:
            var = self._compute_distance_variance(true_range)
            total_variance += var

        # 3. Quantization noise
        if self.config.quantization.enabled:
            var = self._compute_quantization_variance()
            total_variance += var

        # 4. Clock jitter
        if self.config.clock_jitter.enabled:
            var = self._compute_clock_jitter_variance()
            total_variance += var

        # 5. Multipath/NLOS
        if self.config.multipath.enabled:
            bias, var = self._apply_multipath(true_range, node_i, node_j)
            noisy_range += bias
            total_variance += var

        # 6. Shadowing (large-scale fading)
        if self.config.shadowing.enabled:
            # Shadowing adds uncertainty but shouldn't drastically multiply range
            # Convert dB std to linear range uncertainty
            shadowing_db = self._get_shadowing_db(node_i, node_j)
            # For ranging, shadowing primarily adds noise, not multiplicative scaling
            # Convert dB variation to range uncertainty (approximation)
            shadowing_std = true_range * self.config.shadowing.std_db * 0.01  # 1% per dB
            shadowing_noise = self.rng.normal(0, shadowing_std)
            noisy_range += shadowing_noise
            total_variance += shadowing_std**2

        # 7. Small-scale fading
        if self.config.small_scale_fading.enabled:
            fading_factor = self._compute_small_scale_fading()
            # Small-scale fading adds uncertainty around the mean
            # Don't multiply range directly, add noise based on fading
            fading_noise = true_range * (fading_factor - 1) * 0.1  # Reduced impact
            noisy_range += fading_noise
            total_variance += (true_range * 0.05)**2  # 5% uncertainty from fading

        # 8. Frequency offset
        if self.config.frequency_offset.enabled:
            offset_i = self._frequency_offset_cache.get(node_i, 0)
            offset_j = self._frequency_offset_cache.get(node_j, 0)
            freq_error = (offset_i - offset_j) * true_range / self.c
            noisy_range += freq_error

        # 9. Phase noise
        if self.config.phase_noise.enabled:
            phase_error = self._compute_phase_noise_error(true_range)
            noisy_range += phase_error
            total_variance += phase_error**2

        # 10. Antenna delay errors
        if self.config.antenna_delay.enabled:
            delay_i = self._antenna_delay_cache.get(node_i, 0)
            delay_j = self._antenna_delay_cache.get(node_j, 0)
            delay_error = (delay_i + delay_j) * self.c
            noisy_range += delay_error

        # 11. Temperature drift
        if self.config.temperature_drift.enabled:
            drift = self._compute_temperature_drift(node_i, node_j, timestamp)
            noisy_range += drift

        # 12. Node motion
        if self.config.node_motion.enabled and position_i is not None and position_j is not None:
            motion_error = self._compute_motion_error(position_i, position_j, timestamp)
            noisy_range += motion_error
            total_variance += motion_error**2

        # 13. Doppler shift
        if self.config.doppler.enabled:
            doppler_error = self._compute_doppler_error(true_range, timestamp)
            noisy_range += doppler_error

        # 14. Interference
        if self.config.interference.enabled:
            if self.rng.random() < self.config.interference.burst_probability:
                # Add interference burst
                sinr_linear = 10**(-self.config.interference.sinr_db/10)
                interference_var = true_range**2 * sinr_linear
                total_variance += interference_var

        # Add combined Gaussian noise
        if total_variance > 0:
            noise = self.rng.normal(0, np.sqrt(total_variance))
            noisy_range += noise

        # Ensure non-negative range
        noisy_range = max(0, noisy_range)

        # Compute effective standard deviation
        effective_std = np.sqrt(total_variance) if total_variance > 0 else self.config.distance_dependent.min_std_m

        return noisy_range, effective_std

    def _compute_thermal_variance(self, distance: float) -> float:
        """Compute thermal noise variance"""
        # Convert SNR to linear scale
        snr_linear = 10**(self.config.thermal.snr_db/10)
        noise_figure_linear = 10**(self.config.thermal.noise_figure_db/10)

        # Bandwidth in Hz
        bandwidth_hz = self.config.thermal.bandwidth_mhz * 1e6

        # Thermal noise power
        k_b = 1.38e-23  # Boltzmann constant
        T = 290  # Room temperature in Kelvin
        noise_power = k_b * T * bandwidth_hz * noise_figure_linear

        # Range resolution
        range_resolution = self.c / (2 * bandwidth_hz)

        # Variance in range
        variance = (range_resolution**2) / snr_linear
        return variance

    def _compute_distance_variance(self, distance: float) -> float:
        """Compute distance-dependent noise variance"""
        std = self.config.distance_dependent.coefficient * np.sqrt(distance)
        std = max(std, self.config.distance_dependent.min_std_m)
        return std**2

    def _compute_quantization_variance(self) -> float:
        """Compute ADC quantization noise variance"""
        # Quantization step
        lsb = self.config.quantization.full_scale_range_m / (2**self.config.quantization.adc_bits)
        # Uniform quantization noise variance
        variance = (lsb**2) / 12
        return variance

    def _compute_clock_jitter_variance(self) -> float:
        """Compute clock jitter induced range variance"""
        # Convert Allan deviation to range uncertainty
        allan_dev_s = self.config.clock_jitter.allan_deviation_ps * 1e-12
        range_std = self.c * allan_dev_s / np.sqrt(2)
        return range_std**2

    def _apply_multipath(self, true_range: float, node_i: int, node_j: int) -> Tuple[float, float]:
        """Apply multipath/NLOS effects

        Returns:
            Tuple of (bias, additional_variance)
        """
        if self.rng.random() < self.config.multipath.nlos_probability:
            # NLOS condition - add positive bias
            bias = self.rng.uniform(*self.config.multipath.bias_range_m)
            # Increased variance in NLOS
            additional_std = true_range * 0.01 * self.config.multipath.excess_std_factor
            return bias, additional_std**2
        else:
            # LOS condition - minimal bias
            return 0.0, 0.0

    def _get_shadowing_db(self, node_i: int, node_j: int) -> float:
        """Get shadowing in dB (correlated between node pairs)"""
        key = (min(node_i, node_j), max(node_i, node_j))

        if key not in self._shadowing_cache:
            # Generate log-normal shadowing (in dB)
            shadowing_db = self.rng.normal(0, self.config.shadowing.std_db)
            self._shadowing_cache[key] = shadowing_db

        return self._shadowing_cache[key]

    def _compute_small_scale_fading(self) -> float:
        """Compute small-scale fading factor (Rician)"""
        k_linear = 10**(self.config.small_scale_fading.k_factor_db/10)

        # Generate Rician fading
        los_component = np.sqrt(k_linear / (k_linear + 1))
        scatter_component = np.sqrt(1 / (2 * (k_linear + 1)))

        # Complex fading coefficient
        h_real = los_component + scatter_component * self.rng.normal()
        h_imag = scatter_component * self.rng.normal()

        # Magnitude of fading
        fading_magnitude = np.sqrt(h_real**2 + h_imag**2)

        return fading_magnitude

    def _compute_phase_noise_error(self, distance: float) -> float:
        """Compute phase noise induced range error"""
        # Simplified phase noise model
        phase_noise_rad = 10**(self.config.phase_noise.phase_noise_dbc_per_hz/20)

        # Convert phase noise to range error
        wavelength = self.c / (self.config.doppler.carrier_freq_ghz * 1e9)
        range_error = (phase_noise_rad * wavelength) / (4 * np.pi)

        return self.rng.normal(0, range_error)

    def _compute_temperature_drift(self, node_i: int, node_j: int, timestamp: float) -> float:
        """Compute temperature-induced drift"""
        if not self.config.temperature_drift.enabled:
            return 0.0

        # Temperature change over time
        hours = timestamp / 3600.0
        temp_change = self.config.temperature_drift.drift_rate_c_per_hour * hours

        # Antenna delay temperature coefficient
        delay_drift_ps = self.config.antenna_delay.temperature_coefficient_ps_per_c * temp_change
        delay_drift_m = delay_drift_ps * 1e-12 * self.c

        # Frequency offset temperature coefficient
        freq_drift_ppb = self.config.frequency_offset.temperature_coefficient_ppb_per_c * temp_change
        freq_drift = freq_drift_ppb * 1e-9

        return delay_drift_m + freq_drift

    def _compute_motion_error(self, pos_i: np.ndarray, pos_j: np.ndarray, timestamp: float) -> float:
        """Compute error due to node motion"""
        # Random walk motion model
        velocity_i = self.rng.normal(0, self.config.node_motion.velocity_std_mps, 2)
        velocity_j = self.rng.normal(0, self.config.node_motion.velocity_std_mps, 2)

        # Clip to maximum velocity
        velocity_i = np.clip(velocity_i, -self.config.node_motion.max_velocity_mps,
                            self.config.node_motion.max_velocity_mps)
        velocity_j = np.clip(velocity_j, -self.config.node_motion.max_velocity_mps,
                            self.config.node_motion.max_velocity_mps)

        # Relative velocity along line of sight
        los_vector = (pos_j - pos_i) / np.linalg.norm(pos_j - pos_i)
        rel_velocity = np.dot(velocity_j - velocity_i, los_vector)

        # Range rate error (assuming 1ms measurement time)
        measurement_time = 0.001  # 1ms
        range_error = rel_velocity * measurement_time

        return range_error

    def _compute_doppler_error(self, distance: float, timestamp: float) -> float:
        """Compute Doppler-induced range error"""
        # Simplified Doppler model
        max_doppler_hz = (self.config.doppler.max_velocity_mps / self.c) * \
                        (self.config.doppler.carrier_freq_ghz * 1e9)

        # Sinusoidal Doppler variation
        doppler_hz = max_doppler_hz * np.sin(2 * np.pi * 0.1 * timestamp)  # 0.1 Hz variation

        # Convert to range error
        range_error = (doppler_hz / (self.config.doppler.carrier_freq_ghz * 1e9)) * distance

        return range_error

    def get_measurement_covariance(self, distance: float) -> float:
        """Get expected measurement covariance for a given distance

        Args:
            distance: Distance in meters

        Returns:
            Measurement standard deviation in meters
        """
        # Simplified covariance model
        base_std = self.config.distance_dependent.min_std_m

        if self.config.thermal.enabled:
            thermal_std = np.sqrt(self._compute_thermal_variance(distance))
            base_std = np.sqrt(base_std**2 + thermal_std**2)

        if self.config.distance_dependent.enabled:
            dist_std = np.sqrt(self._compute_distance_variance(distance))
            base_std = np.sqrt(base_std**2 + dist_std**2)

        if self.config.multipath.enabled:
            # Average NLOS effect
            nlos_std = distance * 0.01 * self.config.multipath.excess_std_factor * \
                      self.config.multipath.nlos_probability
            base_std = np.sqrt(base_std**2 + nlos_std**2)

        return base_std


def create_preset_config(preset_name: str) -> NoiseConfig:
    """Create a NoiseConfig with a specific preset

    Args:
        preset_name: One of "ideal", "clean", "realistic", "harsh"

    Returns:
        Configured NoiseConfig object
    """
    preset_map = {
        "ideal": NoisePreset.IDEAL,
        "clean": NoisePreset.CLEAN,
        "realistic": NoisePreset.REALISTIC,
        "harsh": NoisePreset.HARSH
    }

    if preset_name.lower() not in preset_map:
        raise ValueError(f"Unknown preset: {preset_name}. Choose from {list(preset_map.keys())}")

    return NoiseConfig(preset=preset_map[preset_name.lower()])


def get_cramer_rao_bound(distances: List[float],
                        measurement_stds: List[float],
                        n_anchors: int) -> float:
    """Compute Cramér-Rao Lower Bound for localization accuracy

    Args:
        distances: List of measured distances
        measurement_stds: List of measurement standard deviations
        n_anchors: Number of anchor nodes

    Returns:
        CRLB in meters
    """
    # Simplified CRLB computation
    # For 2D localization with range measurements
    gdop = np.sqrt(2.0)  # Geometric dilution of precision (simplified)

    # Average measurement uncertainty
    avg_std = np.mean(measurement_stds)

    # CRLB approximation
    crlb = gdop * avg_std / np.sqrt(n_anchors)

    return crlb