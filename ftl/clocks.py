"""
Clock Models with Bias, Drift, and CFO
Realistic oscillator models based on Allan variance and ppm specifications
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class ClockState:
    """State vector for a single node's clock"""
    bias: float = 0.0        # Clock bias in seconds
    drift: float = 0.0       # Clock drift in s/s (dimensionless)
    cfo: float = 0.0         # Carrier frequency offset in Hz
    sco_ppm: float = 0.0     # Sample clock offset in ppm (affects ADC sample rate)

    # Process noise parameters
    bias_noise_std: float = 1e-9    # 1 ns/sqrt(s)
    drift_noise_std: float = 1e-12  # 1 ppb/sqrt(s)
    cfo_noise_std: float = 1.0      # 1 Hz/sqrt(s)
    sco_noise_std: float = 0.01     # 0.01 ppm/sqrt(s)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [bias, drift, cfo, sco_ppm]"""
        return np.array([self.bias, self.drift, self.cfo, self.sco_ppm])

    def from_array(self, arr: np.ndarray):
        """Update from numpy array"""
        self.bias = arr[0]
        self.drift = arr[1]
        self.cfo = arr[2]
        if len(arr) > 3:
            self.sco_ppm = arr[3]


@dataclass
class ClockModel:
    """
    Oscillator model with realistic Allan variance characteristics

    Parameters follow typical TCXO/OCXO specifications:
    - TCXO: ±1-2 ppm frequency accuracy, 1e-10 @ 1s Allan deviation
    - OCXO: ±0.1 ppm frequency accuracy, 1e-11 @ 1s Allan deviation
    """

    oscillator_type: str = "TCXO"  # TCXO, OCXO, or CSAC
    carrier_freq_hz: float = 6.5e9  # UWB center frequency (6.5 GHz)

    # Frequency accuracy (ppm)
    frequency_accuracy_ppm: float = field(default=None)

    # Allan variance parameters
    allan_deviation_1s: float = field(default=None)  # σ_y(τ=1s)

    # Temperature coefficient (ppm/°C)
    temp_coefficient_ppm: float = field(default=None)

    # Aging rate (ppm/year)
    aging_rate_ppm_year: float = field(default=None)

    def __post_init__(self):
        # Set defaults based on oscillator type
        if self.frequency_accuracy_ppm is None:
            defaults = {
                "TCXO": 2.0,
                "OCXO": 0.1,
                "CSAC": 0.005,
                "CRYSTAL": 20.0
            }
            self.frequency_accuracy_ppm = defaults.get(self.oscillator_type, 2.0)

        if self.allan_deviation_1s is None:
            defaults = {
                "TCXO": 1e-10,
                "OCXO": 1e-11,
                "CSAC": 1e-12,
                "CRYSTAL": 1e-9
            }
            self.allan_deviation_1s = defaults.get(self.oscillator_type, 1e-10)

        if self.temp_coefficient_ppm is None:
            defaults = {
                "TCXO": 0.5,
                "OCXO": 0.01,
                "CSAC": 0.001,
                "CRYSTAL": 10.0
            }
            self.temp_coefficient_ppm = defaults.get(self.oscillator_type, 0.5)

        if self.aging_rate_ppm_year is None:
            defaults = {
                "TCXO": 1.0,
                "OCXO": 0.1,
                "CSAC": 0.01,
                "CRYSTAL": 5.0
            }
            self.aging_rate_ppm_year = defaults.get(self.oscillator_type, 1.0)

    def sample_initial_state(self, seed: Optional[int] = None) -> ClockState:
        """
        Sample initial clock state from realistic distributions

        Returns:
            ClockState with sampled bias, drift, and CFO
        """
        if seed is not None:
            np.random.seed(seed)

        # Initial bias: typically microseconds to milliseconds
        # Assume nodes start with some random offset
        bias_std_s = 1e-3  # 1 ms standard deviation
        initial_bias = np.random.normal(0, bias_std_s)

        # Initial drift: based on frequency accuracy
        # Convert ppm to dimensionless drift rate
        drift_ppm = np.random.normal(0, self.frequency_accuracy_ppm)
        initial_drift = drift_ppm * 1e-6  # Convert ppm to s/s

        # Initial CFO: based on frequency accuracy at carrier
        # CFO = carrier_freq * (ppm_error / 1e6)
        cfo_hz = self.carrier_freq_hz * drift_ppm * 1e-6

        # Initial SCO: same PPM error affects sample clock
        # This creates coherent CFO and SCO from same oscillator
        sco_ppm = drift_ppm  # Same oscillator drives both RF and ADC

        # Process noise based on Allan variance
        # For white frequency noise: σ_y²(τ) = σ_y²(1s) / τ
        # This means for discrete time steps: q = σ_y²(1s) * c² * dt
        # where c is speed of light for time-to-distance conversion

        # Bias noise: time noise scales with sqrt(dt) in continuous time
        # σ_bias = c * σ_y(1s) * sqrt(dt) for dt=1s
        bias_noise = self.allan_deviation_1s * 3e8  # Convert time noise to distance

        # Drift noise: frequency noise is white
        drift_noise = self.allan_deviation_1s / np.sqrt(1.0)  # Per second

        # CFO noise: proportional to carrier frequency
        cfo_noise = self.carrier_freq_hz * self.allan_deviation_1s

        # SCO noise: same as drift but in ppm
        sco_noise = self.allan_deviation_1s * 1e6  # Convert to ppm

        return ClockState(
            bias=initial_bias,
            drift=initial_drift,
            cfo=cfo_hz,
            sco_ppm=sco_ppm,
            bias_noise_std=bias_noise,
            drift_noise_std=drift_noise,
            cfo_noise_std=cfo_noise,
            sco_noise_std=sco_noise
        )

    def propagate_state(
        self,
        state: ClockState,
        dt: float,
        add_noise: bool = True
    ) -> ClockState:
        """
        Propagate clock state forward in time

        Uses two-state clock model:
        bias(t+dt) = bias(t) + drift(t) * dt + w_bias
        drift(t+dt) = drift(t) + w_drift
        cfo(t+dt) = cfo(t) + w_cfo

        Args:
            state: Current clock state
            dt: Time step in seconds
            add_noise: Whether to add process noise

        Returns:
            Updated clock state
        """
        new_state = ClockState(
            bias=state.bias,
            drift=state.drift,
            cfo=state.cfo,
            sco_ppm=state.sco_ppm,
            bias_noise_std=state.bias_noise_std,
            drift_noise_std=state.drift_noise_std,
            cfo_noise_std=state.cfo_noise_std,
            sco_noise_std=state.sco_noise_std
        )

        # Propagate bias with drift
        new_state.bias += state.drift * dt

        # Add process noise if requested
        if add_noise:
            # Allan variance-based noise model
            # For white frequency noise: σ_y²(τ) = σ_y²(1s) / τ
            # Clock bias noise accumulates as random walk: σ ∝ sqrt(dt)
            new_state.bias += np.random.normal(0, state.bias_noise_std * np.sqrt(dt))

            # Drift (frequency) has white noise: σ ∝ sqrt(dt)
            new_state.drift += np.random.normal(0, state.drift_noise_std * np.sqrt(dt))

            # CFO follows same model as drift
            new_state.cfo += np.random.normal(0, state.cfo_noise_std * np.sqrt(dt))

            # SCO also follows drift model
            new_state.sco_ppm += np.random.normal(0, state.sco_noise_std * np.sqrt(dt))

        return new_state

    def apply_to_timestamp(
        self,
        true_time: float,
        clock_state: ClockState
    ) -> float:
        """
        Apply clock error to true timestamp

        Args:
            true_time: True time in seconds
            clock_state: Node's clock state

        Returns:
            Observed time including clock error
        """
        # Observed time = true time + bias + drift * true_time
        observed_time = true_time + clock_state.bias + clock_state.drift * true_time
        return observed_time

    def apply_cfo_to_signal(
        self,
        signal: np.ndarray,
        cfo_hz: float,
        sample_rate: float
    ) -> np.ndarray:
        """
        Apply carrier frequency offset to complex baseband signal

        Args:
            signal: Complex baseband signal
            cfo_hz: Carrier frequency offset in Hz
            sample_rate: Sample rate in Hz

        Returns:
            Signal with CFO applied
        """
        n = len(signal)
        t = np.arange(n) / sample_rate
        cfo_phasor = np.exp(1j * 2 * np.pi * cfo_hz * t)
        return signal * cfo_phasor

    def compute_allan_variance(
        self,
        time_series: np.ndarray,
        sample_rate: float,
        tau_values: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Allan variance from time series data

        Args:
            time_series: Clock offset measurements
            sample_rate: Measurement rate in Hz
            tau_values: Integration times to compute (seconds)

        Returns:
            (tau_values, allan_deviation)
        """
        if tau_values is None:
            # Default tau values from 0.1s to 100s
            tau_values = np.logspace(-1, 2, 30)

        n = len(time_series)
        dt = 1.0 / sample_rate
        allan_dev = []

        for tau in tau_values:
            m = int(tau / dt)  # Number of samples per tau
            if m < 1 or m > n // 2:
                allan_dev.append(np.nan)
                continue

            # Compute Allan variance
            # σ²_y(τ) = <(y_{i+1} - y_i)²> / 2
            # where y_i is the average frequency over interval i

            # Average over tau intervals
            n_intervals = n // m
            if n_intervals < 2:
                allan_dev.append(np.nan)
                continue

            y = np.zeros(n_intervals)
            for i in range(n_intervals):
                y[i] = np.mean(time_series[i*m:(i+1)*m])

            # Compute variance of first differences
            diff = np.diff(y)
            allan_var = np.mean(diff**2) / 2
            allan_dev.append(np.sqrt(allan_var))

        return tau_values, np.array(allan_dev)


class ClockEnsemble:
    """Manage multiple node clocks with correlation"""

    def __init__(
        self,
        n_nodes: int,
        model: ClockModel,
        correlation: float = 0.0,
        anchor_indices: Optional[list] = None
    ):
        """
        Initialize ensemble of clocks

        Args:
            n_nodes: Number of nodes
            model: Clock model to use
            correlation: Correlation between node clocks (0-1)
            anchor_indices: Indices of anchor nodes (may have better clocks)
        """
        self.n_nodes = n_nodes
        self.model = model
        self.correlation = correlation
        self.anchor_indices = anchor_indices or []

        # Initialize clock states
        self.states = {}
        for i in range(n_nodes):
            # Anchors might have better clocks
            if i in self.anchor_indices:
                # Use OCXO for anchors
                anchor_model = ClockModel(
                    oscillator_type="OCXO",
                    carrier_freq_hz=model.carrier_freq_hz
                )
                self.states[i] = anchor_model.sample_initial_state()
            else:
                self.states[i] = model.sample_initial_state()

    def propagate_all(self, dt: float):
        """Propagate all clock states forward"""
        for i in range(self.n_nodes):
            self.states[i] = self.model.propagate_state(
                self.states[i], dt, add_noise=True
            )

    def get_relative_clock_offset(self, i: int, j: int) -> float:
        """Get relative clock bias between nodes i and j"""
        return self.states[j].bias - self.states[i].bias

    def get_relative_cfo(self, i: int, j: int) -> float:
        """Get relative CFO between nodes i and j"""
        return self.states[j].cfo - self.states[i].cfo


if __name__ == "__main__":
    # Test clock models
    print("Testing Clock Models...")
    print("=" * 50)

    # Test different oscillator types
    for osc_type in ["CRYSTAL", "TCXO", "OCXO", "CSAC"]:
        model = ClockModel(oscillator_type=osc_type)
        state = model.sample_initial_state()

        print(f"\n{osc_type}:")
        print(f"  Frequency accuracy: ±{model.frequency_accuracy_ppm} ppm")
        print(f"  Allan deviation @ 1s: {model.allan_deviation_1s}")
        print(f"  Initial bias: {state.bias*1e6:.1f} µs")
        print(f"  Initial drift: {state.drift*1e9:.1f} ppb")
        print(f"  Initial CFO: {state.cfo:.1f} Hz")

    # Test propagation
    print("\n" + "="*50)
    print("Testing clock propagation over 10 seconds...")

    model = ClockModel(oscillator_type="TCXO")
    state = model.sample_initial_state()

    times = []
    biases = []

    for step in range(100):
        t = step * 0.1  # 100ms steps
        times.append(t)
        biases.append(state.bias * 1e6)  # Convert to µs
        state = model.propagate_state(state, dt=0.1)

    bias_drift = (biases[-1] - biases[0]) / 10  # µs/s
    print(f"Clock drift over 10s: {bias_drift:.2f} µs/s")
    print(f"Final clock bias: {biases[-1]:.1f} µs")