#!/usr/bin/env python3
"""
Frequency-Compensated TL System Using IQ Phase Tracking

This implementation uses phase information from IQ data to estimate and compensate
for frequency offsets WITHOUT including them in the optimization state vector.
This avoids the c×t gradient explosion problem.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
sys.path.append('/Users/maxburnett/Documents/TL_ship')

from ftl_enhanced import EnhancedFTL, EnhancedFTLConfig
from test_30node_system import Large30NodeFTL
from ftl.signal import SignalConfig, gen_hrp_burst, gen_zc_burst
from ftl.rx_frontend import estimate_cfo, matched_filter, detect_toa


@dataclass
class FrequencyCompensatedConfig(EnhancedFTLConfig):
    """Configuration for frequency-compensated TL"""
    # CFO estimation parameters
    enable_cfo_compensation: bool = True
    cfo_update_interval: int = 10  # Update CFO every N iterations
    cfo_block_separation_s: float = 1e-6  # Time between signal blocks

    # Simulated frequency offsets (ppb)
    min_freq_offset_ppb: float = -50.0
    max_freq_offset_ppb: float = 50.0

    # Phase tracking parameters
    n_signal_blocks: int = 3  # Number of repeated blocks for CFO estimation


class IQSignalGenerator:
    """Generate IQ signals with simulated frequency offsets"""

    def __init__(self, config: SignalConfig):
        self.config = config
        self.template = gen_hrp_burst(config, n_repeats=3)

    def generate_with_cfo(self, cfo_hz: float, timestamp: float) -> np.ndarray:
        """
        Generate signal with carrier frequency offset

        Args:
            cfo_hz: Carrier frequency offset in Hz
            timestamp: Current time in seconds

        Returns:
            Complex baseband signal with CFO
        """
        signal = self.template.copy()

        # Apply frequency offset as phase rotation
        t = np.arange(len(signal)) / self.config.sample_rate
        phase_rotation = np.exp(1j * 2 * np.pi * cfo_hz * t)
        signal_with_cfo = signal * phase_rotation

        # Add accumulated phase from long-term drift
        accumulated_phase = 2 * np.pi * cfo_hz * timestamp
        signal_with_cfo *= np.exp(1j * accumulated_phase)

        return signal_with_cfo

    def extract_blocks(self, signal: np.ndarray, n_blocks: int = 3) -> List[np.ndarray]:
        """
        Extract repeated blocks from signal for CFO estimation

        Args:
            signal: Input signal
            n_blocks: Number of blocks to extract

        Returns:
            List of signal blocks
        """
        block_len = len(signal) // n_blocks
        blocks = []

        for i in range(n_blocks):
            start = i * block_len
            end = start + block_len
            blocks.append(signal[start:end])

        return blocks


class FrequencyCompensatedTL(Large30NodeFTL):
    """
    TL system with IQ-based frequency compensation

    Key innovation: Use phase tracking from IQ data to estimate CFO,
    then compensate measurements BEFORE optimization to avoid gradient explosion.
    """

    def __init__(self, config: Optional[FrequencyCompensatedConfig] = None):
        """Initialize with frequency compensation capability"""
        if config is None:
            config = FrequencyCompensatedConfig()

        super().__init__(config)

        # Frequency tracking
        self.true_freq_offsets_ppb = {}  # Ground truth
        self.estimated_cfos_hz = {}  # Estimated from IQ
        self.cfo_history = []  # Track CFO estimates over time

        # IQ signal generation
        signal_config = SignalConfig()
        self.signal_gen = IQSignalGenerator(signal_config)

        # Measurement timestamps for drift tracking
        self.current_timestamp = 0.0
        self.timestamp_step = 0.1  # 100ms between measurements

        # Initialize frequency offsets
        self._initialize_frequency_offsets()

    def _initialize_frequency_offsets(self):
        """Initialize random frequency offsets for each node"""
        np.random.seed(42)

        for i in range(self.config.n_nodes):
            # Random offset in ppb
            offset_ppb = np.random.uniform(
                self.config.min_freq_offset_ppb,
                self.config.max_freq_offset_ppb
            )
            self.true_freq_offsets_ppb[i] = offset_ppb

            # Convert to Hz (assuming 6.5 GHz carrier)
            carrier_freq = 6.5e9
            self.estimated_cfos_hz[i] = 0.0  # Start with no estimate

        print(f"Initialized frequency offsets: {self.config.min_freq_offset_ppb:.1f} to {self.config.max_freq_offset_ppb:.1f} ppb")

    def estimate_node_cfo(self, node_i: int, node_j: int) -> float:
        """
        Estimate CFO between two nodes using IQ phase tracking

        Args:
            node_i, node_j: Node indices

        Returns:
            Estimated CFO in Hz
        """
        # Get true frequency offsets for simulation
        freq_i_ppb = self.true_freq_offsets_ppb[node_i]
        freq_j_ppb = self.true_freq_offsets_ppb[node_j]

        # Convert to Hz (6.5 GHz carrier)
        carrier_freq = 6.5e9
        cfo_i_hz = freq_i_ppb * 1e-9 * carrier_freq
        cfo_j_hz = freq_j_ppb * 1e-9 * carrier_freq
        relative_cfo = cfo_j_hz - cfo_i_hz

        # Generate signals with CFO
        signal_i = self.signal_gen.generate_with_cfo(cfo_i_hz, self.current_timestamp)
        signal_j = self.signal_gen.generate_with_cfo(cfo_j_hz, self.current_timestamp)

        # Add noise
        snr_db = 30  # High SNR for accurate CFO estimation
        snr_linear = 10**(snr_db/10)
        signal_power = np.mean(np.abs(signal_j)**2)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal_j)) +
                                          1j * np.random.randn(len(signal_j)))
        received = signal_j + noise

        # Extract blocks for CFO estimation
        blocks = self.signal_gen.extract_blocks(received, self.config.n_signal_blocks)

        # Estimate CFO using phase correlation
        estimated_cfo = estimate_cfo(
            blocks,
            self.config.cfo_block_separation_s
        )

        return estimated_cfo

    def update_cfo_estimates(self):
        """Update CFO estimates for all nodes using IQ phase tracking"""
        print("\nUpdating CFO estimates from IQ phase tracking...")

        # For each unknown node, estimate CFO relative to anchors
        for i in range(self.config.n_anchors, self.config.n_nodes):
            cfo_estimates = []

            # Estimate relative to multiple anchors for robustness
            for j in range(min(3, self.config.n_anchors)):  # Use up to 3 anchors
                relative_cfo = self.estimate_node_cfo(j, i)
                cfo_estimates.append(relative_cfo)

            # Average estimates
            self.estimated_cfos_hz[i] = np.mean(cfo_estimates)

        # Track history
        self.cfo_history.append({
            'timestamp': self.current_timestamp,
            'estimates': self.estimated_cfos_hz.copy()
        })

    def compensate_measurement(self, node_i: int, node_j: int, raw_range: float) -> float:
        """
        Compensate range measurement for frequency offset

        Args:
            node_i, node_j: Node indices
            raw_range: Uncorrected range measurement

        Returns:
            Frequency-compensated range
        """
        if not self.config.enable_cfo_compensation:
            return raw_range

        # Get estimated CFOs
        cfo_i = self.estimated_cfos_hz.get(node_i, 0.0)
        cfo_j = self.estimated_cfos_hz.get(node_j, 0.0)

        # Compute drift correction
        # Δrange = c × (f_j - f_i) × t²/2 (accumulated drift)
        c = 299792458.0
        freq_diff_hz = (cfo_j - cfo_i)
        freq_diff_relative = freq_diff_hz / 6.5e9  # Relative to carrier

        # Drift accumulation over time
        drift_m = c * freq_diff_relative * self.current_timestamp**2 / 2

        # Apply compensation
        compensated_range = raw_range - drift_m

        return compensated_range

    def run_with_frequency_compensation(self, max_iterations: int = 100):
        """
        Run optimization with periodic CFO updates

        Args:
            max_iterations: Maximum number of iterations
        """
        print("\n" + "="*60)
        print("FREQUENCY-COMPENSATED TL WITH IQ PHASE TRACKING")
        print("="*60)

        # Track metrics
        rmse_history = []
        freq_error_history = []

        for iteration in range(max_iterations):
            # Advance timestamp
            self.current_timestamp += self.timestamp_step

            # Update CFO estimates periodically
            if iteration % self.config.cfo_update_interval == 0:
                self.update_cfo_estimates()

                # Calculate frequency estimation error
                freq_errors = []
                for i in range(self.config.n_anchors, self.config.n_nodes):
                    true_cfo = self.true_freq_offsets_ppb[i] * 1e-9 * 6.5e9
                    est_cfo = self.estimated_cfos_hz[i]
                    error_hz = abs(true_cfo - est_cfo)
                    freq_errors.append(error_hz)

                mean_freq_error = np.mean(freq_errors)
                freq_error_history.append(mean_freq_error)
                print(f"  CFO estimation error: {mean_freq_error:.2f} Hz")

            # Apply frequency compensation to measurements
            compensated_measurements = []
            for meas in self.measurements:
                i = meas['i']
                j = meas['j']
                raw_range = meas['range']
                std = meas['std']
                comp_range = self.compensate_measurement(i, j, raw_range)
                compensated_measurements.append({
                    'i': i,
                    'j': j,
                    'range': comp_range,
                    'std': std
                })

            # Temporarily replace measurements
            original_measurements = self.measurements
            self.measurements = compensated_measurements

            # Run one iteration of optimization
            if self.config.use_adaptive_lm:
                self.step_with_lm()
            else:
                self.step_basic()

            # Restore original measurements
            self.measurements = original_measurements

            # Calculate RMSE
            errors = []
            for i in range(self.config.n_anchors, self.config.n_nodes):
                err = np.linalg.norm(self.states[i, :2] - self.true_positions[i])
                errors.append(err)
            rmse = np.sqrt(np.mean(np.array(errors)**2))
            rmse_history.append(rmse * 1000)  # Convert to mm

            # Print progress
            if iteration % 10 == 0:
                print(f"Iteration {iteration:3d}: RMSE = {rmse*1000:.2f} mm, "
                      f"Time = {self.current_timestamp:.1f}s")

        return rmse_history, freq_error_history

    def plot_results(self, rmse_history: List[float], freq_error_history: List[float]):
        """Plot convergence with frequency compensation"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # RMSE convergence
        ax = axes[0, 0]
        ax.semilogy(rmse_history, 'b-', linewidth=2)
        ax.axhline(y=18.5, color='r', linestyle='--', label='Target (18.5mm)')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('RMSE (mm)')
        ax.set_title('Position RMSE with IQ-based Frequency Compensation')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Frequency error
        ax = axes[0, 1]
        update_iterations = np.arange(0, len(freq_error_history)) * self.config.cfo_update_interval
        ax.semilogy(update_iterations, freq_error_history, 'g-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('CFO Estimation Error (Hz)')
        ax.set_title('Frequency Offset Tracking Accuracy')
        ax.grid(True, alpha=0.3)

        # Phase tracking visualization
        ax = axes[1, 0]
        # Simulate phase evolution
        t = np.linspace(0, 1e-3, 1000)  # 1ms
        cfo_example = 100  # 100 Hz offset
        phase = 2 * np.pi * cfo_example * t
        ax.plot(t * 1e6, np.unwrap(phase), 'b-', linewidth=2)
        ax.set_xlabel('Time (μs)')
        ax.set_ylabel('Phase (radians)')
        ax.set_title(f'IQ Phase Evolution (CFO = {cfo_example} Hz)')
        ax.grid(True, alpha=0.3)

        # Frequency offset distribution
        ax = axes[1, 1]
        true_freqs = list(self.true_freq_offsets_ppb.values())
        ax.hist(true_freqs, bins=20, alpha=0.7, color='blue', label='True CFO')
        if self.cfo_history:
            final_estimates = [self.cfo_history[-1]['estimates'][i] * 6.5e9 / 1e-9
                             for i in range(self.config.n_anchors, self.config.n_nodes)]
            ax.hist(final_estimates, bins=20, alpha=0.7, color='red', label='Estimated CFO')
        ax.set_xlabel('Frequency Offset (ppb)')
        ax.set_ylabel('Count')
        ax.set_title('CFO Distribution')
        ax.legend()

        plt.suptitle('Frequency-Compensated TL using IQ Phase Tracking',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('frequency_compensation_iq_phase.png', dpi=150)
        print("\nPlot saved as frequency_compensation_iq_phase.png")


def main():
    """Test frequency-compensated TL system"""

    # Configure system
    config = FrequencyCompensatedConfig(
        n_nodes=30,
        n_anchors=5,
        use_adaptive_lm=True,
        use_line_search=False,
        max_iterations=100,
        measurement_std=0.01,
        verbose=False,

        # Frequency compensation
        enable_cfo_compensation=True,
        cfo_update_interval=10,
        min_freq_offset_ppb=-50,
        max_freq_offset_ppb=50
    )

    # Create and initialize system
    print("Initializing frequency-compensated TL system...")
    ftl = FrequencyCompensatedTL(config)

    # Run with frequency compensation
    rmse_history, freq_error_history = ftl.run_with_frequency_compensation(max_iterations=100)

    # Final results
    final_rmse = rmse_history[-1]
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Final RMSE: {final_rmse:.2f} mm")
    print(f"CFO tracking enabled: {config.enable_cfo_compensation}")
    print(f"Time simulated: {ftl.current_timestamp:.1f} seconds")

    if freq_error_history:
        print(f"Final CFO error: {freq_error_history[-1]:.2f} Hz")

    # Compare with uncompensated
    print("\nRunning WITHOUT frequency compensation for comparison...")
    config.enable_cfo_compensation = False
    ftl_uncompensated = FrequencyCompensatedTL(config)
    rmse_uncompensated, _ = ftl_uncompensated.run_with_frequency_compensation(max_iterations=100)

    print(f"\nWithout compensation: {rmse_uncompensated[-1]:.2f} mm")
    print(f"With IQ compensation: {final_rmse:.2f} mm")
    print(f"Improvement: {(rmse_uncompensated[-1] - final_rmse) / rmse_uncompensated[-1] * 100:.1f}%")

    # Plot results
    ftl.plot_results(rmse_history, freq_error_history)

    return ftl, rmse_history


if __name__ == "__main__":
    ftl, history = main()