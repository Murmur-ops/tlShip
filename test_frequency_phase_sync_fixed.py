#!/usr/bin/env python3
"""
Frequency-Compensated TL System Using IQ Phase Tracking (Fixed)

Properly implements phase-based CFO estimation and compensation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
sys.path.append('/Users/maxburnett/Documents/TL_ship')

from ftl_enhanced import EnhancedFTL, EnhancedFTLConfig
from test_30node_system import Large30NodeFTL


@dataclass
class FrequencyCompensatedConfig(EnhancedFTLConfig):
    """Configuration for frequency-compensated TL"""
    # CFO parameters in ppb (parts per billion)
    enable_cfo_compensation: bool = True
    cfo_update_interval: int = 10
    min_freq_offset_ppb: float = -10.0  # More realistic: ±10 ppb
    max_freq_offset_ppb: float = 10.0

    # Phase tracking
    phase_noise_rad: float = 0.01  # Phase measurement noise


class PhaseBasedCFOEstimator:
    """Estimate CFO from phase evolution over time"""

    def __init__(self, carrier_freq: float = 6.5e9):
        self.carrier_freq = carrier_freq
        self.phase_history = {}  # Track phase per node pair

    def add_phase_measurement(self, node_i: int, node_j: int,
                             phase_rad: float, timestamp: float):
        """Record a phase measurement between nodes"""
        key = (node_i, node_j)
        if key not in self.phase_history:
            self.phase_history[key] = []
        self.phase_history[key].append((timestamp, phase_rad))

    def estimate_cfo_ppb(self, node_i: int, node_j: int) -> float:
        """
        Estimate relative CFO in ppb from phase evolution

        Phase evolution: φ(t) = 2π × Δf × t
        Where Δf is the frequency difference
        """
        key = (node_i, node_j)
        if key not in self.phase_history or len(self.phase_history[key]) < 2:
            return 0.0

        history = self.phase_history[key]

        # Use linear regression on phase vs time
        times = np.array([h[0] for h in history])
        phases = np.array([h[1] for h in history])

        # Unwrap phases to handle 2π jumps
        phases = np.unwrap(phases)

        # Linear fit: phase = 2π × Δf × t + φ₀
        if len(times) > 1:
            # Calculate phase rate (rad/s)
            phase_rate = (phases[-1] - phases[0]) / (times[-1] - times[0])

            # Convert to frequency difference
            freq_diff_hz = phase_rate / (2 * np.pi)

            # Convert to ppb relative to carrier
            freq_diff_ppb = (freq_diff_hz / self.carrier_freq) * 1e9

            return freq_diff_ppb

        return 0.0


class FrequencyCompensatedTL(Large30NodeFTL):
    """
    TL system with proper phase-based frequency compensation
    """

    def __init__(self, config: Optional[FrequencyCompensatedConfig] = None):
        """Initialize with frequency compensation"""
        if config is None:
            config = FrequencyCompensatedConfig()

        super().__init__(config)

        # Frequency tracking
        self.true_freq_offsets_ppb = {}
        self.estimated_freq_offsets_ppb = {}
        self.freq_error_history = []

        # Phase-based CFO estimation
        self.cfo_estimator = PhaseBasedCFOEstimator()

        # Time tracking
        self.current_time = 0.0
        self.time_step = 0.1  # 100ms

        # Initialize frequency offsets
        self._initialize_frequency_offsets()

    def _initialize_frequency_offsets(self):
        """Initialize realistic frequency offsets"""
        np.random.seed(42)

        # Anchors have zero offset (reference)
        for i in range(self.config.n_anchors):
            self.true_freq_offsets_ppb[i] = 0.0
            self.estimated_freq_offsets_ppb[i] = 0.0

        # Unknown nodes have random offsets
        for i in range(self.config.n_anchors, self.config.n_nodes):
            offset_ppb = np.random.uniform(
                self.config.min_freq_offset_ppb,
                self.config.max_freq_offset_ppb
            )
            self.true_freq_offsets_ppb[i] = offset_ppb
            self.estimated_freq_offsets_ppb[i] = 0.0  # Start with no estimate

        print(f"Initialized frequency offsets: ±{self.config.max_freq_offset_ppb:.1f} ppb")

    def simulate_phase_measurement(self, node_i: int, node_j: int) -> float:
        """
        Simulate phase measurement between nodes

        Returns phase accumulated due to frequency difference
        """
        # Get true frequency offsets
        freq_i_ppb = self.true_freq_offsets_ppb[node_i]
        freq_j_ppb = self.true_freq_offsets_ppb[node_j]
        freq_diff_ppb = freq_j_ppb - freq_i_ppb

        # Convert to Hz
        carrier_freq = 6.5e9
        freq_diff_hz = freq_diff_ppb * 1e-9 * carrier_freq

        # Phase accumulated over time
        phase = 2 * np.pi * freq_diff_hz * self.current_time

        # Add measurement noise
        phase += np.random.normal(0, self.config.phase_noise_rad)

        return phase

    def update_cfo_estimates(self):
        """Update CFO estimates from phase measurements"""

        # Simulate phase measurements for each unknown node
        for i in range(self.config.n_anchors, self.config.n_nodes):
            # Measure relative to anchors
            cfo_estimates = []

            for j in range(min(3, self.config.n_anchors)):  # Use up to 3 anchors
                # Simulate phase measurement
                phase = self.simulate_phase_measurement(j, i)

                # Record phase
                self.cfo_estimator.add_phase_measurement(j, i, phase, self.current_time)

                # Get CFO estimate
                cfo_ppb = self.cfo_estimator.estimate_cfo_ppb(j, i)
                cfo_estimates.append(cfo_ppb)

            # Average estimates (could use weighted average based on SNR)
            if cfo_estimates:
                self.estimated_freq_offsets_ppb[i] = np.mean(cfo_estimates)

        # Calculate estimation error
        errors = []
        for i in range(self.config.n_anchors, self.config.n_nodes):
            true_cfo = self.true_freq_offsets_ppb[i]
            est_cfo = self.estimated_freq_offsets_ppb[i]
            errors.append(abs(true_cfo - est_cfo))

        mean_error_ppb = np.mean(errors) if errors else 0
        self.freq_error_history.append(mean_error_ppb)

        return mean_error_ppb

    def compensate_measurement(self, node_i: int, node_j: int, raw_range: float) -> float:
        """
        Compensate range measurement for frequency-induced clock drift

        Clock drift causes range error: Δr = c × Δτ
        Where Δτ = freq_offset × time
        """
        if not self.config.enable_cfo_compensation:
            return raw_range

        # Get estimated frequency offsets
        freq_i_ppb = self.estimated_freq_offsets_ppb.get(node_i, 0.0)
        freq_j_ppb = self.estimated_freq_offsets_ppb.get(node_j, 0.0)

        # Relative frequency offset
        freq_diff_ppb = freq_j_ppb - freq_i_ppb

        # Time offset accumulated due to frequency difference
        # Δτ = freq_diff × time (in nanoseconds)
        time_offset_ns = freq_diff_ppb * self.current_time  # ppb × seconds = ns

        # Range correction (c × Δτ)
        c = 299792458.0  # m/s
        range_correction = c * time_offset_ns * 1e-9  # Convert ns to seconds

        # Apply compensation
        compensated_range = raw_range - range_correction

        return compensated_range

    def run_with_frequency_compensation(self, max_iterations: int = 100):
        """Run optimization with frequency compensation"""

        print("\n" + "="*60)
        print("FREQUENCY-COMPENSATED TL WITH IQ PHASE TRACKING")
        print("="*60)

        rmse_history = []

        for iteration in range(max_iterations):
            # Advance time
            self.current_time += self.time_step

            # Update CFO estimates periodically
            if iteration % self.config.cfo_update_interval == 0:
                error_ppb = self.update_cfo_estimates()
                print(f"  Iteration {iteration}: CFO error = {error_ppb:.3f} ppb")

            # Apply frequency compensation to measurements
            if self.config.enable_cfo_compensation:
                compensated_measurements = []
                for meas in self.measurements:
                    i = meas['i']
                    j = meas['j']
                    raw_range = meas['range']
                    std = meas['std']

                    # Apply compensation
                    comp_range = self.compensate_measurement(i, j, raw_range)

                    compensated_measurements.append({
                        'i': i,
                        'j': j,
                        'range': comp_range,
                        'std': std
                    })

                # Use compensated measurements
                original_measurements = self.measurements
                self.measurements = compensated_measurements

            # Run one optimization iteration
            if self.config.use_adaptive_lm:
                self.step_with_lm()
            else:
                self.step_basic()

            # Restore original measurements
            if self.config.enable_cfo_compensation:
                self.measurements = original_measurements

            # Calculate RMSE
            errors = []
            for i in range(self.config.n_anchors, self.config.n_nodes):
                err = np.linalg.norm(self.states[i, :2] - self.true_positions[i])
                errors.append(err)
            rmse = np.sqrt(np.mean(np.array(errors)**2))
            rmse_history.append(rmse * 1000)  # Convert to mm

            # Progress update
            if iteration % 20 == 0:
                print(f"  Iteration {iteration}: RMSE = {rmse*1000:.2f} mm, Time = {self.current_time:.1f}s")

        return rmse_history

    def plot_results(self, rmse_with: List[float], rmse_without: List[float]):
        """Plot comparison results"""

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # RMSE comparison
        ax = axes[0, 0]
        iterations = np.arange(len(rmse_with))
        ax.semilogy(iterations, rmse_with, 'b-', label='With CFO compensation', linewidth=2)
        ax.semilogy(iterations, rmse_without, 'r--', label='Without compensation', linewidth=2)
        ax.axhline(y=18.5, color='g', linestyle=':', label='Target (18.5mm)')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('RMSE (mm)')
        ax.set_title('Position RMSE: With vs Without Frequency Compensation')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # CFO estimation error
        ax = axes[0, 1]
        if self.freq_error_history:
            update_iters = np.arange(len(self.freq_error_history)) * self.config.cfo_update_interval
            ax.semilogy(update_iters, self.freq_error_history, 'g-', linewidth=2)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('CFO Estimation Error (ppb)')
            ax.set_title('Frequency Offset Estimation Accuracy')
            ax.grid(True, alpha=0.3)

        # Frequency offset distribution
        ax = axes[1, 0]
        true_freqs = [self.true_freq_offsets_ppb[i]
                     for i in range(self.config.n_anchors, self.config.n_nodes)]
        est_freqs = [self.estimated_freq_offsets_ppb[i]
                    for i in range(self.config.n_anchors, self.config.n_nodes)]

        x = np.arange(len(true_freqs))
        width = 0.35
        ax.bar(x - width/2, true_freqs, width, label='True CFO', alpha=0.7)
        ax.bar(x + width/2, est_freqs, width, label='Estimated CFO', alpha=0.7)
        ax.set_xlabel('Node Index')
        ax.set_ylabel('Frequency Offset (ppb)')
        ax.set_title('True vs Estimated Frequency Offsets')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Improvement over time
        ax = axes[1, 1]
        time_axis = iterations * self.time_step
        improvement = (np.array(rmse_without) - np.array(rmse_with)) / np.array(rmse_without) * 100
        ax.plot(time_axis, improvement, 'purple', linewidth=2)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Improvement (%)')
        ax.set_title('Relative Improvement with CFO Compensation')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)

        plt.suptitle('Frequency-Compensated TL using IQ Phase Tracking',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('frequency_compensation_fixed.png', dpi=150)
        print("\nPlot saved as frequency_compensation_fixed.png")


def main():
    """Test frequency compensation system"""

    # Test WITH compensation
    config_with = FrequencyCompensatedConfig(
        n_nodes=30,
        n_anchors=5,
        use_adaptive_lm=True,
        use_line_search=False,
        max_iterations=100,
        measurement_std=0.01,
        verbose=False,
        enable_cfo_compensation=True,
        cfo_update_interval=10,
        min_freq_offset_ppb=-10,
        max_freq_offset_ppb=10
    )

    print("\n1. Running WITH frequency compensation...")
    ftl_with = FrequencyCompensatedTL(config_with)
    rmse_with = ftl_with.run_with_frequency_compensation(max_iterations=100)

    # Test WITHOUT compensation
    config_without = FrequencyCompensatedConfig(
        n_nodes=30,
        n_anchors=5,
        use_adaptive_lm=True,
        use_line_search=False,
        max_iterations=100,
        measurement_std=0.01,
        verbose=False,
        enable_cfo_compensation=False  # Disabled
    )

    print("\n2. Running WITHOUT frequency compensation...")
    ftl_without = FrequencyCompensatedTL(config_without)
    rmse_without = ftl_without.run_with_frequency_compensation(max_iterations=100)

    # Results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Without CFO compensation: {rmse_without[-1]:.2f} mm")
    print(f"With CFO compensation: {rmse_with[-1]:.2f} mm")

    if rmse_with[-1] < rmse_without[-1]:
        improvement = (rmse_without[-1] - rmse_with[-1]) / rmse_without[-1] * 100
        print(f"Improvement: {improvement:.1f}%")
        print("\n✅ Frequency compensation improves accuracy!")
    else:
        print("\n⚠️ Frequency compensation needs tuning")

    print(f"\nTime simulated: {ftl_with.current_time:.1f} seconds")
    print(f"Max frequency offset: ±{config_with.max_freq_offset_ppb:.1f} ppb")

    if ftl_with.freq_error_history:
        print(f"Final CFO estimation error: {ftl_with.freq_error_history[-1]:.3f} ppb")

    # Plot results
    ftl_with.plot_results(rmse_with, rmse_without)

    return rmse_with, rmse_without


if __name__ == "__main__":
    rmse_with, rmse_without = main()