#!/usr/bin/env python3
"""
Long-term stability test showing the benefit of frequency compensation

Over long time periods, uncompensated frequency offsets cause significant drift.
This test demonstrates how IQ-based CFO tracking maintains accuracy.
"""

import numpy as np
import matplotlib.pyplot as plt
from test_frequency_phase_sync_fixed import (
    FrequencyCompensatedTL,
    FrequencyCompensatedConfig
)


def test_long_term_stability():
    """Test system stability over extended time periods"""

    print("\n" + "="*60)
    print("LONG-TERM STABILITY TEST")
    print("="*60)
    print("Testing frequency compensation over 200 seconds")
    print("Frequency offsets: ±10 ppb (typical TCXO)")
    print()

    # Configure for long-term test
    config = FrequencyCompensatedConfig(
        n_nodes=30,
        n_anchors=5,
        use_adaptive_lm=True,
        use_line_search=False,
        max_iterations=1000,  # 1000 iterations = 100 seconds at 0.1s/iter
        measurement_std=0.01,
        verbose=False,
        enable_cfo_compensation=True,
        cfo_update_interval=20,  # Update CFO every 2 seconds
        min_freq_offset_ppb=-10,
        max_freq_offset_ppb=10
    )

    # Track drift over time
    time_points = [10, 50, 100, 200]  # Seconds (reduced for faster testing)
    results = {'with': {}, 'without': {}}

    for mode in ['with', 'without']:
        print(f"\n{'='*40}")
        print(f"Testing {mode.upper()} compensation...")
        print(f"{'='*40}")

        config.enable_cfo_compensation = (mode == 'with')
        ftl = FrequencyCompensatedTL(config)

        # Run for specified time points
        iteration = 0
        rmse_history = []

        for target_time in time_points:
            target_iter = int(target_time / 0.1)  # 0.1s per iteration

            # Run until target time
            while iteration < target_iter:
                # Advance time
                ftl.current_time += ftl.time_step

                # Update CFO estimates if enabled
                if config.enable_cfo_compensation and iteration % config.cfo_update_interval == 0:
                    error_ppb = ftl.update_cfo_estimates()

                # Apply compensation if enabled
                if config.enable_cfo_compensation:
                    compensated_measurements = []
                    for meas in ftl.measurements:
                        comp_range = ftl.compensate_measurement(
                            meas['i'], meas['j'], meas['range']
                        )
                        compensated_measurements.append({
                            'i': meas['i'],
                            'j': meas['j'],
                            'range': comp_range,
                            'std': meas['std']
                        })
                    original_measurements = ftl.measurements
                    ftl.measurements = compensated_measurements

                # Run optimization
                if ftl.config.use_adaptive_lm:
                    ftl.step_with_lm()
                else:
                    ftl.step_basic()

                # Restore measurements
                if config.enable_cfo_compensation:
                    ftl.measurements = original_measurements

                # Calculate RMSE
                errors = []
                for i in range(ftl.config.n_anchors, ftl.config.n_nodes):
                    err = np.linalg.norm(ftl.states[i, :2] - ftl.true_positions[i])
                    errors.append(err)
                rmse = np.sqrt(np.mean(np.array(errors)**2))
                rmse_history.append(rmse * 1000)

                iteration += 1

            # Record result at this time point
            results[mode][target_time] = rmse_history[-1]
            print(f"  t={target_time:4d}s: RMSE = {rmse_history[-1]:.2f} mm")

    # Calculate theoretical drift
    print("\n" + "="*60)
    print("THEORETICAL ANALYSIS")
    print("="*60)

    c = 299792458.0  # m/s
    freq_offset_ppb = 10  # ppb
    print(f"Frequency offset: {freq_offset_ppb} ppb")
    print(f"Speed of light: {c/1e6:.1f} m/μs")

    print("\nExpected range error without compensation:")
    for t in time_points:
        # Range error = c × freq_offset × time
        range_error = c * freq_offset_ppb * 1e-9 * t
        print(f"  t={t:4d}s: {range_error*1000:.2f} mm")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # RMSE over time comparison
    ax = axes[0, 0]
    times = list(results['with'].keys())
    rmse_with = list(results['with'].values())
    rmse_without = list(results['without'].values())

    ax.semilogy(times, rmse_with, 'bo-', label='With CFO compensation', linewidth=2, markersize=8)
    ax.semilogy(times, rmse_without, 'rs--', label='Without compensation', linewidth=2, markersize=8)
    ax.axhline(y=18.5, color='g', linestyle=':', label='Target (18.5mm)')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('RMSE (mm)')
    ax.set_title('Long-term Stability: Position RMSE vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Drift accumulation
    ax = axes[0, 1]
    theoretical_drift = [c * freq_offset_ppb * 1e-9 * t * 1000 for t in times]  # mm
    actual_drift = [rmse_without[i] - rmse_with[i] for i in range(len(times))]

    ax.plot(times, theoretical_drift, 'k-', label='Theoretical drift', linewidth=2)
    ax.plot(times, actual_drift, 'mo--', label='Observed drift', linewidth=2, markersize=8)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Drift (mm)')
    ax.set_title('Clock Drift Accumulation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Improvement percentage
    ax = axes[1, 0]
    improvement = [(rmse_without[i] - rmse_with[i]) / rmse_without[i] * 100
                  for i in range(len(times))]
    ax.plot(times, improvement, 'go-', linewidth=2, markersize=8)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Improvement (%)')
    ax.set_title('Relative Improvement with CFO Compensation')
    ax.grid(True, alpha=0.3)

    # Time vs frequency accuracy trade-off
    ax = axes[1, 1]
    ppb_values = [1, 5, 10, 20, 50]
    times_extended = np.logspace(0, 3, 50)  # 1 to 1000 seconds

    for ppb in ppb_values:
        drift = c * ppb * 1e-9 * times_extended * 1000  # mm
        ax.loglog(times_extended, drift, label=f'{ppb} ppb')

    ax.axhline(y=10, color='r', linestyle='--', alpha=0.5, label='10mm threshold')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Range Error (mm)')
    ax.set_title('Drift vs Time for Different Frequency Offsets')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Long-term Stability with IQ-based Frequency Compensation',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('long_term_frequency_stability.png', dpi=150)
    print("\nPlot saved as long_term_frequency_stability.png")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    final_with = results['with'][time_points[-1]]
    final_without = results['without'][time_points[-1]]

    print(f"After {time_points[-1]} seconds:")
    print(f"  Without compensation: {final_without:.2f} mm")
    print(f"  With IQ compensation: {final_with:.2f} mm")
    print(f"  Improvement: {(final_without - final_with) / final_without * 100:.1f}%")

    if final_with < 50:  # 50mm threshold
        print("\n✅ System maintains sub-50mm accuracy over extended time!")
    else:
        print("\n⚠️ Accuracy degrades over time - consider more frequent CFO updates")

    return results


if __name__ == "__main__":
    results = test_long_term_stability()