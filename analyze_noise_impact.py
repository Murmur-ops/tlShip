#!/usr/bin/env python3
"""
Analyze the impact of different noise sources on TL system performance
Performs parameter sweeps and generates comprehensive visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from test_30node_system import Large30NodeFTL
from ftl_enhanced import EnhancedFTLConfig
from ftl.noise_model import NoiseConfig, NoisePreset, create_preset_config
import json
from typing import Dict, List, Tuple
import time


def run_single_test(config: EnhancedFTLConfig, noise_config: NoiseConfig) -> Dict:
    """Run a single test with given configuration

    Returns:
        Dictionary with results (position_rmse_mm, time_rmse_ns, iterations, runtime)
    """
    start_time = time.time()

    ftl = Large30NodeFTL(config, noise_config)
    ftl.run()

    runtime = time.time() - start_time

    return {
        'position_rmse_mm': ftl.position_rmse_history[-1] * 1000 if ftl.position_rmse_history else 999000,
        'time_rmse_ns': ftl.time_rmse_history[-1] if ftl.time_rmse_history else 999,
        'iterations': len(ftl.position_rmse_history),
        'runtime_s': runtime,
        'converged': ftl.position_rmse_history[-1] < 0.1 if ftl.position_rmse_history else False
    }


def sweep_snr_impact():
    """Analyze impact of SNR on performance"""
    print("\n" + "="*60)
    print("SNR IMPACT ANALYSIS")
    print("="*60)

    snr_values = [5, 10, 15, 20, 25, 30, 35, 40]
    results = []

    config = EnhancedFTLConfig(
        n_nodes=30,
        n_anchors=5,
        use_adaptive_lm=True,
        use_line_search=False,
        max_iterations=100,
        verbose=False
    )

    for snr in snr_values:
        print(f"Testing SNR = {snr}dB...")

        # Create noise config with only thermal noise
        noise_config = NoiseConfig(preset=None, enable_noise=True, random_seed=42)
        noise_config.thermal.enabled = True
        noise_config.thermal.snr_db = snr
        # Disable other noise sources
        noise_config.distance_dependent.enabled = False
        noise_config.multipath.enabled = False
        noise_config.shadowing.enabled = False

        result = run_single_test(config, noise_config)
        result['snr_db'] = snr
        results.append(result)

        print(f"  Position RMSE: {result['position_rmse_mm']:.2f}mm")

    return results


def sweep_multipath_impact():
    """Analyze impact of multipath/NLOS on performance"""
    print("\n" + "="*60)
    print("MULTIPATH/NLOS IMPACT ANALYSIS")
    print("="*60)

    nlos_probabilities = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
    results = []

    config = EnhancedFTLConfig(
        n_nodes=30,
        n_anchors=5,
        use_adaptive_lm=True,
        use_line_search=False,
        max_iterations=100,
        verbose=False
    )

    for nlos_prob in nlos_probabilities:
        print(f"Testing NLOS probability = {nlos_prob:.1%}...")

        noise_config = NoiseConfig(preset=None, enable_noise=True, random_seed=42)
        # Enable only multipath
        noise_config.multipath.enabled = True
        noise_config.multipath.nlos_probability = nlos_prob
        noise_config.multipath.bias_range_m = (0.0, 0.5)
        # Disable other sources
        noise_config.thermal.enabled = False
        noise_config.distance_dependent.enabled = False
        noise_config.shadowing.enabled = False

        result = run_single_test(config, noise_config)
        result['nlos_probability'] = nlos_prob
        results.append(result)

        print(f"  Position RMSE: {result['position_rmse_mm']:.2f}mm")

    return results


def sweep_frequency_offset_impact():
    """Analyze impact of frequency offset on performance"""
    print("\n" + "="*60)
    print("FREQUENCY OFFSET IMPACT ANALYSIS")
    print("="*60)

    offset_ppb_values = [0, 5, 10, 20, 50, 100]
    results = []

    config = EnhancedFTLConfig(
        n_nodes=30,
        n_anchors=5,
        use_adaptive_lm=True,
        use_line_search=False,
        max_iterations=100,
        verbose=False
    )

    for offset_ppb in offset_ppb_values:
        print(f"Testing frequency offset = {offset_ppb}ppb...")

        noise_config = NoiseConfig(preset=None, enable_noise=True, random_seed=42)
        # Enable only frequency offset
        noise_config.frequency_offset.enabled = True
        noise_config.frequency_offset.max_offset_ppb = offset_ppb
        # Disable other sources
        noise_config.thermal.enabled = False
        noise_config.multipath.enabled = False
        noise_config.distance_dependent.enabled = False
        noise_config.shadowing.enabled = False

        result = run_single_test(config, noise_config)
        result['frequency_offset_ppb'] = offset_ppb
        results.append(result)

        print(f"  Position RMSE: {result['position_rmse_mm']:.2f}mm")

    return results


def sweep_combined_noise():
    """Analyze combined effect of multiple noise sources"""
    print("\n" + "="*60)
    print("COMBINED NOISE ANALYSIS")
    print("="*60)

    # Test different combinations
    combinations = [
        {'name': 'None', 'thermal': False, 'multipath': False, 'freq': False},
        {'name': 'Thermal Only', 'thermal': True, 'multipath': False, 'freq': False},
        {'name': 'Multipath Only', 'thermal': False, 'multipath': True, 'freq': False},
        {'name': 'Thermal+Multipath', 'thermal': True, 'multipath': True, 'freq': False},
        {'name': 'All Sources', 'thermal': True, 'multipath': True, 'freq': True},
    ]

    results = []

    config = EnhancedFTLConfig(
        n_nodes=30,
        n_anchors=5,
        use_adaptive_lm=True,
        use_line_search=False,
        max_iterations=100,
        verbose=False
    )

    for combo in combinations:
        print(f"Testing {combo['name']}...")

        noise_config = NoiseConfig(preset=None, enable_noise=True, random_seed=42)

        # Configure noise sources
        noise_config.thermal.enabled = combo['thermal']
        noise_config.thermal.snr_db = 20.0

        noise_config.multipath.enabled = combo['multipath']
        noise_config.multipath.nlos_probability = 0.1

        noise_config.frequency_offset.enabled = combo['freq']
        noise_config.frequency_offset.max_offset_ppb = 10.0

        # Always enable distance-dependent as baseline
        noise_config.distance_dependent.enabled = True
        noise_config.distance_dependent.coefficient = 0.001

        # Disable others
        noise_config.shadowing.enabled = False
        noise_config.clock_jitter.enabled = False

        result = run_single_test(config, noise_config)
        result['combination'] = combo['name']
        results.append(result)

        print(f"  Position RMSE: {result['position_rmse_mm']:.2f}mm")

    return results


def create_comprehensive_plots(snr_results, multipath_results, freq_results, combo_results):
    """Create comprehensive visualization of noise impact"""
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Comprehensive Noise Impact Analysis - 30-Node TL System", fontsize=16)

    # 1. SNR Impact
    ax = axes[0, 0]
    snr_values = [r['snr_db'] for r in snr_results]
    pos_rmse = [r['position_rmse_mm'] for r in snr_results]
    ax.plot(snr_values, pos_rmse, 'b-o', linewidth=2, markersize=8)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Position RMSE (mm)")
    ax.set_title("Impact of Thermal Noise (SNR)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    # Add reference line for target
    ax.axhline(y=20, color='g', linestyle='--', alpha=0.5, label='20mm target')
    ax.legend()

    # 2. Multipath Impact
    ax = axes[0, 1]
    nlos_prob = [r['nlos_probability'] * 100 for r in multipath_results]
    pos_rmse = [r['position_rmse_mm'] for r in multipath_results]
    ax.plot(nlos_prob, pos_rmse, 'r-s', linewidth=2, markersize=8)
    ax.set_xlabel("NLOS Probability (%)")
    ax.set_ylabel("Position RMSE (mm)")
    ax.set_title("Impact of Multipath/NLOS")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.axhline(y=20, color='g', linestyle='--', alpha=0.5, label='20mm target')
    ax.legend()

    # 3. Frequency Offset Impact
    ax = axes[0, 2]
    freq_offset = [r['frequency_offset_ppb'] for r in freq_results]
    pos_rmse = [r['position_rmse_mm'] for r in freq_results]
    ax.plot(freq_offset, pos_rmse, 'g-^', linewidth=2, markersize=8)
    ax.set_xlabel("Frequency Offset (ppb)")
    ax.set_ylabel("Position RMSE (mm)")
    ax.set_title("Impact of Crystal Frequency Offset")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.axhline(y=20, color='g', linestyle='--', alpha=0.5, label='20mm target')
    ax.legend()

    # 4. Combined Effects
    ax = axes[1, 0]
    combo_names = [r['combination'] for r in combo_results]
    pos_rmse = [r['position_rmse_mm'] for r in combo_results]
    colors = ['green', 'blue', 'orange', 'red', 'purple']
    bars = ax.bar(range(len(combo_names)), pos_rmse, color=colors, alpha=0.7)
    ax.set_xticks(range(len(combo_names)))
    ax.set_xticklabels(combo_names, rotation=45, ha='right')
    ax.set_ylabel("Position RMSE (mm)")
    ax.set_title("Combined Noise Sources")
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=20, color='g', linestyle='--', alpha=0.5, label='20mm target')
    ax.legend()

    # 5. Runtime Analysis
    ax = axes[1, 1]
    # Compare runtime across SNR sweep
    snr_values = [r['snr_db'] for r in snr_results]
    runtime = [r['runtime_s'] for r in snr_results]
    ax.plot(snr_values, runtime, 'b-o', linewidth=2)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Computational Cost vs Noise Level")
    ax.grid(True, alpha=0.3)

    # 6. Convergence Analysis
    ax = axes[1, 2]
    # Compare convergence across different noise levels
    presets = ['ideal', 'clean', 'realistic', 'harsh']
    preset_results = []

    for preset in presets:
        noise_config = create_preset_config(preset)
        noise_config.random_seed = 42

        config = EnhancedFTLConfig(
            n_nodes=30,
            n_anchors=5,
            use_adaptive_lm=True,
            use_line_search=False,
            max_iterations=100,
            verbose=False
        )

        ftl = Large30NodeFTL(config, noise_config)
        ftl.run()

        if ftl.position_rmse_history:
            ax.semilogy(np.array(ftl.position_rmse_history) * 1000,
                       label=preset.upper(), linewidth=2, alpha=0.8)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Position RMSE (mm)")
    ax.set_title("Convergence Behavior by Preset")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axhline(y=20, color='g', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("noise_impact_analysis.png", dpi=150)
    print("✓ Saved comprehensive analysis to noise_impact_analysis.png")


def save_results_json(snr_results, multipath_results, freq_results, combo_results):
    """Save results to JSON for further analysis"""
    all_results = {
        'snr_sweep': snr_results,
        'multipath_sweep': multipath_results,
        'frequency_offset_sweep': freq_results,
        'combined_effects': combo_results,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open('noise_impact_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("✓ Saved results to noise_impact_results.json")


def print_summary_table(snr_results, multipath_results, freq_results):
    """Print summary table of key findings"""
    print("\n" + "="*60)
    print("SUMMARY OF KEY FINDINGS")
    print("="*60)

    print("\n1. SNR Impact (Thermal Noise):")
    print(f"{'SNR (dB)':<12} {'RMSE (mm)':<12} {'Status':<20}")
    print("-"*44)
    for r in snr_results[::2]:  # Show every other result for brevity
        status = "✓ Meets target" if r['position_rmse_mm'] <= 20 else f"× {r['position_rmse_mm']/20:.1f}x target"
        print(f"{r['snr_db']:<12} {r['position_rmse_mm']:<12.2f} {status:<20}")

    print("\n2. Multipath/NLOS Impact:")
    print(f"{'NLOS %':<12} {'RMSE (mm)':<12} {'Status':<20}")
    print("-"*44)
    for r in multipath_results[::2]:
        status = "✓ Meets target" if r['position_rmse_mm'] <= 20 else f"× {r['position_rmse_mm']/20:.1f}x target"
        print(f"{r['nlos_probability']*100:<12.1f} {r['position_rmse_mm']:<12.2f} {status:<20}")

    print("\n3. Frequency Offset Impact:")
    print(f"{'Offset (ppb)':<12} {'RMSE (mm)':<12} {'Status':<20}")
    print("-"*44)
    for r in freq_results:
        status = "✓ Meets target" if r['position_rmse_mm'] <= 20 else f"× {r['position_rmse_mm']/20:.1f}x target"
        print(f"{r['frequency_offset_ppb']:<12} {r['position_rmse_mm']:<12.2f} {status:<20}")


def main():
    """Run comprehensive noise impact analysis"""
    print("\n" + "="*60)
    print("TL SYSTEM NOISE IMPACT ANALYSIS")
    print("30-node network, 5 anchors, 50x50m area")
    print("="*60)

    start_time = time.time()

    # Run parameter sweeps
    snr_results = sweep_snr_impact()
    multipath_results = sweep_multipath_impact()
    freq_results = sweep_frequency_offset_impact()
    combo_results = sweep_combined_noise()

    # Create visualizations
    create_comprehensive_plots(snr_results, multipath_results, freq_results, combo_results)

    # Save results
    save_results_json(snr_results, multipath_results, freq_results, combo_results)

    # Print summary
    print_summary_table(snr_results, multipath_results, freq_results)

    total_time = time.time() - start_time
    print(f"\n✅ Analysis complete in {total_time:.1f} seconds")
    print(f"   Generated: noise_impact_analysis.png")
    print(f"   Saved data: noise_impact_results.json")

    return {
        'snr': snr_results,
        'multipath': multipath_results,
        'frequency': freq_results,
        'combined': combo_results
    }


if __name__ == "__main__":
    results = main()