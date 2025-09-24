#!/usr/bin/env python3
"""
Unit tests for the comprehensive noise model
Validates each noise source and statistical properties
"""

import numpy as np
import matplotlib.pyplot as plt
from ftl.noise_model import (
    NoiseConfig, NoiseGenerator, NoisePreset,
    create_preset_config, get_cramer_rao_bound,
    ThermalNoiseConfig, MultipathConfig, FrequencyOffsetConfig
)
import sys
import os


def test_noise_presets():
    """Test that presets configure noise correctly"""
    print("="*60)
    print("Testing Noise Presets")
    print("="*60)

    # Test IDEAL preset (no noise)
    config_ideal = create_preset_config("ideal")
    assert not config_ideal.enable_noise, "IDEAL preset should disable all noise"
    print("✓ IDEAL preset: noise disabled")

    # Test CLEAN preset (minimal noise)
    config_clean = create_preset_config("clean")
    assert config_clean.enable_noise, "CLEAN preset should enable noise"
    assert config_clean.thermal.snr_db == 30.0, "CLEAN should have high SNR"
    assert not config_clean.multipath.enabled, "CLEAN should disable multipath"
    print("✓ CLEAN preset: minimal noise configuration")

    # Test REALISTIC preset
    config_realistic = create_preset_config("realistic")
    assert config_realistic.thermal.snr_db == 20.0, "REALISTIC should have moderate SNR"
    assert config_realistic.multipath.nlos_probability == 0.1, "REALISTIC should have 10% NLOS"
    print("✓ REALISTIC preset: typical UWB configuration")

    # Test HARSH preset
    config_harsh = create_preset_config("harsh")
    assert config_harsh.thermal.snr_db == 10.0, "HARSH should have low SNR"
    assert config_harsh.multipath.nlos_probability == 0.3, "HARSH should have 30% NLOS"
    assert config_harsh.interference.enabled, "HARSH should enable interference"
    print("✓ HARSH preset: challenging environment configuration")

    print("\nAll preset tests passed!\n")


def test_thermal_noise():
    """Test thermal noise statistics"""
    print("="*60)
    print("Testing Thermal Noise")
    print("="*60)

    config = NoiseConfig(preset=None, enable_noise=True, random_seed=42)
    # Enable only thermal noise
    config.thermal.enabled = True
    config.thermal.snr_db = 20.0
    config.distance_dependent.enabled = False
    config.quantization.enabled = False
    config.clock_jitter.enabled = False
    config.multipath.enabled = False
    config.shadowing.enabled = False

    generator = NoiseGenerator(config)

    # Generate multiple samples
    n_samples = 10000
    true_range = 10.0
    samples = []

    for _ in range(n_samples):
        noisy_range, std = generator.add_measurement_noise(true_range, 0, 1)
        samples.append(noisy_range - true_range)  # Get noise only

    samples = np.array(samples)

    # Check statistics
    mean_noise = np.mean(samples)
    std_noise = np.std(samples)

    print(f"Thermal noise statistics (SNR={config.thermal.snr_db}dB):")
    print(f"  Mean: {mean_noise*1000:.3f}mm (expected: ~0mm)")
    print(f"  Std:  {std_noise*1000:.3f}mm")

    # Statistical tests (allow for sampling variation)
    assert abs(mean_noise) < 0.005, f"Mean noise should be near zero, got {mean_noise}"
    assert 0.001 < std_noise < 0.1, f"Std should be reasonable for thermal noise, got {std_noise}"

    print("✓ Thermal noise statistics validated\n")


def test_distance_dependent_noise():
    """Test distance-dependent noise scaling"""
    print("="*60)
    print("Testing Distance-Dependent Noise")
    print("="*60)

    config = NoiseConfig(preset=None, enable_noise=True, random_seed=42)
    # Enable only distance-dependent noise
    config.thermal.enabled = False
    config.distance_dependent.enabled = True
    config.distance_dependent.coefficient = 0.001
    config.quantization.enabled = False
    config.clock_jitter.enabled = False
    config.multipath.enabled = False
    config.shadowing.enabled = False

    generator = NoiseGenerator(config)

    distances = [5, 10, 20, 50]
    results = []

    for dist in distances:
        n_samples = 1000
        noise_samples = []

        for _ in range(n_samples):
            noisy_range, std = generator.add_measurement_noise(dist, 0, 1)
            noise_samples.append(noisy_range - dist)

        std_measured = np.std(noise_samples)
        std_expected = config.distance_dependent.coefficient * np.sqrt(dist)
        std_expected = max(std_expected, config.distance_dependent.min_std_m)

        results.append((dist, std_measured, std_expected))
        print(f"  Distance {dist:3.0f}m: std={std_measured*1000:.2f}mm "
              f"(expected: {std_expected*1000:.2f}mm)")

        # Allow 20% tolerance
        assert abs(std_measured - std_expected) / std_expected < 0.2, \
               f"Distance-dependent noise not scaling correctly at {dist}m"

    print("✓ Distance-dependent noise scaling validated\n")


def test_multipath_effects():
    """Test multipath/NLOS effects"""
    print("="*60)
    print("Testing Multipath/NLOS Effects")
    print("="*60)

    config = NoiseConfig(preset=None, enable_noise=True, random_seed=42)
    # Enable only multipath
    config.thermal.enabled = False
    config.distance_dependent.enabled = False
    config.multipath.enabled = True
    config.multipath.nlos_probability = 0.2  # 20% NLOS
    config.multipath.bias_range_m = (0.0, 1.0)
    config.shadowing.enabled = False

    generator = NoiseGenerator(config)

    n_samples = 10000
    true_range = 10.0
    samples = []

    for _ in range(n_samples):
        noisy_range, std = generator.add_measurement_noise(true_range, 0, 1)
        samples.append(noisy_range - true_range)

    samples = np.array(samples)

    # Check for positive bias
    mean_bias = np.mean(samples)
    los_samples = samples[samples < 0.1]  # Approximate LOS samples
    nlos_samples = samples[samples >= 0.1]  # Approximate NLOS samples

    nlos_ratio = len(nlos_samples) / len(samples)

    print(f"Multipath statistics:")
    print(f"  Mean bias: {mean_bias*1000:.2f}mm (should be positive)")
    print(f"  NLOS ratio: {nlos_ratio:.2%} (expected: ~20%)")
    print(f"  LOS samples: {len(los_samples)}")
    print(f"  NLOS samples: {len(nlos_samples)}")

    assert mean_bias > 0, "Multipath should create positive bias"
    assert 0.15 < nlos_ratio < 0.25, f"NLOS ratio should be ~20%, got {nlos_ratio:.1%}"

    print("✓ Multipath/NLOS effects validated\n")


def test_frequency_offset():
    """Test frequency offset effects"""
    print("="*60)
    print("Testing Frequency Offset")
    print("="*60)

    config = NoiseConfig(preset=None, enable_noise=True, random_seed=42)
    # Enable only frequency offset
    config.thermal.enabled = False
    config.distance_dependent.enabled = False
    config.multipath.enabled = False
    config.frequency_offset.enabled = True
    config.frequency_offset.max_offset_ppb = 20.0
    config.shadowing.enabled = False

    generator = NoiseGenerator(config)

    # Test that frequency offset creates consistent bias between same node pairs
    distances = [10, 20, 50]
    node_pairs = [(0, 1), (0, 2), (1, 2)]

    print("Frequency offset effects:")
    valid_pairs = 0
    for i, j in node_pairs:
        biases = []
        for dist in distances:
            noisy_range, _ = generator.add_measurement_noise(dist, i, j)
            bias = noisy_range - dist
            biases.append(bias)

        # Check that bias scales with distance (frequency offset effect)
        bias_range = max(biases) - min(biases)
        if bias_range > 1e-6:  # Only test if there's significant bias
            correlation = np.corrcoef(distances, biases)[0, 1]
            print(f"  Nodes {i}-{j}: bias correlation with distance = {correlation:.3f}")
            # Frequency offset creates linear scaling with distance
            if abs(correlation) > 0.8:
                valid_pairs += 1
        else:
            print(f"  Nodes {i}-{j}: minimal bias (similar frequencies)")

    assert valid_pairs >= 2, "At least 2 node pairs should show frequency offset scaling"

    print("✓ Frequency offset effects validated\n")


def test_noise_combination():
    """Test that multiple noise sources combine correctly"""
    print("="*60)
    print("Testing Combined Noise Sources")
    print("="*60)

    config = NoiseConfig(preset=NoisePreset.REALISTIC, random_seed=42)
    generator = NoiseGenerator(config)

    n_samples = 1000
    true_range = 20.0
    samples = []
    stds = []

    for _ in range(n_samples):
        noisy_range, std = generator.add_measurement_noise(true_range, 0, 1)
        samples.append(noisy_range)
        stds.append(std)

    samples = np.array(samples)
    mean_range = np.mean(samples)
    std_range = np.std(samples)
    mean_reported_std = np.mean(stds)

    print(f"Combined noise statistics (REALISTIC preset):")
    print(f"  True range: {true_range:.1f}m")
    print(f"  Mean measured: {mean_range:.3f}m")
    print(f"  Std measured: {std_range*1000:.2f}mm")
    print(f"  Mean reported std: {mean_reported_std*1000:.2f}mm")

    # Basic sanity checks
    assert abs(mean_range - true_range) < 0.5, "Mean should be close to true range"
    # With all noise sources, std can be higher (up to 1m for realistic preset)
    assert 0.005 < std_range < 1.0, f"Combined std should be reasonable, got {std_range}"

    print("✓ Combined noise sources validated\n")


def test_cramer_rao_bound():
    """Test CRLB computation"""
    print("="*60)
    print("Testing Cramér-Rao Lower Bound")
    print("="*60)

    distances = [10, 15, 20, 25, 30]
    stds = [0.01, 0.015, 0.02, 0.025, 0.03]
    n_anchors = 5

    crlb = get_cramer_rao_bound(distances, stds, n_anchors)

    print(f"CRLB for {n_anchors} anchors:")
    print(f"  Mean distance: {np.mean(distances):.1f}m")
    print(f"  Mean std: {np.mean(stds)*1000:.1f}mm")
    print(f"  CRLB: {crlb*1000:.2f}mm")

    assert 0 < crlb < 0.1, f"CRLB should be reasonable, got {crlb}"

    print("✓ CRLB computation validated\n")


def test_reproducibility():
    """Test that noise generation is reproducible with seed"""
    print("="*60)
    print("Testing Reproducibility")
    print("="*60)

    config1 = NoiseConfig(preset=NoisePreset.REALISTIC, random_seed=123)
    config2 = NoiseConfig(preset=NoisePreset.REALISTIC, random_seed=123)
    config3 = NoiseConfig(preset=NoisePreset.REALISTIC, random_seed=456)

    gen1 = NoiseGenerator(config1)
    gen2 = NoiseGenerator(config2)
    gen3 = NoiseGenerator(config3)

    # Generate samples with same seed
    ranges1 = []
    ranges2 = []
    ranges3 = []

    for i in range(10):
        r1, _ = gen1.add_measurement_noise(10.0, 0, 1)
        r2, _ = gen2.add_measurement_noise(10.0, 0, 1)
        r3, _ = gen3.add_measurement_noise(10.0, 0, 1)

        ranges1.append(r1)
        ranges2.append(r2)
        ranges3.append(r3)

    # Check reproducibility
    assert np.allclose(ranges1, ranges2), "Same seed should produce same results"
    assert not np.allclose(ranges1, ranges3), "Different seeds should produce different results"

    print("✓ Reproducibility with random seed validated\n")


def visualize_noise_distributions():
    """Create visualizations of noise distributions"""
    print("="*60)
    print("Generating Noise Distribution Visualizations")
    print("="*60)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Noise Model Distributions", fontsize=16)

    true_range = 20.0
    n_samples = 5000

    # Test different presets
    presets = ["ideal", "clean", "realistic", "harsh"]
    colors = ["green", "blue", "orange", "red"]

    for preset, color in zip(presets, colors):
        config = create_preset_config(preset)
        generator = NoiseGenerator(config)

        samples = []
        for _ in range(n_samples):
            noisy_range, _ = generator.add_measurement_noise(true_range, 0, 1)
            samples.append(noisy_range - true_range)

        samples = np.array(samples) * 1000  # Convert to mm

        # Plot histogram
        axes[0, 0].hist(samples, bins=50, alpha=0.5, label=preset.upper(), color=color)

    axes[0, 0].set_xlabel("Range Error (mm)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Noise Distribution by Preset")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Distance-dependent noise
    config = NoiseConfig(preset=None, enable_noise=True)
    config.thermal.enabled = False
    config.distance_dependent.enabled = True
    config.multipath.enabled = False
    config.shadowing.enabled = False

    generator = NoiseGenerator(config)

    distances = np.linspace(1, 50, 20)
    stds = []

    for dist in distances:
        noise_samples = []
        for _ in range(100):
            noisy_range, std = generator.add_measurement_noise(dist, 0, 1)
            noise_samples.append(noisy_range - dist)
        stds.append(np.std(noise_samples) * 1000)

    axes[0, 1].plot(distances, stds, 'b-', linewidth=2)
    axes[0, 1].set_xlabel("Distance (m)")
    axes[0, 1].set_ylabel("Noise Std (mm)")
    axes[0, 1].set_title("Distance-Dependent Noise")
    axes[0, 1].grid(True, alpha=0.3)

    # Multipath effects
    config = NoiseConfig(preset=None, enable_noise=True)
    config.thermal.enabled = False
    config.multipath.enabled = True
    config.multipath.nlos_probability = 0.2
    config.distance_dependent.enabled = False
    config.shadowing.enabled = False

    generator = NoiseGenerator(config)

    samples = []
    for _ in range(n_samples):
        noisy_range, _ = generator.add_measurement_noise(true_range, 0, 1)
        samples.append(noisy_range - true_range)

    samples = np.array(samples) * 1000

    axes[0, 2].hist(samples, bins=50, color='orange', alpha=0.7)
    axes[0, 2].set_xlabel("Range Error (mm)")
    axes[0, 2].set_ylabel("Count")
    axes[0, 2].set_title("Multipath/NLOS Effects (20% NLOS)")
    axes[0, 2].axvline(x=0, color='red', linestyle='--', label='Zero bias')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # SNR effect
    snr_values = [10, 15, 20, 25, 30]
    for snr in snr_values:
        config = NoiseConfig(preset=None, enable_noise=True)
        config.thermal.enabled = True
        config.thermal.snr_db = snr
        config.distance_dependent.enabled = False
        config.multipath.enabled = False
        config.shadowing.enabled = False

        generator = NoiseGenerator(config)

        samples = []
        for _ in range(n_samples):
            noisy_range, _ = generator.add_measurement_noise(true_range, 0, 1)
            samples.append(noisy_range - true_range)

        samples = np.array(samples) * 1000

        axes[1, 0].hist(samples, bins=30, alpha=0.5, label=f"SNR={snr}dB")

    axes[1, 0].set_xlabel("Range Error (mm)")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Thermal Noise vs SNR")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Combined realistic noise over time
    config = create_preset_config("realistic")
    generator = NoiseGenerator(config)

    time_steps = 100
    nodes = 5
    range_errors = np.zeros((time_steps, nodes))

    for t in range(time_steps):
        for n in range(nodes):
            noisy_range, _ = generator.add_measurement_noise(true_range, 0, n, timestamp=t*0.1)
            range_errors[t, n] = (noisy_range - true_range) * 1000

    for n in range(nodes):
        axes[1, 1].plot(range_errors[:, n], alpha=0.7, label=f"Node {n}")

    axes[1, 1].set_xlabel("Time Step")
    axes[1, 1].set_ylabel("Range Error (mm)")
    axes[1, 1].set_title("Realistic Noise Over Time")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # CRLB vs measurement noise
    n_anchors = [3, 4, 5, 6, 8]
    noise_levels = np.linspace(0.001, 0.05, 20)

    for n_a in n_anchors:
        crlbs = []
        for noise_std in noise_levels:
            distances = [20] * n_a
            stds = [noise_std] * n_a
            crlb = get_cramer_rao_bound(distances, stds, n_a)
            crlbs.append(crlb * 1000)

        axes[1, 2].plot(noise_levels * 1000, crlbs, label=f"{n_a} anchors")

    axes[1, 2].set_xlabel("Measurement Noise Std (mm)")
    axes[1, 2].set_ylabel("CRLB (mm)")
    axes[1, 2].set_title("Cramér-Rao Lower Bound")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("noise_distributions.png", dpi=150)
    print("✓ Saved visualization to noise_distributions.png\n")


def main():
    """Run all noise model tests"""
    print("\n" + "="*60)
    print("NOISE MODEL VALIDATION TEST SUITE")
    print("="*60 + "\n")

    try:
        # Run all tests
        test_noise_presets()
        test_thermal_noise()
        test_distance_dependent_noise()
        test_multipath_effects()
        test_frequency_offset()
        test_noise_combination()
        test_cramer_rao_bound()
        test_reproducibility()

        # Generate visualizations
        visualize_noise_distributions()

        print("="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nThe noise model is working correctly with:")
        print("  - All presets configured properly")
        print("  - Individual noise sources validated")
        print("  - Statistical properties confirmed")
        print("  - Reproducibility verified")
        print("  - Visualizations generated")

        return True

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)