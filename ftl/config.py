"""
Configuration Management for FTL Simulation
Load and validate YAML configuration files
"""

import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class FTLConfig:
    """Complete FTL simulation configuration"""

    # Geometry settings
    geometry_type: str = 'grid'
    n_nodes: int = 25
    area_size: float = 100.0
    jitter_std: float = 0.0
    n_anchors: int = 4
    anchor_placement: str = 'corner'

    # Clock settings (unknown nodes)
    unknown_oscillator: str = 'TCXO'
    unknown_freq_accuracy_ppm: float = 2.0
    unknown_allan_deviation: float = 1e-10
    unknown_bias_std: float = 1e-6
    unknown_drift_std: float = 1e-9
    unknown_cfo_std: float = 10.0

    # Clock settings (anchors)
    anchor_oscillator: str = 'OCXO'
    anchor_freq_accuracy_ppm: float = 0.1
    anchor_allan_deviation: float = 1e-11
    anchor_bias_std: float = 1e-9
    anchor_drift_std: float = 1e-12
    anchor_cfo_std: float = 1.0

    # Signal settings
    signal_type: str = 'hrp_uwb'
    carrier_freq: float = 6.5e9
    bandwidth: float = 499.2e6
    sample_rate: float = 1e9
    burst_duration: float = 1e-6
    prf: float = 124.8e6
    sequence_length: int = 127
    cyclic_prefix_length: int = 32
    snr_db: float = 20.0

    # Channel settings
    environment: str = 'indoor'
    cluster_arrival_rate: float = 0.0233
    ray_arrival_rate: float = 0.4
    cluster_decay_factor: float = 10.0
    ray_decay_factor: float = 5.0
    k_factor_db: float = 6.0
    nlos_excess_delay_mean_ns: float = 25.0
    nlos_excess_delay_std_ns: float = 10.0
    path_loss_exponent: float = 2.0
    shadowing_std_db: float = 4.0

    # Receiver settings
    toa_detection_mode: str = 'peak'
    toa_threshold: float = 0.5
    enable_subsample_refinement: bool = True
    cfo_estimation_method: str = 'ml'
    nlos_classification_enabled: bool = True

    # Factor graph settings
    toa_weight: float = 1.0
    tdoa_weight: float = 0.5
    twr_weight: float = 2.0
    cfo_weight: float = 0.1
    prior_weight: float = 0.01
    toa_variance: float = 1e-18
    tdoa_variance: float = 1e-18
    twr_variance: float = 0.01
    cfo_variance: float = 1.0

    # Robust settings
    use_huber: bool = True
    huber_delta: float = 1.0
    use_dcs: bool = False
    dcs_phi: float = 1.0
    use_switchable: bool = False
    outlier_threshold: float = 3.0

    # Solver settings
    initialization_method: str = 'trilateration'
    max_iterations: int = 100
    convergence_tolerance: float = 1e-6
    lambda_init: float = 1e-3
    verbose: bool = True

    # Simulation settings
    n_rounds: int = 10
    round_interval: float = 0.1
    max_range: float = 150.0
    los_probability: float = 0.8
    measurement_noise_scale: float = 1.0
    enable_multipath: bool = True
    enable_clock_drift: bool = True

    # Output settings
    save_results: bool = True
    output_dir: str = './results'
    save_plots: bool = True
    plot_format: str = 'png'
    save_trajectories: bool = True
    save_raw_measurements: bool = False
    metrics_to_compute: list = field(default_factory=lambda: [
        'position_rmse', 'position_mae', 'clock_bias_mae',
        'clock_drift_mae', 'cfo_rmse', 'crlb_efficiency'
    ])


def load_config(config_path: str) -> FTLConfig:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to YAML configuration file

    Returns:
        FTLConfig object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    return parse_config(yaml_config)


def parse_config(yaml_config: Dict[str, Any]) -> FTLConfig:
    """
    Parse YAML configuration into FTLConfig object

    Args:
        yaml_config: Dictionary from YAML file

    Returns:
        FTLConfig object
    """
    config = FTLConfig()

    # Parse geometry section
    if 'geometry' in yaml_config:
        geo = yaml_config['geometry']
        config.geometry_type = geo.get('type', config.geometry_type)
        config.n_nodes = geo.get('n_nodes', config.n_nodes)
        config.area_size = geo.get('area_size', config.area_size)
        config.jitter_std = geo.get('jitter_std', config.jitter_std)
        config.n_anchors = geo.get('n_anchors', config.n_anchors)
        config.anchor_placement = geo.get('anchor_placement', config.anchor_placement)

    # Parse clocks section
    if 'clocks' in yaml_config:
        clocks = yaml_config['clocks']

        if 'unknown_nodes' in clocks:
            unk = clocks['unknown_nodes']
            config.unknown_oscillator = unk.get('oscillator_type', config.unknown_oscillator)
            config.unknown_freq_accuracy_ppm = unk.get('frequency_accuracy_ppm', config.unknown_freq_accuracy_ppm)
            config.unknown_allan_deviation = unk.get('allan_deviation_1s', config.unknown_allan_deviation)
            config.unknown_bias_std = unk.get('initial_bias_std', config.unknown_bias_std)
            config.unknown_drift_std = unk.get('initial_drift_std', config.unknown_drift_std)
            config.unknown_cfo_std = unk.get('initial_cfo_std', config.unknown_cfo_std)

        if 'anchor_nodes' in clocks:
            anc = clocks['anchor_nodes']
            config.anchor_oscillator = anc.get('oscillator_type', config.anchor_oscillator)
            config.anchor_freq_accuracy_ppm = anc.get('frequency_accuracy_ppm', config.anchor_freq_accuracy_ppm)
            config.anchor_allan_deviation = anc.get('allan_deviation_1s', config.anchor_allan_deviation)
            config.anchor_bias_std = anc.get('initial_bias_std', config.anchor_bias_std)
            config.anchor_drift_std = anc.get('initial_drift_std', config.anchor_drift_std)
            config.anchor_cfo_std = anc.get('initial_cfo_std', config.anchor_cfo_std)

    # Parse signal section
    if 'signal' in yaml_config:
        sig = yaml_config['signal']
        config.signal_type = sig.get('type', config.signal_type)
        config.carrier_freq = float(sig.get('carrier_freq', config.carrier_freq))
        config.bandwidth = float(sig.get('bandwidth', config.bandwidth))
        config.sample_rate = float(sig.get('sample_rate', config.sample_rate))
        config.burst_duration = sig.get('burst_duration', config.burst_duration)
        config.prf = sig.get('prf', config.prf)
        config.sequence_length = sig.get('sequence_length', config.sequence_length)
        config.cyclic_prefix_length = sig.get('cyclic_prefix_length', config.cyclic_prefix_length)
        config.snr_db = sig.get('snr_db', config.snr_db)

    # Parse channel section
    if 'channel' in yaml_config:
        chan = yaml_config['channel']
        config.environment = chan.get('environment', config.environment)
        config.cluster_arrival_rate = chan.get('cluster_arrival_rate', config.cluster_arrival_rate)
        config.ray_arrival_rate = chan.get('ray_arrival_rate', config.ray_arrival_rate)
        config.cluster_decay_factor = chan.get('cluster_decay_factor', config.cluster_decay_factor)
        config.ray_decay_factor = chan.get('ray_decay_factor', config.ray_decay_factor)
        config.k_factor_db = chan.get('k_factor_db', config.k_factor_db)
        config.nlos_excess_delay_mean_ns = chan.get('nlos_excess_delay_mean_ns', config.nlos_excess_delay_mean_ns)
        config.nlos_excess_delay_std_ns = chan.get('nlos_excess_delay_std_ns', config.nlos_excess_delay_std_ns)
        config.path_loss_exponent = chan.get('path_loss_exponent', config.path_loss_exponent)
        config.shadowing_std_db = chan.get('shadowing_std_db', config.shadowing_std_db)

    # Parse receiver section
    if 'receiver' in yaml_config:
        rx = yaml_config['receiver']
        config.toa_detection_mode = rx.get('toa_detection_mode', config.toa_detection_mode)
        config.toa_threshold = rx.get('toa_threshold', config.toa_threshold)
        config.enable_subsample_refinement = rx.get('enable_subsample_refinement', config.enable_subsample_refinement)
        config.cfo_estimation_method = rx.get('cfo_estimation_method', config.cfo_estimation_method)
        config.nlos_classification_enabled = rx.get('nlos_classification_enabled', config.nlos_classification_enabled)

    # Parse factor graph section
    if 'factor_graph' in yaml_config:
        fg = yaml_config['factor_graph']
        config.toa_weight = fg.get('toa_weight', config.toa_weight)
        config.tdoa_weight = fg.get('tdoa_weight', config.tdoa_weight)
        config.twr_weight = fg.get('twr_weight', config.twr_weight)
        config.cfo_weight = fg.get('cfo_weight', config.cfo_weight)
        config.prior_weight = fg.get('prior_weight', config.prior_weight)
        config.toa_variance = fg.get('toa_variance', config.toa_variance)
        config.tdoa_variance = fg.get('tdoa_variance', config.tdoa_variance)
        config.twr_variance = fg.get('twr_variance', config.twr_variance)
        config.cfo_variance = fg.get('cfo_variance', config.cfo_variance)

    # Parse robust section
    if 'robust' in yaml_config:
        rob = yaml_config['robust']
        config.use_huber = rob.get('use_huber', config.use_huber)
        config.huber_delta = rob.get('huber_delta', config.huber_delta)
        config.use_dcs = rob.get('use_dcs', config.use_dcs)
        config.dcs_phi = rob.get('dcs_phi', config.dcs_phi)
        config.use_switchable = rob.get('use_switchable', config.use_switchable)
        config.outlier_threshold = rob.get('outlier_threshold', config.outlier_threshold)

    # Parse solver section
    if 'solver' in yaml_config:
        sol = yaml_config['solver']
        config.initialization_method = sol.get('initialization_method', config.initialization_method)
        config.max_iterations = sol.get('max_iterations', config.max_iterations)
        config.convergence_tolerance = sol.get('convergence_tolerance', config.convergence_tolerance)
        config.lambda_init = sol.get('lambda_init', config.lambda_init)
        config.verbose = sol.get('verbose', config.verbose)

    # Parse simulation section
    if 'simulation' in yaml_config:
        sim = yaml_config['simulation']
        config.n_rounds = sim.get('n_rounds', config.n_rounds)
        config.round_interval = sim.get('round_interval', config.round_interval)
        config.max_range = sim.get('max_range', config.max_range)
        config.los_probability = sim.get('los_probability', config.los_probability)
        config.measurement_noise_scale = sim.get('measurement_noise_scale', config.measurement_noise_scale)
        config.enable_multipath = sim.get('enable_multipath', config.enable_multipath)
        config.enable_clock_drift = sim.get('enable_clock_drift', config.enable_clock_drift)

    # Parse output section
    if 'output' in yaml_config:
        out = yaml_config['output']
        config.save_results = out.get('save_results', config.save_results)
        config.output_dir = out.get('output_dir', config.output_dir)
        config.save_plots = out.get('save_plots', config.save_plots)
        config.plot_format = out.get('plot_format', config.plot_format)
        config.save_trajectories = out.get('save_trajectories', config.save_trajectories)
        config.save_raw_measurements = out.get('save_raw_measurements', config.save_raw_measurements)
        config.metrics_to_compute = out.get('metrics_to_compute', config.metrics_to_compute)

    return config


def save_config(config: FTLConfig, path: str):
    """
    Save configuration to YAML file

    Args:
        config: FTLConfig object
        path: Output file path
    """
    yaml_dict = {
        'geometry': {
            'type': config.geometry_type,
            'n_nodes': config.n_nodes,
            'area_size': config.area_size,
            'jitter_std': config.jitter_std,
            'n_anchors': config.n_anchors,
            'anchor_placement': config.anchor_placement
        },
        'clocks': {
            'unknown_nodes': {
                'oscillator_type': config.unknown_oscillator,
                'frequency_accuracy_ppm': config.unknown_freq_accuracy_ppm,
                'allan_deviation_1s': config.unknown_allan_deviation,
                'initial_bias_std': config.unknown_bias_std,
                'initial_drift_std': config.unknown_drift_std,
                'initial_cfo_std': config.unknown_cfo_std
            },
            'anchor_nodes': {
                'oscillator_type': config.anchor_oscillator,
                'frequency_accuracy_ppm': config.anchor_freq_accuracy_ppm,
                'allan_deviation_1s': config.anchor_allan_deviation,
                'initial_bias_std': config.anchor_bias_std,
                'initial_drift_std': config.anchor_drift_std,
                'initial_cfo_std': config.anchor_cfo_std
            }
        },
        'signal': {
            'type': config.signal_type,
            'carrier_freq': config.carrier_freq,
            'bandwidth': config.bandwidth,
            'sample_rate': config.sample_rate,
            'burst_duration': config.burst_duration,
            'prf': config.prf,
            'sequence_length': config.sequence_length,
            'cyclic_prefix_length': config.cyclic_prefix_length,
            'snr_db': config.snr_db
        },
        'channel': {
            'environment': config.environment,
            'cluster_arrival_rate': config.cluster_arrival_rate,
            'ray_arrival_rate': config.ray_arrival_rate,
            'cluster_decay_factor': config.cluster_decay_factor,
            'ray_decay_factor': config.ray_decay_factor,
            'k_factor_db': config.k_factor_db,
            'nlos_excess_delay_mean_ns': config.nlos_excess_delay_mean_ns,
            'nlos_excess_delay_std_ns': config.nlos_excess_delay_std_ns,
            'path_loss_exponent': config.path_loss_exponent,
            'shadowing_std_db': config.shadowing_std_db
        },
        'receiver': {
            'toa_detection_mode': config.toa_detection_mode,
            'toa_threshold': config.toa_threshold,
            'enable_subsample_refinement': config.enable_subsample_refinement,
            'cfo_estimation_method': config.cfo_estimation_method,
            'nlos_classification_enabled': config.nlos_classification_enabled
        },
        'factor_graph': {
            'toa_weight': config.toa_weight,
            'tdoa_weight': config.tdoa_weight,
            'twr_weight': config.twr_weight,
            'cfo_weight': config.cfo_weight,
            'prior_weight': config.prior_weight,
            'toa_variance': config.toa_variance,
            'tdoa_variance': config.tdoa_variance,
            'twr_variance': config.twr_variance,
            'cfo_variance': config.cfo_variance
        },
        'robust': {
            'use_huber': config.use_huber,
            'huber_delta': config.huber_delta,
            'use_dcs': config.use_dcs,
            'dcs_phi': config.dcs_phi,
            'use_switchable': config.use_switchable,
            'outlier_threshold': config.outlier_threshold
        },
        'solver': {
            'initialization_method': config.initialization_method,
            'max_iterations': config.max_iterations,
            'convergence_tolerance': config.convergence_tolerance,
            'lambda_init': config.lambda_init,
            'verbose': config.verbose
        },
        'simulation': {
            'n_rounds': config.n_rounds,
            'round_interval': config.round_interval,
            'max_range': config.max_range,
            'los_probability': config.los_probability,
            'measurement_noise_scale': config.measurement_noise_scale,
            'enable_multipath': config.enable_multipath,
            'enable_clock_drift': config.enable_clock_drift
        },
        'output': {
            'save_results': config.save_results,
            'output_dir': config.output_dir,
            'save_plots': config.save_plots,
            'plot_format': config.plot_format,
            'save_trajectories': config.save_trajectories,
            'save_raw_measurements': config.save_raw_measurements,
            'metrics_to_compute': config.metrics_to_compute
        }
    }

    with open(path, 'w') as f:
        yaml.dump(yaml_dict, f, default_flow_style=False, sort_keys=False)


def validate_config(config: FTLConfig) -> bool:
    """
    Validate configuration for consistency

    Args:
        config: FTLConfig object

    Returns:
        True if valid

    Raises:
        ValueError if invalid
    """
    # Check node counts
    if config.n_nodes < config.n_anchors:
        raise ValueError(f"n_anchors ({config.n_anchors}) cannot exceed n_nodes ({config.n_nodes})")

    if config.n_anchors < 3:
        raise ValueError(f"Need at least 3 anchors for localization, got {config.n_anchors}")

    # Check signal parameters
    if config.bandwidth > config.sample_rate:
        raise ValueError(f"Bandwidth ({config.bandwidth}) cannot exceed sample rate ({config.sample_rate})")

    # Check valid options
    valid_geometries = ['grid', 'random', 'poisson']
    if config.geometry_type not in valid_geometries:
        raise ValueError(f"Invalid geometry type: {config.geometry_type}")

    valid_signals = ['hrp_uwb', 'zadoff_chu']
    if config.signal_type not in valid_signals:
        raise ValueError(f"Invalid signal type: {config.signal_type}")

    valid_environments = ['indoor', 'outdoor', 'urban']
    if config.environment not in valid_environments:
        raise ValueError(f"Invalid environment: {config.environment}")

    valid_init_methods = ['trilateration', 'mds', 'grid', 'random']
    if config.initialization_method not in valid_init_methods:
        raise ValueError(f"Invalid initialization method: {config.initialization_method}")

    return True


if __name__ == "__main__":
    # Test configuration loading
    print("Testing Configuration System...")
    print("=" * 50)

    # Load sample config
    config_path = "configs/scene.yaml"
    if os.path.exists(config_path):
        config = load_config(config_path)
        print(f"Loaded config from {config_path}")
        print(f"\nGeometry: {config.n_nodes} nodes in {config.area_size}m area")
        print(f"Signal: {config.signal_type} at {config.bandwidth/1e6:.1f} MHz")
        print(f"Channel: {config.environment} environment")
        print(f"Solver: {config.initialization_method} initialization, max {config.max_iterations} iterations")

        # Validate
        try:
            validate_config(config)
            print("\nConfiguration is valid!")
        except ValueError as e:
            print(f"\nConfiguration error: {e}")
    else:
        print(f"Config file not found: {config_path}")
        print("Creating default config...")

        # Create default config
        default_config = FTLConfig()
        save_config(default_config, "configs/default.yaml")
        print(f"Saved default config to configs/default.yaml")