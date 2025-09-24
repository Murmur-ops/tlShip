#!/usr/bin/env python3
"""
Generic FTL system that can handle any network configuration
Flexible anchor placement and node distribution
"""

import numpy as np
import matplotlib.pyplot as plt
from ftl_enhanced import EnhancedFTL, EnhancedFTLConfig
from ftl.noise_model import NoiseConfig, NoiseGenerator, create_preset_config
from typing import Optional
import yaml
import sys


class GenericFTL(EnhancedFTL):
    """Generic FTL system with flexible configuration"""

    def __init__(self, config: EnhancedFTLConfig,
                 noise_config: Optional[NoiseConfig] = None,
                 anchor_placement: str = "corners_and_center"):
        """Initialize with flexible configuration

        Args:
            config: FTL configuration
            noise_config: Noise model configuration
            anchor_placement: Strategy for placing anchors
        """
        self.noise_config = noise_config
        self.noise_generator = NoiseGenerator(noise_config) if noise_config else None
        self.anchor_placement = anchor_placement
        super().__init__(config)

    def _setup_network(self):
        """Create network with flexible anchor and node placement"""
        area_size = getattr(self.config, 'area_size', 50.0)
        random_seed = getattr(self.config, 'random_seed', None)
        node_margin = getattr(self.config, 'node_margin', 5.0)

        # Initialize positions array
        self.true_positions = np.zeros((self.config.n_nodes, 2))

        # Place anchors based on strategy
        self._place_anchors(area_size)

        # Place unknown nodes
        self._place_unknown_nodes(area_size, node_margin, random_seed)

        # Initialize states with error
        self._initialize_states(random_seed)

        # Create measurements
        self._generate_measurements()

    def _place_anchors(self, area_size: float):
        """Place anchor nodes based on strategy"""
        n_anchors = self.config.n_anchors

        if self.anchor_placement == "corners_and_center":
            # Place anchors at corners and center (works best for 4-5 anchors)
            if n_anchors >= 1:
                self.true_positions[0] = [0, 0]  # Bottom-left
            if n_anchors >= 2:
                self.true_positions[1] = [area_size, 0]  # Bottom-right
            if n_anchors >= 3:
                self.true_positions[2] = [0, area_size]  # Top-left
            if n_anchors >= 4:
                self.true_positions[3] = [area_size, area_size]  # Top-right
            if n_anchors >= 5:
                self.true_positions[4] = [area_size/2, area_size/2]  # Center

            # For more than 5 anchors, distribute along perimeter
            if n_anchors > 5:
                perimeter_anchors = n_anchors - 5
                for i in range(perimeter_anchors):
                    angle = 2 * np.pi * i / perimeter_anchors
                    radius = area_size * 0.4  # 40% of area size
                    self.true_positions[5 + i] = [
                        area_size/2 + radius * np.cos(angle),
                        area_size/2 + radius * np.sin(angle)
                    ]

        elif self.anchor_placement == "perimeter":
            # Distribute anchors evenly around perimeter
            for i in range(n_anchors):
                # Calculate position along perimeter
                perimeter_total = 4 * area_size
                position = (i / n_anchors) * perimeter_total

                if position < area_size:
                    # Bottom edge
                    self.true_positions[i] = [position, 0]
                elif position < 2 * area_size:
                    # Right edge
                    self.true_positions[i] = [area_size, position - area_size]
                elif position < 3 * area_size:
                    # Top edge
                    self.true_positions[i] = [area_size - (position - 2*area_size), area_size]
                else:
                    # Left edge
                    self.true_positions[i] = [0, area_size - (position - 3*area_size)]

        elif self.anchor_placement == "grid":
            # Place anchors in a grid pattern
            grid_size = int(np.sqrt(n_anchors))
            for i in range(n_anchors):
                row = i // grid_size
                col = i % grid_size
                x = (col + 0.5) * area_size / grid_size
                y = (row + 0.5) * area_size / grid_size
                self.true_positions[i] = [x, y]

        else:
            # Default: corners first, then distribute remaining
            self._place_anchors_corners_first(area_size, n_anchors)

    def _place_anchors_corners_first(self, area_size: float, n_anchors: int):
        """Default anchor placement: corners first, then distribute"""
        if n_anchors >= 1:
            self.true_positions[0] = [0, 0]
        if n_anchors >= 2:
            self.true_positions[1] = [area_size, 0]
        if n_anchors >= 3:
            self.true_positions[2] = [area_size, area_size]
        if n_anchors >= 4:
            self.true_positions[3] = [0, area_size]

        # Remaining anchors distributed inside
        for i in range(4, n_anchors):
            angle = 2 * np.pi * (i - 4) / (n_anchors - 4)
            radius = area_size * 0.3
            self.true_positions[i] = [
                area_size/2 + radius * np.cos(angle),
                area_size/2 + radius * np.sin(angle)
            ]

    def _place_unknown_nodes(self, area_size: float, margin: float, seed: Optional[int]):
        """Place unknown nodes randomly within area"""
        if seed is not None:
            np.random.seed(seed)

        n_unknowns = self.config.n_nodes - self.config.n_anchors

        for i in range(self.config.n_anchors, self.config.n_nodes):
            # Random position with margin from edges
            self.true_positions[i] = [
                np.random.uniform(margin, area_size - margin),
                np.random.uniform(margin, area_size - margin)
            ]

    def _initialize_states(self, seed: Optional[int]):
        """Initialize states with error"""
        if seed is not None:
            np.random.seed(seed + 1)  # Different seed for initial errors

        self.states = np.zeros((self.config.n_nodes, 3))  # [x, y, clock_bias]

        for i in range(self.config.n_nodes):
            if i < self.config.n_anchors:
                # Anchors: perfect position, no clock bias
                self.states[i, :2] = self.true_positions[i]
                self.states[i, 2] = 0
            else:
                # Unknowns: add position error and clock bias
                self.states[i, :2] = self.true_positions[i] + np.random.normal(0, 5, 2)
                self.states[i, 2] = np.random.normal(0, 30)  # 30ns clock error

    def _generate_measurements(self):
        """Generate distance measurements with connectivity radius"""
        self.measurements = []

        # Adaptive connectivity radius based on area size
        area_size = getattr(self.config, 'area_size', 50.0)
        default_radius = area_size * 0.5  # 50% of area size
        connectivity_radius = getattr(self.config, 'connectivity_radius', default_radius)

        for i in range(self.config.n_nodes):
            for j in range(i+1, self.config.n_nodes):
                true_dist = np.linalg.norm(self.true_positions[i] - self.true_positions[j])

                # Only create measurement if within connectivity radius
                if true_dist <= connectivity_radius:
                    if self.noise_generator:
                        # Use comprehensive noise model
                        measured_dist, measurement_std = self.noise_generator.add_measurement_noise(
                            true_dist, i, j,
                            position_i=self.true_positions[i],
                            position_j=self.true_positions[j]
                        )
                    else:
                        # Legacy behavior: simple Gaussian noise
                        noise = np.random.normal(0, self.config.measurement_std)
                        measured_dist = true_dist + noise
                        measurement_std = self.config.measurement_std

                    self.measurements.append({
                        'i': i,
                        'j': j,
                        'range': measured_dist,
                        'std': measurement_std
                    })

        print(f"Created {len(self.measurements)} measurements (connectivity radius: {connectivity_radius:.1f}m)")

        # Check connectivity
        avg_connectivity = 2 * len(self.measurements) / self.config.n_nodes
        print(f"Average connectivity: {avg_connectivity:.1f} links per node")

        if avg_connectivity < 3:
            print("⚠️ Warning: Low connectivity may affect convergence")


def load_yaml_config(config_path: str) -> dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_generic_system(config_path: str = "configs/generic_network.yaml"):
    """Run generic FTL system with YAML configuration"""

    print("="*60)
    print("GENERIC FTL SYSTEM")
    print("="*60)

    # Load YAML configuration
    yaml_config = load_yaml_config(config_path)
    print(f"Configuration loaded from: {config_path}")

    # Extract network parameters
    network_config = yaml_config['network']
    n_nodes = network_config['n_nodes']
    n_anchors = network_config['n_anchors']
    area_size = network_config.get('area_size', 50.0)
    random_seed = network_config.get('random_seed', None)
    anchor_placement = network_config.get('anchor_placement', 'corners_and_center')

    print(f"\nNetwork Configuration:")
    print(f"  Nodes: {n_nodes} ({n_anchors} anchors + {n_nodes - n_anchors} unknowns)")
    print(f"  Area: {area_size}m × {area_size}m")
    print(f"  Seed: {random_seed}")
    print(f"  Anchor placement: {anchor_placement}")

    # Create FTL configuration
    config = EnhancedFTLConfig(
        n_nodes=n_nodes,
        n_anchors=n_anchors,
        area_size=area_size,
        random_seed=random_seed,
        use_adaptive_lm=yaml_config['optimization']['use_adaptive_lm'],
        use_line_search=yaml_config['optimization'].get('use_line_search', False),
        max_iterations=yaml_config['optimization']['max_iterations'],
        lm_initial_lambda=yaml_config['optimization']['lm_initial_lambda'],
        gradient_tol=yaml_config['optimization']['gradient_tol'],
        measurement_std=yaml_config['measurements']['measurement_std'],
        verbose=yaml_config['output']['verbose']
    )

    # Add connectivity radius to config
    config.connectivity_radius = yaml_config['measurements'].get('connectivity_radius', area_size * 0.5)
    config.node_margin = network_config.get('node_margin', 5.0)

    # Create noise configuration
    noise_config = None
    if 'noise' in yaml_config:
        noise_preset = yaml_config['noise'].get('preset', 'ideal')
        if noise_preset != 'custom':
            noise_config = create_preset_config(noise_preset)
            if yaml_config['noise'].get('random_seed'):
                noise_config.random_seed = yaml_config['noise']['random_seed']

    # Create and run system
    print("\nInitializing system...")
    ftl = GenericFTL(config, noise_config, anchor_placement)

    # Print initial errors
    print("\nInitial Errors:")
    print("-"*40)
    initial_pos_errors = []
    for i in range(n_anchors, n_nodes):
        pos_error = np.linalg.norm(ftl.states[i, :2] - ftl.true_positions[i])
        initial_pos_errors.append(pos_error)

    initial_pos_rmse = np.sqrt(np.mean(np.array(initial_pos_errors)**2))
    print(f"Position RMSE: {initial_pos_rmse:.3f}m")

    # Run optimization
    print("\nRunning optimization...")
    ftl.run()

    # Final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    final_pos_errors = []
    for i in range(n_anchors, n_nodes):
        pos_error = np.linalg.norm(ftl.states[i, :2] - ftl.true_positions[i])
        final_pos_errors.append(pos_error)

    final_pos_rmse = np.sqrt(np.mean(np.array(final_pos_errors)**2))
    print(f"Position RMSE: {final_pos_rmse*1000:.3f}mm")

    if final_pos_rmse < 0.020:
        print(f"✅ SUCCESS: Achieved {final_pos_rmse*1000:.2f}mm accuracy!")
    else:
        print(f"⚠️ Final RMSE: {final_pos_rmse*1000:.2f}mm")

    # Create visualization
    if yaml_config['output'].get('plot_results', True):
        create_visualization(ftl, area_size, n_anchors)

    return ftl, final_pos_rmse


def create_visualization(ftl, area_size, n_anchors):
    """Create and save visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Convergence plot
    ax1.semilogy(np.array(ftl.position_rmse_history) * 1000, 'b-', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Position RMSE (mm)')
    ax1.set_title('Convergence')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=20, color='g', linestyle='--', alpha=0.5, label='20mm target')
    ax1.legend()

    # Position plot
    n_nodes = ftl.config.n_nodes

    # Plot anchors
    ax2.scatter(ftl.true_positions[:n_anchors, 0],
               ftl.true_positions[:n_anchors, 1],
               c='green', marker='s', s=200, label='Anchors', zorder=5)

    # Plot true positions
    ax2.scatter(ftl.true_positions[n_anchors:, 0],
               ftl.true_positions[n_anchors:, 1],
               c='blue', marker='o', s=100, alpha=0.6, label='True positions')

    # Plot estimated positions
    ax2.scatter(ftl.states[n_anchors:, 0],
               ftl.states[n_anchors:, 1],
               c='red', marker='x', s=100, label='Estimated positions')

    # Draw error lines
    for i in range(n_anchors, n_nodes):
        ax2.plot([ftl.true_positions[i, 0], ftl.states[i, 0]],
                [ftl.true_positions[i, 1], ftl.states[i, 1]],
                'k-', alpha=0.3, linewidth=0.5)

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title(f'Node Positions ({n_nodes} nodes)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-5, area_size + 5)
    ax2.set_ylim(-5, area_size + 5)
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('generic_system_results.png', dpi=150)
    print("\n✓ Saved visualization to generic_system_results.png")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "configs/generic_network.yaml"

    ftl, rmse = run_generic_system(config_path)
    print(f"\nFinal RMSE: {rmse*1000:.3f}mm")