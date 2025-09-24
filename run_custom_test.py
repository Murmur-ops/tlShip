#!/usr/bin/env python3
"""
Run TL system with custom parameters:
- No noise (ideal preset)
- Seed: 22
- Area: 85x85m
- Nodes: 31
"""

import numpy as np
import matplotlib.pyplot as plt
from ftl_enhanced import EnhancedFTL, EnhancedFTLConfig
from ftl.noise_model import create_preset_config


class CustomLargeNodeFTL(EnhancedFTL):
    """Extended FTL for custom network configuration"""

    def __init__(self, config: EnhancedFTLConfig, area_size=85.0, custom_seed=22):
        """Initialize with custom parameters"""
        self.area_size = area_size
        self.custom_seed = custom_seed
        super().__init__(config)

    def _setup_network(self):
        """Create network over 85x85m area with 31 nodes"""
        area_size = self.area_size

        # Generate positions
        self.true_positions = np.zeros((self.config.n_nodes, 2))

        # 5 Anchors at strategic positions
        self.true_positions[0] = [0, 0]           # Bottom-left
        self.true_positions[1] = [area_size, 0]   # Bottom-right
        self.true_positions[2] = [0, area_size]   # Top-left
        self.true_positions[3] = [area_size, area_size]  # Top-right
        self.true_positions[4] = [area_size/2, area_size/2]  # Center

        # 26 Unknown nodes - distributed across area
        np.random.seed(self.custom_seed)  # Seed = 22
        for i in range(5, self.config.n_nodes):
            # Random position with minimum distance from edges
            margin = 5.0
            self.true_positions[i] = [
                np.random.uniform(margin, area_size - margin),
                np.random.uniform(margin, area_size - margin)
            ]

        # Initialize states with error
        np.random.seed(self.custom_seed + 1)  # Seed = 23 for initial errors
        self.states = np.zeros((self.config.n_nodes, 3))  # [x, y, clock_bias]

        for i in range(self.config.n_nodes):
            if i < self.config.n_anchors:
                # Anchors: perfect position, no clock bias
                self.states[i, :2] = self.true_positions[i]
                self.states[i, 2] = 0
            else:
                # Unknowns: add position error and clock bias
                self.states[i, :2] = self.true_positions[i] + np.random.normal(0, 5, 2)  # 5m std error
                self.states[i, 2] = np.random.normal(0, 30)  # 30ns clock error

        # Create measurements
        self._generate_measurements()

    def _generate_measurements(self):
        """Generate distance measurements with connectivity radius"""
        self.measurements = []
        connectivity_radius = 40.0  # Increased for larger area

        for i in range(self.config.n_nodes):
            for j in range(i+1, self.config.n_nodes):
                true_dist = np.linalg.norm(self.true_positions[i] - self.true_positions[j])

                # Only create measurement if within connectivity radius
                if true_dist <= connectivity_radius:
                    # No noise (ideal preset)
                    noise = np.random.normal(0, 0.001)  # 1mm noise only
                    measured_dist = true_dist + noise

                    self.measurements.append({
                        'i': i,
                        'j': j,
                        'range': measured_dist,
                        'std': self.config.measurement_std
                    })

        print(f"Created {len(self.measurements)} measurements (connectivity radius: {connectivity_radius}m)")


def run_custom_test():
    """Run test with custom parameters"""
    print("="*60)
    print("CUSTOM TL SYSTEM TEST")
    print("Parameters:")
    print("  - Noise: NONE (ideal preset)")
    print("  - Random seed: 22")
    print("  - Area: 85m × 85m")
    print("  - Nodes: 31 (5 anchors + 26 unknowns)")
    print("="*60)

    # Configure system
    config = EnhancedFTLConfig(
        n_nodes=31,  # 31 nodes as requested
        n_anchors=5,
        use_adaptive_lm=True,
        use_line_search=False,
        max_iterations=100,
        lm_initial_lambda=1e-3,
        gradient_tol=1e-8,
        measurement_std=0.001,  # 1mm
        verbose=True
    )

    # Create and run system
    ftl = CustomLargeNodeFTL(config, area_size=85.0, custom_seed=22)

    # Print initial errors
    print("\nInitial Errors:")
    print("-"*40)
    initial_pos_errors = []
    initial_time_errors = []

    for i in range(5, 31):  # Unknown nodes only
        pos_error = np.linalg.norm(ftl.states[i, :2] - ftl.true_positions[i])
        time_error = abs(ftl.states[i, 2])
        initial_pos_errors.append(pos_error)
        initial_time_errors.append(time_error)

    initial_pos_rmse = np.sqrt(np.mean(np.array(initial_pos_errors)**2))
    initial_time_rmse = np.sqrt(np.mean(np.array(initial_time_errors)**2))

    print(f"Position RMSE: {initial_pos_rmse:.3f} m")
    print(f"Time RMSE: {initial_time_rmse:.3f} ns")

    # Run optimization
    print("\nRunning optimization...")
    ftl.run()

    # Final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    final_pos_errors = []
    final_time_errors = []

    for i in range(5, 31):
        pos_error = np.linalg.norm(ftl.states[i, :2] - ftl.true_positions[i])
        time_error = abs(ftl.states[i, 2])
        final_pos_errors.append(pos_error)
        final_time_errors.append(time_error)

    final_pos_rmse = np.sqrt(np.mean(np.array(final_pos_errors)**2))
    final_time_rmse = np.sqrt(np.mean(np.array(final_time_errors)**2))

    print(f"\nPosition RMSE: {final_pos_rmse*1000:.3f} mm")
    print(f"Time RMSE: {final_time_rmse*1000:.3f} ps")

    # Check if target met
    if final_pos_rmse < 0.020:
        print(f"\n✅ SUCCESS: Achieved {final_pos_rmse*1000:.2f}mm accuracy (target: 20mm)")
    else:
        print(f"\n⚠️ Position RMSE: {final_pos_rmse*1000:.2f}mm")

    # Create visualization
    print("\nGenerating visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Convergence
    ax1.semilogy(np.array(ftl.position_rmse_history) * 1000, 'b-', linewidth=2, label='Position RMSE')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Position RMSE (mm)')
    ax1.set_title('Convergence (85×85m, 31 nodes, seed=22)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=20, color='g', linestyle='--', alpha=0.5, label='20mm target')
    ax1.legend()

    # Plot 2: Node positions
    ax2.scatter(ftl.true_positions[5:, 0], ftl.true_positions[5:, 1],
               c='blue', marker='o', s=100, alpha=0.6, label='True positions')
    ax2.scatter(ftl.states[5:, 0], ftl.states[5:, 1],
               c='red', marker='x', s=100, label='Estimated positions')
    ax2.scatter(ftl.true_positions[:5, 0], ftl.true_positions[:5, 1],
               c='green', marker='s', s=150, label='Anchors')

    # Draw connections for estimated
    for i in range(5, 31):
        ax2.plot([ftl.true_positions[i, 0], ftl.states[i, 0]],
                [ftl.true_positions[i, 1], ftl.states[i, 1]],
                'k-', alpha=0.3, linewidth=0.5)

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title(f'Final Positions (RMSE: {final_pos_rmse*1000:.2f}mm)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-5, 90)
    ax2.set_ylim(-5, 90)
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('custom_test_results.png', dpi=150)
    print("✓ Saved visualization to custom_test_results.png")

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Area: 85m × 85m")
    print(f"Total nodes: 31 (5 anchors + 26 unknowns)")
    print(f"Random seed: 22")
    print(f"Measurements: {len(ftl.measurements)}")
    print(f"Average connectivity: {2*len(ftl.measurements)/31:.1f} links per node")
    print(f"Convergence iterations: {len(ftl.position_rmse_history)}")
    print(f"Final position RMSE: {final_pos_rmse*1000:.3f}mm")
    print(f"Final time RMSE: {final_time_rmse*1000:.3f}ps")

    # Show worst and best performing nodes
    errors_with_idx = [(i, final_pos_errors[i-5]) for i in range(5, 31)]
    errors_with_idx.sort(key=lambda x: x[1], reverse=True)

    print("\nWorst 3 nodes (highest position error):")
    for node_id, error in errors_with_idx[:3]:
        print(f"  Node {node_id}: {error*1000:.2f}mm")

    print("\nBest 3 nodes (lowest position error):")
    for node_id, error in errors_with_idx[-3:]:
        print(f"  Node {node_id}: {error*1000:.2f}mm")

    return ftl, final_pos_rmse


if __name__ == "__main__":
    ftl, rmse = run_custom_test()
    print(f"\n{'='*60}")
    print(f"Test completed with final RMSE: {rmse*1000:.3f}mm")
    print(f"{'='*60}")