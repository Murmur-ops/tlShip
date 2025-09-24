"""
Test 30-node FTL system with 5 anchors over 50x50m area
Now with comprehensive noise model support
"""

import numpy as np
import matplotlib.pyplot as plt
from ftl_enhanced import EnhancedFTL, EnhancedFTLConfig
from ftl.noise_model import NoiseConfig, NoiseGenerator, NoisePreset, create_preset_config
from typing import Optional


class Large30NodeFTL(EnhancedFTL):
    """Extended FTL for larger network with noise model support"""

    def __init__(self, config: EnhancedFTLConfig = None, noise_config: Optional[NoiseConfig] = None):
        """Initialize with optional noise configuration

        Args:
            config: FTL configuration
            noise_config: Noise model configuration (None for legacy behavior)
        """
        # Set noise configuration before parent init (which calls _setup_network)
        self.noise_config = noise_config
        self.noise_generator = NoiseGenerator(noise_config) if noise_config else None
        # Now call parent init
        super().__init__(config)

    def _setup_network(self):
        """Create network over specified area"""
        # Get area size and seed from config if available
        area_size = getattr(self.config, 'area_size', 50.0)
        custom_seed = getattr(self.config, 'random_seed', None)

        # Generate positions
        self.true_positions = np.zeros((self.config.n_nodes, 2))

        # 5 Anchors at strategic positions
        self.true_positions[0] = [0, 0]           # Bottom-left
        self.true_positions[1] = [area_size, 0]   # Bottom-right
        self.true_positions[2] = [0, area_size]   # Top-left
        self.true_positions[3] = [area_size, area_size]  # Top-right
        self.true_positions[4] = [area_size/2, area_size/2]  # Center

        # Unknown nodes - distributed across area
        np.random.seed(custom_seed if custom_seed is not None else 42)  # For reproducibility
        for i in range(5, self.config.n_nodes):
            # Random position with minimum distance from edges
            margin = 5.0
            self.true_positions[i] = [
                np.random.uniform(margin, area_size - margin),
                np.random.uniform(margin, area_size - margin)
            ]

        # Initialize states with error
        np.random.seed((custom_seed + 1) if custom_seed is not None else 43)  # Different seed for initial errors
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
        """Generate distance measurements with connectivity radius and noise model"""
        self.measurements = []
        connectivity_radius = 30.0  # Only measure within 30m

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
                        noise = np.random.normal(0, 0.001)  # 1mm noise
                        measured_dist = true_dist + noise
                        measurement_std = self.config.measurement_std

                    self.measurements.append({
                        'i': i,
                        'j': j,
                        'range': measured_dist,
                        'std': measurement_std
                    })

        print(f"Created {len(self.measurements)} measurements (connectivity radius: {connectivity_radius}m)")
        if self.noise_generator:
            preset_name = self.noise_config.preset.value if self.noise_config.preset else "custom"
            print(f"Using noise model: {preset_name}")


def test_30_node_system(noise_preset: str = "ideal"):
    """Test 30-node system with configurable noise

    Args:
        noise_preset: One of "ideal", "clean", "realistic", "harsh"
    """
    print("="*60)
    print("30-Node FTL System Test")
    print("Nodes: 30 (5 anchors + 25 unknowns)")
    print("Area: 50m × 50m")
    print(f"Noise preset: {noise_preset.upper()}")
    print("="*60)

    # Configure system
    config = EnhancedFTLConfig(
        n_nodes=30,
        n_anchors=5,
        use_adaptive_lm=True,
        use_line_search=False,
        max_iterations=100,
        lm_initial_lambda=1e-3,
        gradient_tol=1e-8,
        measurement_std=0.01,  # 1cm (used only if no noise model)
        verbose=True
    )

    # Create noise configuration
    if noise_preset.lower() != "legacy":
        noise_config = create_preset_config(noise_preset)
        noise_config.random_seed = 42  # For reproducibility
    else:
        noise_config = None  # Use legacy simple noise

    # Create and run system
    ftl = Large30NodeFTL(config, noise_config)

    # Print initial errors
    print("\nInitial Errors:")
    print("-"*40)
    initial_pos_errors = []
    initial_time_errors = []
    for i in range(5, 30):
        pos_err = np.linalg.norm(ftl.states[i, :2] - ftl.true_positions[i])
        time_err = ftl.states[i, 2]
        initial_pos_errors.append(pos_err)
        initial_time_errors.append(abs(time_err))

    print(f"Position RMSE: {np.sqrt(np.mean(np.array(initial_pos_errors)**2)):.3f} m")
    print(f"Time RMSE: {np.sqrt(np.mean(np.array(initial_time_errors)**2)):.3f} ns")
    print(f"Max position error: {max(initial_pos_errors):.3f} m")
    print(f"Max time error: {max(initial_time_errors):.3f} ns")

    # Run optimization
    print("\nRunning Adaptive LM optimization...")
    print("-"*40)
    ftl.run()

    # Print final errors
    print("\nFinal Errors:")
    print("-"*40)
    final_pos_errors = []
    final_time_errors = []
    for i in range(5, 30):
        pos_err = np.linalg.norm(ftl.states[i, :2] - ftl.true_positions[i])
        time_err = ftl.states[i, 2]
        final_pos_errors.append(pos_err)
        final_time_errors.append(abs(time_err))

    print(f"Position RMSE: {np.sqrt(np.mean(np.array(final_pos_errors)**2))*1000:.3f} mm")
    print(f"Time RMSE: {np.sqrt(np.mean(np.array(final_time_errors)**2))*1000:.3f} ps")
    print(f"Max position error: {max(final_pos_errors)*1000:.3f} mm")
    print(f"Max time error: {max(final_time_errors)*1000:.3f} ps")

    # Create visualization
    fig = plt.figure(figsize=(16, 12))

    # 1. Network topology and results
    ax1 = plt.subplot(2, 3, 1)

    # Plot measurements as lines
    for m in ftl.measurements:
        i, j = m['i'], m['j']
        ax1.plot([ftl.true_positions[i, 0], ftl.true_positions[j, 0]],
                [ftl.true_positions[i, 1], ftl.true_positions[j, 1]],
                'k-', alpha=0.1, linewidth=0.5)

    # Plot nodes
    ax1.scatter(ftl.true_positions[:5, 0], ftl.true_positions[:5, 1],
               c='red', s=200, marker='^', label='Anchors', zorder=5)
    ax1.scatter(ftl.true_positions[5:, 0], ftl.true_positions[5:, 1],
               c='blue', s=100, marker='o', label='True Unknown', zorder=4)
    ax1.scatter(ftl.states[5:, 0], ftl.states[5:, 1],
               c='green', s=50, marker='x', label='Estimated', zorder=6)

    # Draw error lines
    for i in range(5, 30):
        ax1.plot([ftl.true_positions[i, 0], ftl.states[i, 0]],
                [ftl.true_positions[i, 1], ftl.states[i, 1]],
                'r-', alpha=0.5, linewidth=1)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Network Topology and Final Positions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # 2. Position convergence
    ax2 = plt.subplot(2, 3, 2)
    ax2.semilogy(ftl.position_rmse_history, 'b-', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Position RMSE (m)')
    ax2.set_title('Position Convergence')
    ax2.grid(True, alpha=0.3)

    # 3. Time convergence
    ax3 = plt.subplot(2, 3, 3)
    ax3.semilogy(ftl.time_rmse_history, 'g-', linewidth=2)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Time RMSE (ns)')
    ax3.set_title('Time Synchronization Convergence')
    ax3.grid(True, alpha=0.3)

    # 4. Lambda adaptation
    ax4 = plt.subplot(2, 3, 4)
    if ftl.lambda_history:
        ax4.semilogy(ftl.lambda_history, 'r-', linewidth=2)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Damping Parameter λ')
    ax4.set_title('Adaptive Damping Parameter')
    ax4.grid(True, alpha=0.3)

    # 5. Position error distribution
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(np.array(final_pos_errors)*1000, bins=15, edgecolor='black', alpha=0.7)
    ax5.set_xlabel('Position Error (mm)')
    ax5.set_ylabel('Number of Nodes')
    ax5.set_title('Final Position Error Distribution')
    ax5.grid(True, alpha=0.3)

    # 6. Cost function
    ax6 = plt.subplot(2, 3, 6)
    if ftl.cost_history:
        ax6.semilogy(ftl.cost_history, 'k-', linewidth=2)
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Cost Function')
    ax6.set_title('Optimization Cost')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('30node_system_results.png', dpi=150)
    print("\nPlots saved to 30node_system_results.png")

    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    print(f"Convergence iterations: {len(ftl.position_rmse_history)-1}")
    print(f"Final λ: {ftl.lm_optimizer.lambda_current:.2e}")
    print(f"Measurements used: {len(ftl.measurements)}")
    print(f"Average connectivity: {2*len(ftl.measurements)/30:.1f} links per node")

    # Node-by-node results for worst performers
    errors_with_idx = [(i, final_pos_errors[i-5]) for i in range(5, 30)]
    errors_with_idx.sort(key=lambda x: x[1], reverse=True)

    print("\nWorst 5 nodes (highest position error):")
    for idx, (node_id, error) in enumerate(errors_with_idx[:5]):
        print(f"  Node {node_id}: {error*1000:.2f} mm")

    print("\nBest 5 nodes (lowest position error):")
    for idx, (node_id, error) in enumerate(errors_with_idx[-5:]):
        print(f"  Node {node_id}: {error*1000:.2f} mm")

    return ftl


def test_noise_comparison():
    """Test system with different noise presets and compare performance"""
    print("\n" + "="*60)
    print("NOISE MODEL COMPARISON TEST")
    print("="*60 + "\n")

    presets = ["ideal", "clean", "realistic", "harsh"]
    results = {}

    for preset in presets:
        print(f"\n{'='*40}")
        print(f"Testing with {preset.upper()} noise preset")
        print('='*40)

        # Configure system
        config = EnhancedFTLConfig(
            n_nodes=30,
            n_anchors=5,
            use_adaptive_lm=True,
            use_line_search=False,
            max_iterations=100,
            lm_initial_lambda=1e-3,
            gradient_tol=1e-8,
            verbose=False  # Less verbose for comparison
        )

        # Create noise configuration
        noise_config = create_preset_config(preset)
        noise_config.random_seed = 42  # Same seed for fair comparison

        # Create and run system
        ftl = Large30NodeFTL(config, noise_config)

        # Run optimization
        ftl.run()

        # Get final errors
        final_pos_rmse = ftl.position_rmse_history[-1] if ftl.position_rmse_history else 999
        final_time_rmse = ftl.time_rmse_history[-1] if ftl.time_rmse_history else 999

        results[preset] = {
            'position_rmse_mm': final_pos_rmse * 1000,
            'time_rmse_ns': final_time_rmse,
            'iterations': len(ftl.position_rmse_history)
        }

        print(f"Final Position RMSE: {final_pos_rmse*1000:.2f}mm")
        print(f"Final Time RMSE: {final_time_rmse:.3f}ns")

    # Print comparison table
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    print(f"{'Preset':<12} {'Position RMSE':<15} {'Time RMSE':<12} {'Iterations':<10}")
    print("-"*60)
    for preset, result in results.items():
        print(f"{preset.upper():<12} {result['position_rmse_mm']:<15.2f}mm "
              f"{result['time_rmse_ns']:<12.3f}ns {result['iterations']:<10}")

    return results


if __name__ == "__main__":
    import sys

    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--compare":
            # Run comparison test
            test_noise_comparison()
        else:
            # Run with specified noise preset
            noise_preset = sys.argv[1]
            ftl = test_30_node_system(noise_preset)
    else:
        # Default: run with ideal noise (original behavior)
        ftl = test_30_node_system("ideal")