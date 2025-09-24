#!/usr/bin/env python3
"""
Test 30-node FTL system with 500 iterations over 80x80m area
"""

import numpy as np
import matplotlib.pyplot as plt
from ftl_enhanced import EnhancedFTL, EnhancedFTLConfig


class Large30NodeFTL_80m(EnhancedFTL):
    """Extended FTL for 80x80m area"""

    def _setup_network(self):
        """Create 30-node network over 80x80m area"""
        area_size = 80.0  # Increased from 50m to 80m

        # Generate positions
        self.true_positions = np.zeros((self.config.n_nodes, 2))

        # 5 Anchors at strategic positions
        self.true_positions[0] = [0, 0]           # Bottom-left
        self.true_positions[1] = [area_size, 0]   # Bottom-right
        self.true_positions[2] = [0, area_size]   # Top-left
        self.true_positions[3] = [area_size, area_size]  # Top-right
        self.true_positions[4] = [area_size/2, area_size/2]  # Center

        # 25 Unknown nodes - distributed across area
        np.random.seed(42)  # For reproducibility
        for i in range(5, self.config.n_nodes):
            # Random position with minimum distance from edges
            margin = 5.0
            self.true_positions[i] = [
                np.random.uniform(margin, area_size - margin),
                np.random.uniform(margin, area_size - margin)
            ]

        # Initialize states with error
        np.random.seed(43)  # Different seed for initial errors
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
        connectivity_radius = 40.0  # Increased from 30m to 40m for larger area

        for i in range(self.config.n_nodes):
            for j in range(i+1, self.config.n_nodes):
                true_dist = np.linalg.norm(self.true_positions[i] - self.true_positions[j])

                # Only create measurement if within connectivity radius
                if true_dist <= connectivity_radius:
                    # Add small measurement noise for realism
                    noise = np.random.normal(0, 0.001)  # 1mm noise
                    measured_dist = true_dist + noise

                    self.measurements.append({
                        'i': i,
                        'j': j,
                        'range': measured_dist,
                        'std': self.config.measurement_std
                    })

        print(f"Created {len(self.measurements)} measurements (connectivity radius: {connectivity_radius}m)")


def test_500iter_80m():
    """Test 30-node system with 500 iterations over 80x80m area"""
    print("="*60)
    print("30-Node FTL System Test - Extended Configuration")
    print("Nodes: 30 (5 anchors + 25 unknowns)")
    print("Area: 80m × 80m")
    print("Iterations: 500")
    print("="*60)

    # Configure system with 500 iterations
    config = EnhancedFTLConfig(
        n_nodes=30,
        n_anchors=5,
        use_adaptive_lm=True,
        use_line_search=False,
        max_iterations=500,  # Increased from 100 to 500
        lm_initial_lambda=1e-3,
        gradient_tol=1e-8,
        measurement_std=0.01,  # 1cm
        verbose=True
    )

    # Create and run system
    ftl = Large30NodeFTL_80m(config)

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

    # Run optimization
    print("\nRunning Adaptive LM optimization for 500 iterations...")
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

    final_pos_rmse = np.sqrt(np.mean(np.array(final_pos_errors)**2))
    final_time_rmse = np.sqrt(np.mean(np.array(final_time_errors)**2))

    print(f"Position RMSE: {final_pos_rmse*1000:.3f} mm")
    print(f"Time RMSE: {final_time_rmse*1000:.3f} ps")
    print(f"Max position error: {max(final_pos_errors)*1000:.3f} mm")
    print(f"Min position error: {min(final_pos_errors)*1000:.3f} mm")

    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 10))

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

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('80m × 80m Network Topology')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_xlim([-5, 85])
    ax1.set_ylim([-5, 85])

    # 2. Position convergence (full 500 iterations)
    ax2 = plt.subplot(2, 3, 2)
    ax2.semilogy(ftl.position_rmse_history, 'b-', linewidth=2)
    ax2.axhline(y=final_pos_rmse, color='r', linestyle='--', alpha=0.5,
                label=f'Final: {final_pos_rmse*1000:.2f}mm')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Position RMSE (m)')
    ax2.set_title('Position Convergence (500 iterations)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Time convergence
    ax3 = plt.subplot(2, 3, 3)
    ax3.semilogy(ftl.time_rmse_history, 'g-', linewidth=2)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Time RMSE (ns)')
    ax3.set_title('Time Synchronization Convergence')
    ax3.grid(True, alpha=0.3)

    # 4. Lambda adaptation over 500 iterations
    ax4 = plt.subplot(2, 3, 4)
    if ftl.lambda_history:
        ax4.semilogy(ftl.lambda_history, 'r-', linewidth=1, alpha=0.7)
        # Add moving average
        window = 50
        if len(ftl.lambda_history) > window:
            lambda_smooth = np.convolve(ftl.lambda_history,
                                       np.ones(window)/window, mode='valid')
            ax4.semilogy(range(window//2, len(lambda_smooth)+window//2),
                        lambda_smooth, 'k-', linewidth=2, label='Moving avg (50)')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Damping Parameter λ')
    ax4.set_title('Adaptive Damping (500 iterations)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # 5. Position error distribution
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(np.array(final_pos_errors)*1000, bins=15, edgecolor='black', alpha=0.7)
    ax5.axvline(x=final_pos_rmse*1000, color='r', linestyle='--',
                label=f'RMSE: {final_pos_rmse*1000:.2f}mm')
    ax5.set_xlabel('Position Error (mm)')
    ax5.set_ylabel('Number of Nodes')
    ax5.set_title('Final Position Error Distribution')
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    # 6. Convergence rate analysis
    ax6 = plt.subplot(2, 3, 6)
    if len(ftl.position_rmse_history) > 1:
        convergence_rate = np.diff(np.log10(ftl.position_rmse_history))
        ax6.plot(convergence_rate, 'k-', linewidth=1, alpha=0.5)
        # Smooth version
        if len(convergence_rate) > 20:
            window = 20
            rate_smooth = np.convolve(convergence_rate,
                                     np.ones(window)/window, mode='valid')
            ax6.plot(range(window//2, len(rate_smooth)+window//2),
                    rate_smooth, 'b-', linewidth=2)
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Log10 Convergence Rate')
    ax6.set_title('Convergence Rate Analysis')
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=0, color='r', linestyle='--', alpha=0.3)

    plt.suptitle(f'30-Node System: 80×80m, 500 iterations, Final RMSE: {final_pos_rmse*1000:.2f}mm',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('500iter_80m_results.png', dpi=150, bbox_inches='tight')
    print("\nPlots saved to 500iter_80m_results.png")

    # Print detailed statistics
    print("\n" + "="*60)
    print("Detailed Statistics")
    print("="*60)
    print(f"Area: 80m × 80m (6400 m²)")
    print(f"Connectivity radius: 40m")
    print(f"Measurements: {len(ftl.measurements)} links")
    print(f"Average connectivity: {2*len(ftl.measurements)/30:.1f} links per node")
    print(f"Iterations run: {len(ftl.position_rmse_history)-1}")
    print(f"Final damping λ: {ftl.lm_optimizer.lambda_current:.2e}")

    # Convergence analysis
    if len(ftl.position_rmse_history) > 100:
        rmse_100 = ftl.position_rmse_history[100]
        rmse_500 = ftl.position_rmse_history[-1]
        improvement = (rmse_100 - rmse_500) / rmse_100 * 100
        print(f"\nImprovement from iteration 100 to 500: {improvement:.1f}%")
        print(f"  RMSE at iter 100: {rmse_100*1000:.2f}mm")
        print(f"  RMSE at iter 500: {rmse_500*1000:.2f}mm")

    # Check if converged
    if len(ftl.position_rmse_history) > 10:
        last_10 = ftl.position_rmse_history[-10:]
        variance = np.var(last_10)
        if variance < 1e-10:
            print(f"\n✓ System converged (last 10 iterations variance: {variance:.2e})")
        else:
            print(f"\n⚠ System still improving (last 10 iterations variance: {variance:.2e})")

    return ftl


if __name__ == "__main__":
    ftl = test_500iter_80m()