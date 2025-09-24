#!/usr/bin/env python3
"""
Test 30-node FTL system over 100x100m area with 500 iterations
"""

import numpy as np
import matplotlib.pyplot as plt
from ftl_enhanced import EnhancedFTL, EnhancedFTLConfig


class Large30NodeFTL_100m(EnhancedFTL):
    """Extended FTL for 100x100m area"""

    def _setup_network(self):
        """Create 30-node network over 100x100m area"""
        area_size = 100.0  # 100m × 100m

        # Generate positions
        self.true_positions = np.zeros((self.config.n_nodes, 2))

        # 5 Anchors at strategic positions
        self.true_positions[0] = [0, 0]           # Bottom-left
        self.true_positions[1] = [area_size, 0]   # Bottom-right
        self.true_positions[2] = [0, area_size]   # Top-left
        self.true_positions[3] = [area_size, area_size]  # Top-right
        self.true_positions[4] = [area_size/2, area_size/2]  # Center

        # 25 Unknown nodes - distributed across area
        np.random.seed(22)  # For reproducibility
        for i in range(5, self.config.n_nodes):
            # Random position with minimum distance from edges
            margin = 5.0
            self.true_positions[i] = [
                np.random.uniform(margin, area_size - margin),
                np.random.uniform(margin, area_size - margin)
            ]

        # Initialize states with error
        np.random.seed(22)  # Different seed for initial errors
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
        connectivity_radius = 50.0  # Increased to 50m for 100x100m area

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

        # Check connectivity
        connectivity_matrix = np.zeros((self.config.n_nodes, self.config.n_nodes))
        for m in self.measurements:
            connectivity_matrix[m['i'], m['j']] = 1
            connectivity_matrix[m['j'], m['i']] = 1

        # Check if graph is connected
        degrees = np.sum(connectivity_matrix, axis=0)
        min_degree = np.min(degrees)
        avg_degree = np.mean(degrees)
        print(f"Connectivity: min degree={min_degree:.0f}, avg degree={avg_degree:.1f}")

        if min_degree == 0:
            print("WARNING: Some nodes are disconnected!")


def test_100m():
    """Test 31-node system over 100x100m area"""
    print("="*60)
    print("31-Node FTL System Test - 100×100m Configuration")
    print("Nodes: 31 (5 anchors + 26 unknowns)")
    print("Area: 100m × 100m (10,000 m²)")
    print("Iterations: 500")
    print("="*60)

    # Configure system
    config = EnhancedFTLConfig(
        n_nodes=31,
        n_anchors=5,
        use_adaptive_lm=True,
        use_line_search=False,
        max_iterations=500,
        lm_initial_lambda=1e-3,
        gradient_tol=1e-8,
        measurement_std=0.01,  # 1cm
        verbose=True
    )

    # Create and run system
    ftl = Large30NodeFTL_100m(config)

    # Print initial errors
    print("\nInitial Errors:")
    print("-"*40)
    initial_pos_errors = []
    initial_time_errors = []
    for i in range(5, 31):
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
    for i in range(5, 31):
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

    # Draw error vectors (scaled for visibility)
    for i in range(5, 31):
        error_vec = ftl.states[i, :2] - ftl.true_positions[i]
        if np.linalg.norm(error_vec) > 0.001:  # Only draw visible errors
            ax1.arrow(ftl.true_positions[i, 0], ftl.true_positions[i, 1],
                     error_vec[0], error_vec[1],
                     head_width=1, head_length=0.5, fc='r', ec='r', alpha=0.5)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('100m × 100m Network Topology')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_xlim([-5, 105])
    ax1.set_ylim([-5, 105])

    # 2. Position convergence
    ax2 = plt.subplot(2, 3, 2)
    ax2.semilogy(ftl.position_rmse_history, 'b-', linewidth=2)
    ax2.axhline(y=final_pos_rmse, color='r', linestyle='--', alpha=0.5,
                label=f'Final: {final_pos_rmse*1000:.2f}mm')
    ax2.axhline(y=0.0185, color='g', linestyle=':', alpha=0.5,
                label='Target: 18.5mm')
    ax2.axhline(y=0.001, color='orange', linestyle=':', alpha=0.5,
                label='Sub-mm: 1mm')
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

    # 4. Lambda adaptation
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
    if len(ftl.lambda_history) > 0:
        ax4.legend()

    # 5. Position error distribution
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(np.array(final_pos_errors)*1000, bins=15, edgecolor='black', alpha=0.7, color='skyblue')
    ax5.axvline(x=final_pos_rmse*1000, color='r', linestyle='--',
                label=f'RMSE: {final_pos_rmse*1000:.2f}mm', linewidth=2)
    ax5.axvline(x=18.5, color='g', linestyle=':',
                label='Target: 18.5mm', linewidth=2)
    ax5.axvline(x=1.0, color='orange', linestyle=':',
                label='Sub-mm: 1mm', linewidth=2)
    ax5.set_xlabel('Position Error (mm)')
    ax5.set_ylabel('Number of Nodes')
    ax5.set_title('Final Position Error Distribution')
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    # 6. Distance vs Error Analysis
    ax6 = plt.subplot(2, 3, 6)
    # Calculate distance from each node to nearest anchor
    distances_to_anchor = []
    for i in range(5, 31):
        min_dist = min([np.linalg.norm(ftl.true_positions[i] - ftl.true_positions[j])
                       for j in range(5)])
        distances_to_anchor.append(min_dist)

    node_errors_mm = [final_pos_errors[i]*1000 for i in range(26)]

    ax6.scatter(distances_to_anchor, node_errors_mm, alpha=0.7, s=50)
    ax6.set_xlabel('Distance to Nearest Anchor (m)')
    ax6.set_ylabel('Position Error (mm)')
    ax6.set_title('Error vs Distance from Anchors')
    ax6.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(distances_to_anchor, node_errors_mm, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(distances_to_anchor), max(distances_to_anchor), 100)
    ax6.plot(x_trend, p(x_trend), "r--", alpha=0.5, label=f'Trend: {z[0]:.3f}mm/m')
    ax6.legend()

    plt.suptitle(f'31-Node System: 100×100m, 500 iterations, Final RMSE: {final_pos_rmse*1000:.2f}mm',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('100m_results.png', dpi=150, bbox_inches='tight')
    print("\nPlots saved to 100m_results.png")

    # Print detailed statistics
    print("\n" + "="*60)
    print("Detailed Statistics")
    print("="*60)
    print(f"Area: 100m × 100m (10,000 m²)")
    print(f"Connectivity radius: 50m")
    print(f"Measurements: {len(ftl.measurements)} links")
    print(f"Average connectivity: {2*len(ftl.measurements)/31:.1f} links per node")
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

    # Performance vs area analysis
    print(f"\nArea Scaling Analysis:")
    print(f"  50×50m (2,500 m²): ~18.5mm RMSE")
    print(f"  80×80m (6,400 m²): ~0.97mm RMSE")
    print(f"  90×90m (8,100 m²): ~0.83mm RMSE")
    print(f"  100×100m (10,000 m²): {final_pos_rmse*1000:.2f}mm RMSE")
    area_factor = (100*100)/(50*50)
    print(f"  Area increase: {area_factor:.1f}× from baseline")

    # Check if sub-millimeter achieved
    if final_pos_rmse < 0.001:
        print("\n✓ SUB-MILLIMETER accuracy achieved!")
    elif final_pos_rmse < 0.0185:
        print("\n✓ Target 18.5mm accuracy achieved!")
    else:
        print(f"\n⚠ Above target: {final_pos_rmse*1000:.2f}mm > 18.5mm")

    # Check convergence status
    if len(ftl.position_rmse_history) > 10:
        last_10 = ftl.position_rmse_history[-10:]
        variance = np.var(last_10)
        if variance < 1e-10:
            print(f"✓ System converged (last 10 iterations variance: {variance:.2e})")
        else:
            print(f"⚠ System still improving (last 10 iterations variance: {variance:.2e})")

    # Node performance breakdown
    errors_with_idx = [(i, final_pos_errors[i-5]) for i in range(5, 31)]
    errors_with_idx.sort(key=lambda x: x[1], reverse=True)

    print("\nWorst 5 nodes (highest position error):")
    for idx, (node_id, error) in enumerate(errors_with_idx[:5]):
        dist_to_nearest = min([np.linalg.norm(ftl.true_positions[node_id] - ftl.true_positions[j])
                              for j in range(5)])
        print(f"  Node {node_id}: {error*1000:.2f}mm (nearest anchor: {dist_to_nearest:.1f}m)")

    print("\nBest 5 nodes (lowest position error):")
    for idx, (node_id, error) in enumerate(errors_with_idx[-5:]):
        dist_to_nearest = min([np.linalg.norm(ftl.true_positions[node_id] - ftl.true_positions[j])
                              for j in range(5)])
        print(f"  Node {node_id}: {error*1000:.2f}mm (nearest anchor: {dist_to_nearest:.1f}m)")

    # Print final positions
    print("\n" + "="*60)
    print("Final Node Positions")
    print("="*60)
    print("Node ID | True Position (x,y) | Estimated Position (x,y) | Error (mm)")
    print("-"*70)
    for i in range(31):
        true_pos = ftl.true_positions[i]
        est_pos = ftl.states[i, :2]
        if i < 5:
            error = 0  # Anchors have perfect position
            print(f"Anchor {i:2d} | ({true_pos[0]:6.2f}, {true_pos[1]:6.2f}) | "
                  f"({est_pos[0]:6.2f}, {est_pos[1]:6.2f}) | {error:6.2f}")
        else:
            error = np.linalg.norm(true_pos - est_pos) * 1000  # Convert to mm
            print(f"Node {i:4d} | ({true_pos[0]:6.2f}, {true_pos[1]:6.2f}) | "
                  f"({est_pos[0]:6.2f}, {est_pos[1]:6.2f}) | {error:6.2f}")

    # Create a dedicated figure for true vs estimated positions
    plt.figure(figsize=(10, 10))

    # Plot measurements as faint lines
    for m in ftl.measurements:
        i, j = m['i'], m['j']
        plt.plot([ftl.true_positions[i, 0], ftl.true_positions[j, 0]],
                [ftl.true_positions[i, 1], ftl.true_positions[j, 1]],
                'gray', alpha=0.1, linewidth=0.5, zorder=1)

    # Plot true positions
    plt.scatter(ftl.true_positions[:5, 0], ftl.true_positions[:5, 1],
               c='red', s=300, marker='^', label='Anchors', edgecolor='black', linewidth=2, zorder=5)
    plt.scatter(ftl.true_positions[5:, 0], ftl.true_positions[5:, 1],
               c='blue', s=150, marker='o', label='True Positions', edgecolor='black', linewidth=1, zorder=3)

    # Plot estimated positions
    plt.scatter(ftl.states[5:, 0], ftl.states[5:, 1],
               c='green', s=100, marker='x', label='Estimated Positions', linewidth=2, zorder=4)

    # Draw error lines between true and estimated
    for i in range(5, 31):
        plt.plot([ftl.true_positions[i, 0], ftl.states[i, 0]],
                [ftl.true_positions[i, 1], ftl.states[i, 1]],
                'r-', alpha=0.6, linewidth=1, zorder=2)

    # Add node labels
    for i in range(31):
        if i < 5:
            plt.annotate(f'A{i}', (ftl.true_positions[i, 0], ftl.true_positions[i, 1]),
                        xytext=(3, 3), textcoords='offset points', fontsize=8, fontweight='bold')
        else:
            plt.annotate(f'{i}', (ftl.true_positions[i, 0], ftl.true_positions[i, 1]),
                        xytext=(3, 3), textcoords='offset points', fontsize=7)

    plt.xlabel('X Position (m)', fontsize=12)
    plt.ylabel('Y Position (m)', fontsize=12)
    plt.title(f'True vs Estimated Positions - 31 Nodes over 100×100m\nRMSE: {final_pos_rmse*1000:.2f}mm', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlim([-5, 105])
    plt.ylim([-5, 105])

    # Add text box with statistics
    stats_text = f'Nodes: 31 (5 anchors + 26 unknowns)\n'
    stats_text += f'RMSE: {final_pos_rmse*1000:.2f}mm\n'
    stats_text += f'Max Error: {max(final_pos_errors)*1000:.2f}mm\n'
    stats_text += f'Min Error: {min(final_pos_errors)*1000:.2f}mm'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('final_positions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nPosition comparison figure saved to final_positions.png")

    return ftl


if __name__ == "__main__":
    ftl = test_100m()