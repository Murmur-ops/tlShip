"""
Performance Metrics for FTL Evaluation
RMSE, MAE, CRLB efficiency, convergence analysis
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import matplotlib.pyplot as plt


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    position_rmse: float = 0.0
    position_mae: float = 0.0
    position_percentiles: Dict[int, float] = field(default_factory=dict)
    clock_bias_mae: float = 0.0
    clock_drift_mae: float = 0.0
    cfo_rmse: float = 0.0
    crlb_efficiency: float = 0.0
    convergence_iterations: int = 0
    convergence_time: float = 0.0
    outlier_percentage: float = 0.0


def position_rmse(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
    exclude_anchors: bool = True,
    anchor_indices: Optional[List[int]] = None
) -> float:
    """
    Calculate position Root Mean Square Error

    Args:
        estimated: (N, 2) estimated positions
        ground_truth: (N, 2) true positions
        exclude_anchors: Whether to exclude anchors from calculation
        anchor_indices: Indices of anchor nodes

    Returns:
        RMSE in meters
    """
    if exclude_anchors and anchor_indices:
        mask = np.ones(len(estimated), dtype=bool)
        mask[anchor_indices] = False
        estimated = estimated[mask]
        ground_truth = ground_truth[mask]

    errors = np.linalg.norm(estimated - ground_truth, axis=1)
    rmse = np.sqrt(np.mean(errors**2))

    return rmse


def position_mae(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
    exclude_anchors: bool = True,
    anchor_indices: Optional[List[int]] = None
) -> float:
    """
    Calculate position Mean Absolute Error

    Args:
        estimated: (N, 2) estimated positions
        ground_truth: (N, 2) true positions
        exclude_anchors: Whether to exclude anchors
        anchor_indices: Indices of anchor nodes

    Returns:
        MAE in meters
    """
    if exclude_anchors and anchor_indices:
        mask = np.ones(len(estimated), dtype=bool)
        mask[anchor_indices] = False
        estimated = estimated[mask]
        ground_truth = ground_truth[mask]

    errors = np.linalg.norm(estimated - ground_truth, axis=1)
    mae = np.mean(errors)

    return mae


def position_percentiles(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
    percentiles: List[int] = [50, 90, 95, 99],
    exclude_anchors: bool = True,
    anchor_indices: Optional[List[int]] = None
) -> Dict[int, float]:
    """
    Calculate position error percentiles

    Args:
        estimated: (N, 2) estimated positions
        ground_truth: (N, 2) true positions
        percentiles: List of percentiles to compute
        exclude_anchors: Whether to exclude anchors
        anchor_indices: Indices of anchor nodes

    Returns:
        Dictionary of percentile values
    """
    if exclude_anchors and anchor_indices:
        mask = np.ones(len(estimated), dtype=bool)
        mask[anchor_indices] = False
        estimated = estimated[mask]
        ground_truth = ground_truth[mask]

    errors = np.linalg.norm(estimated - ground_truth, axis=1)
    result = {}

    for p in percentiles:
        result[p] = np.percentile(errors, p)

    return result


def clock_mae(
    estimated_clocks: np.ndarray,
    true_clocks: np.ndarray
) -> Tuple[float, float, float]:
    """
    Calculate clock parameter MAE

    Args:
        estimated_clocks: (N, 3) estimated [bias, drift, cfo]
        true_clocks: (N, 3) true [bias, drift, cfo]

    Returns:
        (bias_mae, drift_mae, cfo_mae)
    """
    bias_mae = np.mean(np.abs(estimated_clocks[:, 0] - true_clocks[:, 0]))
    drift_mae = np.mean(np.abs(estimated_clocks[:, 1] - true_clocks[:, 1]))
    cfo_mae = np.mean(np.abs(estimated_clocks[:, 2] - true_clocks[:, 2]))

    return bias_mae, drift_mae, cfo_mae


def crlb_efficiency(
    achieved_variance: float,
    theoretical_crlb: float
) -> float:
    """
    Calculate CRLB efficiency (ratio of theoretical to achieved)

    Args:
        achieved_variance: Measured variance
        theoretical_crlb: Theoretical CRLB

    Returns:
        Efficiency in [0, 1], where 1 is optimal
    """
    if achieved_variance <= 0:
        return 0.0

    efficiency = theoretical_crlb / achieved_variance
    return min(1.0, efficiency)  # Cap at 1.0


def convergence_analysis(
    cost_history: List[float],
    tolerance: float = 1e-6
) -> Dict:
    """
    Analyze convergence behavior

    Args:
        cost_history: List of cost values per iteration
        tolerance: Convergence tolerance

    Returns:
        Dictionary with convergence metrics
    """
    if len(cost_history) < 2:
        return {
            'converged': False,
            'iterations': len(cost_history),
            'final_cost': cost_history[-1] if cost_history else float('inf'),
            'cost_reduction': 0.0,
            'convergence_rate': 0.0
        }

    # Find convergence point
    converged_iter = len(cost_history)
    for i in range(1, len(cost_history)):
        if abs(cost_history[i] - cost_history[i-1]) < tolerance:
            converged_iter = i
            break

    # Calculate convergence rate (exponential fit)
    if len(cost_history) > 2:
        # log(cost_i - cost_final) = log(a) - b*i
        final_cost = cost_history[-1]
        log_errors = []
        for i, c in enumerate(cost_history[:-1]):
            if c > final_cost:
                log_errors.append((i, np.log(c - final_cost)))

        if len(log_errors) > 1:
            # Linear regression for convergence rate
            X = np.array([x[0] for x in log_errors])
            y = np.array([x[1] for x in log_errors])
            rate = -np.polyfit(X, y, 1)[0]
        else:
            rate = 0.0
    else:
        rate = 0.0

    return {
        'converged': converged_iter < len(cost_history),
        'iterations': converged_iter,
        'final_cost': cost_history[-1],
        'cost_reduction': 1 - cost_history[-1] / cost_history[0],
        'convergence_rate': rate
    }


def evaluate_ftl_performance(
    estimated_states: Dict[int, np.ndarray],
    ground_truth_states: Dict[int, np.ndarray],
    anchor_indices: List[int],
    cost_history: Optional[List[float]] = None,
    theoretical_crlb: Optional[float] = None
) -> PerformanceMetrics:
    """
    Comprehensive FTL performance evaluation

    Args:
        estimated_states: Dict mapping node_id to [x, y, b, d, f]
        ground_truth_states: Dict mapping node_id to true [x, y, b, d, f]
        anchor_indices: List of anchor node indices
        cost_history: Optimization cost per iteration
        theoretical_crlb: Theoretical CRLB for comparison

    Returns:
        PerformanceMetrics object
    """
    metrics = PerformanceMetrics()

    # Extract positions and clocks
    n_nodes = len(estimated_states)
    est_pos = np.array([estimated_states[i][:2] for i in range(n_nodes)])
    true_pos = np.array([ground_truth_states[i][:2] for i in range(n_nodes)])
    est_clocks = np.array([estimated_states[i][2:5] for i in range(n_nodes)])
    true_clocks = np.array([ground_truth_states[i][2:5] for i in range(n_nodes)])

    # Position metrics
    metrics.position_rmse = position_rmse(est_pos, true_pos, True, anchor_indices)
    metrics.position_mae = position_mae(est_pos, true_pos, True, anchor_indices)
    metrics.position_percentiles = position_percentiles(est_pos, true_pos,
                                                       [50, 90, 95, 99],
                                                       True, anchor_indices)

    # Clock metrics
    bias_mae, drift_mae, cfo_mae = clock_mae(est_clocks, true_clocks)
    metrics.clock_bias_mae = bias_mae
    metrics.clock_drift_mae = drift_mae
    metrics.cfo_rmse = np.sqrt(np.mean((est_clocks[:, 2] - true_clocks[:, 2])**2))

    # CRLB efficiency
    if theoretical_crlb:
        achieved_var = metrics.position_rmse**2
        metrics.crlb_efficiency = crlb_efficiency(achieved_var, theoretical_crlb)

    # Convergence analysis
    if cost_history:
        conv = convergence_analysis(cost_history)
        metrics.convergence_iterations = conv['iterations']

    return metrics


def plot_performance(
    metrics: PerformanceMetrics,
    save_path: Optional[str] = None
):
    """
    Plot performance metrics

    Args:
        metrics: Performance metrics object
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Position error CDF
    ax = axes[0, 0]
    if metrics.position_percentiles:
        percentiles = sorted(metrics.position_percentiles.keys())
        values = [metrics.position_percentiles[p] for p in percentiles]
        ax.plot(values, percentiles, 'b-', linewidth=2)
        ax.set_xlabel('Position Error (m)')
        ax.set_ylabel('CDF (%)')
        ax.set_title('Position Error Distribution')
        ax.grid(True, alpha=0.3)

    # Clock errors
    ax = axes[0, 1]
    clock_errors = [
        metrics.clock_bias_mae * 1e9,  # Convert to ns
        metrics.clock_drift_mae * 1e12,  # Convert to ps/s
        metrics.cfo_rmse  # Hz
    ]
    labels = ['Bias (ns)', 'Drift (ps/s)', 'CFO (Hz)']
    ax.bar(labels, clock_errors)
    ax.set_title('Clock Parameter Errors')
    ax.set_ylabel('MAE / RMSE')

    # CRLB efficiency
    ax = axes[1, 0]
    efficiency_pct = metrics.crlb_efficiency * 100
    ax.bar(['CRLB Efficiency'], [efficiency_pct])
    ax.set_ylim([0, 100])
    ax.set_ylabel('Efficiency (%)')
    ax.set_title(f'CRLB Efficiency: {efficiency_pct:.1f}%')

    # Summary text
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = f"""Performance Summary:

Position RMSE: {metrics.position_rmse:.3f} m
Position MAE: {metrics.position_mae:.3f} m
Median Error: {metrics.position_percentiles.get(50, 0):.3f} m
90% Error: {metrics.position_percentiles.get(90, 0):.3f} m
95% Error: {metrics.position_percentiles.get(95, 0):.3f} m

Clock Bias MAE: {metrics.clock_bias_mae*1e9:.2f} ns
Clock Drift MAE: {metrics.clock_drift_mae*1e12:.2f} ps/s
CFO RMSE: {metrics.cfo_rmse:.2f} Hz

Convergence: {metrics.convergence_iterations} iterations
CRLB Efficiency: {metrics.crlb_efficiency*100:.1f}%"""

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', family='monospace')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()

    return fig


def compare_methods(
    results: Dict[str, PerformanceMetrics],
    save_path: Optional[str] = None
):
    """
    Compare performance across different methods

    Args:
        results: Dict mapping method name to metrics
        save_path: Optional save path
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    methods = list(results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))

    # Position RMSE comparison
    ax = axes[0, 0]
    rmse_values = [results[m].position_rmse for m in methods]
    bars = ax.bar(methods, rmse_values, color=colors)
    ax.set_ylabel('RMSE (m)')
    ax.set_title('Position RMSE Comparison')
    ax.tick_params(axis='x', rotation=45)

    # Position percentiles
    ax = axes[0, 1]
    x = np.arange(len(methods))
    width = 0.2
    percentiles_to_plot = [50, 90, 95]

    for i, p in enumerate(percentiles_to_plot):
        values = [results[m].position_percentiles.get(p, 0) for m in methods]
        ax.bar(x + i*width, values, width, label=f'{p}th %ile')

    ax.set_xlabel('Method')
    ax.set_ylabel('Error (m)')
    ax.set_title('Position Error Percentiles')
    ax.set_xticks(x + width)
    ax.set_xticklabels(methods, rotation=45)
    ax.legend()

    # CRLB Efficiency
    ax = axes[1, 0]
    efficiency = [results[m].crlb_efficiency * 100 for m in methods]
    ax.bar(methods, efficiency, color=colors)
    ax.set_ylabel('Efficiency (%)')
    ax.set_title('CRLB Efficiency')
    ax.set_ylim([0, 100])
    ax.tick_params(axis='x', rotation=45)

    # Convergence iterations
    ax = axes[1, 1]
    iterations = [results[m].convergence_iterations for m in methods]
    ax.bar(methods, iterations, color=colors)
    ax.set_ylabel('Iterations')
    ax.set_title('Convergence Speed')
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()

    return fig


if __name__ == "__main__":
    # Test metrics calculation
    print("Testing Performance Metrics...")
    print("=" * 50)

    # Generate test data
    n_nodes = 10
    true_positions = np.random.randn(n_nodes, 2) * 10
    est_positions = true_positions + np.random.randn(n_nodes, 2) * 0.5

    true_clocks = np.random.randn(n_nodes, 3) * np.array([1e-6, 1e-9, 10])
    est_clocks = true_clocks + np.random.randn(n_nodes, 3) * np.array([1e-7, 1e-10, 1])

    # Calculate metrics
    rmse = position_rmse(est_positions, true_positions)
    mae = position_mae(est_positions, true_positions)
    percentiles = position_percentiles(est_positions, true_positions)

    print(f"\nPosition Metrics:")
    print(f"  RMSE: {rmse:.3f} m")
    print(f"  MAE: {mae:.3f} m")
    print(f"  Percentiles: {percentiles}")

    bias_mae, drift_mae, cfo_mae = clock_mae(est_clocks, true_clocks)
    print(f"\nClock Metrics:")
    print(f"  Bias MAE: {bias_mae*1e9:.2f} ns")
    print(f"  Drift MAE: {drift_mae*1e12:.2f} ps/s")
    print(f"  CFO MAE: {cfo_mae:.2f} Hz")

    # Test CRLB efficiency
    theoretical_crlb = 0.01  # 1cm theoretical
    achieved_var = rmse**2
    efficiency = crlb_efficiency(achieved_var, theoretical_crlb)
    print(f"\nCRLB Efficiency: {efficiency*100:.1f}%")