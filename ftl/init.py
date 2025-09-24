"""
Initialization Methods for FTL
Trilateration, MDS, and Grid Search for initial position estimates
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy.optimize import least_squares
import warnings


def trilateration(
    anchor_positions: np.ndarray,
    distances: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Closed-form trilateration for position estimation

    Uses linearized least squares for overdetermined systems.

    Args:
        anchor_positions: (M, 2) array of anchor [x, y] positions
        distances: (M,) array of distances to anchors
        weights: Optional (M,) array of measurement weights

    Returns:
        Estimated position [x, y]
    """
    M = len(anchor_positions)
    if M < 3:
        raise ValueError(f"Need at least 3 anchors, got {M}")

    if weights is None:
        weights = np.ones(M)

    # Use first anchor as reference
    ref_pos = anchor_positions[0]
    ref_dist = distances[0]

    # Build linear system: Ax = b
    # From: (x - xi)^2 + (y - yi)^2 = di^2
    # Linearized relative to first anchor
    A = []
    b = []
    W = []

    for i in range(1, M):
        # 2(xi - x0)x + 2(yi - y0)y = di^2 - d0^2 - ||pi||^2 + ||p0||^2
        dx = anchor_positions[i, 0] - ref_pos[0]
        dy = anchor_positions[i, 1] - ref_pos[1]

        A.append([dx, dy])

        dist_diff = ref_dist**2 - distances[i]**2
        pos_diff = np.sum(anchor_positions[i]**2) - np.sum(ref_pos**2)
        b.append((dist_diff + pos_diff) / 2)

        # Weight by geometric mean
        W.append(np.sqrt(weights[0] * weights[i]))

    A = np.array(A)
    b = np.array(b)
    W = np.diag(W)

    # Weighted least squares
    try:
        # x = (A^T W A)^-1 A^T W b
        x = np.linalg.solve(A.T @ W @ A, A.T @ W @ b)
        position = x
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse
        x = np.linalg.pinv(A) @ b
        position = x

    return position


def mds(
    distance_matrix: np.ndarray,
    anchor_positions: Optional[np.ndarray] = None,
    anchor_indices: Optional[List[int]] = None
) -> np.ndarray:
    """
    Multidimensional Scaling for position estimation

    Recovers positions from pairwise distances using eigendecomposition.

    Args:
        distance_matrix: (N, N) symmetric distance matrix
        anchor_positions: Optional (M, 2) known anchor positions
        anchor_indices: Indices of anchors in distance matrix

    Returns:
        (N, 2) array of estimated positions
    """
    N = len(distance_matrix)

    # Classical MDS algorithm
    # 1. Square distances
    D_squared = distance_matrix**2

    # 2. Double centering
    row_mean = np.mean(D_squared, axis=1, keepdims=True)
    col_mean = np.mean(D_squared, axis=0, keepdims=True)
    total_mean = np.mean(D_squared)

    B = -0.5 * (D_squared - row_mean - col_mean + total_mean)

    # 3. Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)

    # Sort by eigenvalue magnitude
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 4. Extract 2D coordinates
    # Use top 2 eigenvalues/vectors
    if eigenvalues[0] < 0 or eigenvalues[1] < 0:
        warnings.warn("Negative eigenvalues in MDS, distance matrix may be inconsistent")
        eigenvalues = np.maximum(eigenvalues, 0)

    coords = eigenvectors[:, :2] @ np.diag(np.sqrt(eigenvalues[:2]))

    # 5. Align with anchors if provided (Procrustes alignment)
    if anchor_positions is not None and anchor_indices is not None:
        coords = align_to_anchors(coords, anchor_positions, anchor_indices)

    return coords


def align_to_anchors(
    positions: np.ndarray,
    anchor_positions: np.ndarray,
    anchor_indices: List[int]
) -> np.ndarray:
    """
    Align MDS solution to known anchor positions

    Uses Procrustes analysis to find optimal rotation, translation, and scale.

    Args:
        positions: (N, 2) MDS positions
        anchor_positions: (M, 2) true anchor positions
        anchor_indices: Indices of anchors

    Returns:
        Aligned positions
    """
    # Extract anchor positions from MDS solution
    mds_anchors = positions[anchor_indices]

    # Center both sets
    mds_center = np.mean(mds_anchors, axis=0)
    true_center = np.mean(anchor_positions, axis=0)

    mds_centered = mds_anchors - mds_center
    true_centered = anchor_positions - true_center

    # Find optimal rotation (Kabsch algorithm)
    H = mds_centered.T @ true_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Find optimal scale
    scale = np.sum(S) / np.sum(mds_centered**2)

    # Apply transformation to all positions
    aligned = scale * (positions - np.mean(positions[anchor_indices], axis=0)) @ R.T
    aligned += true_center

    return aligned


def grid_search(
    anchor_positions: np.ndarray,
    measurements: Dict,
    area_bounds: Tuple[float, float, float, float],
    grid_resolution: float = 1.0
) -> np.ndarray:
    """
    Coarse grid search for initial position

    Evaluates likelihood over spatial grid.

    Args:
        anchor_positions: (M, 2) anchor positions
        measurements: Dict with 'distances', 'variances', etc.
        area_bounds: (xmin, xmax, ymin, ymax)
        grid_resolution: Grid spacing in meters

    Returns:
        Best position estimate [x, y]
    """
    xmin, xmax, ymin, ymax = area_bounds

    # Create grid
    x_grid = np.arange(xmin, xmax, grid_resolution)
    y_grid = np.arange(ymin, ymax, grid_resolution)

    # Evaluate likelihood at each point
    best_pos = None
    best_cost = float('inf')

    distances = measurements.get('distances', [])
    variances = measurements.get('variances', np.ones(len(distances)))

    for x in x_grid:
        for y in y_grid:
            pos = np.array([x, y])

            # Compute residuals
            cost = 0
            for i, (anchor, dist, var) in enumerate(zip(anchor_positions, distances, variances)):
                pred_dist = np.linalg.norm(pos - anchor)
                residual = (pred_dist - dist) / np.sqrt(var)
                cost += residual**2

            if cost < best_cost:
                best_cost = cost
                best_pos = pos

    # Refine with local optimization
    if best_pos is not None:
        def objective(x):
            residuals = []
            for anchor, dist, var in zip(anchor_positions, distances, variances):
                pred_dist = np.linalg.norm(x - anchor)
                residuals.append((pred_dist - dist) / np.sqrt(var))
            return residuals

        result = least_squares(objective, best_pos, bounds=([xmin, ymin], [xmax, ymax]))
        best_pos = result.x

    return best_pos


def initialize_positions(
    n_nodes: int,
    anchor_positions: np.ndarray,
    anchor_indices: List[int],
    measurements: Optional[Dict] = None,
    method: str = 'trilateration'
) -> np.ndarray:
    """
    Initialize node positions for factor graph optimization

    Args:
        n_nodes: Total number of nodes
        anchor_positions: (M, 2) anchor positions
        anchor_indices: Indices of anchors
        measurements: Distance measurements
        method: 'trilateration', 'mds', or 'grid'

    Returns:
        (N, 2) initial position estimates
    """
    positions = np.zeros((n_nodes, 2))

    # Set anchor positions
    for i, idx in enumerate(anchor_indices):
        positions[idx] = anchor_positions[i]

    # Initialize unknown nodes
    unknown_indices = [i for i in range(n_nodes) if i not in anchor_indices]

    if method == 'trilateration' and measurements:
        # Use trilateration for each unknown
        for idx in unknown_indices:
            if f'distances_{idx}' in measurements:
                distances = measurements[f'distances_{idx}']
                weights = measurements.get(f'weights_{idx}')
                try:
                    positions[idx] = trilateration(anchor_positions, distances, weights)
                except:
                    # Fallback to centroid
                    positions[idx] = np.mean(anchor_positions, axis=0)

    elif method == 'mds' and measurements and 'distance_matrix' in measurements:
        # Use MDS for all nodes
        all_positions = mds(
            measurements['distance_matrix'],
            anchor_positions,
            anchor_indices
        )
        positions = all_positions

    elif method == 'grid' and measurements:
        # Grid search for unknowns
        area_size = np.max(anchor_positions) - np.min(anchor_positions)
        bounds = (
            np.min(anchor_positions[:, 0]) - area_size/4,
            np.max(anchor_positions[:, 0]) + area_size/4,
            np.min(anchor_positions[:, 1]) - area_size/4,
            np.max(anchor_positions[:, 1]) + area_size/4
        )

        for idx in unknown_indices:
            if f'distances_{idx}' in measurements:
                node_meas = {
                    'distances': measurements[f'distances_{idx}'],
                    'variances': measurements.get(f'variances_{idx}')
                }
                positions[idx] = grid_search(anchor_positions, node_meas, bounds)

    else:
        # Fallback: random initialization near center
        center = np.mean(anchor_positions, axis=0)
        spread = np.std(anchor_positions) / 2
        for idx in unknown_indices:
            positions[idx] = center + np.random.randn(2) * spread

    return positions


def initialize_clock_states(
    n_nodes: int,
    anchor_indices: List[int],
    oscillator_types: Optional[Dict] = None
) -> np.ndarray:
    """
    Initialize clock states [bias, drift, CFO]

    Args:
        n_nodes: Number of nodes
        anchor_indices: Indices with good clocks
        oscillator_types: Dict mapping node to oscillator type

    Returns:
        (N, 3) clock state array
    """
    clock_states = np.zeros((n_nodes, 3))

    # Anchors have better clocks
    for idx in range(n_nodes):
        if idx in anchor_indices:
            # OCXO-like stability
            clock_states[idx, 0] = np.random.randn() * 1e-9  # 1ns bias
            clock_states[idx, 1] = np.random.randn() * 1e-12  # 1ps/s drift
            clock_states[idx, 2] = np.random.randn() * 0.1  # 0.1 Hz CFO
        else:
            # TCXO-like stability
            clock_states[idx, 0] = np.random.randn() * 1e-6  # 1μs bias
            clock_states[idx, 1] = np.random.randn() * 1e-9  # 1ns/s drift
            clock_states[idx, 2] = np.random.randn() * 10  # 10 Hz CFO

    return clock_states


if __name__ == "__main__":
    # Test initialization methods
    print("Testing Initialization Methods...")
    print("=" * 50)

    # Test trilateration
    anchors = np.array([[0, 0], [10, 0], [5, 8.66]])
    true_pos = np.array([5, 3])
    distances = np.array([
        np.linalg.norm(true_pos - anchors[0]),
        np.linalg.norm(true_pos - anchors[1]),
        np.linalg.norm(true_pos - anchors[2])
    ])

    print("\nTrilateration Test:")
    estimated = trilateration(anchors, distances)
    error = np.linalg.norm(estimated - true_pos)
    print(f"  True position: {true_pos}")
    print(f"  Estimated: {estimated}")
    print(f"  Error: {error:.3f} m")

    # Test MDS
    print("\nMDS Test:")
    # Create small distance matrix
    positions = np.array([[0, 0], [10, 0], [5, 8.66], [5, 3]])
    N = len(positions)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D[i, j] = np.linalg.norm(positions[i] - positions[j])

    recovered = mds(D, positions[:3], [0, 1, 2])
    errors = [np.linalg.norm(recovered[i] - positions[i]) for i in range(N)]
    print(f"  Position errors: {errors}")
    print(f"  Mean error: {np.mean(errors):.3f} m")