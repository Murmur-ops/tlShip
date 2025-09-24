"""
Node Geometry and Placement
Handles grid/Poisson placement and anchor layout for N×N arrays
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from enum import Enum


class PlacementType(Enum):
    GRID = "grid"
    POISSON = "poisson"
    PERIMETER = "perimeter"
    CORNERS = "corners"


@dataclass
class NodeGeometry:
    """Node position and type information"""
    node_id: int
    x: float  # meters
    y: float  # meters
    is_anchor: bool = False

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def distance_to(self, other: 'NodeGeometry') -> float:
        """Euclidean distance to another node"""
        return np.linalg.norm(self.position - other.position)


def place_grid_nodes(
    n: int,
    area_size: float,
    jitter_std: float = 0.0,
    seed: Optional[int] = None
) -> List[NodeGeometry]:
    """
    Place N×N nodes on a regular grid with optional jitter

    Args:
        n: Grid dimension (creates n×n nodes)
        area_size: Total area size in meters (area_size × area_size)
        jitter_std: Standard deviation of position jitter in meters
        seed: Random seed for reproducibility

    Returns:
        List of N^2 NodeGeometry objects
    """
    if seed is not None:
        np.random.seed(seed)

    nodes = []
    spacing = area_size / (n + 1)  # Leave margin at edges
    node_id = 0

    for i in range(n):
        for j in range(n):
            # Base grid position
            x = spacing * (i + 1)
            y = spacing * (j + 1)

            # Add jitter if specified
            if jitter_std > 0:
                x += np.random.normal(0, jitter_std)
                y += np.random.normal(0, jitter_std)

            # Ensure within bounds
            x = np.clip(x, 0, area_size)
            y = np.clip(y, 0, area_size)

            nodes.append(NodeGeometry(
                node_id=node_id,
                x=x,
                y=y,
                is_anchor=False
            ))
            node_id += 1

    return nodes


def place_poisson_nodes(
    n_total: int,
    area_size: float,
    min_distance: float = 1.0,
    seed: Optional[int] = None
) -> List[NodeGeometry]:
    """
    Place nodes using Poisson disk sampling for more natural distribution

    Args:
        n_total: Total number of nodes to place
        area_size: Total area size in meters
        min_distance: Minimum distance between nodes
        seed: Random seed

    Returns:
        List of NodeGeometry objects
    """
    if seed is not None:
        np.random.seed(seed)

    nodes = []
    attempts_per_node = 30

    for node_id in range(n_total):
        placed = False

        for _ in range(attempts_per_node):
            x = np.random.uniform(0, area_size)
            y = np.random.uniform(0, area_size)

            # Check minimum distance constraint
            valid = True
            for existing in nodes:
                dist = np.sqrt((x - existing.x)**2 + (y - existing.y)**2)
                if dist < min_distance:
                    valid = False
                    break

            if valid:
                nodes.append(NodeGeometry(
                    node_id=node_id,
                    x=x,
                    y=y,
                    is_anchor=False
                ))
                placed = True
                break

        # If couldn't place with min distance, place randomly
        if not placed:
            x = np.random.uniform(0, area_size)
            y = np.random.uniform(0, area_size)
            nodes.append(NodeGeometry(
                node_id=node_id,
                x=x,
                y=y,
                is_anchor=False
            ))

    return nodes


def place_anchors(
    nodes: List[NodeGeometry],
    m: int,
    area_size: float,
    placement: PlacementType = PlacementType.PERIMETER
) -> List[NodeGeometry]:
    """
    Designate M nodes as anchors or add new anchor nodes

    Args:
        nodes: Existing nodes
        m: Number of anchors
        area_size: Total area size
        placement: Anchor placement strategy

    Returns:
        Updated list with anchors marked/added
    """
    if placement == PlacementType.CORNERS:
        # Place anchors at corners
        if m < 4:
            raise ValueError("Corner placement requires at least 4 anchors")

        corner_positions = [
            (0, 0),
            (area_size, 0),
            (area_size, area_size),
            (0, area_size)
        ]

        # Add corner anchors
        anchor_nodes = []
        for i, (x, y) in enumerate(corner_positions[:min(4, m)]):
            anchor_nodes.append(NodeGeometry(
                node_id=1000 + i,  # High IDs for anchors
                x=x,
                y=y,
                is_anchor=True
            ))

        # Add mid-edge anchors if m > 4
        if m > 4:
            mid_positions = [
                (area_size/2, 0),
                (area_size, area_size/2),
                (area_size/2, area_size),
                (0, area_size/2)
            ]
            for i, (x, y) in enumerate(mid_positions[:m-4]):
                anchor_nodes.append(NodeGeometry(
                    node_id=1004 + i,
                    x=x,
                    y=y,
                    is_anchor=True
                ))

        return nodes + anchor_nodes

    elif placement == PlacementType.PERIMETER:
        # Evenly space anchors around perimeter
        anchor_nodes = []
        perimeter = 4 * area_size
        spacing = perimeter / m

        for i in range(m):
            distance = i * spacing

            if distance < area_size:
                # Bottom edge
                x, y = distance, 0
            elif distance < 2 * area_size:
                # Right edge
                x, y = area_size, distance - area_size
            elif distance < 3 * area_size:
                # Top edge
                x, y = area_size - (distance - 2 * area_size), area_size
            else:
                # Left edge
                x, y = 0, area_size - (distance - 3 * area_size)

            anchor_nodes.append(NodeGeometry(
                node_id=1000 + i,
                x=x,
                y=y,
                is_anchor=True
            ))

        return nodes + anchor_nodes

    else:  # Convert existing nodes to anchors
        # Select nodes closest to ideal positions
        ideal_positions = []
        perimeter = 4 * area_size
        spacing = perimeter / m

        for i in range(m):
            distance = i * spacing
            if distance < area_size:
                ideal_positions.append((distance, 0))
            elif distance < 2 * area_size:
                ideal_positions.append((area_size, distance - area_size))
            elif distance < 3 * area_size:
                ideal_positions.append((area_size - (distance - 2 * area_size), area_size))
            else:
                ideal_positions.append((0, area_size - (distance - 3 * area_size)))

        # Find closest nodes to ideal positions
        selected_indices = []
        for ideal_x, ideal_y in ideal_positions:
            min_dist = float('inf')
            best_idx = -1

            for idx, node in enumerate(nodes):
                if idx not in selected_indices:
                    dist = np.sqrt((node.x - ideal_x)**2 + (node.y - ideal_y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = idx

            if best_idx >= 0:
                selected_indices.append(best_idx)
                nodes[best_idx].is_anchor = True

        return nodes


def get_connectivity_matrix(
    nodes: List[NodeGeometry],
    comm_radius: float
) -> np.ndarray:
    """
    Generate connectivity matrix based on communication radius

    Args:
        nodes: List of nodes
        comm_radius: Maximum communication distance

    Returns:
        NxN boolean connectivity matrix
    """
    n = len(nodes)
    connectivity = np.zeros((n, n), dtype=bool)

    for i in range(n):
        for j in range(n):
            if i != j:
                dist = nodes[i].distance_to(nodes[j])
                connectivity[i, j] = dist <= comm_radius

    return connectivity


def check_graph_rigidity(
    nodes: List[NodeGeometry],
    connectivity: np.ndarray,
    min_anchor_edges: int = 3
) -> Dict[int, bool]:
    """
    Check if each unknown node has sufficient anchor connectivity

    Args:
        nodes: List of nodes
        connectivity: Connectivity matrix
        min_anchor_edges: Minimum required anchor connections

    Returns:
        Dictionary mapping node_id to rigidity status
    """
    rigidity = {}
    anchor_indices = [i for i, node in enumerate(nodes) if node.is_anchor]

    for i, node in enumerate(nodes):
        if not node.is_anchor:
            # Count connections to anchors
            anchor_connections = sum(
                connectivity[i, j] for j in anchor_indices
            )
            rigidity[node.node_id] = anchor_connections >= min_anchor_edges

    return rigidity


if __name__ == "__main__":
    # Test geometry functions
    print("Testing Node Geometry...")
    print("=" * 50)

    # Create 10x10 grid
    nodes = place_grid_nodes(n=10, area_size=100, jitter_std=0.5)
    print(f"Created {len(nodes)} grid nodes")

    # Add 8 anchors on perimeter
    nodes_with_anchors = place_anchors(nodes, m=8, area_size=100)
    n_anchors = sum(1 for n in nodes_with_anchors if n.is_anchor)
    print(f"Added {n_anchors} anchors")

    # Check connectivity
    connectivity = get_connectivity_matrix(nodes_with_anchors, comm_radius=40)
    print(f"Connectivity matrix shape: {connectivity.shape}")
    print(f"Average connections per node: {connectivity.sum() / len(nodes_with_anchors):.1f}")

    # Check rigidity
    rigidity = check_graph_rigidity(nodes_with_anchors, connectivity)
    rigid_nodes = sum(1 for r in rigidity.values() if r)
    print(f"Rigid nodes: {rigid_nodes}/{len(rigidity)}")