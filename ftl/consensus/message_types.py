"""
Message types for distributed consensus communication
Real message structures with serialization support
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum
import hashlib
import time
import json


class MessageType(Enum):
    """Types of messages in the consensus protocol"""
    STATE = "state"  # Node state broadcast
    MEASUREMENT = "measurement"  # Raw measurement data
    CONVERGENCE = "convergence"  # Convergence status
    HEARTBEAT = "heartbeat"  # Keepalive
    REQUEST_STATE = "request_state"  # Request current state from neighbor


@dataclass
class StateMessage:
    """
    Node state message for consensus
    Contains actual state estimate and metadata
    """
    node_id: int
    state: np.ndarray  # [x_m, y_m, bias_ns, drift_ppb, cfo_ppm]
    iteration: int
    timestamp: float
    covariance: Optional[np.ndarray] = None  # Optional uncertainty estimate
    auth_tag: Optional[bytes] = None  # Authentication tag

    def __post_init__(self):
        """Validate and convert state to numpy array"""
        if not isinstance(self.state, np.ndarray):
            self.state = np.array(self.state)
        assert self.state.shape == (5,), f"State must be 5D, got {self.state.shape}"

        if self.timestamp is None:
            self.timestamp = time.time()

    def serialize(self) -> bytes:
        """Serialize message for network transmission"""
        data = {
            'node_id': self.node_id,
            'state': self.state.tolist(),
            'iteration': self.iteration,
            'timestamp': self.timestamp,
            'covariance': self.covariance.tolist() if self.covariance is not None else None
        }

        # Create JSON representation
        json_str = json.dumps(data, separators=(',', ':'))

        # Add authentication if needed
        if self.auth_tag is None:
            self.auth_tag = self._compute_auth_tag(json_str.encode())

        # Combine data and auth tag
        message = json_str.encode() + b'||' + self.auth_tag
        return message

    @classmethod
    def deserialize(cls, data: bytes) -> 'StateMessage':
        """Deserialize message from network"""
        try:
            # Split data and auth tag
            parts = data.split(b'||')
            json_data = parts[0]
            auth_tag = parts[1] if len(parts) > 1 else None

            # Parse JSON
            msg_dict = json.loads(json_data.decode())

            # Reconstruct numpy arrays
            state = np.array(msg_dict['state'])
            covariance = np.array(msg_dict['covariance']) if msg_dict['covariance'] else None

            return cls(
                node_id=msg_dict['node_id'],
                state=state,
                iteration=msg_dict['iteration'],
                timestamp=msg_dict['timestamp'],
                covariance=covariance,
                auth_tag=auth_tag
            )
        except Exception as e:
            raise ValueError(f"Failed to deserialize StateMessage: {e}")

    def _compute_auth_tag(self, data: bytes) -> bytes:
        """Compute authentication tag (simplified - use HMAC in production)"""
        return hashlib.sha256(data).digest()[:8]

    def verify_auth(self, expected_tag: bytes) -> bool:
        """Verify message authentication"""
        if self.auth_tag is None:
            return False
        return self.auth_tag == expected_tag

    def age(self) -> float:
        """Return age of message in seconds"""
        return time.time() - self.timestamp

    @property
    def position(self) -> np.ndarray:
        """Get position component [x, y]"""
        return self.state[:2]

    @property
    def clock_params(self) -> np.ndarray:
        """Get clock parameters [bias, drift, cfo]"""
        return self.state[2:]


@dataclass
class NetworkMessage:
    """
    Generic network message wrapper
    """
    msg_type: MessageType
    sender_id: int
    receiver_id: int  # -1 for broadcast
    payload: Any
    timestamp: float = field(default_factory=time.time)
    hop_count: int = 0  # For multi-hop routing

    def increment_hop(self):
        """Increment hop count for routing"""
        self.hop_count += 1

    def is_broadcast(self) -> bool:
        """Check if this is a broadcast message"""
        return self.receiver_id == -1

    def is_stale(self, max_age: float = 1.0) -> bool:
        """Check if message is too old"""
        return (time.time() - self.timestamp) > max_age


@dataclass
class MeasurementMessage:
    """
    Raw measurement data for sharing between nodes
    """
    from_node: int
    to_node: int
    measurement_type: str  # "toa", "tdoa", "aoa"
    value: float  # Measurement value in appropriate units
    variance: float  # Measurement variance
    timestamp: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'from_node': self.from_node,
            'to_node': self.to_node,
            'type': self.measurement_type,
            'value': self.value,
            'variance': self.variance,
            'timestamp': self.timestamp
        }


@dataclass
class ConvergenceStatus:
    """
    Convergence status broadcast by nodes
    """
    node_id: int
    iteration: int
    converged: bool
    gradient_norm: float
    step_norm: float
    cost: float

    def has_converged(self, grad_tol: float = 1e-6, step_tol: float = 1e-8) -> bool:
        """Check if node has converged based on tolerances"""
        return (self.gradient_norm < grad_tol and
                self.step_norm < step_tol and
                self.converged)