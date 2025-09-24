"""
Distributed consensus module for FTL synchronization
Implements Consensus-Gauss-Newton algorithm for decentralized state estimation
"""

from .message_types import StateMessage, NetworkMessage, MessageType
from .consensus_node import ConsensusNode, ConsensusNodeConfig
from .consensus_gn import ConsensusGaussNewton, ConsensusGNConfig

__all__ = [
    'StateMessage',
    'NetworkMessage',
    'MessageType',
    'ConsensusNode',
    'ConsensusNodeConfig',
    'ConsensusGaussNewton',
    'ConsensusGNConfig',
]