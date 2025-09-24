"""
FTL (Frequency-Time-Localization) Synchronization Simulation
Physically-accurate waveform-level simulation with joint position/time/frequency estimation
"""

# Import all modules
from .geometry import NodeGeometry, place_grid_nodes, place_anchors
from .clocks import ClockModel, ClockState
from .signal import gen_hrp_burst, gen_zc_burst, SignalConfig
from .channel import SalehValenzuelaChannel, ChannelConfig, propagate_signal
from .rx_frontend import matched_filter, detect_toa, estimate_cfo, toa_crlb
# Import old factors from factors.py if they exist
try:
    from .factors import ToAFactor, TDOAFactor, TWRFactor, CFOFactor
except ImportError:
    # Placeholder if old factors don't exist
    ToAFactor = TDOAFactor = TWRFactor = CFOFactor = None
from .robust import huber_weight, dcs_weight, RobustConfig
from .solver import FactorGraph
from .init import trilateration, mds, grid_search, initialize_positions, initialize_clock_states
from .metrics import (
    PerformanceMetrics, position_rmse, position_mae, clock_mae,
    crlb_efficiency, convergence_analysis, evaluate_ftl_performance
)

__version__ = "1.0.0"

__all__ = [
    # Geometry
    "NodeGeometry", "place_grid_nodes", "place_anchors",
    # Clocks
    "ClockModel", "ClockState",
    # Signal
    "gen_hrp_burst", "gen_zc_burst", "SignalConfig",
    # Channel
    "SalehValenzuelaChannel", "ChannelConfig", "propagate_signal",
    # Receiver
    "matched_filter", "detect_toa", "estimate_cfo", "toa_crlb",
    # Factor Graph
    "ToAFactor", "TDOAFactor", "TWRFactor", "CFOFactor",
    "huber_weight", "dcs_weight", "RobustConfig",
    "FactorGraph",
    # Initialization
    "trilateration", "mds", "grid_search", "initialize_positions", "initialize_clock_states",
    # Metrics
    "PerformanceMetrics", "position_rmse", "position_mae", "clock_mae",
    "crlb_efficiency", "convergence_analysis", "evaluate_ftl_performance",
]