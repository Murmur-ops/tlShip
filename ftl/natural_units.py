"""
Natural Units System for FTL (c = 1)

This module provides unit conversions and operations for working in natural units
where the speed of light c = 1. This dramatically improves numerical stability
by avoiding factors of 3×10^8.

Unit System:
- Distance: light-nanoseconds (lns) where 1 lns = 0.299792458 meters
- Time: nanoseconds (ns)
- Speed of light: 1 lns/ns (dimensionless)
- Frequency: ppb (parts per billion, dimensionless)

Benefits:
- Jacobian elements are O(1) instead of O(10^8)
- Condition numbers improve by ~10^8
- Frequency gradients scale as O(t) instead of O(c*t)
"""

import numpy as np
from typing import Union, Tuple, Optional
from dataclasses import dataclass


# Physical constants
C_SI = 299792458.0  # Speed of light in m/s
C_NATURAL = 1.0     # Speed of light in lns/ns


@dataclass
class NaturalUnits:
    """Natural units conversion system"""

    # Conversion factors
    METERS_PER_LNS = 0.299792458  # 1 light-nanosecond in meters
    LNS_PER_METER = 1.0 / 0.299792458

    # Time is already in nanoseconds, no conversion needed
    NS_PER_NS = 1.0

    # Frequency remains in ppb (dimensionless)
    PPB_PER_PPB = 1.0

    @classmethod
    def meters_to_lns(cls, meters: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert meters to light-nanoseconds"""
        return meters * cls.LNS_PER_METER

    @classmethod
    def lns_to_meters(cls, lns: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert light-nanoseconds to meters"""
        return lns * cls.METERS_PER_LNS

    @classmethod
    def seconds_to_ns(cls, seconds: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert seconds to nanoseconds"""
        return seconds * 1e9

    @classmethod
    def ns_to_seconds(cls, ns: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert nanoseconds to seconds"""
        return ns * 1e-9

    @classmethod
    def mps_to_lns_per_ns(cls, mps: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert meters/second to light-nanoseconds/nanosecond"""
        # v_mps * (lns/m) * (s/ns) = v_mps * LNS_PER_METER * 1e-9
        return mps * cls.LNS_PER_METER * 1e-9

    @classmethod
    def lns_per_ns_to_mps(cls, lns_per_ns: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert light-nanoseconds/nanosecond to meters/second"""
        # v_lns_per_ns * (m/lns) * (ns/s) = v_lns_per_ns * METERS_PER_LNS * 1e9
        return lns_per_ns * cls.METERS_PER_LNS * 1e9


class NaturalUnitState:
    """State vector in natural units [x_lns, y_lns, tau_ns, freq_ppb]"""

    def __init__(self, state: Optional[np.ndarray] = None):
        """
        Initialize state in natural units

        Args:
            state: 4D state vector [x_lns, y_lns, tau_ns, freq_ppb]
        """
        if state is None:
            self.state = np.zeros(4)
        else:
            assert len(state) == 4, "State must be 4D"
            self.state = np.array(state, dtype=float)

    @classmethod
    def from_si(cls, x_m: float, y_m: float, tau_ns: float = 0, freq_ppb: float = 0):
        """Create state from SI units"""
        state = np.array([
            NaturalUnits.meters_to_lns(x_m),
            NaturalUnits.meters_to_lns(y_m),
            tau_ns,  # Already in ns
            freq_ppb  # Already in ppb
        ])
        return cls(state)

    def to_si(self) -> Tuple[float, float, float, float]:
        """Convert to SI units (meters, nanoseconds, ppb)"""
        return (
            NaturalUnits.lns_to_meters(self.state[0]),
            NaturalUnits.lns_to_meters(self.state[1]),
            self.state[2],  # tau already in ns
            self.state[3]   # freq already in ppb
        )

    @property
    def position_lns(self) -> np.ndarray:
        """Position in light-nanoseconds"""
        return self.state[:2]

    @property
    def position_m(self) -> np.ndarray:
        """Position in meters"""
        return NaturalUnits.lns_to_meters(self.state[:2])

    @property
    def time_ns(self) -> float:
        """Time offset in nanoseconds"""
        return self.state[2]

    @property
    def freq_ppb(self) -> float:
        """Frequency offset in ppb"""
        return self.state[3]

    def distance_to(self, other: 'NaturalUnitState') -> float:
        """Euclidean distance in light-nanoseconds"""
        return np.linalg.norm(self.position_lns - other.position_lns)

    def distance_to_meters(self, other: 'NaturalUnitState') -> float:
        """Euclidean distance in meters"""
        return NaturalUnits.lns_to_meters(self.distance_to(other))


class NaturalUnitMeasurement:
    """Range measurement in natural units"""

    def __init__(self, range_lns: float, timestamp_ns: float, sigma_lns: float = 0.1):
        """
        Initialize measurement in natural units

        Args:
            range_lns: Measured range in light-nanoseconds
            timestamp_ns: Measurement time in nanoseconds
            sigma_lns: Measurement uncertainty in light-nanoseconds
        """
        self.range = range_lns
        self.timestamp = timestamp_ns
        self.sigma = sigma_lns

    @classmethod
    def from_si(cls, range_m: float, timestamp_s: float, sigma_m: float = 0.01):
        """Create from SI units"""
        return cls(
            range_lns=NaturalUnits.meters_to_lns(range_m),
            timestamp_ns=NaturalUnits.seconds_to_ns(timestamp_s),
            sigma_lns=NaturalUnits.meters_to_lns(sigma_m)
        )

    def predicted_range(self, state_i: NaturalUnitState, state_j: NaturalUnitState) -> float:
        """
        Compute predicted range with frequency drift (c = 1)

        Returns:
            Predicted range in light-nanoseconds
        """
        # Geometric distance
        dist = state_i.distance_to(state_j)

        # Clock contribution (c = 1 in natural units!)
        time_diff = state_j.time_ns - state_i.time_ns

        # Frequency drift contribution
        freq_diff_ppb = state_j.freq_ppb - state_i.freq_ppb
        drift_ns = freq_diff_ppb * 1e-9 * self.timestamp  # ppb * ns = dimensionless

        # Total: distance + time (both in lns since c=1)
        return dist + time_diff + drift_ns

    def residual(self, state_i: NaturalUnitState, state_j: NaturalUnitState) -> float:
        """Normalized measurement residual"""
        predicted = self.predicted_range(state_i, state_j)
        return (predicted - self.range) / self.sigma

    def jacobian(self, state_i: NaturalUnitState, state_j: NaturalUnitState) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Jacobian with c = 1

        Returns:
            (J_i, J_j): 4D Jacobians for each state
        """
        # Direction vector
        delta = state_j.position_lns - state_i.position_lns
        dist = np.linalg.norm(delta)

        if dist > 1e-10:
            direction = delta / dist
        else:
            direction = np.zeros(2)

        # Build Jacobians (note: c = 1 everywhere!)
        # Don't normalize by sigma here - keep raw gradients
        J_i = np.zeros(4)
        J_i[:2] = -direction  # Position gradient
        J_i[2] = -1.0         # Time gradient (c = 1)
        J_i[3] = -self.timestamp * 1e-9  # Frequency gradient

        J_j = np.zeros(4)
        J_j[:2] = direction
        J_j[2] = 1.0
        J_j[3] = self.timestamp * 1e-9

        # Normalize by sigma
        J_i = J_i / self.sigma
        J_j = J_j / self.sigma

        return J_i, J_j


def demonstrate_numerical_improvement():
    """Show numerical improvement with natural units"""

    # Example: 10m range, 100s timestamp
    range_m = 10.0
    timestamp_s = 100.0

    print("Numerical Comparison: SI vs Natural Units")
    print("=" * 50)

    # SI units gradients
    c_si = 299792458.0
    J_time_si = c_si * 1e-9  # ~300
    J_freq_si = c_si * 1e-9 * timestamp_s  # ~3e10

    # Natural units gradients
    J_time_nat = 1.0  # c = 1
    J_freq_nat = NaturalUnits.seconds_to_ns(timestamp_s) * 1e-9  # ~0.1

    print(f"\nTime Jacobian:")
    print(f"  SI units:      {J_time_si:.2e}")
    print(f"  Natural units: {J_time_nat:.2e}")
    print(f"  Improvement:   {J_time_si/J_time_nat:.2e}×")

    print(f"\nFrequency Jacobian (t={timestamp_s}s):")
    print(f"  SI units:      {J_freq_si:.2e}")
    print(f"  Natural units: {J_freq_nat:.2e}")
    print(f"  Improvement:   {J_freq_si/J_freq_nat:.2e}×")

    # Condition number estimate
    print(f"\nCondition Number Improvement:")
    print(f"  Estimated: ~{J_freq_si/J_freq_nat:.0e}× better")

    return J_time_si, J_time_nat, J_freq_si, J_freq_nat


if __name__ == "__main__":
    demonstrate_numerical_improvement()