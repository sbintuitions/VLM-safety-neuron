"""
Common types and data structures for neuron detection.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class SafetyNeuron:
    """Represents a detected safety neuron."""
    layer: int
    component: str  # "mlp_gate", "mlp_up", "mlp_out", "self_attn"
    index: int
    score: float  # probe z-score (candidates) or mean dP_refuse (confirmed)
    method: str  # "linear_probe", "activation_patch" or "control"
    metadata: Optional[Dict] = None

    def to_tuple(self) -> Tuple[int, str, int]:
        """Return (layer, component, index) tuple for hashing."""
        return (self.layer, self.component, self.index)

    def __hash__(self):
        return hash(self.to_tuple())

    def __eq__(self, other):
        return self.to_tuple() == other.to_tuple()
