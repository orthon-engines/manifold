"""
Minimal signal types stub for backwards compatibility.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class LaplaceField:
    """Stub for LaplaceField - not used in stream architecture."""
    data: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = np.array([])
