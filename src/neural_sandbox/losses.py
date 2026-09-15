"""
Loss function implementations: Focal Loss and Cross-Entropy.
"""

import math
from typing import List

def cross_entropy_loss(probabilities: List[float], target_idx: int, eps: float = 1e-15) -> float:
    """Standard multi-class categorical cross-entropy loss."""
    prob = max(eps, min(1.0 - eps, probabilities[target_idx]))
    return -math.log(prob)

def focal_loss(probabilities: List[float], target_idx: int, gamma: float = 2.0, alpha: float = 0.25) -> float:
    """Focal loss for addressing class imbalance."""
    p = max(1e-15, min(1.0 - 1e-15, probabilities[target_idx]))
    modulating_factor = (1.0 - p) ** gamma
    return -alpha * modulating_factor * math.log(p)
