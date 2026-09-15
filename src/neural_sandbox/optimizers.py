"""
Optimizer utilities and learning rate schedulers.
"""

import math

class CosineAnnealingLR:
    """Cosine Annealing learning rate scheduler."""
    def __init__(self, initial_lr: float, min_lr: float, total_steps: int):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_steps = total_steps

    def get_lr(self, current_step: int) -> float:
        if current_step >= self.total_steps:
            return self.min_lr
        progress = current_step / self.total_steps
        return self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))
