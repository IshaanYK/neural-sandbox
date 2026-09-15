"""
Activation functions: GELU, SwiGLU, and numerically stable Softmax.
"""

import math
from typing import List

def gelu(x: float) -> float:
    """Gaussian Error Linear Unit approximation."""
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))))

def swish(x: float, beta: float = 1.0) -> float:
    """Swish / SiLU activation function."""
    sigmoid = 1.0 / (1.0 + math.exp(-beta * x)) if abs(beta * x) < 30 else (1.0 if beta * x > 0 else 0.0)
    return x * sigmoid

def softmax(logits: List[float]) -> List[float]:
    """Numerically stable softmax computation."""
    max_val = max(logits) if logits else 0.0
    exp_vals = [math.exp(val - max_val) for val in logits]
    sum_exp = sum(exp_vals)
    return [e / sum_exp for e in exp_vals]
