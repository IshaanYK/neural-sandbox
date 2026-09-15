"""
Unit tests for activations, loss functions, and schedulers.
"""

import unittest
from neural_sandbox.activations import gelu, softmax, swish
from neural_sandbox.losses import cross_entropy_loss, focal_loss
from neural_sandbox.optimizers import CosineAnnealingLR

class TestNeuralSandbox(unittest.TestCase):
    def test_activations(self):
        self.assertAlmostEqual(gelu(0.0), 0.0, places=4)
        probs = softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(probs), 1.0, places=5)
        self.assertTrue(probs[2] > probs[1] > probs[0])

    def test_losses(self):
        probs = [0.1, 0.7, 0.2]
        ce = cross_entropy_loss(probs, 1)
        fl = focal_loss(probs, 1, gamma=2.0)
        self.assertTrue(ce > 0)
        self.assertTrue(fl > 0)

    def test_scheduler(self):
        scheduler = CosineAnnealingLR(initial_lr=0.001, min_lr=1e-5, total_steps=100)
        lr_0 = scheduler.get_lr(0)
        lr_50 = scheduler.get_lr(50)
        lr_100 = scheduler.get_lr(100)
        self.assertAlmostEqual(lr_0, 0.001, places=5)
        self.assertTrue(lr_0 > lr_50 > lr_100)

if __name__ == '__main__':
    unittest.main()
