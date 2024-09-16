import unittest
import torch
import numpy as np
from pbdl.torch.phi.loader import *
from phi.torch.flow import *


class TestLoader(unittest.TestCase):
    def test_sample_shape_with_intermediate_time_steps(self):
        loader = Dataloader(
            "sol",
            batch_size=10,
            time_steps=5,
            sel_sims=[0, 1, 2, 3, 4, 5],
            intermediate_time_steps=True,
            shuffle=True,
            normalize=False
        )

        for input_cpu, targets_cpu in loader:
            input = input_cpu.clone().detach()
            targets = targets_cpu.clone().detach()

            self.assertEqual(input.shape, (10, 4, 65, 32))
            self.assertEqual(targets.shape, (10, 5, 3, 65, 32))
