# tests/test_corr.py

import unittest
import os
import sys
import time 

import numpy as np
import torch
import scipy

file_dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, file_dir_path + "/../")

from src import utils


class TestCorr(unittest.TestCase):

    def test_corr_np(self):
        np.random.seed(1)
        x = np.random.rand(900, 100)
        z = utils.backend_corr("numpy", x, x)

        z_gs = np.corrcoef(x.T)
        delta = z - z_gs

        self.assertEqual(np.isclose(delta, 0).all(), True)

    def test_corr_torch(self):
        torch.manual_seed(1)
        x = torch.randn(900, 100)
        z = utils.backend_corr("torch", x, x)

        z_gs = torch.corrcoef(x.T)
        delta = z - z_gs
        self.assertEqual(torch.isclose(z_gs, z, atol=1e-07).all(), True)


    
if __name__ == "__main__":
    unittest.main()
