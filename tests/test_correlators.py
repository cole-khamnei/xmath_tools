import os
import sys
import unittest

import numpy as np
import scipy

TEST_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TEST_DIR_PATH + "/../../")

import xmath_tools as xmt

# ----------------------------------------------------------------------------# 
# --------------------             Constants              --------------------# 
# ----------------------------------------------------------------------------# 

BACKEND = "torch"

N_VOXELS_LARGE = 90_000
N_TRS = 900
BLOCK_SIZE = 5000

np.random.seed(1)
LARGE_VOXEL_DATA = np.random.randn(N_TRS, N_VOXELS_LARGE)

# ----------------------------------------------------------------------------# 
# --------------------                Main                --------------------# 
# ----------------------------------------------------------------------------# 


class TestCorrelators(unittest.TestCase):

    def test_(self):
        pass


class TestRunners(unittest.TestCase):

    # def test_runner(self):
    #     sc = xmt.correlators.Runner.run(LARGE_VOXEL_DATA, mask=None, exclude_index=None,
    #                                                block_size=BLOCK_SIZE, symmetric=True,
    #                                                backend=BACKEND, device="mps")
    
    # def test_maxxer(self):
    #     max_ = xmt.correlators.Maxxer.run(LARGE_VOXEL_DATA, mask=None, exclude_index=None,
    #                                           block_size=BLOCK_SIZE, symmetric=True,
    #                                           backend=BACKEND, device="mps")
    #     assert max_ <= 1
    #     assert max_ >= -1
    
    def test_maxxer(self):
        max_ = xmt.SparseCorrelator.run(LARGE_VOXEL_DATA, mask=None, exclude_index=None,
                                              block_size=BLOCK_SIZE, symmetric=True,
                                              backend=BACKEND, device="mps")
        assert max_ <= 1
        assert max_ >= -1




if __name__ == "__main__":
    unittest.main()

# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
