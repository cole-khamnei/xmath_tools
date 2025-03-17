import os
import sys
import unittest

import numpy as np
import scipy
import torch

TEST_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TEST_DIR_PATH + "/../../")

import xmath_tools as xmt

# ----------------------------------------------------------------------------# 
# --------------------             Options                --------------------# 
# ----------------------------------------------------------------------------# 

PBAR_LEAVE = False
PRINT = False

# ----------------------------------------------------------------------------# 
# --------------------             Constants              --------------------# 
# ----------------------------------------------------------------------------# 

N_VOXELS_LARGE = 100_000
N_VOXELS_SMALL = 10_000
N_TRS = 900
BLOCK_SIZE = 5000

THRESHOLD = 0.1

np.random.seed(1)

# N_VOXELS_LARGE = N_VOXELS_XLARGE if torch.cuda.is_available() else N_VOXELS_LARGE

LARGE_VOXEL_DATA = np.random.randn(N_TRS, N_VOXELS_LARGE)
LARGE_VOXEL_DATA_2 = np.random.randn(N_TRS, N_VOXELS_LARGE)
SMALL_VOXEL_DATA = np.random.randn(N_TRS, N_VOXELS_SMALL)
SMALL_VOXEL_DATA_2 = np.random.randn(N_TRS, N_VOXELS_SMALL)

GPU_STR = "mps" if torch.mps.is_available() else "cuda"

DEVICE_BACKEND_PAIRS = [("torch", GPU_STR), ("torch", "cpu"), ("numpy", "cpu")]


# ----------------------------------------------------------------------------# 
# --------------------                Main                --------------------# 
# ----------------------------------------------------------------------------# 


def printer(*args):
    """ """
    if PRINT:
        print(*args)


class TestCorrelationAggregators(unittest.TestCase):

    def test_pair_correlator(self):
        """ """
        # pair_data = np.stack((SMALL_VOXEL_DATA, SMALL_VOXEL_DATA), axis=2)
        pair_data = np.stack((SMALL_VOXEL_DATA, SMALL_VOXEL_DATA_2), axis=2)
        # pair_data = np.stack((LARGE_VOXEL_DATA, LARGE_VOXEL_DATA_2), axis=2)

        r = xmt.pair_correlation(pair_data, axis=1,
                                 mask=None, exclude_index=None, leave=True,
                                 block_size=2000, symmetric=False,
                                 backend="torch", device=GPU_STR)
        print("Final r", r)


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
