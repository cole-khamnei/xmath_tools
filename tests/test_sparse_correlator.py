import os
import sys

import numpy as np

TEST_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TEST_DIR_PATH + "/../../")

import torch_math_tools as tmt

#\section constants

SAMPLE_DATA_DIR = os.path.join(TEST_DIR_PATH, "sample_data")
SAMPLE_DTSERIES_PATH = os.path.join(SAMPLE_DATA_DIR, "dtseries.npy")

# \section main

def main():
    voxel_data = np.load(SAMPLE_DTSERIES_PATH)
    tmt.matrix.SparseCorrelator.run(voxel_data, sparsity_percent=0.1,
                                    dtype="float32", block_size=4000)

if __name__ == '__main__':
    main()

# \section end
