import os
import sys

import numpy as np
import scipy

TEST_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TEST_DIR_PATH + "/../../")

import xmath_tools as xmt

# ----------------------------------------------------------------------------# 
# --------------------             Constants              --------------------# 
# ----------------------------------------------------------------------------# 

SAMPLE_DATA_DIR = os.path.join(TEST_DIR_PATH, "sample_data")
SAMPLE_DTSERIES_PATH = os.path.join(SAMPLE_DATA_DIR, "dtseries.npy")
SAMPLE_PARTITION_PATH = os.path.join(SAMPLE_DATA_DIR, "partition.npy")


DIST_DIR = "/data/data7/network_control/projects/network_control/resources/brain_distances"
SUBCORTEX_MASK_PATH = os.path.join(DIST_DIR, "subcortex_mask.npy")
GEODESIC_MASK_PATH = os.path.join(DIST_DIR, f"geodesic_mask_10.npz")

# ----------------------------------------------------------------------------# 
# --------------------                Main                --------------------# 
# ----------------------------------------------------------------------------# 


def main():
    """ """

    USE_SYNTHETIC = True
    np.random.seed(1)

    if os.path.exists(SAMPLE_DTSERIES_PATH) and not USE_SYNTHETIC:
        voxel_data = np.load(SAMPLE_DTSERIES_PATH)
        subcortex_index = np.load(SUBCORTEX_MASK_PATH)
        geodesic_mask = scipy.sparse.load_npz(GEODESIC_MASK_PATH)
    else:
        voxel_data = np.random.randn(900, 91_282)
        # voxel_data = np.random.randn(900, 10_000).astype("float32")
        # voxel_data = np.hstack([voxel_data, voxel_data])
        # voxel_data = np.random.randn(900, 1000).astype("float32")
        # z = np.corrcoef(voxel_data.T)
        # z[z > 0.95] = -1
        # print("Z max:", z.max())
        
    
    subcortex_index, geodesic_mask = None, None

    block_size = 5_000
    print(f"BLOCK SIZE :: {block_size}")
    print(voxel_data.shape)

    backend = "torch"
    # backend = "numpy" 

    # sc = xmt.block_aggregators.Runner.run(voxel_data[:, :], mask=geodesic_mask, exclude_index=subcortex_index,
    #                                      block_size=block_size, symmetric=True, backend=backend)

    # sc = xmt.SparseCorrelator.run(voxel_data[:, :], mask=geodesic_mask, exclude_index=subcortex_index,
    #                               block_size=block_size, symmetric=True, backend=backend)

    # torch MPS bug :0
    # https://github.com/pytorch/pytorch/issues/122916

    threshold = 0.1
    sc = xmt.ThresholdCorrelator.run(voxel_data[:, :], threshold=threshold, mask=geodesic_mask, device="mps",
                                     exclude_index=subcortex_index, block_size=block_size,
                                     symmetric=True, backend=backend, skip_diagonal=True)

    # sc = xmt.aggregators.Maxxer.run(voxel_data[:, :], mask=geodesic_mask, exclude_index=subcortex_index,
    #                                      block_size=block_size, symmetric=False, backend=backend)
    # print(sc)


    print("mat max:", sc.max(), "mat min:", sc.min(), "data min:", sc.data.min(), "len data", len(sc.data))
    
    assert sc.max() <= 1
    assert sc.data.min() >= threshold


if __name__ == '__main__':
    main()

# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
