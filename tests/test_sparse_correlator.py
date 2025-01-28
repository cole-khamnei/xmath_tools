import os
import sys

import numpy as np
import scipy

TEST_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TEST_DIR_PATH + "/../../")

import torch_math_tools as tmt

#\section constants

SAMPLE_DATA_DIR = os.path.join(TEST_DIR_PATH, "sample_data")
SAMPLE_DTSERIES_PATH = os.path.join(SAMPLE_DATA_DIR, "dtseries.npy")
SAMPLE_PARTITION_PATH = os.path.join(SAMPLE_DATA_DIR, "partition.npy")


DIST_DIR = "/data/data7/network_control/projects/network_control/resources/brain_distances"
SUBCORTEX_MASK_PATH = os.path.join(DIST_DIR, "subcortex_mask.npy")
GEODESIC_MASK_PATH = os.path.join(DIST_DIR, f"geodesic_mask_10.npz")

# \section main

def main():

    if os.path.exists(SAMPLE_DTSERIES_PATH):
        voxel_data = np.load(SAMPLE_DTSERIES_PATH)
        subcortex_index = np.load(SUBCORTEX_MASK_PATH)
        geodesic_mask = scipy.sparse.load_npz(GEODESIC_MASK_PATH)

    else:
        voxel_data = np.random.randn(900, 91_282)
        subcortex_index, geodesic_mask = None, None

    block_size = 5_000
    print(f"BLOCK SIZE :: {block_size}")
    print(voxel_data.shape)

    # sc = tmt.matrix.Runner.run(voxel_data[:, :], mask=geodesic_mask, exclude_index=subcortex_index,
    #                            block_size=block_size, symmetric=True)

    # assert False
    sc = tmt.matrix.SparseCorrelator.run(voxel_data[:, :], symmetric=True,
                                         mask=geodesic_mask, exclude_index=subcortex_index,
                                         sparsity_percent=0.1,
                                         block_size=block_size)

    # from infomap import Infomap

    # # Create the Infomap instance
    # infomap = Infomap(two_level=True, num_trials=1)

    # # Add edges from the sparse matrix
    # row, col = sc.nonzero()
    # for r, c in zip(row, col):
    #     weight = sc[r, c]
    #     infomap.add_link(r, c, weight=weight)

    # # Run the Infomap algorithm
    # infomap.run()

    # # Get the partition
    # partition = infomap.get_modules()

    # print(type(partition))
    # print(len(partition))

    # index = np.array(list(partition.keys()))
    # values = np.array(list(partition.values()))

    # print(index, len(index))
    # print(values, len(values))
    # np.save(SAMPLE_PARTITION_PATH, [index, values])


if __name__ == '__main__':
    main()

# \section end
