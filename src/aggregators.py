import numpy as np
import torch
import scipy

from tqdm.auto import tqdm

from . import block_analysis
from . import utils


# ----------------------------------------------------------------------------# 
# -                Specific Aggregator And Correlator Classes                -# 
# ----------------------------------------------------------------------------# 


class SparseAggregator(block_analysis.BlockAnalysis):
    """
    keeps running list of coordinates of top p percent (or top n) sparse connections:

    """
    def __init__(self, *args, sparsity_percent=0.1, **kwargs):
        super().__init__(*args, **kwargs)

        self.sparsity_frac = sparsity_percent / 100


        # TODO: determine if sparsity percent should be before or after diagonal skip
        n_items = self.size ** 2 - self.size if self.skip_diagonal else self.size ** 2
        if self.symmetric:
            self.top_n = int(np.ceil(n_items * self.sparsity_frac)) // 2
        else:
            self.top_n = int(np.ceil(n_items * self.sparsity_frac))

        self.create_empty = lambda k=1: [self.backend.ones(0, **self.device_info) for _ in range(k)]

        self.min_tv = -self.backend.inf
        self.cache_tv, self.cache_ri, self.cache_ci = [], [], []
        self.compare_tv, self.compare_ri, self.compare_ci = self.create_empty(3)

    def compare_cache(self, final=False):
        """ """
        backend = self.backend
        self.compare_tv = backend.hstack(self.cache_tv + [self.compare_tv])
        self.compare_ri = backend.hstack(self.cache_ri + [self.compare_ri])
        self.compare_ci = backend.hstack(self.cache_ci + [self.compare_ci])
        self.cache_tv, self.cache_ri, self.cache_ci = [], [], []

        if len(self.compare_tv) > self.top_n:
            # self.compare_tv, update_ti = self.top_k(self.compare_tv, self.top_n)
            self.compare_tv, update_ti = utils.MPS_safe_topk(backend, self.device_info, self.compare_tv, self.top_n)

            self.compare_ri = self.compare_ri[update_ti]
            self.compare_ci = self.compare_ci[update_ti]
            self.min_tv = self.compare_tv[-1]

    def __call__(self, A, B, a_index, b_index, mask=None, exclude_index=None):
        """ """
        backend = self.backend

        threshold_chunk_tv_index = backend.ones(A.shape[1] * B.shape[1], dtype=bool, **self.device_info)
        if exclude_index is not None:
            a_exclude = exclude_index[a_index]
            b_exclude = exclude_index[b_index]

            exclude_chunk = a_exclude.reshape(-1, 1) @ b_exclude.reshape(1, -1)
            threshold_chunk_tv_index &= (exclude_chunk == 0).flatten()

        if mask is not None:
            mask_chunk = mask[a_index, b_index]
            if get_nnz_safe(mask_chunk) > 0:
                mask_flat = ~sparse_to_array(mask_chunk.astype(bool)).ravel()
                mask_flat = utils.to_backend(backend,mask_flat, dtype=bool, **self.device_info)
                threshold_chunk_tv_index &= mask_flat

        if not threshold_chunk_tv_index.any():
            return

        M_chunk = self.core_func(A, B, a_index, b_index)

        # TODO: Fix symmetry:
        if self.symmetric and a_index == b_index:
            # triu_index = backend.triu_indices(*M_chunk.shape, offset=1)
            triu_index = utils.backend_triu_indices(backend, M_chunk.shape, offset=1)
            flat_triu = triu_index[0] * M_chunk.shape[0] + triu_index[1]
            threshold_chunk_tv_index[flat_triu] = 0

        chunk_tv = M_chunk.flatten()
        threshold_chunk_tv_index &= chunk_tv > self.min_tv

        chunk_ti = utils.MPS_safe_where(backend, self.device_info, threshold_chunk_tv_index)
        chunk_ri, chunk_ci = utils.MPS_safe_unravel_index(backend, self.device_info, chunk_ti, M_chunk.shape)

        # chunk_ti = backend.where(threshold_chunk_tv_index)[0]
        if len(chunk_ti) == 0:
            return

        chunk_tv = chunk_tv[chunk_ti]

        # chunk_ri, chunk_ci = backend.unravel_index(chunk_ti, M_chunk.shape)
        chunk_ri += a_index.start
        chunk_ci += b_index.start

        self.cache_tv.append(chunk_tv)
        self.cache_ri.append(chunk_ri)
        self.cache_ci.append(chunk_ci)

        if sum(len(chunk) for chunk in self.cache_tv) > self.top_n:
            self.compare_cache()

    def results(self):
        self.compare_cache(final=True)

        assert len(self.cache_tv) == 0
        # assert len(self.compare_tv) == self.top_n

        tv, ri, ci = utils.to_np((self.compare_tv, self.compare_ri, self.compare_ci))
        if self.symmetric:
            non_diag_index = ri != ci
            tv = np.hstack([tv, tv[non_diag_index]])
            # following is purposely done in one line
            ri, ci = np.hstack([ri, ci[non_diag_index]]), np.hstack([ci, ri[non_diag_index]])

        return scipy.sparse.csr_matrix((tv, (ri.astype(int), ci.astype(int))), shape=self.shape)


# ----------------------------------------------------------------------------# 
# -----------------           Threshold Aggregator           -----------------# 
# ----------------------------------------------------------------------------# 


class ThresholdAggregator(block_analysis.BlockAnalysis):
    """
    keeps running list of coordinates of top p percent (or top n) sparse connections:

    """
    def __init__(self, *args, threshold=0.1, **kwargs):
        super().__init__(*args, **kwargs)

        self.threshold = threshold
        self.cache_tv, self.cache_ri, self.cache_ci = [], [], []

    def __call__(self, A, B, a_index, b_index, mask=None, exclude_index=None):
        """ """
        backend = self.backend

        # TODO: move preselection select index generation to blockanalysis method
        select_index = backend.ones(A.shape[1] * B.shape[1], dtype=bool, **self.device_info)
        select_index = self.get_mask_chunk_index(mask, a_index, b_index, select_index)
        select_index = self.get_exclude_chunk_index(exclude_index, a_index, b_index, select_index)

        if not select_index.any():
            return

        M_chunk = self.core_func(A, B, a_index, b_index)

        # TODO: Add Numpy offset argument version:
        if self.symmetric and a_index == b_index:
            # triu_index = backend.triu_indices(*M_chunk.shape, offset=1)
            triu_index = utils.backend_triu_indices(backend, M_chunk.shape, offset=1)
            flat_triu = triu_index[0] * M_chunk.shape[0] + triu_index[1]
            select_index[flat_triu] = 0

        chunk_tv = M_chunk.flatten()
        select_index &= (chunk_tv > self.threshold)

        chunk_ti = utils.MPS_safe_where(backend, self.device_info, select_index)
        chunk_ri, chunk_ci = utils.MPS_safe_unravel_index(backend, self.device_info, chunk_ti, M_chunk.shape)

        if len(chunk_ti) == 0:
            return

        chunk_tv_reduced = chunk_tv[chunk_ti]

        self.cache_tv.append(chunk_tv_reduced)
        self.cache_ri.append(chunk_ri + a_index.start)  # off set row index by a start #TODO: check if this is right
        self.cache_ci.append(chunk_ci + b_index.start)  # off set col index by b start

    def results(self):

        self.cache_tv = self.backend.hstack(self.cache_tv)
        self.cache_ri = self.backend.hstack(self.cache_ri)
        self.cache_ci = self.backend.hstack(self.cache_ci)

        tv, ri, ci = utils.to_np((self.cache_tv, self.cache_ri, self.cache_ci))
        if self.symmetric:
            non_diag_index = ri != ci
            tv = np.hstack([tv, tv[non_diag_index]])
            # following is purposely done in one line
            ri, ci = np.hstack([ri, ci[non_diag_index]]), np.hstack([ci, ri[non_diag_index]])

        return scipy.sparse.csr_matrix((tv, (ri.astype(int), ci.astype(int))), shape=self.shape)


# \section pair aggregator

class PairComparator(block_analysis.BlockAnalysis):
    """
    keeps running list of coordinates of top p percent (or top n) sparse connections:

    """
    def __init__(self, *args, threshold=0.1, skip_diagonal=False, axis=None, **kwargs):
        super().__init__(*args, skip_diagonal=False, **kwargs)

        self.threshold = threshold
        self.axis = axis
        self.compare_values = []
        self.compare_counts = []

        # self.stds_0 = []
        # self.stds_1 = []

    def __call__(self, A, B, a_index, b_index, mask=None, exclude_index=None):
        """ """
        backend = self.backend

        M_chunk_0 = self.core_func(A[:, :, 0], B[:, :, 0], a_index, b_index)
        M_chunk_1 = self.core_func(A[:, :, 1], B[:, :, 1], a_index, b_index)

        if self.symmetric and a_index == b_index:
            compare_count = (M_chunk_0.shape[0] * (M_chunk_0.shape[0] + 1)) / 2
        else:
            compare_count = M_chunk_0.shape[0] * M_chunk_0.shape[1]

        self.compare_values += [self.compare(M_chunk_0, M_chunk_1, axis=self.axis)]
        self.compare_counts += [compare_count]
        # self.stds_0 += [self.backend.std(M_chunk_0)]
        # self.stds_1 += [self.backend.std(M_chunk_1)]

    def results(self):
        self.compare_values = self.backend.hstack(self.compare_values)
        self.compare_counts = np.hstack(self.compare_counts)

        compare_values = utils.to_np(self.compare_values)
        
        weighted_values = compare_values * self.compare_counts / np.sum(self.compare_counts)
        weighted_values = np.round(weighted_values, 5)

        # print(self.backend.hstack(self.stds_0))
        # print(self.backend.hstack(self.stds_1))
        return np.sum(weighted_values)


class PairComparatorAxis(block_analysis.BlockAnalysis):
    """
    keeps running list of coordinates of top p percent (or top n) sparse connections:

    """
    def __init__(self, *args, threshold=0.1, skip_diagonal=False, axis=None, **kwargs):
        super().__init__(*args, skip_diagonal=False, **kwargs)

        self.threshold = threshold
        self.axis = axis
        
        self.caches_created = False
        self.total_n = args[0][0].shape[0]
        self.compare_values = -2 * self.backend.ones(self.total_n, **self.device_info)

        self.block_ri = 0
        self.block_ci = 0


    def init_caches(self, A):
        """ """
        if self.caches_created:
            return

        self.block_size = A.shape[1]
        n_blocks = int(np.ceil(self.total_n / self.block_size))

        self.block_shape = (n_blocks, n_blocks)
        self.block_cache_0 = [[None] * self.block_shape[1]] * self.block_shape[0]
        self.block_cache_1 = [[None] * self.block_shape[1]] * self.block_shape[0]
        self.caches_created = True

    def __call__(self, A, B, a_index, b_index, mask=None, exclude_index=None):
        """ """
        backend = self.backend

        self.init_caches(A)

        M_chunk_0 = self.core_func(A[:, :, 0], B[:, :, 0], a_index, b_index)
        M_chunk_1 = self.core_func(A[:, :, 1], B[:, :, 1], a_index, b_index)

        if self.symmetric and a_index == b_index:
            raise NotImplementedError
        else:
            self.block_cache_0[self.block_ri][self.block_ci] = M_chunk_0
            self.block_cache_1[self.block_ri][self.block_ci] = M_chunk_1

            self.block_ci += 1
            if self.block_ci >= self.block_shape[1]:

                block_row_0 = self.backend.hstack(self.block_cache_0[self.block_ri])
                block_row_1 = self.backend.hstack(self.block_cache_1[self.block_ri])

                # print(block_row_0.shape)
                row_r = self.compare(block_row_0, block_row_1, axis=self.axis)
                
                # self.compare_values.append(row_r)
                self.compare_values[self.block_ri * self.block_size:(self.block_ri + 1) * self.block_size] = row_r

                self.block_cache_0[self.block_ri] = None
                self.block_cache_1[self.block_ri] = None

                self.block_ci = 0
                self.block_ri += 1
        # print(self.block_ri, self.block_ci)


    def results(self):
        return utils.to_np(self.compare_values)


# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
