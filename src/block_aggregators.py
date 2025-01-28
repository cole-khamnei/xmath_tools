import numpy as np
import torch
import scipy

from tqdm.auto import tqdm

from . import block_analysis

# \section runner

class Runner(block_analysis.BlockAnalysis):
    """ 
    Just runs a correlation to test speeds
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corr_func = utils.get_corr_func(self.backend)

    def core_func(self, A, B, a_index, b_index):
        """ """
        M_chunk = self.corr_func(A, B)
        if a_index == b_index and self.skip_diagonal:
            M_chunk[self.backend.eye(M_chunk.shape[0], dtype=bool)] = -self.backend.inf

        return M_chunk

    def __call__(self, A, B, a_index, b_index, mask=None, exclude_index=None):
        """ """
        backend = self.backend
        M_chunk = self.core_func(A, B, a_index, b_index)

    def results(self):
        return None

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

        if kwargs["use_torch"]:
            self.top_k = lambda values, k, axis=0: torch.topk(values, k, dim=axis)
        else:
            self.top_k = lambda values, k, axis=0: (np.partition(values, -k, axis=axis)[-k:],
                                                    np.argpartition(values, -k, axis=axis)[-k:])

        self.create_empty = lambda k=1: [self.backend.ones(0, **self.devp) for _ in range(k)]

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
            self.compare_tv, update_ti = self.top_k(self.compare_tv, self.top_n)
            self.compare_ri = self.compare_ri[update_ti]
            self.compare_ci = self.compare_ci[update_ti]
            self.min_tv = self.compare_tv[-1]

    def __call__(self, A, B, a_index, b_index, mask=None, exclude_index=None):
        """ """
        backend = self.backend

        threshold_chunk_tv_index = backend.ones(A.shape[1] * B.shape[1], dtype=bool, **self.devp)
        if exclude_index is not None:
            a_exclude = exclude_index[a_index]
            b_exclude = exclude_index[b_index]

            exclude_chunk = a_exclude.reshape(-1, 1) @ b_exclude.reshape(1, -1)
            threshold_chunk_tv_index &= (exclude_chunk == 0).flatten()

        if mask is not None:
            mask_chunk = mask[a_index, b_index]
            if get_nnz_safe(mask_chunk) > 0:
                mask_flat = ~sparse_to_array(mask_chunk.astype(bool)).ravel()
                mask_flat = to_torch(mask_flat, dtype=bool, **self.devp)
                threshold_chunk_tv_index &= mask_flat

        if not threshold_chunk_tv_index.any():
            return

        M_chunk = self.core_func(A, B, a_index, b_index)

        # TODO: Fix symmetry:
        if self.symmetric and a_index == b_index:
            triu_index = backend.triu_indices(*M_chunk.shape, offset=1)
            flat_triu = triu_index[0] * M_chunk.shape[0] + triu_index[1]
            threshold_chunk_tv_index[flat_triu] = 0

        chunk_tv = M_chunk.flatten()
        threshold_chunk_tv_index &= chunk_tv > self.min_tv

        chunk_ti = backend.where(threshold_chunk_tv_index)[0]
        if len(chunk_ti) == 0:
            return

        chunk_tv = chunk_tv[chunk_ti]
        chunk_ri, chunk_ci = backend.unravel_index(chunk_ti, M_chunk.shape)
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

        tv, ri, ci = to_np((self.compare_tv, self.compare_ri, self.compare_ci))
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
        select_index = backend.ones(A.shape[1] * B.shape[1], dtype=bool, **self.devp)
        select_index = self.get_mask_chunk_index(mask, a_index, b_index, select_index)
        select_index = self.get_exclude_chunk_index(exclude_index, a_index, b_index, select_index)

        if not select_index.any():
            return

        M_chunk = self.core_func(A, B, a_index, b_index)

        # TODO: Fix symmetry:
        if self.symmetric and a_index == b_index:
            triu_index = backend.triu_indices(*M_chunk.shape, offset=1)
            flat_triu = triu_index[0] * M_chunk.shape[0] + triu_index[1]
            select_index[flat_triu] = 0

        chunk_tv = M_chunk.flatten()
        select_index &= chunk_tv > self.threshold

        chunk_ti = backend.where(select_index)[0]
        if len(chunk_ti) == 0:
            return

        chunk_tv = chunk_tv[chunk_ti]
        chunk_ri, chunk_ci = backend.unravel_index(chunk_ti, M_chunk.shape)

        self.cache_tv.append(chunk_tv)
        self.cache_ri.append(chunk_ri + a_index.start)  # off set row index by a start #TODO: check if this is right
        self.cache_ci.append(chunk_ci + b_index.start)  # off set col index by b start

    def results(self):

        self.cache_tv = self.backend.hstack(self.cache_tv)
        self.cache_ri = self.backend.hstack(self.cache_ri)
        self.cache_ci = self.backend.hstack(self.cache_ci)

        tv, ri, ci = to_np((self.cache_tv, self.cache_ri, self.cache_ci))
        if self.symmetric:
            non_diag_index = ri != ci
            tv = np.hstack([tv, tv[non_diag_index]])
            # following is purposely done in one line
            ri, ci = np.hstack([ri, ci[non_diag_index]]), np.hstack([ci, ri[non_diag_index]])

        return scipy.sparse.csr_matrix((tv, (ri.astype(int), ci.astype(int))), shape=self.shape)


# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
