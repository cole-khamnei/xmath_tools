import numpy as np
import torch
import scipy

from tqdm.auto import tqdm

# ----------------------------------------------------------------------------# 
# -------------------          Torch Helper Tools          -------------------# 
# ----------------------------------------------------------------------------# 


def to_np(array):
    """ """
    if isinstance(array, tuple):
        return [to_np(ar) for ar in array]
    if isinstance(array, torch.Tensor):
        return array.cpu().numpy()
    return array


def to_torch(array, dtype="float32", **devp):
    """ """
    if scipy.sparse.issparse(array):
        return torch.sparse_coo_tensor(np.array(array.nonzero()),
                                array.data.astype(dtype),
                                array.shape, **devp)

    if not isinstance(array, torch.Tensor):
        return torch.tensor(array.astype(dtype), **devp)

    return array


def get_device(device_str=None):
    """ """
    if device_str:
        return device_str
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def torch_corr(x, y):
    """ """
    x, y = x.T, y.T
    x_demeaned = x - x.mean(axis=1, keepdims=True)
    y_demeaned = y - y.mean(axis=1, keepdims=True)

    x_norm = x_demeaned / torch.sqrt(torch.sum(x_demeaned ** 2, axis=1, keepdims=True))
    y_norm = y_demeaned / torch.sqrt(torch.sum(y_demeaned ** 2, axis=1, keepdims=True))
    return x_norm @ y_norm.T


def np_corr(x, y):
    """ """
    x, y = x.T, y.T
    x_demeaned = x - x.mean(axis=1, keepdims=True)
    y_demeaned = y - y.mean(axis=1, keepdims=True)

    x_norm = x_demeaned / np.sqrt(np.sum(x_demeaned ** 2, axis=1, keepdims=True))
    y_norm = y_demeaned / np.sqrt(np.sum(y_demeaned ** 2, axis=1, keepdims=True))
    return x_norm @ y_norm.T


# ----------------------------------------------------------------------------# 
# -------------------          Scipy Sparse Tools          -------------------# 
# ----------------------------------------------------------------------------# 


def get_nnz_safe(array):
    """ """
    if scipy.sparse.issparse(array):
        return array.nnz

    return np.count_nonzero(array)


def sparse_to_array(arr):
    """ """
    return arr.toarray() if scipy.sparse.issparse(arr) else arr


# ----------------------------------------------------------------------------# 
# --------------------         Old Ev_aggregator          --------------------# 
# ----------------------------------------------------------------------------# 


class ev_aggregator:
    def __init__(self, data, device, use_torch=True):
        self.size = data.shape[1]
        self.device = device
        self.devp = {"device": device} if use_torch else {}
        self.use_torch = use_torch

        if use_torch:
            self.max_f = lambda values, axis: torch.max(values, axis=axis).values
            self.backend = torch
        else:
            self.backend = np
            self.max_f = lambda values, axis: np.max(values, axis=axis)
        self.ev_maxs = -self.backend.ones(self.size, **self.devp)

    def __call__(self, A, B, a_index, b_index):
        """ """
        backend = self.backend
        M_chunk = backend.corrcoef(backend.hstack([A, B]).T)[:A.shape[1], A.shape[1]:] ** 2
        if a_index == b_index:
            M_chunk = M_chunk[~backend.eye(M_chunk.shape[0], dtype=bool)].reshape(A.shape[1], -1)
        
        new_maxes = self.max_f(M_chunk, axis=1)
        self.ev_maxs[a_index] = self.max_f(backend.vstack([self.ev_maxs[a_index], new_maxes]), axis=0)

    def results(self):
        return to_np(self.ev_maxs)


# ----------------------------------------------------------------------------# 
# -----------------           Bock Analysis Module           -----------------# 
# ----------------------------------------------------------------------------# 


# TODO: consider turning into functional versional rather than objects?


class BlockAnalysis:
    """
    """
    def __init__(self, data, device=None, use_torch=True, symmetric=True, skip_diagonal=True):
        self.size = data.shape[1]
        self.shape = (data.shape[1], data.shape[1])
        self.device = get_device(device)
        self.devp = {"device": self.device} if use_torch else {}
        self.use_torch = use_torch
        self.skip_diagonal = skip_diagonal
        self.symmetric=symmetric

        self.backend = torch if use_torch else np
        self.create_empty = lambda k=1: [self.backend.ones(0, **self.devp) for _ in range(k)]

    def core_func(self, A, B, a_index, b_index):
        """ """
        raise NotImplementedError

    def __call__(self, A, B, a_index, b_index, mask=None, exclude_index=None):
        """ """
        raise NotImplementedError

    def get_mask_chunk_index(self, mask, a_index, b_index, select_index):
        """ """
        if mask is not None:
            mask_chunk = mask[a_index, b_index]
            if get_nnz_safe(mask_chunk) > 0:
                mask_flat = ~sparse_to_array(mask_chunk.astype(bool)).ravel()
                # TODO: move to_torch to a to_backend_array() which creates either np or torch
                mask_flat = to_torch(mask_flat, dtype=bool, **self.devp)
                select_index &= mask_flat

        return select_index

    def get_exclude_chunk_index(self, exclude_index, a_index, b_index, select_index):
        """ """
        if exclude_index is not None:
            a_exclude = exclude_index[a_index]
            b_exclude = exclude_index[b_index]

            exclude_chunk = a_exclude.reshape(-1, 1) @ b_exclude.reshape(1, -1)
            select_index &= (exclude_chunk == 0).flatten()

        return select_index

    @classmethod
    def run(cls, data, mask=None, exclude_index=None,
            block_size=4_000, use_torch=True, symmetric=False,
            device=None, dtype="float32", **agg_params):
        """
        TODO: added second data matrix and scan through both matrices
        TODO: add symmetry argument that scans through only half of matrix

        """
        device = get_device(device)

        if use_torch:
            data = to_torch(data, device=device, dtype=dtype)

            if exclude_index is not None:
                exclude_index = to_torch(exclude_index, device=device, dtype="float16")

        if symmetric and mask is not None:
            if get_nnz_safe(mask != mask.T) == 0:
                print("Mask is not symmetric, cannot use symmetric acceleration.")
                symmetric = False

        aggregator = cls(data, device=device, use_torch=use_torch, symmetric=symmetric, **agg_params)

        n_blocks = int(np.ceil(data.shape[1] / block_size))

        total_blocks = n_blocks * (n_blocks + 1) // 2 if symmetric else n_blocks ** 2
        pbar = tqdm(total=total_blocks * (block_size / 1000) ** 2, desc=f'{cls.__name__} Block Analysis')

        for a_count in range(n_blocks):
            a_start = a_count * block_size
            a_n_items = block_size if a_count < n_blocks - 1 else data.shape[1] - a_start
            a_index = slice(a_start, a_start + a_n_items)
            a_block = data[:, a_index]

            b_count_iter = range(a_count + 1) if symmetric else range(n_blocks)

            for b_count in b_count_iter:
                b_start = b_count * block_size
                b_n_items = block_size if b_count < n_blocks - 1 else data.shape[1] - b_start
                b_index = slice(b_start, b_start + b_n_items)
                b_block = data[:, b_index]

                aggregator(a_block, b_block, a_index, b_index, mask=mask, exclude_index=exclude_index)
                pbar.update(a_n_items * b_n_items // 1000 ** 2)
    
        pbar.update(pbar.total - pbar.n)
        pbar.close()
        return aggregator.results()

# ----------------------------------------------------------------------------# 
# -                Specific Aggregator And Correlator Classes                -# 
# ----------------------------------------------------------------------------# 


class SparseAggregator(BlockAnalysis):
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


class SparseCorrelator(SparseAggregator):
    """ """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corr_func = torch_corr if kwargs["use_torch"] else np_corr

    def core_func(self, A, B, a_index, b_index):
        """ """
        backend = self.backend
        M_chunk = self.corr_func(A, B)
        if a_index == b_index and self.skip_diagonal:
            M_chunk[backend.eye(M_chunk.shape[0], dtype=bool)] = -backend.inf

        return M_chunk


class Runner(BlockAnalysis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corr_func = torch_corr if kwargs["use_torch"] else np_corr

    def core_func(self, A, B, a_index, b_index):
        """ """
        backend = self.backend
        M_chunk = self.corr_func(A, B)
        if a_index == b_index and self.skip_diagonal:
            M_chunk[backend.eye(M_chunk.shape[0], dtype=bool)] = -backend.inf

        return M_chunk

    def __call__(self, A, B, a_index, b_index, mask=None, exclude_index=None):
        """ """
        backend = self.backend
        M_chunk = self.core_func(A, B, a_index, b_index)

    def results(self):
        return None


# ----------------------------------------------------------------------------# 
# -----------------           Threshold Aggregator           -----------------# 
# ----------------------------------------------------------------------------# 


class ThresholdAggregator(BlockAnalysis):
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


class ThresholdCorrelator(ThresholdAggregator):
    """ """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corr_func = torch_corr if kwargs["use_torch"] else np_corr

    def core_func(self, A, B, a_index, b_index):
        """ """
        backend = self.backend
        M_chunk = self.corr_func(A, B)
        if a_index == b_index and self.skip_diagonal:
            M_chunk[backend.eye(M_chunk.shape[0], dtype=bool)] = -backend.inf

        return M_chunk


# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
