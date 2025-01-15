import numpy as np
import torch
import scipy

from tqdm.auto import tqdm


def to_np(array):
    """ """
    if isinstance(array, tuple):
        return [to_np(ar) for ar in array]
    if isinstance(array, torch.Tensor):
        return array.cpu().numpy()
    return array


def to_torch(array, dtype="float32", **devp):
    """ """
    if not isinstance(array, torch.Tensor):
        return torch.tensor(array.astype(dtype), **devp)

    return array


def get_device(device_str=None):
    if device_str:
        return device_str
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


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


class SparseAggregator:
    """
    keeps running list of coordinates of top p percent (or top n) sparse connections:

    """
    def __init__(self, data, device=None, use_torch=True, sparsity_percent=0.1, skip_diagonal=True):
        self.size = data.shape[1]
        self.device = get_device(device)
        self.devp = {"device": self.device} if use_torch else {}
        self.use_torch = use_torch

        self.sparsity_frac = sparsity_percent / 100
        self.top_n = int(np.ceil(self.size ** 2 * self.sparsity_frac))
        self.skip_diagonal = skip_diagonal

        if use_torch:
            self.top_k = lambda values, k, axis=0: torch.topk(values, k, dim=axis)
            self.backend = torch
        else:
            self.backend = np
            self.top_k = lambda values, k, axis=0: (np.partition(values, -k, axis=axis)[-k:],
                                                    np.argpartition(values, -k, axis=axis)[-k:])

        self.create_empty = lambda k=1: [self.backend.ones(0, **self.devp) for _ in range(k)]

        self.min_tv = -self.backend.inf
        self.cache_tv, self.cache_ri, self.cache_ci = [], [], []
        self.compare_tv, self.compare_ri, self.compare_ci = self.create_empty(3)

    def compare_cache(self):
        """ """
        backend = self.backend
        self.compare_tv = backend.hstack(self.cache_tv + [self.compare_tv])
        self.compare_ri = backend.hstack(self.cache_ri + [self.compare_ri])
        self.compare_ci = backend.hstack(self.cache_ci + [self.compare_ci])

        if len(self.compare_tv) > self.top_n:
            # print("compare")
            self.compare_tv, update_ti = self.top_k(self.compare_tv, self.top_n)
            self.compare_ri = self.compare_ri[update_ti]
            self.compare_ci = self.compare_ci[update_ti]

            self.min_tv = self.compare_tv[-1]
            self.cache_tv, self.cache_ri, self.cache_ci = [], [], []

    def __call__(self, A, B, a_index, b_index):
        """ """
        backend = self.backend
        M_chunk = self.core_func(A, B, a_index, b_index)

        chunk_tv = M_chunk.flatten()
        chunk_ti = backend.where(chunk_tv > self.min_tv)[0]
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
        self.compare_cache()

        assert len(self.cache_tv) == 0
        assert len(self.compare_tv) == self.top_n

        tv, ri, ci = to_np((self.compare_tv, self.compare_ri, self.compare_ci))
        return scipy.sparse.csr_matrix((tv, (ri.astype(int), ci.astype(int))))

    def core_func(self, A, B, a_index, b_index):
        """ """
        raise NotImplementedError

    @classmethod
    def run(cls, data, block_size=1000, use_torch=True, device=None, dtype="float32", **agg_params):
        """ """
        device = get_device(device)

        if use_torch:
            data = to_torch(data, device=device, dtype=dtype)
        aggregator = cls(data, device=device, use_torch=use_torch, **agg_params)

        n_blocks = int(np.ceil(data.shape[1] / block_size))
        pbar = tqdm(total=(n_blocks * block_size / 1000) ** 2, desc=f'{cls.__name__} Block Analysis')
        for a_count in range(n_blocks):
            a_start = a_count * block_size
            a_n_items = block_size if a_count < n_blocks - 1 else data.shape[1] - a_start
            a_index = slice(a_start, a_start + a_n_items)
            a_block = data[:, a_index]

            for b_count in range(n_blocks):
                b_start = b_count * block_size
                b_n_items = block_size if b_count < n_blocks - 1 else data.shape[1] - b_start
                b_index = slice(b_start, b_start + b_n_items)
                b_block = data[:, b_index]

                aggregator(a_block, b_block, a_index, b_index)
                pbar.update(a_n_items * b_n_items // 1000 ** 2)

        return aggregator.results()


class SparseCorrelator(SparseAggregator):
    """ """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def core_func(self, A, B, a_index, b_index):
        """ """
        backend = self.backend
        M_chunk = backend.corrcoef(backend.hstack([A, B]).T)[:A.shape[1], A.shape[1]:]
        if a_index == b_index and self.skip_diagonal:
            M_chunk[backend.eye(M_chunk.shape[0], dtype=bool)] = -backend.inf

        return M_chunk
