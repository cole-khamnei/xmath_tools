import numpy as np
import torch

from tqdm.auto import tqdm


def to_np(array):
    """ """
    if isinstance(array, tuple):
        return [to_np(ar) for ar in array]
    if isinstance(array, torch.Tensor):
        return array.cpu().numpy()
    return array


def to_torch(array, **devp):
    """ """
    if not isinstance(array, torch.Tensor):
        return torch.tensor(array.astype("float32"), **devp)

    return array


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


class sparse_aggregator:
    """
    keeps running list of coordinates of top p percent (or top n) sparse connections:

    """
    def __init__(self, data, device, use_torch=True, sparsity_percent=0.1, skip_diagonal=True):
        self.size = data.shape[1]
        self.device = device
        self.devp = {"device": device} if use_torch else {}
        self.use_torch = use_torch

        self.sparsity_frac = sparsity_percent / 100
        self.top_n = int(np.ceil(self.size ** 2 * self.sparsity_frac))
        self.skip_diagonal = skip_diagonal

        if use_torch:
            self.top_k = lambda values, k, axis: torch.topk(values, k, dim=axis)
            self.backend = torch
        else:
            self.backend = np
            raise NotImplementedError
            self.top_k = lambda values, k, axis: np.partition(values, -k, axis=axis)[-k:]

        self.compare_tv = self.backend.ones(0, **self.devp)
        self.compare_ri = self.backend.ones(0, **self.devp)
        self.compare_ci = self.backend.ones(0, **self.devp)

    def __call__(self, A, B, a_index, b_index):
        """ """
        backend = self.backend

        M_chunk = backend.corrcoef(backend.hstack([A, B]).T)[:A.shape[1], A.shape[1]:] ** 2
        if a_index == b_index and self.skip_diagonal:
            M_chunk[backend.eye(M_chunk.shape[0], dtype=bool)] = -backend.inf

        chunk_tv = M_chunk.flatten()
        chunk_ti = backend.arange(len(chunk_tv), **self.devp)

        chunk_ri, chunk_ci = backend.unravel_index(chunk_ti, M_chunk.shape)
        chunk_ri = backend.arange(a_index.start, a_index.stop, **self.devp)[chunk_ri]
        chunk_ci = backend.arange(b_index.start, b_index.stop, **self.devp)[chunk_ci]

        self.compare_tv = backend.hstack([chunk_tv, self.compare_tv])
        self.compare_ri = backend.hstack([chunk_ri, self.compare_ri])
        self.compare_ci = backend.hstack([chunk_ci, self.compare_ci])

        if len(self.compare_tv) > self.top_n:
            self.compare_tv, update_ti = self.top_k(self.compare_tv, self.top_n, axis=0) # TODO: need axis?

            self.compare_ri = self.compare_ri[update_ti]
            self.compare_ci = self.compare_ci[update_ti]

    def results(self):
        return self.compare_tv, self.compare_ri, self.compare_ci


def block_analysis(data, aggregator_class, block_size=1000, use_torch=True, device="mps", **agg_params):
    """ """

    if use_torch:
        data = to_torch(data, device=device)
    aggregator = aggregator_class(data, device=device, use_torch=use_torch, **agg_params)

    n_blocks = int(np.ceil(data.shape[1] / block_size))

    overhang_size = data.shape[1] % (n_blocks - 1) + 1 if n_blocks > 1 else data.shape[1]

    for a_count in tqdm(range(n_blocks), leave=True):
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
    
    return aggregator.results()
