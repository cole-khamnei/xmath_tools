import numpy as np
import torch

from tqdm.auto import tqdm

def to_np(array):
    """ """
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
    def __init__(self, data, device, use_torch=True, sparsity_percent=0.1):
        self.size = data.shape[1]
        self.device = device
        self.devp = {"device": device} if use_torch else {}
        self.use_torch = use_torch

        self.sparsity_frac = sparsity_percent / 100
        self.top_n = int(np.ceil(self.size ** 2 * self.sparsity_frac))
        print(self.top_n)

        if use_torch:
            self.top_k = lambda values, k, axis: torch.topk(values, k, dim=axis)
            self.backend = torch
        else:
            self.backend = np
            self.top_k = lambda values, k, axis: np.partition(values, -k, axis=axis)[-k:]

        self.top_values = -self.backend.ones(self.top_n, **self.devp)
        self.top_ci = -self.backend.ones(self.top_n, **self.devp)
        self.top_ri = -self.backend.ones(self.top_n, **self.devp)

    def __call__(self, A, B, a_index, b_index):
        """ """
        backend = self.backend
        M_chunk = backend.corrcoef(backend.hstack([A, B]).T)[:A.shape[1], A.shape[1]:] ** 2
        if a_index == b_index:
            M_chunk[~backend.eye(M_chunk.shape[0], dtype=bool)] = -10
            # M_chunk = M_chunk[~backend.eye(M_chunk.shape[0], dtype=bool)].reshape(A.shape[1], -1)
            # raise NotImplementedError

        M_chunk_flat = M_chunk.flatten()

        adjusted_k = min(self.top_n, to_np(M_chunk.shape[0] * M_chunk.shape[1]))
        # Get top n values and their indices
        chunk_tv, chunk_ti = self.top_k(M_chunk_flat, adjusted_k, axis=0)

        # Convert the flat indices back to row and column indices
        chunk_ri, chunk_ci = backend.unravel_index(chunk_ti, M_chunk.shape)

        chunk_ri = backend.arange(a_index.start, a_index.stop, **self.devp)[chunk_ri]
        chunk_ci = backend.arange(b_index.start, b_index.stop, **self.devp)[chunk_ci]

        compare_ri = backend.hstack([chunk_ri, self.top_ri])
        compare_ci = backend.hstack([chunk_ci, self.top_ci])
        compare_tv = backend.hstack([chunk_tv, self.top_values])

        self.top_values, update_ti = self.top_k(compare_tv, self.top_n, axis=0) # TODO: need axis?
        self.top_ci[update_ti] = compare_ci[update_ti]
        self.top_ri[update_ti] = compare_ri[update_ti]

    def results(self):
        return self.top_values, self.top_ri, self.top_ci
        return to_np(self.top_values)


def block_analysis(data, aggregator_class, block_size=1000, use_torch=True, device="mps", **agg_params):
    """ """

    if use_torch:
        data = to_torch(data, device=device)
    aggregator = aggregator_class(data, device=device, use_torch=use_torch, **agg_params)

    n_blocks = int(np.ceil(data.shape[1] / block_size))
    overhang_size = data.shape[1] % (n_blocks - 1) + 1
    for a_count in tqdm(range(n_blocks), leave=True):
        a_start = a_count * block_size
        a_n_items = block_size if a_count < n_blocks - 1 else overhang_size
        a_index = slice(a_start, a_start + a_n_items)
        a_block = data[:, a_index]
    
        for b_count in range(n_blocks):
            b_start = b_count * block_size
            b_n_items = block_size if b_count < n_blocks - 1 else overhang_size
            b_index = slice(b_start, b_start + b_n_items)
            b_block = data[:, b_index]

            aggregator(a_block, b_block, a_index, b_index)
    
    return aggregator.results()
