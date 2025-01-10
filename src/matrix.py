import numpy as np
import torch

from tqdm.auto import tqdm

def to_np(array):
    """ """
    if isinstance(array, torch.tensor):
        return array.cpu().numpy()
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


def block_analysis(data, aggregator_class, block_size=1000, use_torch=True, device="mps"):
    """ """
    if use_torch:
        data = torch.tensor(data.astype("float32"), device=device)
    
    aggregator = aggregator_class(data, device=device, use_torch=use_torch)

    n_blocks = int(np.ceil(data.shape[1] / block_size))
    overhang_size = data.shape[1] % (n_blocks - 1) + 1
    for a_count in tqdm(range(n_blocks), leave=True):
        a_start = a_count * block_size
        a_n_items = block_size if a_count < n_blocks - 1 else overhang_size
        a_index = slice(a_start, a_start + a_n_items)
        a_block = data[:, a_index]
    
        for b_count in tqdm(range(n_blocks), leave=False):
            b_start = b_count * block_size
            b_n_items = block_size if b_count < n_blocks - 1 else overhang_size
            b_index = slice(b_start, b_start + b_n_items)
            b_block = data[:, b_index]

            aggregator(a_block, b_block, a_index, b_index)
    
    return aggregator.results()
