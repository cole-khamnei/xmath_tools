import numpy as np
import torch
import scipy

from tqdm.auto import tqdm

from . import utils

# ----------------------------------------------------------------------------# 
# -----------------           Bock Analysis Module           -----------------# 
# ----------------------------------------------------------------------------# 


# TODO: consider turning into functional versional rather than objects?


class BlockAnalysis:
    """
    """
    def __init__(self, data, backend="torch", device=None, symmetric=True, skip_diagonal=True):
        self.size = data.shape[1]
        self.shape = (data.shape[1], data.shape[1])
        
        #TODO: refactor device handling
        self.backend = utils.get_backend(backend)
        self.device_info = utils.get_device_info(backend, device)

        self.skip_diagonal = skip_diagonal
        self.symmetric = symmetric
        self.preprocessed = False

        self.create_empty = lambda k=1: [self.backend.ones(0, **self.device_info) for _ in range(k)]

    @classmethod
    def run(cls, data, mask=None, exclude_index=None, block_size=4_000,
            backend="torch", device=None,
            symmetric=False, dtype="float32", **block_params):
        """
        TODO: added second data matrix and scan through both matrices
        TODO: add symmetry argument that scans through only half of matrix

        """

        aggregator = cls(data, backend=backend, device=device, symmetric=symmetric, **block_params)
        device_info = utils.get_device_info(backend, device)

        data = utils.to_backend(backend, data, dtype=dtype, **device_info)

        # TODO: Fully test
        data = aggregator.preprocess(data)

        if exclude_index is not None:
            exclude_index = utils.to_backend(backend, exclude_index, dtype="float16", **device_info)

        if symmetric and mask is not None:
            if get_nnz_safe(mask != mask.T) == 0:
                print("Mask is not symmetric, cannot use symmetric acceleration.")
                symmetric = False

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

    def get_mask_chunk_index(self, mask, a_index, b_index, select_index):
        """ """
        if mask is not None:
            mask_chunk = mask[a_index, b_index]
            if get_nnz_safe(mask_chunk) > 0:
                mask_flat = ~sparse_to_array(mask_chunk.astype(bool)).ravel()
                # TODO: move to_backend to a to_backend_array() which creates either np or torch
                mask_flat = utils.to_backend(self.backend, mask_flat, dtype=bool, **self.device_info)
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

    def preprocess(self, data):
        """ Can be modified child classes"""
        self.preprocessed = False
        return data

    def core_func(self, A, B, a_index, b_index):
        """ """
        raise NotImplementedError

    def __call__(self, A, B, a_index, b_index, mask=None, exclude_index=None):
        """ """
        raise NotImplementedError

    def results(self):
        return None


# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
