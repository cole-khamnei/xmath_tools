from . import block_analysis
from . import aggregators
from . import utils

import gc
import torch

#\section Correlator class


class Correlator:
    def __init__(self, *args, backend="numpy", skip_diagonal="None", **kwargs):
        self.corr_func = utils.backend_corr
        self.backend = utils.get_backend(backend)
        self.skip_diagonal = skip_diagonal

    def preprocess(self, data):
        """ norm data so correlation is just multiplication """
        data = utils.backend_norm(self.backend, data)
        self.preprocessed = True
        return data

    def core_func(self, A, B, a_index=None, b_index=None, preprocess_overide=False):
        """ """
        if self.preprocessed and not preprocess_overide:
            M_chunk = A.T @ B
        else:
            M_chunk = self.corr_func(self.backend, A, B)
        
        if a_index == b_index and self.skip_diagonal:
            M_chunk[self.backend.eye(M_chunk.shape[0], dtype=bool)] = -self.backend.inf

        return M_chunk


#\section Runners


class Runner(Correlator, block_analysis.BlockAnalysis):
    """ 
    Runs a correlation to test speeds
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, A, B, a_index, b_index, mask=None, exclude_index=None):
        """ """
        M_chunk = self.core_func(A, B, a_index, b_index)


class Maxxer(Correlator, block_analysis.BlockAnalysis):
    """ 
    Runs a correlation to test speeds
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max = -self.backend.inf

    def __call__(self, A, B, a_index, b_index, mask=None, exclude_index=None):
        """ """
        M_chunk = self.core_func(A, B, a_index, b_index)
        self.max = max(M_chunk.max(), self.max)
    
    def results(self):
        return self.max


# ----------------------------------------------------------------------------# 
# --------------------            Correlators             --------------------# 
# ----------------------------------------------------------------------------# 


class SparseCorrelator(Correlator, aggregators.SparseAggregator):
    """ """
    def __init__(self, *args, **kwargs):
        Correlator.__init__(self, *args, **kwargs)
        aggregators.SparseAggregator.__init__(self, *args, **kwargs)


class ThresholdCorrelator(Correlator, aggregators.ThresholdAggregator):
    """ """
    def __init__(self, *args, **kwargs):
        Correlator.__init__(self, *args, **kwargs)
        aggregators.ThresholdAggregator.__init__(self, *args, **kwargs)



class PairCorrelator(Correlator, aggregators.PairComparator):
    """ """
    def __init__(self, *args, **kwargs):
        Correlator.__init__(self, *args, **kwargs)
        aggregators.PairComparator.__init__(self, *args, **kwargs)

    def preprocess(self, data):
        """ norm data so correlation is just multiplication """

        d1, d2 = data[:, :, 0], data[:, :, 1]
        d1 = utils.backend_norm(self.backend, d1)
        d2 = utils.backend_norm(self.backend, d2)
        data = self.backend.stack((d1, d2), axis=2)
        self.preprocessed = True
        return data

    def compare(self, d1, d2, axis=None):
        """ """
        if axis is None:
            r = self.core_func(d1.reshape(-1, 1), d2.reshape(-1, 1), preprocess_overide=True)
        else:
            raise NotImplementedError

        return r[0]


class PairCorrelatorAxis(Correlator, aggregators.PairComparatorAxis):
    """ """
    def __init__(self, *args, **kwargs):
        Correlator.__init__(self, *args, **kwargs)
        aggregators.PairComparatorAxis.__init__(self, *args, **kwargs)

    def preprocess(self, data):
        """ norm data so correlation is just multiplication """

        d1, d2 = data[:, :, 0], data[:, :, 1]
        d1 = utils.backend_norm(self.backend, d1)
        d2 = utils.backend_norm(self.backend, d2)
        data = self.backend.stack((d1, d2), axis=2)
        self.preprocessed = True
        return data

    def compare(self, d1, d2, axis=None):
        """ """
        if axis is None:
            d1, d2 = d1.reshape(1, -1), d2.reshape(1, -1)

        d1_normed = utils.backend_norm(self.backend, d1.T)
        d2_normed = utils.backend_norm(self.backend, d2.T)
        r_axis = (d1_normed * d2_normed).sum(axis=0)

        return r_axis


def pair_correlation(*args, **kwargs):
    """ """
    if kwargs["axis"] is None:
        pair_correlator = PairCorrelator
    else:
        kwargs["block_size"] = min(kwargs.get("block_size", 2000), 2000)
        pair_correlator = PairCorrelatorAxis

    results = pair_correlator.run(*args, **kwargs)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
