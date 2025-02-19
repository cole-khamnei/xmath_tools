from . import block_analysis
from . import aggregators
from . import utils


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

    def core_func(self, A, B, a_index, b_index):
        """ """
        if self.preprocessed:
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


# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
