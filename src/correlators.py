from . import aggregators
from . import utils

# ----------------------------------------------------------------------------# 
# --------------------            Correlators             --------------------# 
# ----------------------------------------------------------------------------# 


class SparseCorrelator(aggregators.SparseAggregator):
    """ """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corr_func = utils.backend_corr

    def core_func(self, A, B, a_index, b_index):
        """ """
        M_chunk = self.corr_func(self.backend, A, B)
        if a_index == b_index and self.skip_diagonal:
            M_chunk[self.backend.eye(M_chunk.shape[0], dtype=bool)] = -self.backend.inf

        return M_chunk


class ThresholdCorrelator(aggregators.ThresholdAggregator):
    """ """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corr_func = utils.backend_corr

    def core_func(self, A, B, a_index, b_index):
        """ """
        M_chunk = self.corr_func(self.backend, A, B)
        if a_index == b_index and self.skip_diagonal:
            M_chunk[self.backend.eye(M_chunk.shape[0], dtype=bool)] = -self.backend.inf

        return M_chunk


# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
