import numpy as np


from typing import Union

Module = None

VALID_BACKENDS = {"np": np}

try:
    import torch
    VALID_BACKENDS["torch"] = torch

except:
    pass

try:
    import mlx
    VALID_BACKENDS["mlx"] = mlx
except:
    pass


# ----------------------------------------------------------------------------# 
# --------------------          Backend Helpers           --------------------# 
# ----------------------------------------------------------------------------# 


def get_backend(backend: Union[str, Module]) -> Module:
    """ """
    if isinstance(backend, str):
        assert backend in VALID_BACKENDS, f"Invalid backend '{backend}', valid options are {VALID_BACKENDS.keys()}"
        return VALID_BACKENDS[backend]

    assert any(backend is in list(VALID_BACKENDS.values())), f"Invalid backend '{backend.__name__}',"
                                                             f"valid options are {VALID_BACKENDS.keys()}"
    return 


def get_devp():
    """ """
    return 

# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
