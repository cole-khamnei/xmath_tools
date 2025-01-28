import numpy as np


from typing import Union

Module = None

VALID_BACKENDS = {"numpy": np}

try:
    import torch
    VALID_BACKENDS["torch"] = torch
    USE_CUDA = torch.cuda.is_available()
    USE_MPS = torch.mps.is_available()
    DEFAULT_DEVICE = "cuda" if USE_CUDA else "mps" if USE_MPS else "cpu"

except:
    pass

try:
    import mlx.core as mx
    VALID_BACKENDS["mlx"] = mx
except:
    pass


# ----------------------------------------------------------------------------# 
# --------------------          Backend Helpers           --------------------# 
# ----------------------------------------------------------------------------# 


def get_backend_name(backend: Union[str, Module]) -> str:
    """ """
    if isinstance(backend, str):
        return backend

    return backend.__name__.replace(".core", "")


def assert_valid_backend(backend: Union[str, Module]) -> None:
    """"""
    backend_name = get_backend_name(backend)
    assert backend_name in VALID_BACKENDS, f"Invalid backend '{backend_name}', valid options are {VALID_BACKENDS.keys()}"


def get_backend(backend: Union[str, Module]) -> Module:
    """ """
    
    assert_valid_backend(backend)
    if isinstance(backend, str):
        return VALID_BACKENDS[backend_name]

    return backend


def get_device_info(backend: Union[str, Module], device=None) -> dict:
    """ """

    assert_valid_backend(backend)
    backend_name = get_backend_name(backend)

    if backend_name == "numpy":
        return {}

    elif backend_name == "torch":
        device = DEFAULT_DEVICE if device is None else device
        return {"device": device}

    elif backend_name == "mlx":
        device = mx.gpu if device is None else device
        device = mx.cpu if device.lower() == "cpu" else mx.gpu if isinstance(device, str) else device
        return {"stream": device}
    
    raise ValueError, backend


def backend_array(backend):
    """ """
    backend_name = get_backend_name(backend)

    if backend_name == "numpy":
        return np.array

    elif backend_name == "torch":
        return torch.tensor

    elif backend_name == "mlx":
        return mlx.array

    raise ValueError, backend


def to_backend(backend, values, **kwargs):
    """ """
    array_func = backend_array(backend)
    return array_func(values, **kwargs)


# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
