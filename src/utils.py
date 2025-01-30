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


def check_backend(backend: Union[str, Module], name) -> bool:
    """ """
    backend_name = get_backend_name(backend)
    return backend_name == name


def get_backend(backend: Union[str, Module]) -> Module:
    """ """
    
    assert_valid_backend(backend)
    if isinstance(backend, str):
        return VALID_BACKENDS[backend]

    return backend


def get_device_info(backend: Union[str, Module], device) -> dict:
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
        if isinstance(device, str):
            device = mx.cpu if device.lower() == "cpu" else mx.gpu

        # TODO: Arrays are shared in MLX, so no need to move to gpu, however, need to specify stream
        return {}
        # return {"stream": device}
    
    raise NotImplementedError


def backend_array(backend: Union[str, Module]):
    """ """
    backend_name = get_backend_name(backend)
    if backend_name == "numpy":
        return np.array

    elif backend_name == "torch":
        return torch.tensor

    elif backend_name == "mlx":
        return mx.array

    raise NotImplementedError


def to_backend(backend: Union[str, Module], values: np.ndarray, dtype = None, **kwargs):
    """ """
    values = values if dtype is None else values.astype(dtype) 
    array_func = backend_array(backend)
    return array_func(values, **kwargs)


def to_np(array):
    """ """
    if isinstance(array, tuple):
        return [to_np(ar) for ar in array]
    if isinstance(array, torch.Tensor):
        return array.cpu().numpy()
    return np.array(array)

# ----------------------------------------------------------------------------# 
# -------------------          Backend Math Tools          -------------------# 
# ----------------------------------------------------------------------------# 


def backend_corr(backend, x, y):
    """ """
    backend = get_backend(backend)

    x, y = x.T, y.T
    x_demeaned = x - x.mean(axis=1, keepdims=True)
    y_demeaned = y - y.mean(axis=1, keepdims=True)

    x_norm = x_demeaned / backend.sqrt(backend.sum(x_demeaned ** 2, axis=1, keepdims=True))
    y_norm = y_demeaned / backend.sqrt(backend.sum(y_demeaned ** 2, axis=1, keepdims=True))
    return x_norm @ y_norm.T


# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
