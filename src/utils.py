import numpy as np


from typing import Union

Module = None

VALID_BACKENDS = {"numpy": np}

import torch
VALID_BACKENDS["torch"] = torch
USE_CUDA = torch.cuda.is_available()
USE_MPS = torch.mps.is_available()
DEFAULT_DEVICE = "cuda" if USE_CUDA else "mps" if USE_MPS else "cpu"


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

def backend_triu_indices(backend, shape, offset=0):
    """ """
    if check_backend(backend, "torch"):
        return backend.triu_indices(*shape, offset=offset) 
    
    return backend.triu_indices(shape[0], k=offset)


def backend_norm(backend, x):
    """ """
    backend = get_backend(backend)

    x = x.T
    x_demeaned = x - x.mean(axis=1, keepdims=True)
    x_norm = x_demeaned / backend.sqrt(backend.sum(x_demeaned ** 2, axis=1, keepdims=True))
    return x_norm.T


def backend_corr(backend, x, y):
    """ """
    backend = get_backend(backend)

    x, y = x.T, y.T
    x_demeaned = x - x.mean(axis=1, keepdims=True)
    y_demeaned = y - y.mean(axis=1, keepdims=True)

    x_norm = x_demeaned / backend.sqrt(backend.sum(x_demeaned ** 2, axis=1, keepdims=True))
    y_norm = y_demeaned / backend.sqrt(backend.sum(y_demeaned ** 2, axis=1, keepdims=True))
    return x_norm @ y_norm.T


# \section MPS Torch Safe Actions

def needs_MPS_safety(backend, device_info):
    """ """
    return check_backend(backend, "torch") and device_info == {"device": "mps"}


def MPS_safe_where(backend, device_info, array):
    """ only works for flat arrays"""
    if needs_MPS_safety(backend, device_info):
        cpu_array = array.to(backend.device("cpu"))
        return backend.where(cpu_array)[0].to(torch.device("mps"))

    return backend.where(array)[0]


def MPS_safe_unravel_index(backend, device_info, array, shape):
    """ """
    if needs_MPS_safety(backend, device_info):
        ri_cpu, ci_cpu = backend.unravel_index(array.to(backend.device("cpu")), shape)
        return ri_cpu.to(torch.device("mps")), ci_cpu.to(torch.device("mps"))

    return backend.unravel_index(array, shape)


def MPS_safe_topk(backend, device_info, array, k, axis=0):
    """ """
    if needs_MPS_safety(backend, device_info):
        cpu_array = array.to(backend.device("cpu"))
        top_values_cpu, top_index_cpu = torch.topk(cpu_array, k, dim=axis)
        return top_values_cpu.to(torch.device("mps")), top_index_cpu.to(torch.device("mps"))

    elif check_backend(backend, "torch"):
        return torch.topk(array, k, dim=axis)

    elif check_backend(backend, "numpy"):
        return (np.partition(array, -k, axis=axis)[-k:], np.argpartition(array, -k, axis=axis)[-k:])

    else:
        raise NotImplementedError

# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
