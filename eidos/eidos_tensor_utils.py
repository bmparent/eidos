"""Small tensor/array conversion helpers used by optional Eidos modules."""

from typing import Any, Optional

import numpy as np


def to_cpu_numpy(value: Any, *, dtype: Optional[Any] = None, copy: bool = False) -> np.ndarray:
    """Return ``value`` as a CPU NumPy array.

    PyTorch CUDA tensors cannot be converted by ``np.asarray`` directly. This
    helper keeps the optional modules independent from torch while still doing
    the safe ``detach().cpu().numpy()`` path when tensor-like inputs appear.
    """
    current = value
    detach = getattr(current, "detach", None)
    if callable(detach):
        current = detach()

    cpu = getattr(current, "cpu", None)
    if callable(cpu):
        current = cpu()

    numpy = getattr(current, "numpy", None)
    if callable(numpy):
        arr = numpy()
    else:
        arr = np.asarray(current)

    if dtype is not None:
        return np.asarray(arr, dtype=dtype)
    if copy:
        return np.array(arr, copy=True)
    return np.asarray(arr)


def to_cpu_numpy_1d(value: Any, *, dtype: Optional[Any] = np.float64) -> np.ndarray:
    """Return ``value`` as a flattened CPU NumPy array."""
    return to_cpu_numpy(value, dtype=dtype).reshape(-1)


def to_cpu_list(value: Any) -> Any:
    """Return a JSON-friendly Python scalar/list from tensor-like input."""
    arr = to_cpu_numpy(value)
    return arr.tolist()
