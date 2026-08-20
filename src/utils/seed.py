"""Reproducibility seed module for deterministic experiments.

This module provides a single function to set the global RNG state for all
randomness sources used in the repository: Python's built-in `random`, NumPy,
PyTorch (including CUDA), and JAX.

Reproducibility Contract
------------------------
- All environment variables are set BEFORE importing torch/jax to ensure
  deterministic behavior in CUDA kernels and hash-based operations.
- For libraries with global state (`random`, `numpy`, `torch`), the global
  RNG is seeded directly.
- JAX has no global RNG state. Instead, this function returns a fresh
  `jax.random.PRNGKey(seed)` that should be threaded through JAX code.
- If a library is not installed, it is skipped gracefully without raising
  ImportError.

Usage
-----
    from src.utils.seed import seed_everything

    # Set seed and get a JAX PRNGKey
    key = seed_everything(42)

    # Use the key in JAX code
    from jax import random
    subkey1, subkey2 = random.split(key)

Environment Variables
---------------------
The following environment variables are set via `os.environ.setdefault`:
- `PYTHONHASHSEED`: Ensures deterministic hash-based operations in Python.
- `CUDA_DETERMINISTIC`: Forces deterministic CUDA kernel selection.
- `CUBLAS_WORKSPACE_CONFIG`: Configures cuBLAS workspace for determinism.

"""

from __future__ import annotations

import os
import random as py_random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def seed_everything(seed: int = 42) -> object:
    """Set global RNG state for reproducibility across all libraries.

    This function configures the random number generators for Python's
    built-in `random`, NumPy, PyTorch (including CUDA), and JAX. It also
    sets environment variables required for deterministic behavior in
    hash-based operations and CUDA kernels.

    Parameters
    ----------
    seed : int, optional
        The seed value for all RNGs. Default is 42.

    Returns
    -------
    object
        A JAX PRNGKey array if JAX is installed, otherwise None.
        The return type is `object` to avoid requiring JAX at runtime.

    Notes
    -----
    - Environment variables are set using `os.environ.setdefault`, which
      only sets them if they are not already defined.
    - PyTorch CUDA seeding is performed only if `torch.cuda.is_available()`.
    - JAX has no global state; the returned key must be explicitly passed
      to JAX functions that require randomness.
    - If any library is not installed, it is silently skipped.

    Examples
    --------
    >>> key = seed_everything(42)
    >>> import numpy as np
    >>> np.random.rand()  # doctest: +SKIP
    0.3745401188473625

    """
    # Set environment variables BEFORE importing torch/jax
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUDA_DETERMINISTIC", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # Seed Python's built-in random
    py_random.seed(seed)

    # Seed NumPy
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    # Seed PyTorch (and CUDA if available)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Enforce deterministic behavior in CUDA operations
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass

    # JAX: No global state; generate and return a PRNGKey
    try:
        import jax.random as jrandom

        return jrandom.PRNGKey(seed)
    except ImportError:
        pass

    return None
