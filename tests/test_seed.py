"""Tests for the reproducibility seed module.

This test suite verifies that `seed_everything` correctly initializes
the RNG state for all supported libraries and produces deterministic,
reproducible results across multiple invocations.
"""

from __future__ import annotations

import random as py_random

import pytest


class TestSeedDeterminism:
    """Test determinism of seed_everything across multiple calls."""

    def test_python_random_determinism(self) -> None:
        """Verify Python's random produces identical values after reseeding."""
        from src.utils.seed import seed_everything

        seed_everything(42)
        val1 = py_random.random()

        seed_everything(42)
        val2 = py_random.random()

        assert val1 == val2, "Python random should be deterministic"

    def test_numpy_determinism(self) -> None:
        """Verify NumPy produces identical arrays after reseeding."""
        import numpy as np

        from src.utils.seed import seed_everything

        seed_everything(42)
        arr1 = np.random.rand(3, 3)

        seed_everything(42)
        arr2 = np.random.rand(3, 3)

        assert np.array_equal(arr1, arr2), "NumPy should be deterministic"

    def test_torch_determinism(self) -> None:
        """Verify PyTorch produces identical tensors after reseeding."""
        import torch

        from src.utils.seed import seed_everything

        seed_everything(42)
        t1 = torch.randn(3, 3)

        seed_everything(42)
        t2 = torch.randn(3, 3)

        assert torch.equal(t1, t2), "PyTorch should be deterministic"


class TestJAXKeyReproducibility:
    """Test JAX PRNGKey reproducibility."""

    def test_jax_key_same_seed(self) -> None:
        """Verify JAX returns identical keys for the same seed."""
        from src.utils.seed import seed_everything

        key1 = seed_everything(42)
        key2 = seed_everything(42)

        if key1 is not None and key2 is not None:
            import jax.numpy as jnp

            assert jnp.array_equal(key1, key2), "JAX keys should match for same seed"

    def test_jax_key_different_seeds(self) -> None:
        """Verify JAX returns different keys for different seeds."""
        from src.utils.seed import seed_everything

        key1 = seed_everything(42)
        key2 = seed_everything(123)

        if key1 is not None and key2 is not None:
            import jax.numpy as jnp

            assert not jnp.array_equal(key1, key2), "JAX keys should differ for different seeds"


class TestDifferentSeeds:
    """Test that different seeds produce different random draws."""

    @pytest.mark.parametrize("seed1,seed2", [(42, 123), (0, 1), (999, 1000)])
    def test_different_seeds_yield_different_values(
        self, seed1: int, seed2: int
    ) -> None:
        """Verify different seeds produce different random values."""
        import numpy as np
        import torch

        from src.utils.seed import seed_everything

        # Test Python random
        seed_everything(seed1)
        py_val1 = py_random.random()

        seed_everything(seed2)
        py_val2 = py_random.random()

        assert py_val1 != py_val2, f"Python random should differ for seeds {seed1}, {seed2}"

        # Test NumPy
        seed_everything(seed1)
        np_val1 = np.random.rand()

        seed_everything(seed2)
        np_val2 = np.random.rand()

        assert np_val1 != np_val2, f"NumPy should differ for seeds {seed1}, {seed2}"

        # Test PyTorch
        seed_everything(seed1)
        torch_val1 = torch.randn(1).item()

        seed_everything(seed2)
        torch_val2 = torch.randn(1).item()

        assert torch_val1 != torch_val2, f"PyTorch should differ for seeds {seed1}, {seed2}"


class TestIdempotency:
    """Test idempotency of seed_everything."""

    def test_idempotent_reset(self) -> None:
        """Verify calling seed_everything multiple times resets state identically."""
        import numpy as np
        import torch

        from src.utils.seed import seed_everything

        # First call
        seed_everything(42)
        py_val1 = py_random.random()
        np_val1 = np.random.rand()
        torch_val1 = torch.randn(1).item()

        # Second call with same seed
        seed_everything(42)
        py_val2 = py_random.random()
        np_val2 = np.random.rand()
        torch_val2 = torch.randn(1).item()

        # Third call with same seed
        seed_everything(42)
        py_val3 = py_random.random()
        np_val3 = np.random.rand()
        torch_val3 = torch.randn(1).item()

        assert py_val1 == py_val2 == py_val3, "Python random should be idempotent"
        assert np_val1 == np_val2 == np_val3, "NumPy should be idempotent"
        assert torch_val1 == torch_val2 == torch_val3, "PyTorch should be idempotent"

    def test_idempotent_jax_key(self) -> None:
        """Verify JAX key is identical across multiple calls with same seed."""
        from src.utils.seed import seed_everything

        key1 = seed_everything(42)
        key2 = seed_everything(42)
        key3 = seed_everything(42)

        if key1 is not None and key2 is not None and key3 is not None:
            import jax.numpy as jnp

            assert jnp.array_equal(key1, key2), "JAX key should be idempotent (1 vs 2)"
            assert jnp.array_equal(key2, key3), "JAX key should be idempotent (2 vs 3)"
