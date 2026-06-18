"""Array construction helpers for memory benchmarks."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def create_test_array(
    shape: tuple[int, ...],
    dtype=jnp.float32,
    fill_value: float = 1.0,
    device: Any | None = None,
) -> jnp.ndarray:
    """Create a constant test array and optionally place it on one device."""
    # Build on host first, then place on the requested device. This keeps the
    # benchmark kernels simple and makes device placement explicit.
    host_value = np.full(shape, fill_value, dtype=np.float32)
    if device is not None:
        return jax.device_put(host_value, device).astype(dtype)
    return jnp.asarray(host_value).astype(dtype)
