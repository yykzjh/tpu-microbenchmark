"""Runtime cleanup helpers for ICI benchmark executions."""

from __future__ import annotations

import gc
import logging
from typing import Any

import jax

from utils.profiling import delete_device_object

logger = logging.getLogger(__name__)

def clear_runtime_memory() -> None:
    """Best-effort cleanup of JAX runtime caches and Python references."""
    clear_caches = getattr(jax, "clear_caches", None)
    if callable(clear_caches):
        try:
            clear_caches()
        except Exception as exc:
            logger.debug("Failed to clear JAX caches: %s", exc)
    gc.collect()


def release_compiled_cache(compiled_cache: dict[tuple[Any, ...], Any]) -> None:
    """Release compiled executables held by the benchmark cache."""
    seen = set()
    for compiled_value in compiled_cache.values():
        compiled_items = (
            compiled_value
            if isinstance(compiled_value, (list, tuple))
            else (compiled_value,)
        )
        for compiled_fn in compiled_items:
            if compiled_fn is None or id(compiled_fn) in seen:
                continue
            seen.add(id(compiled_fn))
            delete_device_object(compiled_fn)
    compiled_cache.clear()
    clear_runtime_memory()
