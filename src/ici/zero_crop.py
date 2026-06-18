"""ZeroCrop live-out consumer for ICI collective outputs."""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp

from utils.profiling import delete_device_object

logger = logging.getLogger(__name__)

try:
    from jax import core
    from jax import ffi
    from jax.interpreters import mlir
    _ZERO_CROP_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on installed JAX version.
    core = None
    ffi = None
    mlir = None
    _ZERO_CROP_IMPORT_ERROR = exc

zero_crop_p = None


if _ZERO_CROP_IMPORT_ERROR is None:
    Primitive = type(jax.lax.add_p)
    zero_crop_p = Primitive("zero_crop")

    def zero_crop_abstract_eval(x):
        """Keep the abstract result shape/dtype identical to the input."""
        return core.ShapedArray(x.shape, x.dtype)

    zero_crop_p.def_abstract_eval(zero_crop_abstract_eval)

    def zero_crop_lowering(ctx, x):
        """Lower zero_crop to the TPU-side ZeroCrop FFI custom call."""
        return ffi.ffi_lowering("ZeroCrop", has_side_effect=True)(ctx, x)

    mlir.register_lowering(zero_crop_p, zero_crop_lowering)


def zero_crop(x):
    """Consume a collective result through a side-effecting custom call.

    This mirrors AI-Hypercomputer/accelerator-microbenchmarks. The intent is
    to keep the collective result observable without treating the full output
    buffer as an ordinary benchmark result. CPU returns the input unchanged so
    local import/syntax checks do not require a TPU FFI target.
    """
    if jax.default_backend() == "cpu" or _ZERO_CROP_IMPORT_ERROR is not None:
        return x
    return ffi.ffi_call(
        "ZeroCrop",
        result_shape_dtypes=jax.ShapeDtypeStruct(x.shape, x.dtype),
        has_side_effect=True,
    )(x)


def detect_zero_crop_available() -> bool:
    """Return whether the current backend can compile the ZeroCrop FFI call."""
    if _ZERO_CROP_IMPORT_ERROR is not None:
        logger.warning(
            "ZeroCrop FFI helpers are unavailable in this JAX install; "
            "collective outputs will remain ordinary live-out buffers: %s",
            _ZERO_CROP_IMPORT_ERROR,
        )
        return False
    if jax.default_backend() == "cpu":
        return False

    try:
        probe = (
            jax.jit(lambda x: zero_crop(x))
            .lower(jax.ShapeDtypeStruct((1,), jnp.float32))
            .compile()
        )
        delete_device_object(probe)
        return True
    except Exception as exc:
        logger.warning(
            "ZeroCrop FFI is unavailable; collective outputs will remain "
            "ordinary live-out buffers: %s",
            exc,
        )
        return False


def maybe_zero_crop(x, enabled: bool):
    """Apply ZeroCrop when available for this run."""
    return zero_crop(x) if enabled else x
