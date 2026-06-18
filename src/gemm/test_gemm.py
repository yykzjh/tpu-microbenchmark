"""TPU GEMM peak compute benchmark.

Each JAX device runs one local matrix multiplication:

    C[m, n] = A[m, k] @ B[k, n]

The input dtype is controlled by ``--dtype``. The timed xprof marker wraps only
the matrix multiplication.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils.runtime import (
    configure_logging,
    configure_dump_hlo_from_argv,
    configure_tpu_benchmark_env,
    device_metadata,
    get_local_devices_or_raise,
    initialize_jax_runtime,
    log_device_separator,
    prepare_benchmark_dirs,
    validate_non_negative,
    validate_positive,
)

# ---------------------------------------------------------------------------
# Environment flags MUST be set before JAX is imported.
# - LIBTPU_INIT_ARGS: TPU GEMM flags adapted from accelerator-microbenchmarks.
# - XLA_FLAGS: --xla_dump_to for HLO graph dumps (only if --dump-hlo).
# ---------------------------------------------------------------------------
configure_tpu_benchmark_env("gemm")

_DUMP_HLO = configure_dump_hlo_from_argv("gemm")

from utils.metrics import MetricsStatistics, average_min_max, write_jsonl_metrics

logger = logging.getLogger(__name__)


def load_jax_runtime_deps() -> None:
    """Import JAX-dependent modules after CLI parsing."""
    global jax, jnp, MARKER, collect_hlo_dumps_if_requested
    global TraceTimingConfig, delete_device_object, run_profiled_iterations

    import jax as _jax
    import jax.numpy as _jnp
    from utils.profiling import (
        MARKER as _MARKER,
        TraceTimingConfig as _TraceTimingConfig,
        collect_hlo_dumps_if_requested as _collect_hlo_dumps_if_requested,
        delete_device_object as _delete_device_object,
        run_profiled_iterations as _run_profiled_iterations,
    )

    jax = _jax
    jnp = _jnp
    MARKER = _MARKER
    TraceTimingConfig = _TraceTimingConfig
    collect_hlo_dumps_if_requested = _collect_hlo_dumps_if_requested
    delete_device_object = _delete_device_object
    run_profiled_iterations = _run_profiled_iterations


@dataclass(frozen=True)
class GemmShape:
    """Matrix multiplication shape."""

    m: int
    n: int
    k: int


def parse_input_dtype(dtype_name: str):
    """Return a JAX dtype from a user-facing dtype name."""
    normalized = dtype_name.strip().lower()
    aliases = {
        "float32": jnp.float32,
        "fp32": jnp.float32,
        "bf16": jnp.bfloat16,
        "fp16": jnp.float16,
        "fp8": "float8_e4m3fn",
    }
    if normalized == "fp8":
        dtype = getattr(jnp, aliases[normalized], None)
        if dtype is None:
            raise ValueError("JAX does not expose dtype float8_e4m3fn")
        return dtype

    if normalized not in aliases:
        raise ValueError(
            f"Unsupported dtype {dtype_name!r}; supported values: "
            "fp32, fp16, bf16, fp8"
        )
    return aliases[normalized]


def dtype_name(dtype) -> str:
    """Return a stable dtype display name."""
    return jnp.dtype(dtype).name


def validate_shape(shape: GemmShape) -> None:
    """Validate matrix dimensions."""
    for name, value in (("m", shape.m), ("n", shape.n), ("k", shape.k)):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")


def make_device_operand(
    device: Any,
    matrix_shape: tuple[int, int],
    dtype,
) -> jax.Array:
    """Create one matrix operand on a single JAX device."""
    # Keep operand creation outside the timed region. The benchmark measures
    # only the matmul kernel, not host allocation or H2D transfer.
    host_value = np.ones(matrix_shape, dtype=np.float32)
    return jax.device_put(host_value, device).astype(dtype)


def compile_gemm_kernel_for_device(
    device: Any,
    shape: GemmShape,
    input_dtype,
):
    """Compile a GEMM kernel pinned to one JAX device."""
    def kernel(lhs, rhs):
        with jax.named_scope(MARKER):
            # Let XLA choose the TPU matmul lowering and result dtype for the
            # input dtype. Keeping the marker around matmul only avoids counting
            # an explicit output cast in the compute duration.
            return jnp.matmul(lhs, rhs)

    lhs_spec = jax.ShapeDtypeStruct((shape.m, shape.k), input_dtype)
    rhs_spec = jax.ShapeDtypeStruct((shape.k, shape.n), input_dtype)
    # Pin the compiled executable to one device. This benchmark intentionally
    # does not shard one GEMM over multiple TPU devices.
    return jax.jit(kernel, device=device).lower(lhs_spec, rhs_spec).compile()


def compute_tflops(flops: int | float, duration_ms: float) -> float:
    """Compute TFLOP/s from floating-point operation count and duration."""
    if duration_ms <= 0:
        return 0.0
    return flops / 1e12 / (duration_ms / 1e3)


def release_compiled_cache(compiled_cache: dict[str, Any]) -> None:
    """Release compiled executables."""
    for compiled_fn in compiled_cache.values():
        delete = getattr(compiled_fn, "delete", None)
        if callable(delete):
            try:
                delete()
            except Exception as exc:
                logger.debug("Failed to delete compiled executable: %s", exc)
    compiled_cache.clear()
    gc.collect()


def run_gemm_per_device(
    device: Any,
    device_index: int,
    shape: GemmShape,
    input_dtype,
    input_dtype_display: str,
    warmup: int,
    iteration: int,
    output_directory: str,
    metrics_dir: str,
    trace_dir: str,
    dump_hlo_dir: str | None,
    dump_hlo: bool = False,
    cleanup_trace: bool = False,
) -> dict[str, Any]:
    """Run GEMM benchmark on one local JAX device."""
    lhs = rhs = None
    compiled_cache: dict[str, Any] = {}
    try:
        lhs = make_device_operand(device, (shape.m, shape.k), input_dtype)
        rhs = make_device_operand(device, (shape.k, shape.n), input_dtype)
        # Make sure operands are resident before warmup starts.
        jax.block_until_ready((lhs, rhs))

        compiled_cache["gemm"] = compile_gemm_kernel_for_device(
            device,
            shape,
            input_dtype,
        )
        compiled_fn = compiled_cache["gemm"]

        def data_generator():
            # Reuse the same resident operands for every timed iteration. That
            # keeps the xprof marker around the GEMM only.
            return lhs, rhs

        logger.info("Phase 1: warmup (%d iterations)", warmup)
        for _ in range(warmup):
            result = None
            try:
                result = compiled_fn(lhs, rhs)
                jax.block_until_ready(result)
            finally:
                delete_device_object(result)

        logger.info(
            "Phase 2: xprof marker timed iterations (%d iterations)",
            iteration,
        )
        test_name = (
            f"tpu_gemm_device{device_index}_m{shape.m}_n{shape.n}_"
            f"k{shape.k}_{input_dtype_display}"
        )
        durations_ms = run_profiled_iterations(
            compiled_fn=compiled_fn,
            data_generator=data_generator,
            iteration=iteration,
            config=TraceTimingConfig(
                task_name=test_name,
                trace_dir=trace_dir,
                dest_name=f"{test_name}_timed",
                cleanup_trace=cleanup_trace,
                kernel_name_contains=["convolution", "dot_general", "dot"],
                thread_name_contains="XLA Ops",
                per_iteration_reducer="max",
                require_marker=False,
            ),
        )

        collect_hlo_dumps_if_requested(
            dump_hlo,
            _DUMP_HLO.temp_dir,
            dump_hlo_dir,
            test_name,
            clear_source=_DUMP_HLO.owned,
        )

        duration_stats = MetricsStatistics(durations_ms, "duration", unit="ms")
        metrics = duration_stats.serialize()
        avg_ms = float(metrics.get("duration_avg_ms", 0))
        flops_per_device = 2 * shape.m * shape.n * shape.k
        metrics.update({
            "flops": flops_per_device,
            "timing_source": "xprof_gemm_device_kernel_duration",
            "output_dtype": "jax_matmul_inferred",
            "tflops": round(compute_tflops(flops_per_device, avg_ms), 4),
        })

        metadata = {
            "benchmark": "gemm",
            **device_metadata(device, device_index),
            "m": shape.m,
            "n": shape.n,
            "k": shape.k,
            "input_dtype": input_dtype_display,
            "output_dtype": "jax_matmul_inferred",
            "lhs_shape": [shape.m, shape.k],
            "rhs_shape": [shape.k, shape.n],
            "output_shape": [shape.m, shape.n],
            "operation": "jax.numpy.matmul",
            "flops_formula": "2 * m * n * k",
            "warmup": warmup,
            "iteration": iteration,
            "dump_hlo": dump_hlo,
        }
        write_jsonl_metrics(metrics_dir, test_name, metadata, metrics)

        return {
            "metadata": metadata,
            "metrics": metrics,
            "output_directory": output_directory,
        }
    finally:
        delete_device_object(lhs)
        delete_device_object(rhs)
        release_compiled_cache(compiled_cache)


def run_gemm(
    shape: GemmShape,
    dtype_name_input: str,
    warmup: int,
    iteration: int,
    result_dir: str,
    dump_hlo: bool = False,
    cleanup_trace: bool = False,
) -> dict[str, Any]:
    """Run GEMM sequentially on every local JAX device."""
    validate_shape(shape)
    validate_non_negative("warmup", warmup)
    validate_positive("iteration", iteration)

    input_dtype = parse_input_dtype(dtype_name_input)
    input_dtype_display = dtype_name(input_dtype)
    devices = get_local_devices_or_raise()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{ts}_gemm_m{shape.m}_n{shape.n}_k{shape.k}_"
        f"dtype{input_dtype_display}_{len(devices)}devices"
    )
    dirs = prepare_benchmark_dirs(result_dir, run_name, dump_hlo=dump_hlo)
    out_dir = dirs.output_dir
    metrics_dir = dirs.metrics_dir
    trace_dir = dirs.trace_dir
    dump_hlo_dir = dirs.dump_hlo_dir

    per_device_results = []
    for device_index, device in enumerate(devices):
        # Devices are measured one by one, so per-device TFLOP/s is not inflated
        # by parallel execution across the full local TPU host.
        log_device_separator(logger, "GEMM local device", device_index, device)
        device_result = run_gemm_per_device(
            device=device,
            device_index=device_index,
            shape=shape,
            input_dtype=input_dtype,
            input_dtype_display=input_dtype_display,
            warmup=warmup,
            iteration=iteration,
            output_directory=out_dir,
            metrics_dir=metrics_dir,
            trace_dir=trace_dir,
            dump_hlo_dir=dump_hlo_dir,
            dump_hlo=dump_hlo,
            cleanup_trace=cleanup_trace,
        )
        per_device_results.append(device_result)
        logger.info(
            "GEMM local device %d result: duration_avg_ms=%.4f, tflops=%.4f",
            device_index,
            float(device_result["metrics"].get("duration_avg_ms", 0)),
            float(device_result["metrics"].get("tflops", 0)),
        )

    tflops_values = [
        float(result["metrics"]["tflops"])
        for result in per_device_results
    ]
    aggregate_metrics = average_min_max(
        tflops_values,
        avg_key="avg_tflops",
        min_key="min_tflops",
        max_key="max_tflops",
        count_key="device_count",
    )
    for key in ("avg_tflops", "min_tflops", "max_tflops"):
        aggregate_metrics[key] = round(aggregate_metrics[key], 4)

    metadata = {
        "benchmark": "gemm",
        "m": shape.m,
        "n": shape.n,
        "k": shape.k,
        "input_dtype": input_dtype_display,
        "output_dtype": "jax_matmul_inferred",
        "device_count": len(devices),
        "operation": "jax.numpy.matmul",
        "execution": "sequential_per_local_device",
        "warmup": warmup,
        "iteration": iteration,
        "dump_hlo": dump_hlo,
    }

    return {
        "metadata": metadata,
        "per_device_results": per_device_results,
        "aggregate_metrics": aggregate_metrics,
        "output_directory": out_dir,
    }


def format_cli_summary(results: dict[str, Any]) -> dict[str, Any]:
    """Return a compact JSON-safe summary for terminal output."""
    metadata = results.get("metadata", {})
    per_device_results = results.get("per_device_results", [])
    aggregate_metrics = results.get("aggregate_metrics", {})
    output_directory = results.get("output_directory")
    return {
        "benchmark": "gemm",
        "shape": {
            "m": metadata.get("m"),
            "n": metadata.get("n"),
            "k": metadata.get("k"),
        },
        "dtype": {
            "input": metadata.get("input_dtype"),
            "output": metadata.get("output_dtype"),
        },
        "devices": {
            "count": metadata.get("device_count"),
        },
        "iterations": {
            "warmup": metadata.get("warmup"),
            "timed": metadata.get("iteration"),
        },
        "per_device": [
            {
                "device_index": result["metadata"].get("device_index"),
                "device": result["metadata"].get("device"),
                "coords": result["metadata"].get("coords"),
                "core_on_chip": result["metadata"].get("core_on_chip"),
                "duration_avg_ms": result["metrics"].get("duration_avg_ms"),
                "tflops": result["metrics"].get("tflops"),
            }
            for result in per_device_results
        ],
        "aggregate": aggregate_metrics,
        "output": {
            "run_dir": output_directory,
            "metrics_dir": (
                os.path.join(output_directory, "metrics")
                if output_directory else None
            ),
            "trace_dir": (
                os.path.join(output_directory, "trace")
                if output_directory else None
            ),
            "dump_hlo_enabled": metadata.get("dump_hlo"),
            "dump_hlo_dir": (
                os.path.join(output_directory, "dump_hlo")
                if output_directory and metadata.get("dump_hlo") else None
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="TPU GEMM peak compute benchmark"
    )
    parser.add_argument("--m", type=int, required=True, help="GEMM M dimension")
    parser.add_argument("--n", type=int, required=True, help="GEMM N dimension")
    parser.add_argument("--k", type=int, required=True, help="GEMM K dimension")
    parser.add_argument(
        "--dtype",
        default="bf16",
        help=(
            "Input dtype: fp32, fp16, bf16, or fp8 "
            "(fp8 maps to float8_e4m3fn; default: bf16)"
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup iterations (default: 2)",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=5,
        help="Timed iterations (default: 5)",
    )
    parser.add_argument(
        "--result-dir",
        default="./results",
        help="Root directory for results (default: ./results)",
    )
    parser.add_argument(
        "--dump-hlo",
        action="store_true",
        help="Collect XLA HLO dumps",
    )
    parser.add_argument(
        "--cleanup-trace",
        action="store_true",
        help="Delete xprof trace directory after extracting durations to save disk space",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    configure_logging()
    load_jax_runtime_deps()
    initialize_jax_runtime(logger)

    try:
        results = run_gemm(
            shape=GemmShape(m=args.m, n=args.n, k=args.k),
            dtype_name_input=args.dtype,
            warmup=args.warmup,
            iteration=args.iteration,
            result_dir=args.result_dir,
            dump_hlo=args.dump_hlo,
            cleanup_trace=args.cleanup_trace,
        )
        print(json.dumps(format_cli_summary(results), indent=2, default=str))
    finally:
        _DUMP_HLO.cleanup()


if __name__ == "__main__":
    main()
