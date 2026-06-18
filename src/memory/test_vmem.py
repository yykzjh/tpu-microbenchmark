"""VMEM (Vector Memory) copy bandwidth benchmark using Pallas.

Tests VMEM read-write bandwidth with a copy pattern:
- copy: VMEM read-write bandwidth using independent multi-buffer copies

Kernel optimizations for approaching hardware peak:
- Inner loop for repeated VMEM copy operations
- Four independent VMEM A/B buffer pairs per loop iteration
- Data stays entirely in VMEM during computation (no HBM round-trips)
- Final writeback consumes every copy pair to prevent dead-code elimination
- Large VMEM buffers to maximize bandwidth utilization

Test workflow (aligned with ICI test):
- Phase 1: Warmup iterations
- Phase 2: Timed iterations with xprof traces
- Phase 3: HLO dump collection (if --dump-hlo)
"""

from __future__ import annotations

import os
import sys
import argparse
import functools
import gc
import logging
from datetime import datetime
from typing import Any

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils.runtime import (
    configure_logging,
    configure_dump_hlo_from_argv,
    configure_tpu_benchmark_env,
    device_metadata,
    initialize_jax_runtime,
    prepare_benchmark_dirs,
)
from utils.units import parse_data_size

# Set TPU/XLA environment variables BEFORE importing JAX.
configure_tpu_benchmark_env("vmem")

_DUMP_HLO = configure_dump_hlo_from_argv("vmem")

from memory.options import (
    add_memory_common_args,
    parse_memory_dtype,
    print_memory_results,
    validate_block_shape,
)

logger = logging.getLogger(__name__)


def load_jax_runtime_deps() -> None:
    """Import JAX/Pallas-dependent modules after CLI parsing."""
    global jax, jnp, pl, pltpu, MARKER
    global TraceTimingConfig
    global vmem_copy_kernel
    global run_memory_benchmark_across_devices
    global run_traced_bandwidth_phases

    import jax as _jax
    import jax.numpy as _jnp
    from jax.experimental import pallas as _pl
    import jax.experimental.pallas.tpu as _pltpu
    from utils.profiling import (
        MARKER as _MARKER,
        TraceTimingConfig as _TraceTimingConfig,
    )
    from memory.kernels import (
        vmem_copy_kernel as _vmem_copy_kernel,
    )
    from memory.benchmark import (
        run_memory_benchmark_across_devices as _run_memory_benchmark_across_devices,
        run_traced_bandwidth_phases as _run_traced_bandwidth_phases,
    )

    jax = _jax
    jnp = _jnp
    pl = _pl
    pltpu = _pltpu
    MARKER = _MARKER
    TraceTimingConfig = _TraceTimingConfig
    vmem_copy_kernel = _vmem_copy_kernel
    run_memory_benchmark_across_devices = _run_memory_benchmark_across_devices
    run_traced_bandwidth_phases = _run_traced_bandwidth_phases


# ---------------------------------------------------------------------------
# Benchmark orchestrator
# ---------------------------------------------------------------------------

def run_vmem_per_device(
    device: Any,
    device_index: int,
    data_size: str | int,
    block_shape: tuple[int, int],
    dtype: jnp.dtype,
    warmup: int,
    iteration: int,
    result_dir: str,
    dump_hlo: bool = False,
    cleanup_trace: bool = False,
) -> dict[str, Any]:
    """
    Run VMEM bandwidth benchmark on a single device.

    Args:
        device: The specific TPU chiplet device to test
        device_index: Index of this device in the device list
        data_size: Effective data size in bytes, with optional binary unit suffix
        block_shape: Block shape for Pallas kernel
        dtype: Data type
        warmup: Number of warmup iterations
        iteration: Number of timed iterations
        result_dir: Base directory for results
        dump_hlo: Enable HLO dump collection
        cleanup_trace: Cleanup trace directory after extracting durations

    Returns:
        Dict with metadata, metrics, and output_directory
    """
    validate_block_shape(block_shape)

    elem_size = jnp.dtype(dtype).itemsize
    block_elems = block_shape[0] * block_shape[1]
    block_bytes = block_elems * elem_size
    copy_pairs = 4

    bytes_per_inner_iter = block_bytes * copy_pairs * 2
    # --data-size is effective VMEM traffic. Each inner iteration performs four
    # independent VMEM copies, and each copy counts read + write bytes.
    parsed_data_size = parse_data_size(
        data_size,
        alignment_bytes=bytes_per_inner_iter,
        alignment_description=(
            f"one VMEM copy inner iteration: "
            f"4 copy pairs * read+write * {block_bytes} bytes = "
            f"{bytes_per_inner_iter} bytes"
        ),
    )
    actual_data_bytes = parsed_data_size.bytes
    n_inner_iters = actual_data_bytes // bytes_per_inner_iter
    data_size_label = parsed_data_size.label

    logger.info(
        f"VMEM copy benchmark on device {device_index}: {actual_data_bytes / 1e6:.2f} MB, "
        f"block_shape={block_shape}, {n_inner_iters} inner iters"
    )

    # Prepare per-device output directories
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dtype_name = jnp.dtype(dtype).name
    run_name = (
        f"{ts}_vmem_copy_device{device_index}_bs{block_shape[0]}x{block_shape[1]}_"
        f"{dtype_name}_{data_size_label}"
    )
    dirs = prepare_benchmark_dirs(result_dir, run_name, dump_hlo=dump_hlo)
    out_dir = dirs.output_dir
    metrics_dir = dirs.metrics_dir
    trace_dir = dirs.trace_dir
    dump_hlo_dir = dirs.dump_hlo_dir

    # Compile kernel
    grid = (1,)  # Single grid iteration (inner loop does the work)

    compiled_fn = pl.pallas_call(
        functools.partial(vmem_copy_kernel, n_iters=n_inner_iters),
        in_specs=[],
        out_shape=jax.ShapeDtypeStruct(block_shape, dtype),
        out_specs=pl.BlockSpec(
            block_shape, lambda i: (0, 0),
            memory_space=pl.ANY,
        ),
        scratch_shapes=(
            pltpu.MemorySpace.VMEM(block_shape, dtype),
            pltpu.MemorySpace.VMEM(block_shape, dtype),
            pltpu.MemorySpace.VMEM(block_shape, dtype),
            pltpu.MemorySpace.VMEM(block_shape, dtype),
            pltpu.MemorySpace.VMEM(block_shape, dtype),
            pltpu.MemorySpace.VMEM(block_shape, dtype),
            pltpu.MemorySpace.VMEM(block_shape, dtype),
            pltpu.MemorySpace.VMEM(block_shape, dtype),
            pltpu.SemaphoreType.DMA,
        ),
        grid=grid,
    )

    def data_generator():
        return ()

    data_bytes = block_bytes * n_inner_iters * copy_pairs * 2

    raw_compiled_fn = compiled_fn

    def marked_fn(*args):
        with jax.named_scope(MARKER):
            return raw_compiled_fn(*args)

    compiled_fn = jax.jit(marked_fn, device=device)

    test_name = f"vmem_copy_device{device_index}_bs{block_shape[0]}x{block_shape[1]}_{dtype_name}"
    metadata = {
        "mode": "copy",
        **device_metadata(device, device_index),
        "block_shape": list(block_shape),
        "dtype": dtype_name,
        **parsed_data_size.metadata(),
        "data_size_mib": actual_data_bytes / (1024 * 1024),
        "actual_data_bytes": actual_data_bytes,
        "n_inner_iters": n_inner_iters,
        "copy_pairs": copy_pairs,
        "bytes_per_inner_iter": bytes_per_inner_iter,
        "copy_pattern": "eight_phase_permuted_four_copy_ring",
        "warmup": warmup,
        "iteration": iteration,
        "dump_hlo": dump_hlo,
        "cleanup_trace": cleanup_trace,
        "timing_note": (
            "VMEM Pallas custom-call device_duration_ps is not used because "
            "it can report only outer custom-call bookkeeping time."
        ),
    }
    # The xprof XLA Ops device_duration_ps for this Pallas VMEM custom-call can
    # be only a few microseconds even when the annotated call blocks much longer.
    # Use the StepTrace duration as a conservative denominator until VMEM has a
    # lower-level kernel event that represents the Mosaic body itself.
    timing_config = TraceTimingConfig(
        task_name=f"timed_{test_name}",
        trace_dir=trace_dir,
        dest_name=f"trace_{test_name}",
        cleanup_trace=cleanup_trace,
        duration_source="trace",
        require_marker=False,
        event_name_contains=f"timed_{test_name}",
        per_iteration_reducer="max",
    )
    metrics = run_traced_bandwidth_phases(
        compiled_fn=compiled_fn,
        data_generator=data_generator,
        data_bytes=data_bytes,
        test_name=test_name,
        metadata=metadata,
        warmup=warmup,
        iteration=iteration,
        metrics_dir=metrics_dir,
        trace_dir=trace_dir,
        dump_hlo=dump_hlo,
        dump_hlo_source_dir=_DUMP_HLO.temp_dir,
        dump_hlo_dir=dump_hlo_dir,
        clear_hlo_source=_DUMP_HLO.owned,
        cleanup_trace=cleanup_trace,
        logger=logger,
        timing_config=timing_config,
    )
    gc.collect()
    return {
        "metadata": metadata,
        "metrics": metrics,
        "output_directory": out_dir,
    }


def run_vmem(
    data_size: str | int,
    block_shape: tuple[int, int],
    dtype: jnp.dtype,
    warmup: int,
    iteration: int,
    result_dir: str,
    dump_hlo: bool = False,
    cleanup_trace: bool = False,
) -> dict[str, Any]:
    """
    Run VMEM bandwidth benchmark on all local devices.

    Phases:
      1. Warmup iterations
      2. Timed iterations with xprof traces
      3. HLO dump collection (if dump_hlo=True)

    Args:
        data_size: Effective data size in bytes, with optional binary unit suffix
        block_shape: Block shape for Pallas kernel
        dtype: Data type
        warmup: Number of warmup iterations
        iteration: Number of timed iterations
        result_dir: Base directory for results
        dump_hlo: Enable HLO dump collection
        cleanup_trace: Cleanup trace directory after extracting durations

    Returns:
        Dict with metadata, per_device_results, aggregate_metrics, and output_directory
    """
    elem_size = jnp.dtype(dtype).itemsize
    block_bytes = block_shape[0] * block_shape[1] * elem_size
    bytes_per_inner_iter = block_bytes * 4 * 2
    dtype_name = jnp.dtype(dtype).name
    parsed_data_size = parse_data_size(
        data_size,
        alignment_bytes=bytes_per_inner_iter,
        alignment_description=(
            f"one VMEM copy inner iteration: {bytes_per_inner_iter} bytes"
        ),
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run_name_builder(num_devices: int) -> str:
        return (
            f"{ts}_vmem_copy_bs{block_shape[0]}x{block_shape[1]}_"
            f"{dtype_name}_{parsed_data_size.label}_{num_devices}devices"
        )

    def metadata_builder(num_devices: int) -> dict[str, Any]:
        return {
            "mode": "copy",
            "device_count": num_devices,
            "block_shape": list(block_shape),
            "dtype": dtype_name,
            "copy_pairs": 4,
            "copy_pattern": "eight_phase_permuted_four_copy_ring",
            **parsed_data_size.metadata(),
            "warmup": warmup,
            "iteration": iteration,
            "dump_hlo": dump_hlo,
            "cleanup_trace": cleanup_trace,
        }

    def per_device_runner(device_index: int, device: Any, out_dir: str) -> dict[str, Any]:
        return run_vmem_per_device(
            device=device,
            device_index=device_index,
            data_size=data_size,
            block_shape=block_shape,
            dtype=dtype,
            warmup=warmup,
            iteration=iteration,
            result_dir=out_dir,
            dump_hlo=dump_hlo,
            cleanup_trace=cleanup_trace,
        )

    return run_memory_benchmark_across_devices(
        benchmark="vmem",
        mode="copy",
        result_dir=result_dir,
        run_name_builder=run_name_builder,
        metadata_builder=metadata_builder,
        per_device_runner=per_device_runner,
        warmup=warmup,
        iteration=iteration,
        logger=logger,
    )


def main():
    parser = argparse.ArgumentParser(description='VMEM bandwidth benchmark')
    add_memory_common_args(
        parser,
        default_data_size='64MiB',
        data_size_description='Effective data size in bytes',
        include_block_shape=True,
        include_dtype=True,
        include_mode=False,
    )

    args = parser.parse_args()

    configure_logging()
    load_jax_runtime_deps()
    initialize_jax_runtime(logger)

    dtype = parse_memory_dtype(args.dtype)
    block_shape = tuple(args.block_shape)

    logger.info("Running VMEM copy benchmark on all devices...")
    results = run_vmem(
        data_size=args.data_size,
        block_shape=block_shape,
        dtype=dtype,
        warmup=args.warmup,
        iteration=args.iteration,
        result_dir=args.result_dir,
        dump_hlo=args.dump_hlo,
        cleanup_trace=args.cleanup_trace,
    )
    print_memory_results("VMEM", results, show_block_shape=True)


if __name__ == '__main__':
    main()
