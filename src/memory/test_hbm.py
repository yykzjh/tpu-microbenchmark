"""HBM bandwidth benchmark using Pallas.

Tests HBM (High Bandwidth Memory) bandwidth with three patterns:
- read: HBM read bandwidth (HBM → VMEM → touch)
- write: HBM write bandwidth (VMEM fill → HBM)
- copy: HBM read-write bandwidth (HBM → VMEM → HBM)

Kernel optimizations for approaching hardware peak:
- Four-bank VMEM scratch with DMA semaphores
- make_async_copy for non-blocking DMA transfers
- Bank-rotated VMEM scratch buffers to overlap adjacent HBM DMA requests
- A single Pallas program iterates over the HBM tensor's first dimension so
  HBM BlockSpec lowering stays trivial while touching different HBM chunks
- Data dependency chains to prevent dead-code elimination
- Large per-kernel data sizes to saturate DMA engines

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
configure_tpu_benchmark_env("hbm")

_DUMP_HLO = configure_dump_hlo_from_argv("hbm")

from memory.options import (
    add_memory_common_args,
    expand_memory_modes,
    parse_memory_dtype,
    print_memory_mode_summary,
    print_memory_results,
    validate_memory_mode,
    validate_block_shape,
)

logger = logging.getLogger(__name__)


def load_jax_runtime_deps() -> None:
    """Import JAX/Pallas-dependent modules after CLI parsing."""
    global jax, jnp, pl, pltpu, MARKER, delete_device_object
    global hbm_copy_kernel, hbm_read_kernel, hbm_write_kernel
    global create_test_array, run_memory_benchmark_across_devices
    global run_traced_bandwidth_phases

    import jax as _jax
    import jax.numpy as _jnp
    from jax.experimental import pallas as _pl
    import jax.experimental.pallas.tpu as _pltpu
    from utils.profiling import (
        MARKER as _MARKER,
        delete_device_object as _delete_device_object,
    )
    from memory.kernels import (
        hbm_copy_kernel as _hbm_copy_kernel,
        hbm_read_kernel as _hbm_read_kernel,
        hbm_write_kernel as _hbm_write_kernel,
    )
    from memory.arrays import create_test_array as _create_test_array
    from memory.benchmark import (
        run_memory_benchmark_across_devices as _run_memory_benchmark_across_devices,
        run_traced_bandwidth_phases as _run_traced_bandwidth_phases,
    )

    jax = _jax
    jnp = _jnp
    pl = _pl
    pltpu = _pltpu
    MARKER = _MARKER
    delete_device_object = _delete_device_object
    hbm_copy_kernel = _hbm_copy_kernel
    hbm_read_kernel = _hbm_read_kernel
    hbm_write_kernel = _hbm_write_kernel
    create_test_array = _create_test_array
    run_memory_benchmark_across_devices = _run_memory_benchmark_across_devices
    run_traced_bandwidth_phases = _run_traced_bandwidth_phases


# ---------------------------------------------------------------------------
# Benchmark orchestrator
# ---------------------------------------------------------------------------

def run_hbm_per_device(
    mode: str,
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
    Run HBM bandwidth benchmark on a single device.

    Args:
        mode: Test mode ('read', 'write', 'copy')
        device: The specific TPU device to test
        device_index: Index of this device in the device list
        data_size: Total data size in bytes, with optional binary unit suffix
        block_shape: Single VMEM buffer shape as (n, k)
        dtype: Data type
        warmup: Number of warmup iterations
        iteration: Number of timed iterations
        result_dir: Base directory for results
        dump_hlo: Enable HLO dump collection
        cleanup_trace: Cleanup trace directory after extracting durations

    Returns:
        Dict with metadata, metrics, and output_directory
    """
    validate_memory_mode(mode)
    validate_block_shape(block_shape)

    vmem_buffer_shape = tuple(block_shape)
    pipeline_banks = 4
    vmem_buffer_count = pipeline_banks

    elem_size = jnp.dtype(dtype).itemsize
    vmem_buffer_elems = vmem_buffer_shape[0] * vmem_buffer_shape[1]
    vmem_buffer_bytes = vmem_buffer_elems * elem_size
    # --block-shape is one VMEM buffer shape. HBM is represented as
    # (m, n, k), where each HBM chunk has exactly one VMEM buffer worth of
    # elements. HBM read/write rotate across 4 VMEM banks. Copy uses one
    # program with 2 read banks and 2 write banks.

    parsed_data_size = parse_data_size(
        data_size,
        alignment_bytes=vmem_buffer_bytes,
        alignment_description=(
            f"one VMEM buffer: {vmem_buffer_shape[0]} * "
            f"{vmem_buffer_shape[1]} * {elem_size} = "
            f"{vmem_buffer_bytes} bytes"
        ),
    )
    data_size_bytes = parsed_data_size.bytes
    hbm_chunks = data_size_bytes // vmem_buffer_bytes
    hbm_iterations = hbm_chunks
    actual_data_bytes = hbm_chunks * vmem_buffer_bytes
    data_size_label = parsed_data_size.label
    hbm_tensor_shape = (hbm_chunks, vmem_buffer_shape[0], vmem_buffer_shape[1])
    read_placeholder_shape = (1, 1, 128)

    logger.info(
        f"HBM {mode} benchmark: {actual_data_bytes / 1e6:.2f} MB, "
        f"{hbm_chunks} HBM chunks with tensor shape {hbm_tensor_shape}; "
        f"{pipeline_banks} rotating VMEM banks"
    )

    # Prepare per-device output directories
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dtype_name = jnp.dtype(dtype).name
    run_name = (
        f"{ts}_hbm_{mode}_device{device_index}_bs{vmem_buffer_shape[0]}x{vmem_buffer_shape[1]}_"
        f"{dtype_name}_{data_size_label}"
    )
    dirs = prepare_benchmark_dirs(result_dir, run_name, dump_hlo=dump_hlo)
    out_dir = dirs.output_dir
    metrics_dir = dirs.metrics_dir
    trace_dir = dirs.trace_dir
    dump_hlo_dir = dirs.dump_hlo_dir

    # Use one Pallas program per launch. The kernels loop over HBM tensor dim-0
    # and select VMEM-sized HBM chunks with x_hbm_ref.at[chunk_id, ...].
    grid = (1,)
    x = None

    if mode == 'read':
        x = create_test_array(
            hbm_tensor_shape,
            dtype,
            device=device,
        )

        compiled_fn = pl.pallas_call(
            functools.partial(
                hbm_read_kernel,
                n_iters=hbm_iterations,
            ),
            in_specs=[
                pl.BlockSpec(
                    hbm_tensor_shape, lambda i: (0, 0, 0),
                    memory_space=pltpu.HBM,
                ),
            ],
            out_shape=jax.ShapeDtypeStruct(read_placeholder_shape, dtype),
            out_specs=pl.BlockSpec(
                read_placeholder_shape, lambda i: (0, 0, 0),
                memory_space=pltpu.HBM,
            ),
            scratch_shapes=(
                pltpu.MemorySpace.VMEM(vmem_buffer_shape, dtype),
                pltpu.MemorySpace.VMEM(vmem_buffer_shape, dtype),
                pltpu.MemorySpace.VMEM(vmem_buffer_shape, dtype),
                pltpu.MemorySpace.VMEM(vmem_buffer_shape, dtype),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
            ),
            grid=grid,
        )

        def data_generator():
            return (x,)

        data_bytes = actual_data_bytes

    elif mode == 'write':
        # Write mode has no input operand; it materializes VMEM data and stores
        # it to HBM output.
        compiled_fn = pl.pallas_call(
            functools.partial(
                hbm_write_kernel,
                n_iters=hbm_iterations,
            ),
            in_specs=[],
            out_shape=jax.ShapeDtypeStruct(hbm_tensor_shape, dtype),
            out_specs=pl.BlockSpec(
                hbm_tensor_shape, lambda i: (0, 0, 0),
                memory_space=pltpu.HBM,
            ),
            scratch_shapes=(
                pltpu.MemorySpace.VMEM(vmem_buffer_shape, dtype),
                pltpu.MemorySpace.VMEM(vmem_buffer_shape, dtype),
                pltpu.MemorySpace.VMEM(vmem_buffer_shape, dtype),
                pltpu.MemorySpace.VMEM(vmem_buffer_shape, dtype),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
            ),
            grid=grid,
        )

        def data_generator():
            return ()

        data_bytes = actual_data_bytes

    elif mode == 'copy':
        x = create_test_array(
            hbm_tensor_shape,
            dtype,
            device=device,
        )

        # Copy mode moves bytes in both directions, so the bandwidth numerator
        # counts read + write traffic.
        compiled_fn = pl.pallas_call(
            functools.partial(
                hbm_copy_kernel,
                n_iters=hbm_iterations,
            ),
            in_specs=[
                pl.BlockSpec(
                    hbm_tensor_shape, lambda i: (0, 0, 0),
                    memory_space=pltpu.HBM,
                ),
            ],
            out_shape=jax.ShapeDtypeStruct(hbm_tensor_shape, dtype),
            out_specs=pl.BlockSpec(
                hbm_tensor_shape, lambda i: (0, 0, 0),
                memory_space=pltpu.HBM,
            ),
            scratch_shapes=(
                pltpu.MemorySpace.VMEM(vmem_buffer_shape, dtype),
                pltpu.MemorySpace.VMEM(vmem_buffer_shape, dtype),
                pltpu.MemorySpace.VMEM(vmem_buffer_shape, dtype),
                pltpu.MemorySpace.VMEM(vmem_buffer_shape, dtype),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
            ),
            grid=grid,
        )

        def data_generator():
            return (x,)

        data_bytes = actual_data_bytes * 2  # read + write

    else:
        raise ValueError(f"Unknown mode: {mode}")

    raw_compiled_fn = compiled_fn

    def marked_fn(*args):
        with jax.named_scope(MARKER):
            return raw_compiled_fn(*args)

    compiled_fn = jax.jit(marked_fn, device=device)

    test_name = f"hbm_{mode}_device{device_index}_bs{vmem_buffer_shape[0]}x{vmem_buffer_shape[1]}_{dtype_name}"
    metadata = {
        "mode": mode,
        **device_metadata(device, device_index),
        "block_shape": list(vmem_buffer_shape),
        "block_shape_semantics": "single_vmem_buffer",
        "vmem_buffer_shape": list(vmem_buffer_shape),
        "vmem_buffer_bytes": vmem_buffer_bytes,
        "vmem_bank_bytes": vmem_buffer_bytes,
        "hbm_tensor_shape": list(hbm_tensor_shape),
        "hbm_chunks": hbm_chunks,
        "hbm_window_iterations": hbm_iterations,
        "read_placeholder_shape": list(read_placeholder_shape) if mode == "read" else None,
        "hbm_bank_shape": list(vmem_buffer_shape),
        "hbm_grid_programs": 1,
        "hbm_pipeline_banks": pipeline_banks,
        "vmem_buffer_count": vmem_buffer_count,
        "copy_grid_layout": (
            "single_program_interleaved_read_write" if mode == "copy" else None
        ),
        "copy_semaphore_layout": (
            "two_read_two_write" if mode == "copy" else "single_direction"
        ),
        "copy_vmem_layout": (
            "two_read_banks_two_write_banks" if mode == "copy" else "single_direction"
        ),
        "copy_semantics": (
            "interleaved_two_bank_read_write_streams" if mode == "copy" else None
        ),
        "num_blocks": hbm_chunks,
        "dtype": dtype_name,
        **parsed_data_size.metadata(),
        "data_size_mib": data_size_bytes / (1024 * 1024),
        "actual_data_bytes": actual_data_bytes,
        "warmup": warmup,
        "iteration": iteration,
        "dump_hlo": dump_hlo,
        "cleanup_trace": cleanup_trace,
    }
    try:
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
        )
        return {
            "metadata": metadata,
            "metrics": metrics,
            "output_directory": out_dir,
        }
    finally:
        delete_device_object(x)
        gc.collect()


def run_hbm(
    mode: str,
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
    Run HBM bandwidth benchmark across all local TPU devices.

    Tests at TPU chiplet device granularity: each chiplet device runs the
    kernel independently.

    Phases per device:
      1. Warmup iterations
      2. Timed iterations with xprof traces
      3. HLO dump collection (if dump_hlo=True)

    Args:
        mode: Test mode ('read', 'write', 'copy')
        data_size: Total data size in bytes, with optional binary unit suffix
        block_shape: Single VMEM buffer shape as (n, k)
        dtype: Data type
        warmup: Number of warmup iterations
        iteration: Number of timed iterations
        result_dir: Base directory for results
        dump_hlo: Enable HLO dump collection
        cleanup_trace: Cleanup trace directory after extracting durations

    Returns:
        Dict with metadata, per_device_results, aggregate_metrics, and output_directory
    """
    validate_memory_mode(mode)
    vmem_buffer_shape = tuple(block_shape)
    pipeline_banks = 4
    vmem_buffer_count = pipeline_banks
    elem_size = jnp.dtype(dtype).itemsize
    vmem_buffer_bytes = vmem_buffer_shape[0] * vmem_buffer_shape[1] * elem_size
    dtype_name = jnp.dtype(dtype).name
    parsed_data_size = parse_data_size(
        data_size,
        alignment_bytes=vmem_buffer_bytes,
        alignment_description=(
            f"one VMEM buffer: {vmem_buffer_shape[0]} * "
            f"{vmem_buffer_shape[1]} * {elem_size} = "
            f"{vmem_buffer_bytes} bytes"
        ),
    )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run_name_builder(num_devices: int) -> str:
        return (
            f"{ts}_hbm_{mode}_bs{vmem_buffer_shape[0]}x{vmem_buffer_shape[1]}_"
            f"{dtype_name}_{parsed_data_size.label}_{num_devices}devices"
        )

    def metadata_builder(num_devices: int) -> dict[str, Any]:
        return {
            "mode": mode,
            "device_count": num_devices,
            "block_shape": list(vmem_buffer_shape),
            "block_shape_semantics": "single_vmem_buffer",
            "vmem_buffer_shape": list(vmem_buffer_shape),
            "vmem_buffer_bytes": vmem_buffer_bytes,
            "vmem_bank_bytes": vmem_buffer_bytes,
            "hbm_bank_shape": list(vmem_buffer_shape),
            "hbm_grid_programs": 1,
            "hbm_pipeline_banks": pipeline_banks,
            "vmem_buffer_count": vmem_buffer_count,
            "copy_grid_layout": (
                "single_program_interleaved_read_write" if mode == "copy" else None
            ),
            "copy_semaphore_layout": (
                "two_read_two_write" if mode == "copy" else "single_direction"
            ),
            "copy_vmem_layout": (
                "two_read_banks_two_write_banks" if mode == "copy" else "single_direction"
            ),
            "dtype": dtype_name,
            **parsed_data_size.metadata(),
            "warmup": warmup,
            "iteration": iteration,
            "dump_hlo": dump_hlo,
            "cleanup_trace": cleanup_trace,
        }

    def per_device_runner(device_index: int, device: Any, out_dir: str) -> dict[str, Any]:
        return run_hbm_per_device(
            mode=mode,
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
        benchmark="hbm",
        mode=mode,
        result_dir=result_dir,
        run_name_builder=run_name_builder,
        metadata_builder=metadata_builder,
        per_device_runner=per_device_runner,
        warmup=warmup,
        iteration=iteration,
        logger=logger,
    )


def main():
    parser = argparse.ArgumentParser(description='HBM bandwidth benchmark')
    add_memory_common_args(
        parser,
        default_data_size='256MiB',
        data_size_description='Data size in bytes',
        include_block_shape=True,
        include_dtype=True,
        block_shape_help=(
            "Single VMEM buffer shape n k; HBM tensor is (m, n, k), "
            "where m is derived from --data-size (default: 128 1024)"
        ),
    )
    args = parser.parse_args()

    configure_logging()
    load_jax_runtime_deps()
    initialize_jax_runtime(logger)

    dtype = parse_memory_dtype(args.dtype)
    block_shape = tuple(args.block_shape)

    all_results = []
    for mode in expand_memory_modes(args.mode):
        logger.info(f"Running HBM {mode} benchmark...")
        results = run_hbm(
            mode=mode,
            data_size=args.data_size,
            block_shape=block_shape,
            dtype=dtype,
            warmup=args.warmup,
            iteration=args.iteration,
            result_dir=args.result_dir,
            dump_hlo=args.dump_hlo,
            cleanup_trace=args.cleanup_trace,
        )
        all_results.append(results)

        print_memory_results("HBM", results, show_block_shape=True)

    if args.mode == 'all':
        print_memory_mode_summary(all_results)


if __name__ == '__main__':
    main()
