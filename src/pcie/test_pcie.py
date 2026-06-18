"""TPU PCIe transfer bandwidth benchmark.

Measures host-to-device (H2D) and device-to-host (D2H) transfer bandwidth
over the PCIe link between CPU and local TPU chiplet devices:

  - h2d: host-to-device transfer with ``jax.device_put``
  - d2h: device-to-host transfer with ``jax.device_get``

Each mode runs two test items:

  - one_to_one: CPU transfers with each selected local TPU device one by one.
  - one_to_many: CPU transfers with all selected local TPU devices concurrently.

Use ``python test_pcie.py --help`` for CLI usage.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Iterator

import numpy as np

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils.runtime import (
    configure_logging,
    configure_tpu_benchmark_env,
    get_local_devices_or_raise,
    initialize_jax_runtime,
    log_device_separator,
    prepare_benchmark_dirs,
    validate_non_negative,
    validate_positive,
)
from utils.units import data_size_help, parse_data_size

# ---------------------------------------------------------------------------
# Environment flags MUST be set before JAX is imported.
# These mirror the reference benchmark's high-frequency TPU and premapped
# transfer-buffer settings while still allowing callers to override.
# ---------------------------------------------------------------------------
configure_tpu_benchmark_env("pcie")

from utils.metrics import (
    MetricsStatistics,
    average_min_max,
    compute_bandwidth_GBps,
    write_jsonl_metrics,
)

logger = logging.getLogger(__name__)

FLOAT32_BYTES = 4
TRANSFER_ROW_ELEMENTS = 128
TRANSFER_ROW_BYTES = TRANSFER_ROW_ELEMENTS * FLOAT32_BYTES


def load_jax_runtime_deps() -> None:
    """Import JAX-dependent modules after CLI parsing."""
    global jax, delete_device_object

    import jax as _jax
    from utils.profiling import (
        delete_device_object as _delete_device_object,
    )

    jax = _jax
    delete_device_object = _delete_device_object


@dataclass(frozen=True)
class TargetDevice:
    """One selected local TPU chiplet device."""

    target_index: int
    local_device_index: int
    global_device_id: int | None
    process_index: int | None
    coords: tuple[int, ...] | None
    core_on_chip: int | None
    device: Any


def make_host_buffer(size_bytes: int) -> np.ndarray:
    """Allocate one contiguous float32 host transfer buffer."""
    num_elements = size_bytes // FLOAT32_BYTES
    # Shape as [rows, 128] so every transfer is naturally aligned to the same
    # 512-byte row size used by the CLI alignment check.
    return np.ones(
        (num_elements // TRANSFER_ROW_ELEMENTS, TRANSFER_ROW_ELEMENTS),
        dtype=np.float32,
    )


def device_coords(device: Any) -> tuple[int, ...] | None:
    """Return device coordinates when exposed by JAX."""
    coords = getattr(device, "coords", None)
    if coords is None:
        return None
    return tuple(int(value) for value in coords)


def device_id(device: Any) -> int | None:
    """Return a stable numeric device id when exposed by JAX."""
    value = getattr(device, "id", None)
    if value is None:
        return None
    return int(value)


def device_process_index(device: Any) -> int | None:
    """Return process index when exposed by JAX."""
    value = getattr(device, "process_index", None)
    if value is None:
        return None
    return int(value)


def device_core_on_chip(device: Any) -> int | None:
    """Return core_on_chip when exposed by JAX."""
    value = getattr(device, "core_on_chip", None)
    if value is None:
        return None
    return int(value)


def select_local_tpu_targets(target_devices: int) -> list[TargetDevice]:
    """Select JAX devices for per-chiplet-device testing.

    Selects up to ``target_devices`` individual TPU chiplet devices.
    Devices are grouped by TPU chip (same coordinates), then within each chip
    sorted by ``core_on_chip`` so all chiplet devices on the same chip are tested
    before moving to the next chip.

    Args:
        target_devices: Maximum number of TPU chiplet devices to test.

    Returns:
        List of TargetDevice, one per selected chiplet device.
    """
    validate_positive("target_devices", target_devices)

    local_devices = get_local_devices_or_raise()

    # Group devices by TPU chip (same physical coordinates)
    groups: dict[Any, list[tuple[int, Any]]] = {}
    for local_index, device in enumerate(local_devices):
        coords = device_coords(device)
        # On TPU, the two JAX devices with the same coords are the two chiplets
        # of one physical TPU chip. If coords are unavailable, fall back to the
        # common adjacent-pair ordering used by jax.devices().
        key = coords if coords is not None else local_index // 2
        groups.setdefault(key, []).append((local_index, device))

    # Within each chip, sort by core_on_chip so all chiplet devices on the same
    # chip are adjacent in the output
    selected: list[tuple[int, Any]] = []
    for group in groups.values():
        decorated = [
            (device_core_on_chip(item[1]) or 0, item[0], item)
            for item in group
        ]
        decorated.sort(key=lambda x: (x[0], x[1]))
        # Select ALL chiplet devices on this chip
        for _, _, item in decorated:
            selected.append(item)

    if len(selected) < target_devices:
        raise ValueError(
            f"Requested {target_devices} TPU chiplet devices, but only found "
            f"{len(selected)} local JAX devices "
            f"({len(groups)} TPU chips)"
        )

    targets = []
    for target_index, (local_index, device) in enumerate(selected[:target_devices]):
        targets.append(
            TargetDevice(
                target_index=target_index,
                local_device_index=local_index,
                global_device_id=device_id(device),
                process_index=device_process_index(device),
                coords=device_coords(device),
                core_on_chip=device_core_on_chip(device),
                device=device,
            )
        )
    return targets


def format_target(target: TargetDevice) -> dict[str, Any]:
    """Return JSON-friendly target metadata."""
    return {
        "target_index": target.target_index,
        "local_device_index": target.local_device_index,
        "global_device_id": target.global_device_id,
        "process_index": target.process_index,
        "coords": list(target.coords) if target.coords is not None else None,
        "core_on_chip": target.core_on_chip,
        "device": str(target.device),
    }


def h2d_transfer(host_buffer: np.ndarray, target: TargetDevice):
    """Run one host-to-device transfer."""
    return jax.device_put(host_buffer, target.device)


def d2h_transfer(device_array: Any):
    """Run one device-to-host transfer."""
    return jax.device_get(device_array)


def prepare_h2d_host_batch(host_buffer: np.ndarray, count: int) -> list[np.ndarray]:
    """Create host-side H2D buffers for one measured case.

    H2D uses the same timing path as D2H: every warmup/timed run consumes one
    prepared input, and synchronization happens in the shared timing wrapper.
    The copies are prepared outside the measured region.
    """
    return [host_buffer.copy() for _ in range(count)]


def next_h2d_host_buffer(buffer_iter: Iterator[np.ndarray]) -> np.ndarray:
    """Return the next prepared H2D host buffer."""
    try:
        return next(buffer_iter)
    except StopIteration:
        raise RuntimeError(
            "Ran out of prepared H2D host buffers. Increase the host buffer "
            "batch size to cover warmup + timed iterations."
        ) from None


def prepare_d2h_source_batch(
    host_buffer: np.ndarray,
    target: TargetDevice,
    count: int,
) -> list[Any]:
    """Create device-resident D2H buffers for one measured case.

    This follows AI-Hypercomputer/accelerator-microbenchmarks' PCIe D2H
    pattern: each warmup/timed run consumes a fresh device array. Reusing one
    device array across many device_get calls can measure runtime/cache reuse
    rather than a clean D2H payload transfer, while copying the returned NumPy
    buffer would include extra CPU memory bandwidth.
    """
    sources = []
    try:
        for _ in range(count):
            device_array = jax.device_put(host_buffer, target.device)
            device_array.block_until_ready()
            sources.append(device_array)
        return sources
    except Exception:
        for device_array in sources:
            delete_device_object(device_array)
        raise


def next_d2h_source(source_iter: Iterator[Any]) -> Any:
    """Return the next preallocated D2H source buffer."""
    try:
        return next(source_iter)
    except StopIteration:
        raise RuntimeError(
            "Ran out of preallocated D2H source buffers. Increase the source "
            "batch size to cover warmup + timed iterations."
        ) from None


def summarize_transfer_durations(
    durations: list[float],
    bytes_per_measurement: int,
) -> dict[str, Any]:
    """Return duration and PCIe transfer statistics for one item."""
    duration_stats = MetricsStatistics(durations, "duration", unit="ms")
    metrics = duration_stats.serialize()
    avg_ms = float(metrics.get("duration_avg_ms", 0))
    metrics.update({
        "bytes_per_measurement": bytes_per_measurement,
        "bandwidth_GBps": round(compute_bandwidth_GBps(bytes_per_measurement, avg_ms), 4),
        "timing_source": "wall_clock_with_block_until_ready",
    })
    return metrics


def block_transfer_result(result: Any) -> None:
    """Synchronize transfer outputs using JAX's pytree-aware blocker."""
    jax.block_until_ready(result)


def run_transfer_warmup(operation: Callable[[], Any], warmup: int) -> None:
    """Run unmeasured warmup iterations for one transfer operation."""
    for _ in range(warmup):
        result = None
        try:
            result = operation()
            block_transfer_result(result)
        finally:
            delete_device_object(result)
            result = None


def run_transfer_iterations(
    operation: Callable[[], Any],
    iteration: int,
    trace_dir: str,
    trace_name: str,
    cleanup_trace: bool = False,
) -> list[float]:
    """Run one transfer case and return wall-clock durations in milliseconds.

    Host-device transfer APIs are runtime calls rather than compiled TPU
    kernels. Following AI-Hypercomputer/accelerator-microbenchmarks, the primary
    timing is Python wall-clock around the transfer plus
    ``jax.block_until_ready``. Xprof trace is retained only for diagnosis.
    """
    task_name = f"timed_{trace_name}"
    case_trace_dir = os.path.join(trace_dir, trace_name)
    os.makedirs(case_trace_dir, exist_ok=True)
    durations = []
    with jax.profiler.trace(case_trace_dir, create_perfetto_link=False):
        for i in range(iteration):
            result = None
            try:
                start = time.perf_counter()
                with jax.profiler.TraceAnnotation(task_name):
                    with jax.profiler.StepTraceAnnotation(task_name, step_num=i):
                        result = operation()
                        block_transfer_result(result)
                end = time.perf_counter()
                durations.append((end - start) * 1000.0)
            finally:
                delete_device_object(result)
                result = None

    if cleanup_trace:
        import shutil

        shutil.rmtree(case_trace_dir, ignore_errors=True)
    return durations


def run_h2d_one_to_one(
    host_buffer: np.ndarray,
    targets: list[TargetDevice],
    warmup: int,
    iteration: int,
    data_size_bytes: int,
    trace_dir: str,
    cleanup_trace: bool,
) -> list[dict[str, Any]]:
    """Run sequential CPU-to-one-TPU H2D tests."""
    results = []
    for target in targets:
        log_device_separator(
            logger,
            "PCIe H2D one-to-one target",
            target.target_index,
            target.device,
        )
        host_batch = prepare_h2d_host_batch(host_buffer, warmup + iteration)
        try:
            host_iter = iter(host_batch)
            operation = lambda target=target, host_iter=host_iter: h2d_transfer(
                next_h2d_host_buffer(host_iter),
                target,
            )
            run_transfer_warmup(operation, warmup)
            durations = run_transfer_iterations(
                operation,
                iteration,
                trace_dir,
                trace_name=f"trace_h2d_one_to_one_target{target.target_index}",
                cleanup_trace=cleanup_trace,
            )
        finally:
            del host_batch
        results.append({
            "test_item": "one_to_one",
            "target": format_target(target),
            "metrics": summarize_transfer_durations(durations, data_size_bytes),
        })
    return results


def run_d2h_one_to_one(
    host_buffer: np.ndarray,
    targets: list[TargetDevice],
    warmup: int,
    iteration: int,
    data_size_bytes: int,
    trace_dir: str,
    cleanup_trace: bool,
) -> list[dict[str, Any]]:
    """Run sequential one-TPU-to-CPU D2H tests."""
    results = []
    for target in targets:
        log_device_separator(
            logger,
            "PCIe D2H one-to-one target",
            target.target_index,
            target.device,
        )
        source_batch = prepare_d2h_source_batch(host_buffer, target, warmup + iteration)
        try:
            source_iter = iter(source_batch)
            operation = lambda source_iter=source_iter: d2h_transfer(
                next_d2h_source(source_iter)
            )
            run_transfer_warmup(operation, warmup)
            durations = run_transfer_iterations(
                operation,
                iteration,
                trace_dir,
                trace_name=f"trace_d2h_one_to_one_target{target.target_index}",
                cleanup_trace=cleanup_trace,
            )
        finally:
            delete_device_object(source_batch)
        results.append({
            "test_item": "one_to_one",
            "target": format_target(target),
            "metrics": summarize_transfer_durations(durations, data_size_bytes),
        })
    return results


def run_h2d_one_to_many(
    host_buffer: np.ndarray,
    targets: list[TargetDevice],
    warmup: int,
    iteration: int,
    data_size_bytes: int,
    trace_dir: str,
    cleanup_trace: bool,
) -> dict[str, Any]:
    """Run concurrent CPU-to-many-TPU H2D tests."""
    host_batch = prepare_h2d_host_batch(host_buffer, warmup + iteration)

    try:
        host_iter = iter(host_batch)
        with ThreadPoolExecutor(max_workers=len(targets)) as executor:

            def run_once() -> list[Any]:
                # 1:N means one host buffer is sent to all selected chiplet
                # devices at roughly the same time.
                current_host_buffer = next_h2d_host_buffer(host_iter)
                futures = [
                    executor.submit(h2d_transfer, current_host_buffer, target)
                    for target in targets
                ]
                device_arrays = [future.result() for future in futures]
                return device_arrays

            run_transfer_warmup(run_once, warmup)
            durations = run_transfer_iterations(
                run_once,
                iteration,
                trace_dir,
                trace_name="trace_h2d_one_to_many",
                cleanup_trace=cleanup_trace,
            )
    finally:
        del host_batch

    total_bytes = data_size_bytes * len(targets)
    metrics = summarize_transfer_durations(durations, total_bytes)
    metrics["per_target_bytes"] = data_size_bytes
    metrics["target_count"] = len(targets)
    return {
        "test_item": "one_to_many",
        "targets": [format_target(target) for target in targets],
        "metrics": metrics,
    }


def run_d2h_one_to_many(
    host_buffer: np.ndarray,
    targets: list[TargetDevice],
    warmup: int,
    iteration: int,
    data_size_bytes: int,
    trace_dir: str,
    cleanup_trace: bool,
) -> dict[str, Any]:
    """Run concurrent many-TPU-to-CPU D2H tests."""
    source_batches = {
        target.target_index: prepare_d2h_source_batch(
            host_buffer,
            target,
            warmup + iteration,
        )
        for target in targets
    }

    try:
        source_iters = {
            target_index: iter(batch)
            for target_index, batch in source_batches.items()
        }
        with ThreadPoolExecutor(max_workers=len(targets)) as executor:

            def run_once() -> list[Any]:
                # Run D2H reads concurrently from fresh device-resident
                # sources prepared before the timed loop.
                futures = [
                    executor.submit(
                        d2h_transfer,
                        next_d2h_source(source_iters[target.target_index]),
                    )
                    for target in targets
                ]
                host_arrays = [future.result() for future in futures]
                return host_arrays

            run_transfer_warmup(run_once, warmup)
            durations = run_transfer_iterations(
                run_once,
                iteration,
                trace_dir,
                trace_name="trace_d2h_one_to_many",
                cleanup_trace=cleanup_trace,
            )
    finally:
        delete_device_object(list(source_batches.values()))

    total_bytes = data_size_bytes * len(targets)
    metrics = summarize_transfer_durations(durations, total_bytes)
    metrics["per_target_bytes"] = data_size_bytes
    metrics["target_count"] = len(targets)
    return {
        "test_item": "one_to_many",
        "targets": [format_target(target) for target in targets],
        "metrics": metrics,
    }


def write_case_metrics(
    metrics_dir: str,
    benchmark: str,
    metadata: dict[str, Any],
    case: dict[str, Any],
) -> None:
    """Write one test item to metrics JSONL."""
    test_item = case["test_item"]
    if test_item == "one_to_one":
        target = case["target"]
        test_name = (
            f"tpu_{benchmark}_{test_item}_target{target['target_index']}"
        )
        dimensions = {
            **metadata,
            "test_item": test_item,
            "target": target,
        }
    else:
        test_name = f"tpu_{benchmark}_{test_item}"
        dimensions = {
            **metadata,
            "test_item": test_item,
            "targets": case["targets"],
        }
    write_jsonl_metrics(metrics_dir, test_name, dimensions, case["metrics"])


def aggregate_one_to_one(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize one-to-one per-target PCIe throughput."""
    gbps_values = [
        float(case["metrics"]["bandwidth_GBps"])
        for case in case_results
        if case["test_item"] == "one_to_one"
    ]
    if not gbps_values:
        return {}
    summary = average_min_max(
        gbps_values,
        avg_key="avg_bandwidth_GBps",
        min_key="min_bandwidth_GBps",
        max_key="max_bandwidth_GBps",
        count_key="target_count",
    )
    for key in (
        "avg_bandwidth_GBps",
        "min_bandwidth_GBps",
        "max_bandwidth_GBps",
    ):
        summary[key] = round(summary[key], 4)
    return summary


def run_pcie(
    benchmark: str,
    data_size: str | int,
    warmup: int,
    iteration: int,
    result_dir: str,
    target_devices: int,
    cleanup_trace: bool = False,
) -> dict[str, Any]:
    """Run H2D or D2H PCIe benchmark."""
    if benchmark not in {"h2d", "d2h"}:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    validate_non_negative("warmup", warmup)
    validate_positive("iteration", iteration)

    parsed_data_size = parse_data_size(
        data_size,
        alignment_bytes=TRANSFER_ROW_BYTES,
        alignment_description=(
            f"one transfer row: {TRANSFER_ROW_ELEMENTS} float32 values = "
            f"{TRANSFER_ROW_BYTES} bytes"
        ),
    )
    data_size_bytes = parsed_data_size.bytes
    host_buffer = make_host_buffer(data_size_bytes)
    targets = select_local_tpu_targets(target_devices)

    data_size_label = parsed_data_size.label
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{ts}_{benchmark}_pcie_{target_devices}devices_{data_size_label}"
    )
    dirs = prepare_benchmark_dirs(
        result_dir,
        run_name,
        dump_hlo=False,
        create_trace=True,
    )
    out_dir = dirs.output_dir
    metrics_dir = dirs.metrics_dir
    trace_dir = dirs.trace_dir
    if metrics_dir is None or trace_dir is None:
        raise RuntimeError("metrics and trace directories are required")

    metadata = {
        "benchmark": benchmark,
        "transfer_direction": (
            "host_to_device" if benchmark == "h2d" else "device_to_host"
        ),
        **parsed_data_size.metadata(),
        "data_dtype": "float32",
        "host_role": "single_cpu_process",
        "target_device_count": len(targets),
        "target_selection": "all_chiplet_devices_per_chip",
        "test_domain": "pcie",
        "warmup": warmup,
        "iteration": iteration,
        "cleanup_trace": cleanup_trace,
    }

    logger.info(
        "Starting %s PCIe benchmark: data_size=%d bytes, targets=%s",
        benchmark,
        data_size_bytes,
        [format_target(target) for target in targets],
    )
    try:
        if benchmark == "h2d":
            one_to_one_cases = run_h2d_one_to_one(
                host_buffer,
                targets,
                warmup,
                iteration,
                data_size_bytes,
                trace_dir,
                cleanup_trace,
            )
            one_to_many_case = run_h2d_one_to_many(
                host_buffer,
                targets,
                warmup,
                iteration,
                data_size_bytes,
                trace_dir,
                cleanup_trace,
            )
        else:
            one_to_one_cases = run_d2h_one_to_one(
                host_buffer,
                targets,
                warmup,
                iteration,
                data_size_bytes,
                trace_dir,
                cleanup_trace,
            )
            one_to_many_case = run_d2h_one_to_many(
                host_buffer,
                targets,
                warmup,
                iteration,
                data_size_bytes,
                trace_dir,
                cleanup_trace,
            )

        case_results = [*one_to_one_cases, one_to_many_case]
        for case in case_results:
            write_case_metrics(metrics_dir, benchmark, metadata, case)

        return {
            "metadata": metadata,
            "one_to_one": one_to_one_cases,
            "one_to_many": one_to_many_case,
            "one_to_one_summary": aggregate_one_to_one(one_to_one_cases),
            "output_directory": out_dir,
        }
    finally:
        del host_buffer
        gc.collect()


def format_cli_summary(results: dict[str, Any]) -> dict[str, Any]:
    """Return a compact JSON-safe terminal summary."""
    metadata = results["metadata"]
    one_to_many = results["one_to_many"]
    return {
        "benchmark": metadata["benchmark"],
        "transfer_direction": metadata["transfer_direction"],
        "data_size": {
            "input": metadata["data_size_input"],
            "bytes": metadata["data_size_bytes"],
            "dtype": metadata["data_dtype"],
        },
        "targets": {
            "count": metadata["target_device_count"],
            "selection": metadata["target_selection"],
            "devices": [
                case["target"] for case in results["one_to_one"]
            ],
        },
        "iterations": {
            "warmup": metadata["warmup"],
            "timed": metadata["iteration"],
        },
        "one_to_one_summary": results["one_to_one_summary"],
        "one_to_many": {
            "bandwidth_GBps": one_to_many["metrics"].get("bandwidth_GBps"),
            "bytes_per_measurement": one_to_many["metrics"].get(
                "bytes_per_measurement"
            ),
            "target_count": one_to_many["metrics"].get("target_count"),
        },
        "output": {
            "run_dir": results["output_directory"],
            "metrics_dir": os.path.join(results["output_directory"], "metrics"),
            "trace_dir": os.path.join(results["output_directory"], "trace"),
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="TPU PCIe transfer benchmark"
    )
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--data-size",
        required=True,
        help=data_size_help(
            "Transfer size per target TPU chiplet device in bytes",
            alignment_bytes=TRANSFER_ROW_BYTES,
        ),
    )
    common_parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup iterations (default: 2)",
    )
    common_parser.add_argument(
        "--iteration",
        type=int,
        default=5,
        help="Timed iterations (default: 5)",
    )
    common_parser.add_argument(
        "--result-dir",
        default="./results",
        help="Root directory for results (default: ./results)",
    )
    common_parser.add_argument(
        "--target-devices",
        type=int,
        default=8,
        help="Number of TPU chiplet devices to test (default: 8, i.e. 4 chips × 2 chiplets)",
    )
    common_parser.add_argument(
        "--cleanup-trace",
        action="store_true",
        help="Delete diagnostic xprof traces after timed iterations",
    )
    subparsers = parser.add_subparsers(dest="benchmark", required=True)
    subparsers.add_parser(
        "h2d",
        parents=[common_parser],
        help="Measure CPU to local TPU PCIe transfer",
    )
    subparsers.add_parser(
        "d2h",
        parents=[common_parser],
        help="Measure local TPU to CPU PCIe transfer",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    configure_logging()
    load_jax_runtime_deps()
    initialize_jax_runtime(logger)

    results = run_pcie(
        benchmark=args.benchmark,
        data_size=args.data_size,
        warmup=args.warmup,
        iteration=args.iteration,
        result_dir=args.result_dir,
        target_devices=args.target_devices,
        cleanup_trace=args.cleanup_trace,
    )
    print(json.dumps(format_cli_summary(results), indent=2, default=str))


if __name__ == "__main__":
    main()
