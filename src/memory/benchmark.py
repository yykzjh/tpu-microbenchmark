"""Execution helpers for TPU memory bandwidth benchmarks."""

from __future__ import annotations

from typing import Any, Callable

import jax

from memory.options import summarize_per_device_bandwidth, validate_memory_mode
from utils.metrics import MetricsStatistics, compute_bandwidth_GBps, write_jsonl_metrics
from utils.profiling import (
    TraceTimingConfig,
    collect_hlo_dumps_if_requested,
    delete_device_object,
    run_profiled_iterations,
)
from utils.runtime import (
    get_local_devices_or_raise,
    log_device_separator,
    prepare_benchmark_dirs,
    validate_non_negative,
    validate_positive,
)


def run_memory_benchmark_across_devices(
    *,
    benchmark: str,
    mode: str,
    result_dir: str,
    run_name_builder: Callable[[int], str],
    metadata_builder: Callable[[int], dict[str, Any]],
    per_device_runner: Callable[[int, Any, str], dict[str, Any]],
    warmup: int,
    iteration: int,
    logger: Any,
) -> dict[str, Any]:
    """Run one memory benchmark mode sequentially on all local devices."""
    validate_memory_mode(mode)
    validate_non_negative("warmup", warmup)
    validate_positive("iteration", iteration)

    devices = get_local_devices_or_raise()
    num_devices = len(devices)
    # The mode-level directory contains one aggregate metrics file and one
    # subdirectory per device. Example: hbm/read/device0... plus aggregate.
    dirs = prepare_benchmark_dirs(
        result_dir,
        run_name_builder(num_devices),
        dump_hlo=False,
        create_trace=False,
    )
    out_dir = dirs.output_dir
    metrics_dir = dirs.metrics_dir

    logger.info(
        "Running %s %s benchmark on %d devices",
        benchmark.upper(),
        mode,
        num_devices,
    )

    per_device_results = []
    for device_index, device in enumerate(devices):
        # Run devices sequentially so one device's kernel does not borrow
        # bandwidth or compiler resources from another device's measurement.
        log_device_separator(
            logger,
            f"{benchmark.upper()} {mode} local device",
            device_index,
            device,
        )
        result = per_device_runner(device_index, device, out_dir)
        per_device_results.append(result)
        metrics = result.get("metrics", {})
        logger.info(
            "%s %s local device %d result: duration_avg_ms=%.4f, bandwidth=%.4f GB/s",
            benchmark.upper(),
            mode,
            device_index,
            float(metrics.get("duration_avg_ms", 0)),
            float(metrics.get("bandwidth_GBps", 0)),
        )

    aggregate_metrics = summarize_per_device_bandwidth(per_device_results)
    metadata = metadata_builder(num_devices)
    write_jsonl_metrics(
        metrics_dir,
        f"{benchmark}_{mode}_aggregate",
        metadata,
        aggregate_metrics,
    )

    return {
        "metadata": metadata,
        "per_device_results": per_device_results,
        "aggregate_metrics": aggregate_metrics,
        "output_directory": out_dir,
    }


def run_traced_bandwidth_phases(
    *,
    compiled_fn: Callable,
    data_generator: Callable,
    data_bytes: int | float,
    test_name: str,
    metadata: dict[str, Any],
    warmup: int,
    iteration: int,
    metrics_dir: str,
    trace_dir: str,
    dump_hlo: bool,
    dump_hlo_source_dir: str | None,
    dump_hlo_dir: str | None,
    clear_hlo_source: bool,
    cleanup_trace: bool,
    logger: Any,
    timing_config: TraceTimingConfig | None = None,
) -> dict[str, Any]:
    """Run warmup, traced timing, JSONL output, and optional HLO collection."""
    logger.info("Phase 1: Warmup (%d iterations)", warmup)
    for _ in range(warmup):
        args = data_generator()
        result = None
        try:
            # Warmup compiles and primes caches; its duration is not recorded.
            result = compiled_fn(*args)
            jax.block_until_ready(result)
        finally:
            delete_device_object(result)

    logger.info("Phase 2: Timed iterations (%d iterations)", iteration)
    # Most memory kernels are represented by one XLA marker event with a
    # device_duration_ps field. Some Pallas Mosaic kernels only expose a tiny
    # custom-call bookkeeping duration there, so callers may override the trace
    # extraction policy when the device marker is not a trustworthy denominator.
    if timing_config is None:
        timing_config = TraceTimingConfig(
            task_name=f"timed_{test_name}",
            trace_dir=trace_dir,
            dest_name=f"trace_{test_name}",
            cleanup_trace=cleanup_trace,
            thread_name_contains="XLA Ops",
        )
    durations_ms = run_profiled_iterations(
        compiled_fn=compiled_fn,
        data_generator=data_generator,
        iteration=iteration,
        config=timing_config,
    )

    stats = MetricsStatistics(durations_ms, "duration", "ms")
    metrics = stats.serialize()
    avg_duration_ms = stats.get_stat("avg")
    # data_bytes is provided by the caller because HBM/VMEM count traffic
    # differently. For example, copy usually counts read + write bytes.
    bandwidth_gbps = compute_bandwidth_GBps(data_bytes, avg_duration_ms)
    metrics["data_size_bytes"] = data_bytes
    metrics["data_size_mib"] = data_bytes / (1024 * 1024)
    if timing_config.require_marker and timing_config.duration_source == "device":
        timing_source = "xprof_marker_device_duration"
    elif timing_config.duration_source == "trace":
        timing_source = "xprof_trace_duration"
    else:
        timing_source = f"xprof_{timing_config.duration_source}_duration"
    metrics["timing_source"] = timing_source
    metrics["trace_task_name"] = timing_config.task_name
    metrics["trace_event_name_contains"] = timing_config.event_name_contains
    metrics["trace_thread_name_contains"] = timing_config.thread_name_contains
    metrics["bandwidth_GBps"] = round(bandwidth_gbps, 4)

    write_jsonl_metrics(metrics_dir, test_name, metadata, metrics)

    collect_hlo_dumps_if_requested(
        dump_hlo,
        dump_hlo_source_dir,
        dump_hlo_dir,
        test_name,
        phase_label="Phase 3",
        clear_source=clear_hlo_source,
    )
    logger.info("Benchmark completed. Bandwidth: %.2f GB/s", bandwidth_gbps)
    return metrics
