"""Warmup, trace timing, and metrics phases for ICI benchmarks."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Callable

import jax

from utils.metrics import (
    MetricsStatistics,
    compute_bandwidth_GBps,
    write_jsonl_metrics,
)
from utils.profiling import (
    TraceTimingConfig,
    delete_device_object,
    run_profiled_iterations,
)

logger = logging.getLogger(__name__)

def run_timed_iterations(
    compiled_fn,
    data_generator: Callable,
    iteration: int,
    trace_dir: str | None,
    trace_name: str,
    cleanup_trace: bool = False,
    per_iteration_reducer: str = "max",
    source_contains: str | None = None,
    kernel_name_contains: Sequence[str] | None = None,
) -> list[float]:
    """Run synchronized timed iterations and parse MARKER device durations."""
    return run_profiled_iterations(
        compiled_fn=compiled_fn,
        data_generator=data_generator,
        iteration=iteration,
        config=TraceTimingConfig(
            task_name=trace_name,
            trace_dir=trace_dir,
            dest_name=f"{trace_name}_timed" if trace_dir else None,
            cleanup_trace=cleanup_trace,
            per_iteration_reducer=per_iteration_reducer,
            source_contains=source_contains,
            kernel_name_contains=kernel_name_contains,
            thread_name_contains="XLA Ops",
        ),
    )


# ---------------------------------------------------------------------------
# Benchmark phase runner shared by all traffic cases
# ---------------------------------------------------------------------------

def _run_benchmark_phases(
    compiled_fn,
    data_generator: Callable,
    warmup: int,
    iteration: int,
    trace_dir: str | None,
    metrics_dir: str,
    active_links: int,
    bytes_per_measurement: int | float,
    total_traffic_bytes: int | float | None,
    bandwidth_scope: str,
    test_name: str,
    metadata: dict[str, Any],
    cleanup_trace: bool = False,
    trace_event_reducer: str = "max",
    trace_event_source_contains: str | None = None,
    trace_event_kernel_name_contains: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Execute warmup -> timed trace phases and compute metrics.

    Shared by all benchmark paths after they are lowered to traffic cases.

    Args:
        compiled_fn: the compiled JAX function to benchmark.
        data_generator: callable returning a tuple of positional arguments
            to pass to ``compiled_fn`` on each invocation.
        warmup: number of warmup iterations.
        iteration: number of timed iterations.
        trace_dir: directory where timed xprof trace files are saved.
        metrics_dir: directory for metrics JSONL output.
        active_links: number of active P2P links.
        bytes_per_measurement: bytes used as the bandwidth numerator.
        total_traffic_bytes: total logical traffic bytes per single call to
            ``compiled_fn``. Defaults to ``bytes_per_measurement``.
        bandwidth_scope: semantic scope of the reported bandwidth.
        test_name: name for this test run.
        metadata: metadata dict to include in metrics output.
        cleanup_trace: whether to cleanup trace directory after parsing.
    """
    # Phase 1: warmup
    logger.info("Phase 1: warmup (%d iterations)", warmup)
    for _ in range(warmup):
        result = None
        try:
            result = compiled_fn(*data_generator())
            jax.block_until_ready(result)
        finally:
            delete_device_object(result)

    # Phase 2: timed iterations from xprof MARKER device durations.
    logger.info(
        "Phase 2: xprof marker timed iterations (%d iterations)",
        iteration,
    )
    durations_ms = run_timed_iterations(
        compiled_fn,
        data_generator,
        iteration,
        trace_dir=trace_dir,
        trace_name=test_name,
        cleanup_trace=cleanup_trace,
        per_iteration_reducer=trace_event_reducer,
        source_contains=trace_event_source_contains,
        kernel_name_contains=trace_event_kernel_name_contains,
    )

    stats = MetricsStatistics(durations_ms, "duration", unit="ms")
    metrics = stats.serialize()

    bandwidth_duration_ms = metrics.get(
        "duration_p50_ms",
        metrics.get("duration_avg_ms", 0),
    )
    bw = compute_bandwidth_GBps(bytes_per_measurement, bandwidth_duration_ms)

    metrics["total_traffic_bytes"] = total_traffic_bytes or bytes_per_measurement
    metrics["bandwidth_traffic_bytes"] = bytes_per_measurement
    metrics["bandwidth_scope"] = bandwidth_scope
    metrics["timing_source"] = "xprof_marker_device_duration"
    selection_prefix = (
        "kernel_name_matched_"
        if trace_event_kernel_name_contains else ""
    )
    metrics["trace_event_selection"] = (
        f"{selection_prefix}{trace_event_reducer}_marker_duration_per_iteration"
    )
    if trace_event_source_contains:
        metrics["trace_event_source_filter"] = trace_event_source_contains
    if trace_event_kernel_name_contains:
        metrics["trace_event_kernel_name_filter"] = list(
            trace_event_kernel_name_contains
        )
    metrics["bandwidth_duration_stat"] = "p50"
    metrics["bandwidth_duration_ms"] = bandwidth_duration_ms
    metrics["active_links"] = active_links
    metrics["timed_iterations_kept"] = len(durations_ms)
    metrics["bandwidth_GBps"] = round(bw, 4)

    write_jsonl_metrics(metrics_dir, test_name, metadata, metrics)

    logger.info("Done. %s bandwidth=%.2f GB/s", bandwidth_scope, bw)
    return {"metadata": metadata, "metrics": metrics, "durations_ms": durations_ms}


def _run_benchmark_phases_without_recording(
    compiled_fn,
    data_generator: Callable,
    warmup: int,
    iteration: int,
) -> None:
    """Execute phases without local trace parsing or metrics output."""
    logger.info(
        "This process does not own metrics for this split-pair case; "
        "executing warmup/timed calls only for collective synchronization"
    )
    logger.info("Phase 1: warmup (%d iterations)", warmup)
    for _ in range(warmup):
        result = None
        try:
            result = compiled_fn(*data_generator())
            jax.block_until_ready(result)
        finally:
            delete_device_object(result)

    logger.info("Phase 2: synchronized timed calls (%d iterations)", iteration)
    for _ in range(iteration):
        result = None
        try:
            result = compiled_fn(*data_generator())
            jax.block_until_ready(result)
        finally:
            delete_device_object(result)
