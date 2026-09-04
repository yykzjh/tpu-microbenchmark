"""Profiling and tracing utilities for TPU benchmarks.

Provides:
  - MARKER constant for trace event annotation
  - timed xprof trace capture
  - best-effort JAX device object release
  - HLO dump copying
"""

from __future__ import annotations

import os
import glob
import shutil
import logging
import gzip
import json
import pathlib
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax

from utils.runtime import benchmark_temporary_directory

logger = logging.getLogger(__name__)

MARKER = "!!MARKER!!"


@dataclass(frozen=True)
class TraceTimingConfig:
    """Profile-trace timing settings shared by TPU benchmarks.

    The defaults match single-kernel device benchmarks. ICI callers override
    the reducer and kernel-name filters for multi-device communication events.
    """

    task_name: str = "timed_iterations"
    trace_dir: str | None = None
    dest_name: str | None = None
    cleanup_trace: bool = False
    marker: str = MARKER
    duration_source: str = "device"
    trace_only_xla: bool = True
    per_iteration_reducer: str = "median"
    event_name_contains: str | None = None
    source_contains: str | None = None
    kernel_name_contains: Sequence[str] | None = None
    thread_name_contains: str | None = None
    require_marker: bool = True
    exclude_zero_crop: bool = True

    def __post_init__(self) -> None:
        if self.duration_source not in {"device", "trace"}:
            raise ValueError(
                f"Unsupported duration_source {self.duration_source!r}; "
                "expected 'device' or 'trace'"
            )
        if self.per_iteration_reducer not in {"median", "max"}:
            raise ValueError(
                "Unsupported per_iteration_reducer "
                f"{self.per_iteration_reducer!r}; expected 'median' or 'max'"
            )
        if not self.require_marker and not any(
            [
                self.event_name_contains,
                self.source_contains,
                self.kernel_name_contains,
                self.thread_name_contains,
            ]
        ):
            raise ValueError(
                "TraceTimingConfig(require_marker=False) requires at least "
                "one event/source/kernel filter"
            )


def _trace_context(trace_dir: str, trace_only_xla: bool = True):
    """Return a JAX trace context, preferring TPU XLA-only traces when supported."""
    profile_options_cls = getattr(jax.profiler, "ProfileOptions", None)
    if profile_options_cls is None or not trace_only_xla:
        return jax.profiler.trace(trace_dir, create_perfetto_link=False)

    options = profile_options_cls()
    try:
        # TRACE_ONLY_XLA keeps the timeline focused on device execution. That
        # makes the MARKER duration closer to the TPU kernel time than a full
        # host/device trace with Python and runtime bookkeeping mixed in.
        options.advanced_configuration = {
            "tpu_trace_mode": "TRACE_ONLY_XLA",
            "tpu_num_sparse_cores_to_trace": 0,
            "tpu_num_sparse_core_tiles_to_trace": 0,
        }
        return jax.profiler.trace(
            trace_dir,
            create_perfetto_link=False,
            profiler_options=options,
        )
    except (TypeError, AttributeError):
        return jax.profiler.trace(trace_dir, create_perfetto_link=False)


def get_trace(trace_dir: str) -> dict[str, Any]:
    """Load the latest xprof trace JSON from a JAX profiler trace directory."""
    profile_root = pathlib.Path(trace_dir).absolute() / "plugins" / "profile"
    if not profile_root.exists():
        raise FileNotFoundError("xprof profile directory not found")

    trace_files = list(profile_root.glob("**/*.trace.json.gz"))
    trace_files.extend(profile_root.glob("**/*.trace.json"))
    if not trace_files:
        raise FileNotFoundError("No xprof trace JSON found")

    trace_file = max(trace_files, key=lambda path: path.stat().st_mtime)
    if trace_file.suffix == ".gz":
        with gzip.open(trace_file, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(trace_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _device_duration_ms(event: dict[str, Any]) -> float | None:
    """Return a device event duration in milliseconds when available."""
    args = event.get("args", {})
    if "device_duration_ps" in args:
        return float(args["device_duration_ps"]) / 1e9
    return None


def _trace_duration_ms(event: dict[str, Any]) -> float | None:
    """Return a Chrome trace event duration in milliseconds when available."""
    if "dur" not in event:
        return None
    # Chrome trace ``dur`` is expressed in microseconds.
    return float(event["dur"]) / 1e3


def _event_duration_ms(
    event: dict[str, Any],
    duration_source: str,
) -> float | None:
    """Return the requested duration value from one trace event."""
    if duration_source == "device":
        return _device_duration_ms(event)
    if duration_source == "trace":
        return _trace_duration_ms(event)
    raise ValueError(
        f"Unsupported duration_source {duration_source!r}; "
        "expected 'device' or 'trace'"
    )


def _marker_events(
    trace: dict[str, Any],
    marker: str | None,
    duration_source: str,
    event_name_contains: str | None = None,
    source_contains: str | None = None,
    kernel_name_contains: Sequence[str] | None = None,
    thread_name_contains: str | None = None,
    require_marker: bool = True,
    exclude_zero_crop: bool = True,
) -> list[dict[str, Any]]:
    """Return trace events containing *marker* with duration information."""
    kernel_name_terms = [
        term.lower() for term in (kernel_name_contains or []) if term
    ]
    thread_names = {
        (event.get("pid"), event.get("tid")): str(
            event.get("args", {}).get("name", "")
        )
        for event in trace.get("traceEvents", [])
        if event.get("ph") == "M" and event.get("name") == "thread_name"
    }
    events = []
    for event in trace.get("traceEvents", []):
        if thread_name_contains:
            thread_name = thread_names.get((event.get("pid"), event.get("tid")), "")
            if thread_name_contains not in thread_name:
                continue
        args = event.get("args", {})
        tf_op = str(args.get("tf_op", ""))
        event_name = str(event.get("name", ""))
        source = str(args.get("source", ""))
        long_name = str(args.get("long_name", ""))
        # The marker may appear either in the TensorFlow op metadata or in the
        # event name, depending on JAX/jaxlib versions.
        if require_marker and marker not in tf_op and marker not in event_name:
            continue
        if exclude_zero_crop and (
            "ZeroCrop" in long_name or "zero_crop.py" in source
        ):
            continue
        if event_name_contains and event_name_contains not in event_name:
            continue
        if source_contains and source_contains not in source:
            continue
        if kernel_name_terms:
            searchable_text = " ".join(
                [event_name, tf_op, long_name, source]
            ).lower()
            if not any(term in searchable_text for term in kernel_name_terms):
                continue
        if _event_duration_ms(event, duration_source) is not None:
            events.append(event)

    return events


def _select_marker_duration_ms(
    events: list[dict[str, Any]],
    duration_source: str,
    reducer: str,
) -> float:
    """Select one marker duration from one logical iteration."""
    if not events:
        raise ValueError("No marker events to select from")
    durations = sorted(
        duration for event in events
        if (duration := _event_duration_ms(event, duration_source)) is not None
    )
    if not durations:
        raise ValueError("No duration-bearing marker events")
    if reducer == "max":
        return durations[-1]
    if reducer != "median":
        raise ValueError(
            f"Unsupported marker duration reducer {reducer!r}; "
            "expected 'median' or 'max'"
        )
    mid = len(durations) // 2
    if len(durations) % 2 == 1:
        return durations[mid]
    return (durations[mid - 1] + durations[mid]) / 2


def marker_durations_ms_from_trace(
    trace: dict[str, Any],
    marker: str | None = MARKER,
    expected_iterations: int | None = None,
    duration_source: str = "device",
    per_iteration_reducer: str = "median",
    event_name_contains: str | None = None,
    source_contains: str | None = None,
    kernel_name_contains: Sequence[str] | None = None,
    thread_name_contains: str | None = None,
    require_marker: bool = True,
    exclude_zero_crop: bool = True,
) -> list[float]:
    """Extract marker durations from a loaded xprof trace."""
    events = _marker_events(
        trace,
        marker,
        duration_source,
        event_name_contains=event_name_contains,
        source_contains=source_contains,
        kernel_name_contains=kernel_name_contains,
        thread_name_contains=thread_name_contains,
        require_marker=require_marker,
        exclude_zero_crop=exclude_zero_crop,
    )
    if not events:
        filter_parts = []
        if kernel_name_contains:
            filter_parts.append(f"kernel terms={list(kernel_name_contains)!r}")
        if source_contains:
            filter_parts.append(f"source contains={source_contains!r}")
        if event_name_contains:
            filter_parts.append(f"event name contains={event_name_contains!r}")
        if thread_name_contains:
            filter_parts.append(f"thread name contains={thread_name_contains!r}")
        filters = f" ({', '.join(filter_parts)})" if filter_parts else ""
        marker_desc = (
            f"marker {marker!r} and "
            if require_marker else ""
        )
        raise RuntimeError(
            f"No xprof events with {marker_desc}{duration_source!r} "
            f"durations were found in the trace{filters}"
        )

    sorted_events = sorted(
        events,
        key=lambda event: event.get("ts", 0),
    )
    if expected_iterations is None:
        return [
            _select_marker_duration_ms(
                [event],
                duration_source,
                per_iteration_reducer,
            )
            for event in sorted_events
        ]

    if len(sorted_events) % expected_iterations != 0:
        raise RuntimeError(
            f"Expected marker events to split evenly across "
            f"{expected_iterations} iterations, collected {len(sorted_events)}"
        )
    events_per_iteration = len(sorted_events) // expected_iterations
    if events_per_iteration == 0:
        raise RuntimeError("No marker events were found for traced iterations")

    durations = []
    for iteration in range(expected_iterations):
        # One logical iteration may emit multiple marker events. The caller
        # chooses how to collapse them: max for multi-device communication
        # elapsed time, median for single-device duplicate trace events.
        start = iteration * events_per_iteration
        end = start + events_per_iteration
        durations.append(
            _select_marker_duration_ms(
                sorted_events[start:end],
                duration_source,
                per_iteration_reducer,
            )
        )
    return durations


def run_profiled_iterations(
    compiled_fn,
    data_generator,
    iteration: int,
    config: TraceTimingConfig,
    extract_trace_durations: bool = True,
) -> list[float]:
    """Run iterations under Xprof and return trace or synchronized CPU durations."""
    with benchmark_temporary_directory("xprof_") as tmp:
        cpu_durations_ms = []
        with _trace_context(tmp, trace_only_xla=config.trace_only_xla):
            for i in range(iteration):
                data_args = data_generator()
                # Prepare inputs before entering the annotated region so H2D
                # staging or host-side argument construction is not measured.
                jax.block_until_ready(data_args)
                result = None
                try:
                    with jax.profiler.StepTraceAnnotation(
                        config.task_name,
                        step_num=i,
                    ):
                        started_at = time.perf_counter()
                        result = compiled_fn(*data_args)
                        jax.block_until_ready(result)
                        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                    cpu_durations_ms.append(elapsed_ms)
                finally:
                    delete_device_object(result)
                    result = None

        if extract_trace_durations:
            trace = get_trace(tmp)
            durations_ms = marker_durations_ms_from_trace(
                trace,
                marker=config.marker,
                expected_iterations=iteration,
                duration_source=config.duration_source,
                per_iteration_reducer=config.per_iteration_reducer,
                event_name_contains=config.event_name_contains,
                source_contains=config.source_contains,
                kernel_name_contains=config.kernel_name_contains,
                thread_name_contains=config.thread_name_contains,
                require_marker=config.require_marker,
                exclude_zero_crop=config.exclude_zero_crop,
            )
        else:
            durations_ms = cpu_durations_ms

        if not config.cleanup_trace and config.trace_dir and config.dest_name:
            dest = os.path.join(config.trace_dir, config.dest_name)
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.copytree(tmp, dest)

    return durations_ms


def run_synchronized_iterations(
    compiled_fn,
    data_generator,
    iteration: int,
) -> list[float]:
    """Measure host elapsed time around fully synchronized device execution.

    Inputs are made ready before the timer starts. The result is explicitly
    blocked before the stop timestamp so asynchronous JAX dispatch cannot turn
    this into a kernel-issue latency measurement. This path does not start
    Xprof and does not create trace or temporary files.
    """
    durations_ms = []
    for _ in range(iteration):
        data_args = data_generator()
        jax.block_until_ready(data_args)
        result = None
        try:
            started_at = time.perf_counter()
            result = compiled_fn(*data_args)
            jax.block_until_ready(result)
            durations_ms.append((time.perf_counter() - started_at) * 1000.0)
        finally:
            delete_device_object(result)
            result = None
    return durations_ms


def delete_device_object(obj: Any) -> None:
    """Best-effort release for JAX device-backed objects."""
    if obj is None:
        return
    if isinstance(obj, dict):
        for value in obj.values():
            delete_device_object(value)
        return
    if isinstance(obj, (list, tuple)):
        for value in obj:
            delete_device_object(value)
        return

    delete = getattr(obj, "delete", None)
    if callable(delete):
        try:
            delete()
        except Exception as exc:
            logger.debug("Failed to delete device object %r: %s", type(obj), exc)


# ---------------------------------------------------------------------------
# HLO dump management
# ---------------------------------------------------------------------------

def copy_hlo_dumps(
    src_dir: str,
    dest_dir: str,
    test_name: str,
    clear_source: bool = False,
) -> list[str]:
    """Copy HLO dump files from *src_dir* to *dest_dir* with a test prefix."""
    os.makedirs(dest_dir, exist_ok=True)
    if os.path.realpath(src_dir) == os.path.realpath(dest_dir):
        retained = []
        for pattern in ("*.before_optimizations.txt", "*.after_optimizations.txt"):
            retained.extend(glob.glob(os.path.join(src_dir, pattern)))
        if not retained:
            logger.warning("No HLO dump files were generated")
        return sorted(retained)

    copied = []

    for pattern in ("*.before_optimizations.txt", "*.after_optimizations.txt"):
        for src_file in glob.glob(os.path.join(src_dir, pattern)):
            new_name = f"{test_name}_{os.path.basename(src_file)}"
            dest_file = os.path.join(dest_dir, new_name)
            shutil.copy2(src_file, dest_file)
            copied.append(dest_file)
            if clear_source:
                try:
                    os.remove(src_file)
                except OSError as exc:
                    logger.debug("Failed to remove a copied HLO dump: %s", exc)

    if not copied:
        logger.warning("No HLO dump files were generated")
    return copied


def collect_hlo_dumps_if_requested(
    dump_hlo: bool,
    src_dir: str | None,
    dest_dir: str | None,
    test_name: str,
    phase_label: str = "Phase 3",
    clear_source: bool = False,
) -> list[str]:
    """Copy HLO dumps for one test when dump-HLO mode is enabled."""
    if not dump_hlo:
        logger.info("%s: dump HLO skipped", phase_label)
        return []

    logger.info("%s: dump HLO", phase_label)
    if src_dir and dest_dir:
        return copy_hlo_dumps(
            src_dir,
            dest_dir,
            test_name,
            clear_source=clear_source,
        )

    logger.warning(
        "Dump-HLO was requested, but no HLO dump source directory was configured"
    )
    return []
