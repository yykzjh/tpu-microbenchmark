"""Metrics calculation utilities for TPU benchmarks.

Provides:
  - MetricsStatistics: percentile analysis (p50/p90/p95/p99/avg/max/min)
  - Bandwidth computation (GB/s)
  - Output writer: metrics JSONL
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any


METRIC_LOG_PREFIX = "[COMMPILOT_METRIC]"


def emit_log_metric(
    *,
    testcase: str,
    metric: str,
    value: int | float,
    unit: str,
    device: str | int | None = None,
    dimensions: dict[str, Any] | None = None,
) -> None:
    """Print one stable JSON metric record into the benchmark log."""
    record: dict[str, Any] = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "testcase": testcase,
        "metric": metric,
        "value": round(float(value), 4),
        "unit": unit,
    }
    if device is not None:
        record["device"] = str(device)
    if dimensions:
        record["dimensions"] = dimensions
    print(f"{METRIC_LOG_PREFIX} {json.dumps(record, separators=(',', ':'))}")


class MetricsStatistics:
    """Compute percentile statistics from a list of metric values.

    The *unit* parameter controls the suffix appended to metric keys in
    :meth:`serialize` output (e.g. ``"ms"``, ``"bytes"``, ``"gbps"``).
    """

    def __init__(
        self,
        metrics_list: list[float],
        metrics_name: str,
        unit: str = "ms",
    ):
        self.metrics_list = sorted(metrics_list)
        self.metrics_name = metrics_name
        self.unit = unit
        self.n = len(metrics_list)

    def _percentile(self, p: float) -> float:
        if self.n == 0:
            return 0.0
        k = (self.n - 1) * p / 100.0
        f = int(k)
        c = f + 1 if f + 1 < self.n else f
        d = k - f
        return self.metrics_list[f] + d * (self.metrics_list[c] - self.metrics_list[f])

    def _key(self, suffix: str) -> str:
        """Build a metric key like ``duration_avg_ms`` or ``bytes_p50_bytes``."""
        return f"{self.metrics_name}_{suffix}_{self.unit}"

    def get_stat(self, stat: str) -> float:
        """Return one statistic by name."""
        if self.n == 0:
            return 0.0
        if stat == "min":
            return self.metrics_list[0]
        if stat == "max":
            return self.metrics_list[-1]
        if stat == "avg":
            return sum(self.metrics_list) / self.n
        if stat.startswith("p"):
            return self._percentile(float(stat[1:]))
        raise ValueError(f"Unsupported statistic: {stat}")

    def serialize(self) -> dict[str, Any]:
        """Return a flat dict of statistics keyed by ``{name}_{stat}_{unit}``."""
        if self.n == 0:
            return {f"{self.metrics_name}_count": 0}

        return {
            f"{self.metrics_name}_count": self.n,
            self._key("min"): round(self.metrics_list[0], 4),
            self._key("max"): round(self.metrics_list[-1], 4),
            self._key("avg"): round(sum(self.metrics_list) / self.n, 4),
            self._key("p50"): round(self._percentile(50), 4),
            self._key("p90"): round(self._percentile(90), 4),
            self._key("p95"): round(self._percentile(95), 4),
            self._key("p99"): round(self._percentile(99), 4),
        }


def compute_bandwidth_GBps(data_size_bytes: int | float, duration_ms: float) -> float:
    """Compute achieved bandwidth in GB/s from data size and duration."""
    if duration_ms <= 0:
        return 0.0
    return data_size_bytes / 1e9 / (duration_ms / 1e3)


def average_min_max(
    values: list[int | float],
    avg_key: str,
    min_key: str,
    max_key: str,
    count_key: str | None = None,
) -> dict[str, Any]:
    """Return average/min/max summary for a metric series."""
    numeric_values = [float(value) for value in values]
    if not numeric_values:
        summary = {
            avg_key: 0.0,
            min_key: 0.0,
            max_key: 0.0,
        }
        if count_key:
            summary[count_key] = 0
        return summary

    summary = {
        avg_key: sum(numeric_values) / len(numeric_values),
        min_key: min(numeric_values),
        max_key: max(numeric_values),
    }
    if count_key:
        summary[count_key] = len(numeric_values)
    return summary


def write_jsonl_metrics(
    metrics_dir: str | None,
    test_name: str,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    """Write metrics in JSONL format for pipeline consumption.

    Appends to ``metrics_report.jsonl`` so multiple test runs in the same
    directory accumulate in a single file.
    """
    if metrics_dir is None:
        return
    os.makedirs(metrics_dir, exist_ok=True)
    filepath = os.path.join(metrics_dir, "metrics_report.jsonl")

    record = {
        "metrics": metrics,
        "dimensions": {
            "test_name": test_name,
            **metadata,
        },
    }

    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")
