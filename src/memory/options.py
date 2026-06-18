"""CLI options and result presentation helpers for memory benchmarks."""

from __future__ import annotations

import argparse
from typing import Any

from utils.metrics import average_min_max
from utils.units import data_size_help


MEMORY_MODES = ("read", "write", "copy")
MEMORY_MODE_CHOICES = (*MEMORY_MODES, "all")


def validate_memory_mode(mode: str) -> None:
    """Raise if *mode* is not a supported memory bandwidth mode."""
    if mode not in MEMORY_MODES:
        raise ValueError(f"Unknown mode: {mode}")


def expand_memory_modes(mode: str) -> list[str]:
    """Expand ``all`` to the supported memory benchmark modes."""
    if mode == "all":
        return list(MEMORY_MODES)
    validate_memory_mode(mode)
    return [mode]


def parse_memory_dtype(dtype_name: str):
    """Return a JAX dtype for common memory benchmark dtypes."""
    import jax.numpy as jnp

    dtype_map = {
        "float32": jnp.float32,
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16,
    }
    try:
        return dtype_map[dtype_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported dtype {dtype_name!r}; supported values: "
            + ", ".join(dtype_map)
        ) from exc


def summarize_per_device_bandwidth(
    per_device_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return rounded avg/min/max bandwidth summary across local devices."""
    # Each entry is one independently timed local JAX device. The aggregate is
    # a summary table for quick comparison; it is not a cross-device parallel
    # throughput number.
    bandwidths = [
        result["metrics"]["bandwidth_GBps"]
        for result in per_device_results
    ]
    summary = average_min_max(
        bandwidths,
        avg_key="avg_bandwidth_GBps",
        min_key="min_bandwidth_GBps",
        max_key="max_bandwidth_GBps",
        count_key="device_count",
    )
    for key in ("avg_bandwidth_GBps", "min_bandwidth_GBps", "max_bandwidth_GBps"):
        summary[key] = round(summary[key], 4)
    return summary


def add_memory_common_args(
    parser: argparse.ArgumentParser,
    *,
    default_data_size: str,
    data_size_description: str,
    include_block_shape: bool,
    include_dtype: bool,
    block_shape_help: str | None = None,
    mode_choices: tuple[str, ...] = MEMORY_MODE_CHOICES,
    include_mode: bool = True,
) -> None:
    """Add common memory benchmark CLI arguments."""
    if include_mode:
        parser.add_argument(
            "--mode",
            type=str,
            choices=mode_choices,
            default="all",
            help="Test mode (default: all)",
        )
    parser.add_argument(
        "--data-size",
        default=default_data_size,
        help=data_size_help(data_size_description, default=default_data_size),
    )
    if include_block_shape:
        parser.add_argument(
            "--block-shape",
            type=int,
            nargs=2,
            default=[128, 1024],
            help=block_shape_help or "Pallas block shape (default: 128 1024)",
        )
    if include_dtype:
        parser.add_argument(
            "--dtype",
            type=str,
            default="float32",
            choices=["float32", "float16", "bfloat16"],
            help="Data type (default: float32)",
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
        type=str,
        default="./results",
        help="Result directory (default: ./results)",
    )
    parser.add_argument(
        "--dump-hlo",
        action="store_true",
        help="Enable HLO dump collection",
    )
    parser.add_argument(
        "--cleanup-trace",
        action="store_true",
        help="Cleanup trace directory after extracting durations",
    )


def print_memory_results(
    benchmark: str,
    results: dict[str, Any],
    *,
    data_size_label: str = "Data size",
    show_block_shape: bool = False,
) -> None:
    """Print a compact per-device memory benchmark result table."""
    print(f"\n{'=' * 60}")
    print(f"{benchmark.upper()} {results['metadata']['mode'].upper()} Results:")
    print(f"{'=' * 60}")
    print(
        f"{data_size_label}: {results['metadata']['data_size_input']} "
        f"({results['metadata']['data_size_bytes']} bytes)"
    )
    if show_block_shape:
        shape_label = (
            "VMEM buffer shape"
            if results["metadata"].get("block_shape_semantics") == "single_vmem_buffer"
            else "Block shape"
        )
        print(f"{shape_label}: {results['metadata']['block_shape']}")
    print(f"Number of devices: {results['metadata']['device_count']}")
    for device_result in results["per_device_results"]:
        meta = device_result["metadata"]
        metrics = device_result["metrics"]
        print(
            f"Device {meta['device_index']:2d}: "
            f"{metrics['bandwidth_GBps']:8.2f} GB/s, "
            f"avg {metrics['duration_avg_ms']:.3f} ms, "
            f"min {metrics['duration_min_ms']:.3f} ms, "
            f"max {metrics['duration_max_ms']:.3f} ms"
        )
    print(
        "Aggregate: "
        f"avg {results['aggregate_metrics']['avg_bandwidth_GBps']:.2f} GB/s, "
        f"min {results['aggregate_metrics']['min_bandwidth_GBps']:.2f} GB/s, "
        f"max {results['aggregate_metrics']['max_bandwidth_GBps']:.2f} GB/s"
    )


def print_memory_mode_summary(results: list[dict[str, Any]]) -> None:
    """Print one-line summaries when a memory benchmark runs all modes."""
    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"{'=' * 60}")
    for result in results:
        print(
            f"{result['metadata']['mode'].upper():6s}: "
            f"{result['aggregate_metrics']['avg_bandwidth_GBps']:8.2f} GB/s "
            f"(avg across {result['metadata']['device_count']} devices)"
        )


def validate_block_shape(shape: tuple[int, ...], required_rank: int = 2):
    """Validate Pallas block shape constraints."""
    if len(shape) != required_rank:
        raise ValueError(
            f"Block shape must have rank {required_rank}, got {len(shape)}"
        )

    if required_rank >= 2:
        if shape[-1] % 128 != 0:
            raise ValueError(
                f"Last dimension must be divisible by 128, got {shape[-1]}"
            )
        if shape[-2] % 8 != 0:
            raise ValueError(
                f"Second-to-last dimension must be divisible by 8, got {shape[-2]}"
            )
