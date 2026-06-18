"""ICI (Inter-Chip Interconnect) link performance benchmark for TPU.

ICI is the high-bandwidth torus interconnect between TPU chips. Each TPU chip
contains 2 logical devices (chiplets), so a slice of N TPU chips exposes 2N
JAX devices.

CLI subcommands:
  - **raw**: user-provided traffic matrix and execution shape
  - **p2p**: built-in all-ones TPU traffic matrix expanded into split pairs
  - **p2p-rdma**: built-in P2P split pairs using Pallas remote DMA
  - **a2a**: built-in all-to-all over all JAX devices with inter-TPU ICI metrics
  - **ar**: built-in all-reduce over all JAX devices with bus-bandwidth metrics

Use ``python test_ici.py --help`` for CLI usage.
"""

from __future__ import annotations

import json
import logging
import os
import sys

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ici.constants import PAYLOAD_ROW_BYTES
from utils.runtime import (
    configure_dump_hlo_from_argv,
    configure_logging,
    configure_tpu_benchmark_env,
    initialize_jax_runtime,
)
from utils.units import data_size_help


def select_ici_env_profile(argv: list[str]) -> str:
    """Choose XLA flags before importing JAX based on the ICI subcommand."""
    for token in argv[1:]:
        if token in {"raw", "p2p", "p2p-rdma"}:
            return "ici_send_recv"
        if token == "a2a":
            return "ici_all_to_all"
        if token == "ar":
            return "ici_psum"
    return "ici"


# Environment flags MUST be set before importing JAX-backed ICI modules.
_ICI_ENV_PROFILE = select_ici_env_profile(sys.argv)
configure_tpu_benchmark_env(_ICI_ENV_PROFILE)
_DUMP_HLO = configure_dump_hlo_from_argv("ici")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ICI link performance benchmark for TPU"
    )

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--data-size", required=True,
        help=data_size_help(
            "Payload size in bytes; a2a uses pre-split bytes per chiplet",
            alignment_bytes=PAYLOAD_ROW_BYTES,
        ),
    )
    common_parser.add_argument(
        "--warmup", type=int, default=2,
        help="Warmup iterations (default: 2)",
    )
    common_parser.add_argument(
        "--iteration", type=int, default=5,
        help="Timed iterations (default: 5)",
    )
    common_parser.add_argument(
        "--result-dir", default="./results",
        help="Root directory for results (default: ./results)",
    )
    common_parser.add_argument(
        "--dump-hlo", action="store_true",
        help="Collect XLA HLO dumps",
    )
    common_parser.add_argument(
        "--cleanup-trace", action="store_true",
        help="Delete xprof trace directory after extracting durations to save disk space",
    )

    subparsers = parser.add_subparsers(dest="benchmark", required=True)

    raw_parser = subparsers.add_parser(
        "raw",
        parents=[common_parser],
        help="Run a user-defined traffic matrix",
    )
    raw_parser.add_argument(
        "--traffic-matrix", required=True,
        help=(
            'Traffic matrix defining P2P communication pattern at TPU level. '
            'Square matrix where row=src, col=dst, value 1=send, 0=no send. '
            'Use "#" to start each row, "," to separate values. '
            'Matrix size must match the inferred TPU chip count. '
            'Matrix indices map to consecutive TPU chip groups in jax.devices(). '
            'Each TPU has 2 chiplets, so TPU[i][j]=1 expands to 4 chiplet links. '
            'Example: "#0,1#1,0" means TPU0->TPU1 and TPU1->TPU0'
        ),
    )
    raw_parser.add_argument(
        "--concurrent", action="store_true", default=False,
        help=(
            "Run the provided matrix as one traffic case "
            "(default: split active entries into one-link cases)"
        ),
    )

    subparsers.add_parser(
        "p2p",
        parents=[common_parser],
        help="Run built-in all-ones TPU traffic matrix as split one-link cases",
    )
    subparsers.add_parser(
        "p2p-rdma",
        parents=[common_parser],
        help=(
            "Run built-in split one-link cases using Pallas remote DMA"
        ),
    )
    subparsers.add_parser(
        "a2a",
        parents=[common_parser],
        help="Run built-in all-to-all with inter-TPU ICI bandwidth metrics",
    )
    subparsers.add_parser(
        "ar",
        parents=[common_parser],
        help="Run built-in all-reduce with bus-bandwidth metrics",
    )

    args = parser.parse_args()

    configure_logging()
    initialize_jax_runtime(logging.getLogger(__name__))

    from ici.runner import format_cli_summary, run_ici

    if args.benchmark == "raw":
        traffic_matrix_str = args.traffic_matrix
        execution_shape = "single_matrix" if args.concurrent else "split_pairs"
    elif args.benchmark == "p2p":
        traffic_matrix_str = None
        execution_shape = "split_pairs"
    elif args.benchmark == "p2p-rdma":
        traffic_matrix_str = None
        execution_shape = "split_pairs"
    elif args.benchmark == "a2a":
        traffic_matrix_str = None
        execution_shape = "single_matrix"
    elif args.benchmark == "ar":
        traffic_matrix_str = None
        execution_shape = "single_matrix"
    else:
        raise ValueError(f"Unknown benchmark subcommand: {args.benchmark}")

    try:
        results = run_ici(
            benchmark=args.benchmark,
            traffic_matrix_str=traffic_matrix_str,
            data_size=args.data_size,
            execution_shape=execution_shape,
            warmup=args.warmup,
            iteration=args.iteration,
            result_dir=args.result_dir,
            dump_hlo=args.dump_hlo,
            cleanup_trace=args.cleanup_trace,
            xla_flag_profile=_ICI_ENV_PROFILE,
            dump_hlo_source_dir=_DUMP_HLO.temp_dir,
            clear_hlo_source=_DUMP_HLO.owned,
        )
        print(json.dumps(format_cli_summary(results), indent=2, default=str))
    finally:
        _DUMP_HLO.cleanup()


if __name__ == "__main__":
    main()
