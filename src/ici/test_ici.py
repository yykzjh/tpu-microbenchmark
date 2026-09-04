"""ICI (Inter-Chip Interconnect) link performance benchmark for TPU.

ICI is the high-bandwidth torus interconnect between TPU chips. Each TPU chip
contains 2 logical devices (chiplets), so a slice of N TPU chips exposes 2N
JAX devices.

CLI subcommands:
  - **raw**: user-provided traffic matrix and execution shape
  - **p2p**: built-in all-ones TPU traffic matrix expanded into split pairs
  - **p2p-rdma**: built-in P2P split pairs using Pallas remote DMA
  - **a2a**: built-in all-to-all with optional parallel chiplet groups
  - **ar**: built-in all-reduce with optional parallel chiplet groups

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
    initialize_local_jax_runtime,
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
        "--runtime-scope",
        choices=("local", "slice"),
        default="slice",
        help=(
            "local runs independently on this host's local devices; slice "
            "joins every selected host in one distributed collective"
        ),
    )
    common_parser.add_argument(
        "--coordinator-address",
        default=None,
        help="JAX distributed coordinator for --runtime-scope slice",
    )
    common_parser.add_argument(
        "--process-count",
        type=int,
        default=None,
        help="Number of selected hosts for --runtime-scope slice",
    )
    common_parser.add_argument(
        "--process-id",
        type=int,
        default=None,
        help="Zero-based rank of this host for --runtime-scope slice",
    )
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
        "--result-dir", default=None,
        help="Optional output directory for JSONL metrics and retained profile artifacts",
    )
    common_parser.add_argument(
        "--block-range",
        default=None,
        help=(
            "Optional x,y,z half-open TPU chip slices, for example "
            "0:2,0:2,2:4"
        ),
    )
    common_parser.add_argument(
        "--xprof-timing", action="store_true",
        help=(
            "Use a temporary Xprof trace for timing; the trace is always "
            "deleted after parsing"
        ),
    )
    common_parser.add_argument(
        "--profile", "--dump-hlo", dest="profile", action="store_true",
        help="Retain Xprof trace and XLA HLO dumps for performance analysis",
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

    p2p_parser = subparsers.add_parser(
        "p2p",
        parents=[common_parser],
        help="Run built-in all-ones TPU traffic matrix as split one-link cases",
    )
    p2p_parser.add_argument(
        "--p2p-pair-mode", choices=["all", "neighbors"], default="all",
        help="neighbors: all die-to-die links and adjacent-chip core0 links, no self copies",
    )
    subparsers.add_parser(
        "p2p-rdma",
        parents=[common_parser],
        help=(
            "Run built-in split one-link cases using Pallas remote DMA"
        ),
    )
    a2a_parser = subparsers.add_parser(
        "a2a",
        parents=[common_parser],
        help="Run built-in all-to-all with one-way bisection bandwidth metrics",
    )
    a2a_parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help=(
            "Run two chiplet-partitioned all-to-all groups in parallel and "
            "report their combined one-way bisection bandwidth"
        ),
    )
    ar_parser = subparsers.add_parser(
        "ar",
        parents=[common_parser],
        help="Run built-in all-reduce with bus-bandwidth metrics",
    )
    ar_parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help=(
            "Run two chiplet-partitioned all-reduce groups in parallel and "
            "report total effective unidirectional ICI bandwidth per TPU chip"
        ),
    )

    args = parser.parse_args()

    configure_logging()
    logger = logging.getLogger(__name__)
    distributed_values = (
        args.coordinator_address,
        args.process_count,
        args.process_id,
    )
    if args.runtime_scope == "local":
        if any(value is not None for value in distributed_values):
            parser.error(
                "distributed coordinator/process arguments are invalid with "
                "--runtime-scope local"
            )
        initialize_local_jax_runtime(logger)
    else:
        if any(value is None for value in distributed_values):
            parser.error(
                "--runtime-scope slice requires --coordinator-address, "
                "--process-count, and --process-id"
            )
        assert args.coordinator_address is not None
        assert args.process_count is not None
        assert args.process_id is not None
        if args.process_count < 1:
            parser.error("--process-count must be positive")
        if not 0 <= args.process_id < args.process_count:
            parser.error("--process-id must be in [0, process-count)")
        initialize_jax_runtime(
            logger,
            coordinator_address=args.coordinator_address,
            process_count=args.process_count,
            process_id=args.process_id,
        )

    from ici.runner import format_cli_summary, iter_commpilot_log_metrics, run_ici
    from utils.metrics import emit_log_metric

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

    results = run_ici(
        benchmark=args.benchmark,
        traffic_matrix_str=traffic_matrix_str,
        data_size=args.data_size,
        execution_shape=execution_shape,
        warmup=args.warmup,
        iteration=args.iteration,
        result_dir=args.result_dir,
        profile_artifacts=args.profile,
        xla_flag_profile=_ICI_ENV_PROFILE,
        dump_hlo_source_dir=_DUMP_HLO.dump_dir,
        clear_hlo_source=False,
        ar_parallel=(
            bool(args.parallel) if args.benchmark == "ar" else False
        ),
        a2a_parallel=(
            bool(args.parallel) if args.benchmark == "a2a" else False
        ),
        use_xprof_timing=args.xprof_timing,
        runtime_scope=args.runtime_scope,
        block_range=args.block_range,
        p2p_pair_mode=getattr(args, "p2p_pair_mode", "all"),
    )
    if args.runtime_scope == "slice":
        from jax.experimental import multihost_utils

        multihost_utils.sync_global_devices("ici_block_complete")
    for metric, value, dimensions in iter_commpilot_log_metrics(results):
        emit_log_metric(
            testcase="ici",
            metric=metric,
            value=value,
            unit="GB/s",
            device=(
                f"{args.runtime_scope}_process"
                f"{__import__('jax').process_index()}"
            ),
            dimensions=dimensions,
        )
    print(json.dumps(format_cli_summary(results), indent=2, default=str))


if __name__ == "__main__":
    main()
