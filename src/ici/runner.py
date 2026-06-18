"""Main ICI benchmark orchestration and CLI summary formatting."""

from __future__ import annotations

import gc
import logging
import os
from datetime import datetime
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ici.cleanup import release_compiled_cache
from ici.constants import (
    CHIPLETS_PER_TPU,
    FLOAT32_BYTES,
    PAYLOAD_LAST_DIM,
    PAYLOAD_MID_DIM,
    PAYLOAD_ROW_BYTES,
)
from ici.kernels import (
    compile_all_reduce_kernel,
    compile_all_to_all_kernel,
    compile_remote_dma_p2p_kernel,
    compile_traffic_matrix_kernel,
    create_device_mesh,
    make_axis_sharded_payload,
)
from ici.phases import (
    _run_benchmark_phases,
    _run_benchmark_phases_without_recording,
)
from ici.traffic import (
    TrafficCase,
    average_bandwidth,
    build_builtin_tpu_traffic_matrix_str,
    build_split_pair_bandwidth_groups,
    build_traffic_cases,
    collect_case_bandwidths,
    compute_case_traffic_bytes,
    expand_to_chiplet_level,
    format_topology,
    format_traffic_case,
    format_traffic_matrix_count_label,
    format_tpu_chip_order,
    get_active_pairs,
    infer_tpu_topology,
    local_chiplet_ids,
    manhattan_distance,
    parse_traffic_matrix,
    should_execute_split_case,
    should_record_split_case,
    split_case_local_endpoint_role,
    torus_axis_distances,
    validate_ragged_all_to_all_bounds,
)
from ici.zero_crop import detect_zero_crop_available
from utils.metrics import MetricsStatistics, write_jsonl_metrics
from utils.profiling import collect_hlo_dumps_if_requested, delete_device_object
from utils.runtime import prepare_benchmark_dirs, validate_non_negative, validate_positive
from utils.units import parse_data_size

logger = logging.getLogger(__name__)

def run_ici(
    benchmark: str,
    traffic_matrix_str: str | None,
    data_size: str | int,
    execution_shape: str,
    warmup: int,
    iteration: int,
    result_dir: str,
    dump_hlo: bool = False,
    cleanup_trace: bool = False,
    xla_flag_profile: str | None = None,
    dump_hlo_source_dir: str | None = None,
    clear_hlo_source: bool = False,
) -> dict[str, Any]:
    """Run ICI link performance benchmark.

    Phases per measurement:
      1. **Warmup** -- prime the TPU and let XLA compilation settle.
      2. **Timed iterations** -- save xprof traces and parse MARKER durations.
      3. **Dump HLO** -- copy XLA HLO dumps if ``dump_hlo=True``.

    Args:
        benchmark: benchmark subcommand name, e.g. "raw", "p2p",
            "p2p-rdma", "a2a", or "ar".
        traffic_matrix_str: traffic matrix string, e.g. "#1,0#0,1". Built-in
            p2p/a2a modes generate the matrix from inferred topology. ar does
            not use a traffic matrix.
        data_size: byte size per active chiplet link. Accepts plain integer
            bytes or K/KB/KiB/M/MB/MiB/G/GB/GiB suffixes.
        execution_shape: either ``single_matrix`` or ``split_pairs``.
        warmup: number of warmup iterations.
        iteration: number of timed iterations.
        result_dir: base directory for results.
        dump_hlo: if True, enable HLO dump collection.
        xla_flag_profile: TPU/XLA flag profile selected by the CLI entrypoint.
        dump_hlo_source_dir: temporary XLA dump directory configured before
            JAX import.
        clear_hlo_source: whether copied HLO dump files should be removed from
            the temporary source directory.

    Returns:
        Dict with ``metadata``, ``metrics``, and ``output_directory``. Split
        pair runs also include ``per_case``.
    """
    # --- Validate inputs ---------------------------------------------------
    if benchmark not in {"raw", "p2p", "p2p-rdma", "a2a", "ar"}:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    if benchmark == "raw" and traffic_matrix_str is None:
        raise ValueError("raw benchmark requires traffic_matrix_str")
    if (
        benchmark in {"p2p", "p2p-rdma", "a2a", "ar"}
        and traffic_matrix_str is not None
    ):
        raise ValueError(f"{benchmark} benchmark uses a built-in traffic matrix")
    if execution_shape not in {"single_matrix", "split_pairs"}:
        raise ValueError(f"Unknown traffic execution shape: {execution_shape}")
    if benchmark in {"a2a", "ar"} and execution_shape != "single_matrix":
        raise ValueError(f"{benchmark} benchmark requires single_matrix execution")
    parsed_data_size = parse_data_size(
        data_size,
        alignment_bytes=PAYLOAD_ROW_BYTES,
        alignment_description=(
            f"one payload row: {PAYLOAD_MID_DIM} * {PAYLOAD_LAST_DIM} * "
            f"{FLOAT32_BYTES} = {PAYLOAD_ROW_BYTES} bytes"
        ),
    )
    data_size_bytes = parsed_data_size.bytes
    data_size_elements = data_size_bytes // FLOAT32_BYTES
    payload_rows = data_size_bytes // PAYLOAD_ROW_BYTES
    payload_rows_per_link = payload_rows
    data_size_label = parsed_data_size.label
    validate_non_negative("warmup", warmup)
    validate_positive("iteration", iteration)

    topology, topology_source, tpu_chip_order = infer_tpu_topology()
    topology_str = format_topology(topology)
    n_tpus = topology[0] * topology[1] * topology[2]
    n_chiplets = n_tpus * CHIPLETS_PER_TPU
    all_to_all_chunk_rows = None
    all_to_all_chunk_bytes = None
    if benchmark == "a2a":
        if payload_rows % n_chiplets != 0:
            raise ValueError(
                "a2a --data-size is the pre-split payload bytes per chiplet "
                "and must divide evenly into one all_to_all chunk per JAX "
                f"device. Got payload_rows={payload_rows}, "
                f"n_devices={n_chiplets}, row_bytes={PAYLOAD_ROW_BYTES}. "
                f"Use a size divisible by {n_chiplets * PAYLOAD_ROW_BYTES} bytes."
            )
        all_to_all_chunk_rows = payload_rows // n_chiplets
        all_to_all_chunk_bytes = all_to_all_chunk_rows * PAYLOAD_ROW_BYTES

    if benchmark == "ar":
        # AllReduce does not need an explicit traffic matrix. Every chiplet is
        # a participant in one collective group.
        traffic_matrix_source = "not_applicable"
        traffic_matrix_label = "allreduce"
        tpu_traffic_matrix: list[list[int]] = []
        tpu_matrix_np = np.zeros((n_tpus, n_tpus), dtype=np.int32)
        chiplet_matrix_np = np.zeros((n_chiplets, n_chiplets), dtype=np.int32)
        chiplet_active_pairs: list[tuple[int, int]] = []
        logger.info(
            "Built-in all-reduce uses jax.lax.psum across all %d JAX devices; "
            "traffic matrix is not used.",
            n_chiplets,
        )
    else:
        if traffic_matrix_str is None and benchmark == "p2p":
            # p2p intentionally includes diagonal TPU entries so self and
            # same-chip chiplet cases appear in the split-pair scan.
            traffic_matrix_str = build_builtin_tpu_traffic_matrix_str(
                n_tpus,
                include_diagonal=True,
            )
            traffic_matrix_source = "builtin_all_tpu"
        elif traffic_matrix_str is None and benchmark == "p2p-rdma":
            # p2p-rdma scans P2P paths like p2p, including same-TPU cross
            # chiplet cases. Exact self chiplet pairs are filtered after
            # expansion because the Pallas remote-DMA kernel requires distinct
            # source and destination chiplets.
            traffic_matrix_str = build_builtin_tpu_traffic_matrix_str(
                n_tpus,
                include_diagonal=True,
            )
            traffic_matrix_source = "builtin_all_tpu_rdma"
        elif traffic_matrix_str is None:
            # a2a uses dense all_to_all for execution, but this matrix is still
            # useful for reporting the inter-TPU traffic pattern separately.
            traffic_matrix_str = build_builtin_tpu_traffic_matrix_str(
                n_tpus,
                include_diagonal=False,
            )
            traffic_matrix_source = "builtin_inter_tpu"
        else:
            traffic_matrix_source = "provided"

        tpu_traffic_matrix = parse_traffic_matrix(traffic_matrix_str)
        if len(tpu_traffic_matrix) != n_tpus:
            raise ValueError(
                f"Traffic matrix size {len(tpu_traffic_matrix)} != "
                f"inferred TPU chip count {n_tpus} "
                f"(topology={topology_str}, source={topology_source}). "
                "If this is a multi-host run, ensure all hosts started and "
                "JAX distributed initialization sees the full slice."
            )
        traffic_matrix_label = format_traffic_matrix_count_label(
            tpu_traffic_matrix
        )

        # Expand to chiplet level (each TPU chip has 2 chiplets). For example,
        # an 8x8 TPU matrix becomes a 16x16 JAX-device matrix.
        tpu_matrix_np = np.array(tpu_traffic_matrix, dtype=np.int32)

        diag = np.diag(tpu_matrix_np)
        if traffic_matrix_source == "builtin_inter_tpu" and benchmark == "a2a":
            logger.info(
                "Built-in a2a executes jax.lax.all_to_all across all JAX "
                "devices and reports inter-TPU ICI payload. The TPU-level "
                "matrix keeps diagonal entries disabled to describe the "
                "inter-TPU traffic pattern."
            )
        elif traffic_matrix_source == "builtin_all_tpu_rdma":
            logger.info(
                "Built-in p2p-rdma uses a %dx%d all-ones TPU traffic matrix; "
                "self chiplet pairs are skipped because Pallas remote DMA "
                "requires distinct endpoints. Same-TPU cross-chiplet and "
                "inter-TPU cases are executed as directed transfers.",
                n_tpus, n_tpus,
            )
        elif traffic_matrix_source == "builtin_inter_tpu":
            logger.info(
                "Built-in inter-TPU traffic matrix is %dx%d with diagonal "
                "entries disabled; chiplets on the same TPU chip do not "
                "communicate.",
                n_tpus, n_tpus,
            )
        elif traffic_matrix_source == "builtin_all_tpu":
            logger.info(
                "Built-in all-ones TPU traffic matrix is %dx%d with diagonal "
                "entries enabled; p2p scans self, same-chip, and inter-TPU "
                "chiplet cases. Aggregate ICI P2P bandwidth uses only "
                "inter-TPU cases.",
                n_tpus, n_tpus,
            )
        elif np.any(diag):
            self_tpus = np.where(diag)[0].tolist()
            logger.warning(
                "Traffic matrix has diagonal entries for TPU indices %s. "
                "These expand to local/intra-chip communication, not "
                "inter-chip ICI links.",
                self_tpus,
            )

        chiplet_matrix_np = expand_to_chiplet_level(tpu_matrix_np)
        if benchmark == "p2p-rdma":
            # Pallas remote DMA needs distinct source/destination chiplets.
            # Keep same-TPU cross-chiplet links, but remove exact self links.
            np.fill_diagonal(chiplet_matrix_np, 0)
        chiplet_active_pairs = get_active_pairs(chiplet_matrix_np.tolist())

        if not chiplet_active_pairs:
            raise ValueError("No active P2P pairs in traffic matrix")

    logger.info("Starting ICI benchmark")
    payload_shape_name = (
        "payload_shape_per_chiplet"
        if benchmark == "a2a" else "payload_shape_per_link"
    )
    logger.info(
        "  tpu_chip_topology=%s (%s)  tpu_chips=%d  chiplets=%d  "
        "data_size=%d bytes (%d float32 elements)  %s=(%d,%d,%d)  "
        "execution_shape=%s  traffic_units=%s  "
        "tpu_units=%d  chiplet_units=%d",
        topology_str, topology_source, n_tpus, n_chiplets, data_size_bytes,
        data_size_elements, payload_shape_name, payload_rows, PAYLOAD_MID_DIM,
        PAYLOAD_LAST_DIM, execution_shape,
        "all_reduce_participants" if benchmark == "ar" else "active_pairs",
        n_tpus if benchmark == "ar" else len(get_active_pairs(tpu_traffic_matrix)),
        n_chiplets if benchmark == "ar" else len(chiplet_active_pairs),
    )
    if benchmark == "a2a":
        logger.info(
            "  all_to_all_chunk=%d bytes (%d payload rows) per destination",
            all_to_all_chunk_bytes, all_to_all_chunk_rows,
        )

    # --- Prepare output directories ----------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{ts}_{benchmark}_topo{topology_str}_{traffic_matrix_label}_"
        f"{execution_shape}_{data_size_label}"
    )
    dirs = prepare_benchmark_dirs(result_dir, run_name, dump_hlo=dump_hlo)
    out_dir = dirs.output_dir
    metrics_dir = dirs.metrics_dir
    trace_dir = dirs.trace_dir
    dump_hlo_dir = dirs.dump_hlo_dir

    mesh = create_device_mesh(n_chiplets)
    zero_crop_enabled = detect_zero_crop_available()
    logger.info(
        "ZeroCrop live-out consumer is %s for ICI collective outputs",
        "enabled" if zero_crop_enabled else "disabled",
    )
    traffic_matrix_role = {
        "a2a": "inter_tpu_ici_bandwidth_accounting",
        "ar": "all_reduce_collective_group",
        "p2p-rdma": "pallas_remote_dma_link_scan",
    }.get(benchmark, "collective_execution_pattern")
    communication_primitive = {
        "a2a": "jax.lax.all_to_all",
        "ar": "jax.lax.psum",
        "p2p-rdma": "pallas.remote_dma",
    }.get(benchmark, "jax.lax.ragged_all_to_all")
    trace_event_kernel_terms = {
        "a2a": ["all_to_all", "all-to-all", "alltoall"],
        "ar": ["psum", "all_reduce", "all-reduce", "allreduce"],
        "p2p-rdma": ["pallas", "remote_dma", "remote-dma", "dma"],
    }.get(benchmark, ["ragged_all_to_all", "ragged-all-to-all", "all_to_all"])

    metadata = {
        "benchmark": benchmark,
        "topology": topology_str,
        "topology_source": topology_source,
        "tpu_chip_order": format_tpu_chip_order(tpu_chip_order),
        "traffic_matrix": traffic_matrix_str,
        "traffic_matrix_label": traffic_matrix_label,
        "traffic_matrix_source": traffic_matrix_source,
        "traffic_matrix_role": traffic_matrix_role,
        "communication_primitive": communication_primitive,
        "trace_event_kernel_name_filter": trace_event_kernel_terms,
        "xla_flag_profile": xla_flag_profile,
        "collective_output_consumer": (
            "ZeroCrop" if zero_crop_enabled else "ordinary_live_out"
        ),
        "zero_crop_live_out_consumer": zero_crop_enabled,
        "tpu_count": n_tpus,
        "chiplet_count": n_chiplets,
        **parsed_data_size.metadata(),
        "data_size_semantics": (
            "pre_split_payload_bytes_per_chiplet"
            if benchmark == "a2a" else "bytes_per_active_chiplet_link"
        ),
        "data_size_elements": data_size_elements,
        "payload_dtype": "float32",
        "payload_rows_per_link": payload_rows_per_link,
        "payload_shape_per_link": [
            payload_rows_per_link,
            PAYLOAD_MID_DIM,
            PAYLOAD_LAST_DIM,
        ],
        "payload_row_bytes": PAYLOAD_ROW_BYTES,
        "execution_shape": execution_shape,
        "warmup": warmup,
        "iteration": iteration,
        "dump_hlo": dump_hlo,
    }
    if benchmark == "a2a":
        metadata.update({
            "payload_rows_per_link": None,
            "payload_shape_per_link": None,
            "payload_rows_per_chiplet": payload_rows,
            "payload_shape_per_chiplet": [
                payload_rows,
                PAYLOAD_MID_DIM,
                PAYLOAD_LAST_DIM,
            ],
            "all_to_all_chunk_rows": all_to_all_chunk_rows,
            "all_to_all_chunk_bytes": all_to_all_chunk_bytes,
            "all_to_all_chunk_shape": [
                all_to_all_chunk_rows,
                PAYLOAD_MID_DIM,
                PAYLOAD_LAST_DIM,
            ],
        })

    if benchmark == "ar":
        traffic_cases = [
            TrafficCase(
                label="all_reduce",
                matrix=chiplet_matrix_np,
                active_pairs=[],
                metadata={
                    "participant_tpu_chips": n_tpus,
                    "participant_chiplets": n_chiplets,
                },
            )
        ]
    else:
        traffic_cases = build_traffic_cases(
            chiplet_matrix_np,
            execution_shape,
            tpu_chip_order=tpu_chip_order,
            topology=topology,
        )
    compiled_cache: dict[tuple[Any, ...], Any] = {}
    case_results = []
    all_durations = []
    local_ids = local_chiplet_ids()
    unrecorded_case_count = 0

    try:
        for case in traffic_cases:
            logical_pair = case.metadata.get("logical_pair")
            if logical_pair:
                logger.info(
                    "=== traffic case: %s  %s  delta=%s  "
                    "torus_manhattan=%s ===",
                    case.label,
                    logical_pair,
                    case.metadata.get("tpu_coord_delta"),
                    case.metadata.get("tpu_torus_manhattan_distance"),
                )
            else:
                logger.info("=== traffic case: %s ===", case.label)
            traffic_jnp = None
            rdma_payload = None
            if benchmark == "ar":
                # ar has one compiled executable for the whole collective.
                cache_key = (
                    "ar",
                    n_chiplets,
                    payload_rows_per_link,
                    zero_crop_enabled,
                )
                if cache_key not in compiled_cache:
                    compiled_cache[cache_key] = compile_all_reduce_kernel(
                        mesh,
                        n_chiplets,
                        payload_rows_per_link,
                        zero_crop_enabled,
                    )
                compiled_fn = compiled_cache[cache_key]

                def data_generator():
                    return (
                        make_axis_sharded_payload(
                            mesh,
                            n_chiplets,
                            payload_rows_per_link,
                        ),
                    )

            elif benchmark == "a2a":
                # a2a uses all_to_all directly; no traffic matrix argument is
                # needed at runtime.
                cache_key = ("a2a", n_chiplets, payload_rows, zero_crop_enabled)
                if cache_key not in compiled_cache:
                    compiled_cache[cache_key] = compile_all_to_all_kernel(
                        mesh,
                        n_chiplets,
                        payload_rows,
                        zero_crop_enabled,
                    )
                compiled_fn = compiled_cache[cache_key]

                def data_generator():
                    return ()
            elif benchmark == "p2p-rdma":
                if len(case.active_pairs) != 1:
                    raise ValueError(
                        "p2p-rdma split cases must contain exactly one "
                        f"chiplet link, got {len(case.active_pairs)}"
                    )
                src_chiplet, dst_chiplet = case.active_pairs[0]
                cache_key = (
                    "p2p-rdma",
                    src_chiplet,
                    dst_chiplet,
                    payload_rows_per_link,
                    zero_crop_enabled,
                )
                if cache_key not in compiled_cache:
                    compiled_cache[cache_key] = compile_remote_dma_p2p_kernel(
                        mesh,
                        n_chiplets,
                        payload_rows_per_link,
                        src_chiplet,
                        dst_chiplet,
                        zero_crop_enabled,
                    )
                compiled_fn = compiled_cache[cache_key]
                rdma_payload = make_axis_sharded_payload(
                    mesh,
                    n_chiplets,
                    payload_rows_per_link,
                )
                jax.block_until_ready(rdma_payload)

                def data_generator(payload=rdma_payload):
                    return (payload,)
            else:
                # raw/p2p use ragged_all_to_all. The same executable can be
                # reused when max_send/max_recv buffer shapes match.
                tm_rows = validate_ragged_all_to_all_bounds(
                    case.matrix,
                    payload_rows_per_link,
                    f"{benchmark} {case.label}",
                )
                max_send = max(1, int(tm_rows.sum(axis=1).max()))
                max_recv = max(1, int(tm_rows.sum(axis=0).max()))
                cache_key = ("ragged", max_send, max_recv, zero_crop_enabled)
                if cache_key not in compiled_cache:
                    compiled_cache[cache_key] = compile_traffic_matrix_kernel(
                        mesh,
                        n_chiplets,
                        max_send,
                        max_recv,
                        zero_crop_enabled,
                    )
                compiled_fn = compiled_cache[cache_key]
                traffic_jnp = jnp.asarray(tm_rows, dtype=jnp.int32)

                def data_generator(tm=traffic_jnp):
                    return (tm,)

            try:
                case_trace_dir = trace_dir
                if len(traffic_cases) > 1:
                    # split-pair scans write one trace directory per link so a
                    # slow pair can be inspected without opening a huge trace.
                    case_trace_dir = os.path.join(trace_dir, case.label)
                    os.makedirs(case_trace_dir, exist_ok=True)

                case_metadata = {
                    **metadata,
                    **case.metadata,
                    "case_label": case.label,
                    "case_active_links": (
                        n_chiplets if benchmark == "ar"
                        else len(case.active_pairs)
                    ),
                    "case_active_link_semantics": (
                        "all_reduce_participants"
                        if benchmark == "ar" else "p2p_links"
                    ),
                }
                src_tpu_id = case.metadata.get("src_tpu_id")
                dst_tpu_id = case.metadata.get("dst_tpu_id")
                if (
                    src_tpu_id is not None
                    and dst_tpu_id is not None
                    and "src_tpu_coord" not in case_metadata
                ):
                    src_coord = tpu_chip_order[int(src_tpu_id)]
                    dst_coord = tpu_chip_order[int(dst_tpu_id)]
                    case_metadata["src_tpu_coord"] = list(src_coord)
                    case_metadata["dst_tpu_coord"] = list(dst_coord)
                    case_metadata["tpu_coord_delta"] = [
                        dst_axis - src_axis
                        for src_axis, dst_axis in zip(src_coord, dst_coord)
                    ]
                    case_metadata["tpu_torus_axis_distances"] = (
                        torus_axis_distances(src_coord, dst_coord, topology)
                    )
                    case_metadata["tpu_distance_metric"] = "torus_manhattan"
                    case_metadata["tpu_torus_manhattan_distance"] = (
                        manhattan_distance(
                            src_coord,
                            dst_coord,
                            topology,
                        )
                    )

                (
                    total_case_bytes,
                    bandwidth_case_bytes,
                    bandwidth_scope,
                    bandwidth_metadata,
                ) = compute_case_traffic_bytes(
                    benchmark=benchmark,
                    case=case,
                    data_size_bytes=data_size_bytes,
                    n_chiplets=n_chiplets,
                    n_tpus=n_tpus,
                )
                case_metadata.update(bandwidth_metadata)

                test_name = f"ici_{benchmark}_{case.label}_topo{topology_str}"
                local_endpoint_role = split_case_local_endpoint_role(
                    case,
                    local_ids,
                )
                if (
                    benchmark in {"p2p", "p2p-rdma"}
                    and execution_shape == "split_pairs"
                ):
                    record_case = should_execute_split_case(case, local_ids)
                else:
                    record_case = (
                        execution_shape != "split_pairs"
                        or should_record_split_case(case, local_ids)
                    )
                if record_case:
                    trace_event_source_contains = (
                        "/ici/kernels.py" if benchmark == "p2p-rdma" else None
                    )
                    trace_event_kernel_name_contains = trace_event_kernel_terms
                    case_result = _run_benchmark_phases(
                        compiled_fn=compiled_fn,
                        data_generator=data_generator,
                        warmup=warmup,
                        iteration=iteration,
                        trace_dir=case_trace_dir,
                        metrics_dir=metrics_dir,
                        active_links=(
                            n_chiplets if benchmark == "ar"
                            else len(case.active_pairs)
                        ),
                        bytes_per_measurement=bandwidth_case_bytes,
                        total_traffic_bytes=total_case_bytes,
                        bandwidth_scope=bandwidth_scope,
                        test_name=test_name,
                        metadata={
                            **case_metadata,
                            "metrics_recorder_process_index": (
                                jax.process_index()
                            ),
                            "metrics_recorder_endpoint_role": (
                                local_endpoint_role
                            ),
                        },
                        cleanup_trace=cleanup_trace,
                        trace_event_source_contains=trace_event_source_contains,
                        trace_event_kernel_name_contains=(
                            trace_event_kernel_name_contains
                        ),
                    )
                else:
                    unrecorded_case_count += 1
                    _run_benchmark_phases_without_recording(
                        compiled_fn=compiled_fn,
                        data_generator=data_generator,
                        warmup=warmup,
                        iteration=iteration,
                    )
                    case_result = None
            finally:
                delete_device_object(traffic_jnp)
                delete_device_object(rdma_payload)
                gc.collect()

            if case_result is not None:
                case_result["traffic_case"] = format_traffic_case(case)
                case_results.append(case_result)
                all_durations.extend(case_result["durations_ms"])

        collect_hlo_dumps_if_requested(
            dump_hlo,
            dump_hlo_source_dir,
            dump_hlo_dir,
            f"ici_{benchmark}_{execution_shape}_topo{topology_str}_kernel",
            clear_source=clear_hlo_source,
        )

        if len(traffic_cases) == 1:
            results = case_results[0]
            results["output_directory"] = out_dir
            return results

        if not case_results:
            logger.info(
                "No split-pair metrics are owned by process %d; "
                "skipping aggregate metrics on this host",
                jax.process_index(),
            )
            return {
                "metadata": metadata,
                "metrics": {
                    "traffic_cases": 0,
                    "unrecorded_no_local_endpoint_cases": unrecorded_case_count,
                    "timed_iterations_kept": 0,
                    "bandwidth_scope": "not_recorded_on_this_process",
                    "metrics_recording_policy": "local_endpoint_processes",
                },
                "per_case": [],
                "output_directory": out_dir,
            }

        agg_stats = MetricsStatistics(all_durations, "duration", unit="ms")
        agg_metrics = agg_stats.serialize()
        case_bandwidths = collect_case_bandwidths(case_results)
        all_split_pair_avg_bw = average_bandwidth(case_bandwidths)
        agg_bw = all_split_pair_avg_bw
        bandwidth_scope = "average_per_split_pair"
        bandwidth_aggregation = "arithmetic_mean_of_case_bandwidths"

        if benchmark in {"p2p", "p2p-rdma"} and execution_shape == "split_pairs":
            inter_tpu_case_bandwidths = collect_case_bandwidths(
                case_results,
                inter_tpu_only=True,
            )
            inter_tpu_avg_bw = average_bandwidth(inter_tpu_case_bandwidths)
            agg_bw = inter_tpu_avg_bw
            bandwidth_scope = "average_inter_tpu_split_pair"
            bandwidth_aggregation = (
                "arithmetic_mean_of_inter_tpu_case_bandwidths"
            )
            agg_metrics["avg_all_split_pair_bandwidth_GBps"] = (
                all_split_pair_avg_bw
            )
            agg_metrics["avg_ici_p2p_bandwidth_GBps"] = inter_tpu_avg_bw
            if benchmark == "p2p-rdma":
                agg_metrics["avg_ici_p2p_rdma_bandwidth_GBps"] = (
                    inter_tpu_avg_bw
                )
            agg_metrics["ici_p2p_case_count"] = len(inter_tpu_case_bandwidths)
            agg_metrics["local_or_same_tpu_case_count"] = (
                len(case_bandwidths) - len(inter_tpu_case_bandwidths)
            )
            if not inter_tpu_case_bandwidths:
                logger.warning(
                    "%s scan did not contain inter-TPU chiplet pairs; "
                    "aggregate ICI P2P bandwidth is reported as 0.",
                    benchmark,
                )

        recorded_active_links = sum(
            int(result.get("metadata", {}).get("case_active_links", 0))
            for result in case_results
        )
        agg_metrics["total_traffic_bytes"] = (
            recorded_active_links * data_size_bytes
        )
        agg_metrics["active_links"] = recorded_active_links
        agg_metrics["traffic_cases"] = len(case_results)
        agg_metrics["discovered_traffic_cases"] = len(traffic_cases)
        agg_metrics["unrecorded_no_local_endpoint_cases"] = unrecorded_case_count
        agg_metrics["timed_iterations_kept"] = len(all_durations)
        agg_metrics["bandwidth_traffic_bytes"] = data_size_bytes
        agg_metrics["bandwidth_scope"] = bandwidth_scope
        agg_metrics["bandwidth_aggregation"] = bandwidth_aggregation
        agg_metrics["timing_source"] = "xprof_marker_device_duration"
        agg_metrics["avg_bandwidth_GBps"] = round(agg_bw, 4)
        if execution_shape == "split_pairs":
            agg_metrics["split_pair_bandwidth_grouping"] = (
                "tpu_torus_manhattan_distance_and_core_pair"
            )
            agg_metrics["split_pair_bandwidth_groups"] = (
                build_split_pair_bandwidth_groups(case_results)
            )

        if benchmark in {"p2p", "p2p-rdma"} and execution_shape == "split_pairs":
            group_prefix = "p2p_rdma" if benchmark == "p2p-rdma" else "p2p"
            agg_metrics[f"{group_prefix}_bandwidth_grouping"] = (
                "inter_tpu_torus_manhattan_distance_and_core_pair"
            )
            agg_metrics[f"{group_prefix}_bandwidth_groups"] = (
                build_split_pair_bandwidth_groups(
                    case_results,
                    inter_tpu_only=True,
                )
            )
        elif benchmark == "raw" and execution_shape == "split_pairs":
            agg_metrics[f"{benchmark}_bandwidth_grouping"] = (
                agg_metrics["split_pair_bandwidth_grouping"]
            )
            agg_metrics[f"{benchmark}_bandwidth_groups"] = (
                agg_metrics["split_pair_bandwidth_groups"]
            )

        test_name = f"ici_{benchmark}_{execution_shape}_topo{topology_str}"
        write_jsonl_metrics(metrics_dir, test_name + "_aggregate", metadata, agg_metrics)

        logger.info(
            "Aggregate: %d traffic cases, %d chiplet pairs, %s bandwidth=%.2f GB/s",
            len(traffic_cases), len(chiplet_active_pairs), bandwidth_scope, agg_bw,
        )

        return {
            "metadata": metadata,
            "metrics": agg_metrics,
            "per_case": case_results,
            "output_directory": out_dir,
        }
    finally:
        release_compiled_cache(compiled_cache)


def format_cli_summary(results: dict[str, Any]) -> dict[str, Any]:
    """Return a compact JSON-safe summary for terminal output."""
    metadata = results.get("metadata", {})
    metrics = results.get("metrics", {})
    output_directory = results.get("output_directory")

    duration_ms = {
        key.removeprefix("duration_").removesuffix("_ms"): value
        for key, value in metrics.items()
        if key.startswith("duration_") and key.endswith("_ms")
    }
    bandwidth = metrics.get("bandwidth_GBps", metrics.get("avg_bandwidth_GBps"))

    summary: dict[str, Any] = {
        "benchmark": metadata.get("benchmark"),
        "topology": {
            "tpu_chip_topology": metadata.get("topology"),
            "source": metadata.get("topology_source"),
            "tpu_chips": metadata.get("tpu_count"),
            "chiplets": metadata.get("chiplet_count"),
        },
        "traffic": {
            "matrix_source": metadata.get("traffic_matrix_source"),
            "matrix_label": metadata.get("traffic_matrix_label"),
            "matrix_role": metadata.get("traffic_matrix_role"),
            "communication_primitive": metadata.get("communication_primitive"),
            "execution_shape": metadata.get("execution_shape"),
            "active_links": metrics.get(
                "active_links",
                metadata.get("case_active_links"),
            ),
            "active_link_semantics": metadata.get("case_active_link_semantics"),
            "traffic_cases": metrics.get("traffic_cases", 1),
        },
        "data_size": {
            "input": metadata.get("data_size_input"),
            "semantics": metadata.get("data_size_semantics"),
            "elements": metadata.get("data_size_elements"),
            "bytes": metadata.get("data_size_bytes"),
            "payload_shape": (
                metadata.get("payload_shape_per_chiplet")
                or metadata.get("payload_shape_per_link")
            ),
            "all_to_all_chunk_shape": metadata.get("all_to_all_chunk_shape"),
        },
        "iterations": {
            "warmup": metadata.get("warmup"),
            "timed": metadata.get("iteration"),
            "kept": metrics.get("timed_iterations_kept"),
        },
        "metrics": {
            "duration_ms": duration_ms,
            "bandwidth_GBps": bandwidth,
            "bandwidth_scope": metrics.get("bandwidth_scope"),
            "bandwidth_aggregation": metrics.get("bandwidth_aggregation"),
            "bandwidth_traffic_bytes": metrics.get("bandwidth_traffic_bytes"),
            "bandwidth_duration_stat": metrics.get("bandwidth_duration_stat"),
            "bandwidth_duration_ms": metrics.get("bandwidth_duration_ms"),
            "bandwidth_formula": metadata.get("bandwidth_formula"),
            "bandwidth_formula_n_devices": metadata.get(
                "bandwidth_formula_n_devices"
            ),
            "bandwidth_formula_scope": metadata.get("bandwidth_formula_scope"),
            "bandwidth_formula_participants": metadata.get(
                "bandwidth_formula_participants"
            ),
            "timing_source": metrics.get("timing_source"),
            "trace_event_selection": metrics.get("trace_event_selection"),
            "total_traffic_bytes": metrics.get("total_traffic_bytes"),
        },
        "output": {
            "run_dir": output_directory,
            "metrics_dir": (
                os.path.join(output_directory, "metrics")
                if output_directory else None
            ),
            "trace_dir": (
                os.path.join(output_directory, "trace")
                if output_directory else None
            ),
            "dump_hlo_enabled": metadata.get("dump_hlo"),
            "dump_hlo_dir": (
                os.path.join(output_directory, "dump_hlo")
                if output_directory and metadata.get("dump_hlo") else None
            ),
        },
    }

    if "per_case" in results:
        summary["traffic"]["traffic_cases"] = len(results["per_case"])
    if "avg_all_split_pair_bandwidth_GBps" in metrics:
        summary["metrics"]["avg_all_split_pair_bandwidth_GBps"] = metrics.get(
            "avg_all_split_pair_bandwidth_GBps"
        )
    if "avg_ici_p2p_bandwidth_GBps" in metrics:
        summary["metrics"]["avg_ici_p2p_bandwidth_GBps"] = metrics.get(
            "avg_ici_p2p_bandwidth_GBps"
        )
    if "avg_ici_p2p_rdma_bandwidth_GBps" in metrics:
        summary["metrics"]["avg_ici_p2p_rdma_bandwidth_GBps"] = metrics.get(
            "avg_ici_p2p_rdma_bandwidth_GBps"
        )
    if "ici_p2p_case_count" in metrics:
        summary["metrics"]["ici_p2p_case_count"] = metrics.get(
            "ici_p2p_case_count"
        )
    if "local_or_same_tpu_case_count" in metrics:
        summary["metrics"]["local_or_same_tpu_case_count"] = metrics.get(
            "local_or_same_tpu_case_count"
        )
    if "split_pair_bandwidth_groups" in metrics:
        summary["metrics"]["split_pair_bandwidth_grouping"] = metrics.get(
            "split_pair_bandwidth_grouping"
        )
        summary["metrics"]["split_pair_bandwidth_groups"] = metrics.get(
            "split_pair_bandwidth_groups"
        )
    if "p2p_bandwidth_groups" in metrics:
        summary["metrics"]["p2p_bandwidth_grouping"] = metrics.get(
            "p2p_bandwidth_grouping"
        )
        summary["metrics"]["p2p_bandwidth_groups"] = metrics.get(
            "p2p_bandwidth_groups"
        )
    if "p2p_rdma_bandwidth_groups" in metrics:
        summary["metrics"]["p2p_rdma_bandwidth_grouping"] = metrics.get(
            "p2p_rdma_bandwidth_grouping"
        )
        summary["metrics"]["p2p_rdma_bandwidth_groups"] = metrics.get(
            "p2p_rdma_bandwidth_groups"
        )
    if "raw_bandwidth_groups" in metrics:
        summary["metrics"]["raw_bandwidth_grouping"] = metrics.get(
            "raw_bandwidth_grouping"
        )
        summary["metrics"]["raw_bandwidth_groups"] = metrics.get(
            "raw_bandwidth_groups"
        )

    return summary
