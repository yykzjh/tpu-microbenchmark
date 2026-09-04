"""Main ICI benchmark orchestration and CLI summary formatting."""

from __future__ import annotations

import gc
import logging
import os
import time
from datetime import datetime
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.distributed import global_state as distributed_global_state
from jax.sharding import PartitionSpec as P

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
    create_parallel_chiplet_mesh,
    compile_remote_dma_p2p_kernel,
    compile_self_copy_kernel,
    compile_traffic_matrix_kernel,
    create_device_mesh,
    make_axis_sharded_payload,
)
from ici.phases import (
    _run_benchmark_phases,
    _run_benchmark_phases_without_recording,
)
from ici.traffic import (
    TpuBlockRange,
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
    neighboring_p2p_chiplet_pairs,
    parse_traffic_matrix,
    parse_tpu_block_range,
    should_record_split_case,
    split_case_local_endpoint_role,
    torus_axis_distances,
    validate_ragged_all_to_all_bounds,
)
from utils.metrics import MetricsStatistics, write_jsonl_metrics
from utils.profiling import collect_hlo_dumps_if_requested, delete_device_object
from utils.runtime import (
    default_profile_result_dir,
    prepare_benchmark_dirs,
    validate_non_negative,
    validate_positive,
)
from utils.units import parse_data_size

logger = logging.getLogger(__name__)


def _wait_at_process_barrier(name: str) -> None:
    """Synchronize every initialized JAX process, including idle block hosts."""
    client = distributed_global_state.client
    if client is None:
        raise RuntimeError("JAX distributed client is unavailable for ICI barrier")
    client.wait_at_barrier(name, timeout_in_ms=600_000)


def _case_completion_key(case_index: int) -> str:
    """Return the coordination-service key for one benchmark case."""
    return f"commpilot_ici_case_{case_index}_work_complete"


def _signal_case_completion(case_index: int) -> None:
    """Tell idle Slice hosts that all participants completed one case."""
    client = distributed_global_state.client
    if client is None:
        raise RuntimeError("JAX distributed client is unavailable for ICI signal")
    client.key_value_set(_case_completion_key(case_index), "done")


def _wait_for_case_completion(
    case_index: int,
    timeout_seconds: int = 1_800,
) -> None:
    """Wait without holding idle hosts in a long coordination barrier.

    TPU compilation can exceed the coordination service's effective barrier
    timeout. Idle processes therefore poll a completion key while the selected
    block executes, then join the short completion barrier with active hosts.
    """
    client = distributed_global_state.client
    if client is None:
        raise RuntimeError("JAX distributed client is unavailable for ICI signal")
    key = _case_completion_key(case_index)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            if client.key_value_try_get(key) == "done":
                return
        except Exception as exc:
            if "NOT_FOUND" not in str(exc):
                raise
        time.sleep(1)
    raise TimeoutError(
        f"Timed out waiting for active TPU block to complete case {case_index}"
    )


def iter_commpilot_log_metrics(results: dict[str, Any]):
    """Yield catalog metric name, value, and dimensions for one ICI run."""
    metadata = results.get("metadata", {})
    benchmark = metadata.get("benchmark")
    if benchmark == "p2p":
        for result in results.get("per_case", []):
            case_metadata = result.get("metadata", {})
            src_tpu = case_metadata.get("src_tpu_id")
            dst_tpu = case_metadata.get("dst_tpu_id")
            src_core = case_metadata.get("src_core")
            dst_core = case_metadata.get("dst_core")
            if src_tpu is None or dst_tpu is None:
                continue
            if int(src_tpu) == int(dst_tpu):
                if src_core == dst_core:
                    continue
                metric = "p2p_die_to_die"
                link_type = "die_to_die"
            else:
                metric = "p2p_chip_to_chip"
                link_type = "chip_to_chip"
            bandwidth = result.get("metrics", {}).get("bandwidth_GBps")
            if bandwidth is None:
                continue
            yield metric, bandwidth, {
                "runtime_scope": metadata.get("runtime_scope"),
                "p2p_pair_mode": metadata.get("p2p_pair_mode", "all"),
                "slice_topology": metadata.get("slice_topology"),
                "topology": metadata.get("topology"),
                "block_range": metadata.get("block_range"),
                "src_tpu_id": src_tpu,
                "dst_tpu_id": dst_tpu,
                "src_core": src_core,
                "dst_core": dst_core,
                "link_type": link_type,
                "src_tpu_coord": case_metadata.get("src_tpu_coord"),
                "dst_tpu_coord": case_metadata.get("dst_tpu_coord"),
                "axis_distances": case_metadata.get("tpu_torus_axis_distances"),
                "distance": case_metadata.get("tpu_torus_manhattan_distance"),
            }
        return

    metrics = results.get("metrics", {})
    bandwidth = metrics.get("bandwidth_GBps", metrics.get("avg_bandwidth_GBps"))
    if bandwidth is None:
        return
    if benchmark == "a2a":
        metric = (
            "alltoall_parallel_bisection"
            if metadata.get("all_to_all_parallel") else "alltoall_bisection"
        )
    elif benchmark == "ar":
        metric = "allreduce_parallel" if metadata.get("all_reduce_parallel") else "allreduce"
    else:
        return
    yield metric, bandwidth, {
        "runtime_scope": metadata.get("runtime_scope"),
        "slice_topology": metadata.get("slice_topology"),
        "topology": metadata.get("topology"),
        "block_range": metadata.get("block_range"),
        "tpu_count": metadata.get("tpu_count"),
        "chiplet_count": metadata.get("chiplet_count"),
        "bandwidth_accounting_unit": metadata.get("bandwidth_accounting_unit"),
        "bandwidth_accounting_mode": metadata.get("bandwidth_accounting_mode"),
        "bandwidth_scope": metrics.get("bandwidth_scope"),
        "bandwidth_traffic_bytes": metrics.get("bandwidth_traffic_bytes"),
        "bandwidth_formula": metadata.get("bandwidth_formula"),
        "bisection_traffic_bytes": metadata.get("bisection_traffic_bytes"),
        "ranks_per_group": metadata.get("bandwidth_formula_n_devices_per_group"),
        "parallel_groups": metadata.get("bandwidth_formula_parallel_groups"),
    }


def is_self_copy_split_case(benchmark: str, execution_shape: str, case: TrafficCase) -> bool:
    """Return whether a raw/p2p split case should run as local HBM copy."""
    return (
        benchmark in {"raw", "p2p"}
        and execution_shape == "split_pairs"
        and len(case.active_pairs) == 1
        and case.active_pairs[0][0] == case.active_pairs[0][1]
    )


def run_ici(
    benchmark: str,
    traffic_matrix_str: str | None,
    data_size: str | int,
    execution_shape: str,
    warmup: int,
    iteration: int,
    result_dir: str | None,
    profile_artifacts: bool = False,
    xla_flag_profile: str | None = None,
    dump_hlo_source_dir: str | None = None,
    clear_hlo_source: bool = False,
    ar_parallel: bool = False,
    a2a_parallel: bool = False,
    use_xprof_timing: bool = False,
    runtime_scope: str = "slice",
    block_range: str | None = None,
    p2p_pair_mode: str = "all",
) -> dict[str, Any]:
    """Run ICI link performance benchmark.

    Phases per measurement:
      1. **Warmup** -- prime the TPU and let XLA compilation settle.
      2. **Timed iterations** -- synchronized CPU timing by default; optional Xprof.
      3. **Profile artifacts** -- retain Xprof traces and HLO dumps when enabled.

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
        profile_artifacts: retain Xprof traces and HLO dumps for analysis.
        xla_flag_profile: TPU/XLA flag profile selected by the CLI entrypoint.
        dump_hlo_source_dir: persistent XLA dump directory configured before
            JAX import as part of ``--profile`` output.
        clear_hlo_source: whether copied HLO dump files should be removed from
            a caller-provided source directory.
        ar_parallel: for ``benchmark="ar"``, split the mesh by chiplet index
            and run two all-reduce groups in parallel across TPU chips.
        a2a_parallel: for ``benchmark="a2a"``, split the mesh by chiplet index
            and run two all-to-all groups in parallel across TPU chips.
        runtime_scope: ``local`` selects only devices addressable by this host;
            ``slice`` selects the global devices from all distributed processes.
        block_range: optional x,y,z half-open TPU chip range. Distributed JAX
            still initializes every Slice host, while the benchmark mesh uses
            only devices whose chip coordinates fall in this block.

    Returns:
        Dict with ``metadata``, ``metrics``, and ``output_directory``. Split
        pair runs also include ``per_case``.
    """
    # --- Validate inputs ---------------------------------------------------
    if benchmark not in {"raw", "p2p", "p2p-rdma", "a2a", "ar"}:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    if p2p_pair_mode not in {"all", "neighbors"}:
        raise ValueError(f"Unknown P2P pair mode: {p2p_pair_mode}")
    if p2p_pair_mode != "all" and benchmark != "p2p":
        raise ValueError("P2P pair selection is only supported by p2p")
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
    if ar_parallel and benchmark != "ar":
        raise ValueError("--parallel is only supported by ar benchmark")
    if a2a_parallel and benchmark != "a2a":
        raise ValueError("a2a_parallel is only supported by a2a benchmark")
    if runtime_scope not in {"local", "slice"}:
        raise ValueError(f"Unknown runtime scope: {runtime_scope}")
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
    if profile_artifacts and result_dir is None:
        result_dir = default_profile_result_dir("ici")
    validate_positive("iteration", iteration)

    slice_devices = list(
        jax.local_devices() if runtime_scope == "local" else jax.devices()
    )
    slice_topology, topology_source, slice_tpu_chip_order = infer_tpu_topology(
        slice_devices,
        normalize_origin=runtime_scope == "local",
    )
    selected_block: TpuBlockRange = parse_tpu_block_range(
        block_range,
        slice_topology,
    )
    selected_chip_indices = [
        index
        for index, coord in enumerate(slice_tpu_chip_order)
        if selected_block.contains(coord)
    ]
    if len(selected_chip_indices) != selected_block.chip_count:
        raise ValueError(
            f"--block-range {selected_block.spec} selected "
            f"{len(selected_chip_indices)} TPU chips, expected "
            f"{selected_block.chip_count}"
        )
    benchmark_devices = [
        device
        for chip_index in selected_chip_indices
        for device in slice_devices[
            chip_index * CHIPLETS_PER_TPU:(chip_index + 1) * CHIPLETS_PER_TPU
        ]
    ]
    topology = selected_block.shape
    tpu_chip_order = [
        selected_block.normalize(slice_tpu_chip_order[index])
        for index in selected_chip_indices
    ]
    topology_str = format_topology(topology)
    slice_topology_str = format_topology(slice_topology)
    n_tpus = topology[0] * topology[1] * topology[2]
    n_chiplets = n_tpus * CHIPLETS_PER_TPU
    metrics_recorder_process_index = min(
        int(device.process_index) for device in benchmark_devices
    )
    all_to_all_chunk_rows = None
    all_to_all_chunk_bytes = None
    if benchmark == "a2a":
        all_to_all_group_size = n_tpus if a2a_parallel else n_chiplets
        if payload_rows % all_to_all_group_size != 0:
            raise ValueError(
                "a2a --data-size is the pre-split payload bytes per chiplet "
                "and must divide evenly into one all_to_all chunk per JAX "
                f"device. Got payload_rows={payload_rows}, "
                f"n_devices_per_group={all_to_all_group_size}, "
                f"row_bytes={PAYLOAD_ROW_BYTES}. Use a size divisible by "
                f"{all_to_all_group_size * PAYLOAD_ROW_BYTES} bytes."
            )
        all_to_all_chunk_rows = payload_rows // all_to_all_group_size
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
            # Keep diagonal TPU entries so same-chip cross-chiplet links are
            # measured; exact chiplet self-copies are removed after expansion.
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
                "TPU entries enabled; p2p scans same-chip cross-chiplet and "
                "inter-TPU links, excluding exact self-copies. Aggregate ICI "
                "P2P bandwidth uses only "
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
        if benchmark == "p2p" and p2p_pair_mode == "neighbors":
            chiplet_matrix_np[:] = 0
            pairs = neighboring_p2p_chiplet_pairs(
                [slice_tpu_chip_order[index] for index in selected_chip_indices],
                slice_topology,
            )
            for src, dst in pairs:
                chiplet_matrix_np[src, dst] = 1
            tpu_traffic_matrix = [[0] * n_tpus for _ in range(n_tpus)]
            for src, dst in pairs:
                tpu_traffic_matrix[src // 2][dst // 2] = 1
            traffic_matrix_source = "builtin_neighbor_tpu"
            traffic_matrix_label = format_traffic_matrix_count_label(tpu_traffic_matrix)
            logger.info(
                "P2P neighbors: all %d directed die-to-die links; %d "
                "directed adjacent-chip core0-to-core0 links; no self copies",
                n_tpus * 2, len(pairs) - n_tpus * 2,
            )
        if benchmark == "a2a" and a2a_parallel:
            for src_chiplet in range(n_chiplets):
                for dst_chiplet in range(n_chiplets):
                    if (
                        src_chiplet % CHIPLETS_PER_TPU
                        != dst_chiplet % CHIPLETS_PER_TPU
                    ):
                        chiplet_matrix_np[src_chiplet, dst_chiplet] = 0
        if benchmark in {"p2p", "p2p-rdma"}:
            # Self-copy measures local HBM rather than a communication link;
            # Pallas remote DMA likewise requires distinct endpoints. Keep
            # same-TPU cross-chiplet links but remove exact self links.
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
        "  slice_topology=%s  block_range=%s  "
        "tpu_chip_topology=%s (%s)  tpu_chips=%d  chiplets=%d  "
        "data_size=%d bytes (%d float32 elements)  %s=(%d,%d,%d)  "
        "execution_shape=%s  traffic_units=%s  "
        "tpu_units=%d  chiplet_units=%d",
        slice_topology_str, selected_block.spec,
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
    block_label = selected_block.label
    run_name = (
        f"{ts}_{benchmark}_topo{topology_str}_block_{block_label}_"
        f"{traffic_matrix_label}_"
        f"{execution_shape}_{data_size_label}"
    )
    dirs = prepare_benchmark_dirs(
        result_dir,
        run_name,
        dump_hlo=profile_artifacts,
        create_trace=profile_artifacts,
    )
    out_dir = dirs.output_dir
    metrics_dir = dirs.metrics_dir
    trace_dir = dirs.trace_dir
    dump_hlo_dir = dirs.dump_hlo_dir

    parallel_collective = (
        (benchmark == "ar" and ar_parallel)
        or (benchmark == "a2a" and a2a_parallel)
    )
    mesh = (
        create_parallel_chiplet_mesh(n_chiplets, benchmark_devices)
        if parallel_collective
        else create_device_mesh(n_chiplets, benchmark_devices)
    )
    # All kernels return their actual device result. Synchronizing that result
    # keeps communication live without a separately timed ZeroCrop FFI call.
    traffic_matrix_role = {
        "a2a": "uniform_alltoall_one_way_bisection_accounting",
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
        "p2p_pair_mode": p2p_pair_mode if benchmark == "p2p" else None,
        "runtime_scope": runtime_scope,
        "slice_topology": slice_topology_str,
        "topology": topology_str,
        "topology_source": topology_source,
        "block_range": selected_block.spec,
        "block_label": block_label,
        "block_origin": list(selected_block.origin),
        "block_shape": list(selected_block.shape),
        "tpu_chip_order": format_tpu_chip_order(tpu_chip_order),
        "traffic_matrix": traffic_matrix_str,
        "traffic_matrix_label": traffic_matrix_label,
        "traffic_matrix_source": traffic_matrix_source,
        "traffic_matrix_role": traffic_matrix_role,
        "communication_primitive": communication_primitive,
        "trace_event_kernel_name_filter": trace_event_kernel_terms,
        "xla_flag_profile": xla_flag_profile,
        "collective_output_consumer": "ordinary_live_out",
        "zero_crop_live_out_consumer": False,
        "tpu_count": n_tpus,
        "chiplet_count": n_chiplets,
        "all_reduce_parallel": bool(ar_parallel),
        "all_to_all_parallel": bool(a2a_parallel),
        "metrics_recorder_process_index": metrics_recorder_process_index,
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
        "profile_artifacts": profile_artifacts,
        "dump_hlo_dir": dump_hlo_dir,
        "timing_mode": "xprof" if use_xprof_timing else "cpu",
        "ar_parallel": ar_parallel if benchmark == "ar" else None,
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
            "all_to_all_mesh_shape": (
                [n_tpus, CHIPLETS_PER_TPU]
                if a2a_parallel else [n_chiplets]
            ),
            "all_to_all_mesh_axes": (
                ["d", "chiplet"] if a2a_parallel else ["d"]
            ),
            "all_to_all_collective_axis": "d",
            "all_to_all_chiplet_axis_semantics": (
                "replicated_parallel_groups" if a2a_parallel else "single_group"
            ),
        })

    if benchmark == "ar":
        if ar_parallel:
            metadata.update({
                "all_reduce_mesh_shape": [n_tpus, CHIPLETS_PER_TPU],
                "all_reduce_mesh_axes": ["d", "chiplet"],
                "all_reduce_collective_axis": "d",
                "all_reduce_payload_partition_spec": "P('d', None, None)",
                "all_reduce_chiplet_axis_semantics": "replicated_parallel_groups",
            })
        else:
            metadata.update({
                "all_reduce_mesh_shape": [n_chiplets],
                "all_reduce_mesh_axes": ["d"],
                "all_reduce_collective_axis": "d",
                "all_reduce_payload_partition_spec": "P('d')",
                "all_reduce_chiplet_axis_semantics": "single_group",
            })
        traffic_cases = [
            TrafficCase(
                label="all_reduce",
                matrix=chiplet_matrix_np,
                active_pairs=[],
                metadata={
                    "participant_tpu_chips": n_tpus,
                    "participant_chiplets": n_chiplets,
                    "parallel_group_count": (
                        CHIPLETS_PER_TPU if ar_parallel else 1
                    ),
                    "parallel_group_size": (
                        n_tpus if ar_parallel else n_chiplets
                    ),
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
    # Keep only one send-buffer capacity across split pairs, rather than
    # uploading GiB inputs for every link or retaining unbounded raw buffers.
    ragged_buffers: list[Any] = [None]
    ragged_buffer_shape: tuple[int, int] | None = None
    case_results = []
    all_durations = []
    local_ids = local_chiplet_ids(benchmark_devices)
    unrecorded_case_count = 0

    if runtime_scope == "slice" and jax.process_count() > 1 and not local_ids:
        logger.info(
            "This process owns no TPU chip in block %s; waiting while the "
            "selected block is tested",
            selected_block.spec,
        )
        for case_index in range(len(traffic_cases)):
            _wait_at_process_barrier(f"ici_case_{case_index}_start")
            _wait_for_case_completion(case_index)
            _wait_at_process_barrier(f"ici_case_{case_index}_complete")
        return {
            "metadata": metadata,
            "metrics": {
                "traffic_cases": 0,
                "timed_iterations_kept": 0,
                "bandwidth_scope": "not_in_selected_block",
            },
            "per_case": [],
            "output_directory": out_dir,
        }

    try:
        for case_index, case in enumerate(traffic_cases):
            if runtime_scope == "slice" and jax.process_count() > 1:
                _wait_at_process_barrier(f"ici_case_{case_index}_start")
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
            all_reduce_payload = None
            all_to_all_payload = None
            if benchmark == "ar":
                # ar has one compiled executable for the whole collective.
                cache_key = (
                    "ar",
                    n_chiplets,
                    payload_rows_per_link,
                    ar_parallel,
                )
                if cache_key not in compiled_cache:
                    compiled_cache[cache_key] = compile_all_reduce_kernel(
                        mesh,
                        n_chiplets,
                        payload_rows_per_link,
                        parallel=ar_parallel,
                    )
                compiled_fn = compiled_cache[cache_key]
                # The input is immutable and the kernel does not donate it.
                # Preparing it for every invocation adds large host allocation
                # and H2D skew between processes, which the first process then
                # observes as collective wait time inside its CPU timer.
                all_reduce_payload = make_axis_sharded_payload(
                    mesh,
                    n_chiplets,
                    payload_rows_per_link,
                    partition_spec=(
                        P("d", None, None) if ar_parallel else P("d")
                    ),
                    row_shard_count=(n_tpus if ar_parallel else n_chiplets),
                )
                jax.block_until_ready(all_reduce_payload)

                def data_generator(payload=all_reduce_payload):
                    return (payload,)

            elif benchmark == "a2a":
                # a2a uses all_to_all directly; no traffic matrix argument is
                # needed at runtime.
                all_to_all_group_size = n_tpus if a2a_parallel else n_chiplets
                cache_key = (
                    "a2a",
                    n_chiplets,
                    payload_rows,
                    a2a_parallel,
                )
                if cache_key not in compiled_cache:
                    compiled_cache[cache_key] = compile_all_to_all_kernel(
                        mesh,
                        all_to_all_group_size,
                        payload_rows,
                        parallel=a2a_parallel,
                    )
                compiled_fn = compiled_cache[cache_key]

                all_to_all_payload = make_axis_sharded_payload(
                    mesh, n_chiplets, payload_rows,
                    partition_spec=P("d", None, None) if a2a_parallel else P("d"),
                    row_shard_count=all_to_all_group_size,
                )
                jax.block_until_ready(all_to_all_payload)

                def data_generator(payload=all_to_all_payload):
                    return (payload,)
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
                )
                if cache_key not in compiled_cache:
                    compiled_cache[cache_key] = compile_remote_dma_p2p_kernel(
                        mesh,
                        n_chiplets,
                        payload_rows_per_link,
                        src_chiplet,
                        dst_chiplet,
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
            elif is_self_copy_split_case(benchmark, execution_shape, case):
                src_chiplet, _ = case.active_pairs[0]
                cache_key = (
                    "self-copy",
                    src_chiplet,
                    payload_rows_per_link,
                )
                if cache_key not in compiled_cache:
                    compiled_cache[cache_key] = compile_self_copy_kernel(
                        mesh,
                        n_chiplets,
                        payload_rows_per_link,
                        src_chiplet,
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
                cache_key = ("ragged", max_send, max_recv)
                if cache_key not in compiled_cache:
                    compiled_cache[cache_key] = compile_traffic_matrix_kernel(
                        mesh,
                        n_chiplets,
                        max_send,
                        max_recv,
                    )
                compiled_fn = compiled_cache[cache_key]
                traffic_jnp = jnp.asarray(tm_rows, dtype=jnp.int32)
                if ragged_buffer_shape != (max_send, max_recv):
                    delete_device_object(ragged_buffers)
                    ragged_buffers = [None]
                    ragged_buffers[0] = make_axis_sharded_payload(mesh, n_chiplets, max_send)
                    ragged_buffer_shape = (max_send, max_recv)
                ragged_payload = ragged_buffers[0]
                jax.block_until_ready((traffic_jnp, ragged_payload))

                def data_generator(tm=traffic_jnp, payload=ragged_payload):
                    return (tm, payload)

            try:
                case_trace_dir = trace_dir
                if len(traffic_cases) > 1 and trace_dir:
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
                        if benchmark == "ar"
                        else (
                            "all_to_all_directed_inter_tpu_pairs"
                            if benchmark == "a2a" else "p2p_links"
                        )
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
                    ar_parallel=ar_parallel,
                    a2a_parallel=a2a_parallel,
                )
                case_metadata.update(bandwidth_metadata)
                if is_self_copy_split_case(benchmark, execution_shape, case):
                    case_metadata.update({
                        "communication_primitive": "jax_elementwise_add",
                        "traffic_matrix_role": "self_hbm_read_write",
                        "pair_scope": "self_chiplet_hbm_read_write",
                        "bandwidth_formula": "data_size",
                        "bandwidth_formula_scope": (
                            "single_chiplet_hbm_read_write_effective"
                        ),
                    })

                test_name = (
                    f"ici_{benchmark}_{case.label}_topo{topology_str}_"
                    f"block_{block_label}"
                )
                local_endpoint_role = split_case_local_endpoint_role(
                    case,
                    local_ids,
                )
                if (
                    benchmark in {"p2p", "p2p-rdma"}
                    and execution_shape == "split_pairs"
                ):
                    record_case = should_record_split_case(case, local_ids)
                else:
                    record_case = (
                        jax.process_index() == metrics_recorder_process_index
                    )
                if record_case:
                    trace_event_source_contains = (
                        "/ici/kernels.py" if benchmark == "p2p-rdma" else None
                    )
                    if is_self_copy_split_case(benchmark, execution_shape, case):
                        trace_event_kernel_name_contains = None
                    else:
                        trace_event_kernel_name_contains = (
                            trace_event_kernel_terms
                        )
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
                        trace_event_source_contains=trace_event_source_contains,
                        trace_event_kernel_name_contains=(
                            trace_event_kernel_name_contains
                        ),
                        use_xprof_timing=use_xprof_timing,
                        profile_artifacts=profile_artifacts,
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
                delete_device_object(all_reduce_payload)
                delete_device_object(all_to_all_payload)
                gc.collect()

            if runtime_scope == "slice" and jax.process_count() > 1:
                if jax.process_index() == metrics_recorder_process_index:
                    _signal_case_completion(case_index)
                _wait_at_process_barrier(f"ici_case_{case_index}_complete")

            if case_result is not None:
                case_result["traffic_case"] = format_traffic_case(case)
                case_results.append(case_result)
                all_durations.extend(case_result["durations_ms"])

        collect_hlo_dumps_if_requested(
            profile_artifacts,
            dump_hlo_source_dir,
            dump_hlo_dir,
            f"ici_{benchmark}_{execution_shape}_topo{topology_str}_"
            f"block_{block_label}_kernel",
            clear_source=clear_hlo_source,
        )

        if not case_results:
            logger.info(
                "No benchmark metrics are owned by process %d; "
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

        if len(traffic_cases) == 1:
            results = case_results[0]
            results["output_directory"] = out_dir
            return results

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
        agg_metrics["timing_source"] = (
            "xprof_marker_device_duration"
            if use_xprof_timing else "cpu_wall_clock_with_block_until_ready"
        )
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
        delete_device_object(ragged_buffers)
        release_compiled_cache(compiled_cache)


def format_cli_summary(results: dict[str, Any]) -> dict[str, Any]:
    """Return a compact JSON-safe summary for terminal output."""
    metadata = results.get("metadata", {})
    metrics = results.get("metrics", {})

    duration_ms = {
        key.removeprefix("duration_").removesuffix("_ms"): value
        for key, value in metrics.items()
        if key.startswith("duration_") and key.endswith("_ms")
    }
    bandwidth = metrics.get("bandwidth_GBps", metrics.get("avg_bandwidth_GBps"))

    summary: dict[str, Any] = {
        "benchmark": metadata.get("benchmark"),
        "topology": {
            "slice_topology": metadata.get("slice_topology"),
            "tpu_chip_topology": metadata.get("topology"),
            "source": metadata.get("topology_source"),
            "block_range": metadata.get("block_range"),
            "block_origin": metadata.get("block_origin"),
            "block_shape": metadata.get("block_shape"),
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
