"""Topology, traffic-matrix, and bandwidth metadata helpers for ICI."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np

from ici.constants import (
    CHIPLETS_PER_TPU,
    CORE_PAIR_LABELS,
    PAYLOAD_ROW_BYTES,
    RAGGED_INDEX_MAX,
)

logger = logging.getLogger(__name__)


def neighboring_p2p_chiplet_pairs(
    chip_order: list[tuple[int, int, int]],
    slice_topology: tuple[int, int, int],
) -> list[tuple[int, int]]:
    """Directed die links plus core-0 links to physical Slice neighbors.

    Coordinates belong to the full Slice, even when chip_order is a subset.
    Slices up to 4x4x4 are meshes without wraparound links. Larger Slice
    adjacency must be confirmed before using this restricted test mode.
    """
    if any(dim > 4 for dim in slice_topology):
        raise ValueError(
            "--p2p-pair-mode neighbors requires a Slice no larger than "
            "4x4x4; larger-Slice physical adjacency is not configured"
        )
    pairs = []
    for src, src_coord in enumerate(chip_order):
        pairs.extend([(2 * src, 2 * src + 1), (2 * src + 1, 2 * src)])
        for dst, dst_coord in enumerate(chip_order):
            if src != dst and sum(
                abs(a - b) for a, b in zip(src_coord, dst_coord)
            ) == 1:
                pairs.append((2 * src, 2 * dst))
    return pairs


@dataclass
class TrafficCase:
    """One concrete traffic matrix execution case."""

    label: str
    matrix: np.ndarray
    active_pairs: list[tuple[int, int]]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TpuBlockRange:
    """One half-open cuboid over TPU chip coordinates."""

    axes: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]

    @property
    def origin(self) -> tuple[int, int, int]:
        return tuple(start for start, _ in self.axes)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(end - start for start, end in self.axes)

    @property
    def chip_count(self) -> int:
        x, y, z = self.shape
        return x * y * z

    @property
    def spec(self) -> str:
        return ",".join(f"{start}:{end}" for start, end in self.axes)

    @property
    def label(self) -> str:
        return "_".join(
            f"{axis}{start}-{end}"
            for axis, (start, end) in zip("xyz", self.axes)
        )

    def contains(self, coord: tuple[int, int, int]) -> bool:
        return all(
            start <= value < end
            for value, (start, end) in zip(coord, self.axes)
        )

    def normalize(self, coord: tuple[int, int, int]) -> tuple[int, int, int]:
        return tuple(
            value - start
            for value, (start, _) in zip(coord, self.axes)
        )


_TPU_BLOCK_RANGE = re.compile(
    r"^(\d+):(\d+),(\d+):(\d+),(\d+):(\d+)$"
)


def parse_tpu_block_range(
    value: str | None,
    topology: tuple[int, int, int],
) -> TpuBlockRange:
    """Parse and validate an x,y,z half-open TPU chip block range."""
    if value is None:
        return TpuBlockRange(tuple((0, dim) for dim in topology))
    match = _TPU_BLOCK_RANGE.fullmatch(value.strip())
    if match is None:
        raise ValueError(
            "--block-range must use x,y,z half-open slices such as "
            "0:2,0:2,2:4"
        )
    numbers = tuple(int(item) for item in match.groups())
    axes = tuple(
        (numbers[index], numbers[index + 1])
        for index in range(0, len(numbers), 2)
    )
    for axis_name, (start, end), dimension in zip("xyz", axes, topology):
        if start < 0 or start >= end or end > dimension:
            raise ValueError(
                f"--block-range axis {axis_name}={start}:{end} is outside "
                f"slice dimension 0:{dimension}"
            )
    return TpuBlockRange(axes)


def format_active_pairs_compact(active_pairs: list[tuple[int, int]]) -> str:
    """Return comma-separated src->dst pairs for compact JSON output."""
    return ",".join(f"{src}->{dst}" for src, dst in active_pairs)


def format_tpu_chip_order(
    chip_order: list[tuple[int, int, int]],
) -> list[list[int]]:
    """Return TPU chip coordinate order as JSON-friendly coordinate arrays."""
    return [list(coord) for coord in chip_order]


def format_coord_label(coord: tuple[int, int, int]) -> str:
    """Return a filesystem-safe TPU coordinate label."""
    return f"x{coord[0]}_y{coord[1]}_z{coord[2]}"


def format_endpoint_label(coord: tuple[int, int, int], core: int) -> str:
    """Return a filesystem-safe TPU chiplet endpoint label."""
    return f"{format_coord_label(coord)}_core{core}"


def format_endpoint_display(coord: tuple[int, int, int], core: int) -> str:
    """Return a readable TPU chiplet endpoint display string."""
    return f"(x={coord[0]},y={coord[1]},z={coord[2]},core={core})"


def torus_axis_distances(
    src_coord: tuple[int, int, int],
    dst_coord: tuple[int, int, int],
    topology: tuple[int, int, int],
) -> list[int]:
    """Return per-axis torus-aware distances between TPU chip coordinates."""
    distances = []
    for src, dst, dim in zip(src_coord, dst_coord, topology):
        axis_distance = abs(src - dst)
        distances.append(min(axis_distance, dim - axis_distance))
    return distances


def format_traffic_case(case: TrafficCase) -> dict[str, Any]:
    """Return metadata for a traffic case without expanded pair arrays."""
    return {
        "label": case.label,
        "active_links": len(case.active_pairs),
        "active_pairs": format_active_pairs_compact(case.active_pairs),
        **case.metadata,
    }


def local_chiplet_ids(devices: list[Any] | None = None) -> set[int]:
    """Return selected-mesh indices addressable by this process."""
    current_process = jax.process_index()
    return {
        index
        for index, device in enumerate(jax.devices() if devices is None else devices)
        if int(device.process_index) == current_process
    }


def should_record_split_case(case: TrafficCase, local_ids: set[int]) -> bool:
    """Return whether this process owns metrics for a split-pair case."""
    if len(case.active_pairs) != 1:
        return True
    src_chiplet, _ = case.active_pairs[0]
    # Split-pair P2P/raw scans are directed links. Use the source chiplet as
    # the owner so exactly one host writes metrics for each pair while every
    # host still executes the collective calls needed by multi-host JAX.
    return int(src_chiplet) in local_ids


def should_execute_split_case(case: TrafficCase, local_ids: set[int]) -> bool:
    """Return whether this process has a local endpoint in a split-pair case."""
    if len(case.active_pairs) != 1:
        return True
    src_chiplet, dst_chiplet = case.active_pairs[0]
    # P2P split-pair cases only need to be launched on hosts that own either
    # endpoint. A host that owns neither endpoint would only collect local
    # marker overhead and must be excluded from both execution and metrics.
    return int(src_chiplet) in local_ids or int(dst_chiplet) in local_ids


def split_case_local_endpoint_role(
    case: TrafficCase,
    local_ids: set[int],
) -> str:
    """Return this process's endpoint role in one split-pair case."""
    if len(case.active_pairs) != 1:
        return "not_split_pair"
    src_chiplet, dst_chiplet = case.active_pairs[0]
    has_src = int(src_chiplet) in local_ids
    has_dst = int(dst_chiplet) in local_ids
    if has_src and has_dst:
        return "src_and_dst"
    if has_src:
        return "src"
    if has_dst:
        return "dst"
    return "none"


def manhattan_distance(
    src_coord: tuple[int, int, int],
    dst_coord: tuple[int, int, int],
    topology: tuple[int, int, int],
) -> int:
    """Return torus-aware Manhattan distance between TPU chip coordinates."""
    return sum(torus_axis_distances(src_coord, dst_coord, topology))


def average_bandwidth(values: list[float]) -> float:
    """Return a rounded average bandwidth for a group."""
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def summarize_bandwidth_group(values: list[float]) -> dict[str, Any]:
    """Return compact average-bandwidth summary for one group."""
    return {
        "count": len(values),
        "avg_bandwidth_GBps": average_bandwidth(values),
    }


def case_bandwidth(result: dict[str, Any]) -> float | None:
    """Return per-case bandwidth if the result carries one."""
    bandwidth = result.get("metrics", {}).get("bandwidth_GBps")
    return None if bandwidth is None else float(bandwidth)


def is_inter_tpu_split_case(result: dict[str, Any]) -> bool:
    """Return whether a split-pair result crosses two TPU chips."""
    case_metadata = result.get("metadata", {})
    src_tpu_id = case_metadata.get("src_tpu_id")
    dst_tpu_id = case_metadata.get("dst_tpu_id")
    if src_tpu_id is None or dst_tpu_id is None:
        return False
    return int(src_tpu_id) != int(dst_tpu_id)


def collect_case_bandwidths(
    case_results: list[dict[str, Any]],
    *,
    inter_tpu_only: bool = False,
) -> list[float]:
    """Collect per-case bandwidths, optionally filtering local TPU-chip cases."""
    values = []
    for result in case_results:
        if inter_tpu_only and not is_inter_tpu_split_case(result):
            continue
        bandwidth = case_bandwidth(result)
        if bandwidth is not None:
            values.append(bandwidth)
    return values


def build_split_pair_bandwidth_groups(
    case_results: list[dict[str, Any]],
    *,
    inter_tpu_only: bool = False,
) -> list[dict[str, Any]]:
    """Group split-pair bandwidth by torus-aware TPU distance and core pair."""
    groups: dict[int, dict[str, list[float]]] = {}
    for result in case_results:
        if inter_tpu_only and not is_inter_tpu_split_case(result):
            continue
        case_metadata = result.get("metadata", {})
        distance = case_metadata.get("tpu_torus_manhattan_distance")
        core_pair = case_metadata.get("core_pair")
        bandwidth = case_bandwidth(result)
        if distance is None or core_pair is None or bandwidth is None:
            continue
        # Split-pair results are single chiplet-link measurements. Grouping by
        # torus distance and 0/1 core direction makes local, same-chip, and
        # cross-chip behavior easy to compare.
        distance = int(distance)
        groups.setdefault(
            distance,
            {label: [] for label in CORE_PAIR_LABELS},
        )
        groups[distance].setdefault(core_pair, []).append(bandwidth)

    summaries = []
    for distance in sorted(groups):
        core_pairs = {
            label: summarize_bandwidth_group(groups[distance].get(label, []))
            for label in CORE_PAIR_LABELS
        }
        all_values = [
            bandwidth
            for values in groups[distance].values()
            for bandwidth in values
        ]
        summaries.append({
            "tpu_torus_manhattan_distance": distance,
            **summarize_bandwidth_group(all_values),
            "core_pairs": core_pairs,
        })
    return summaries


def compute_case_traffic_bytes(
    benchmark: str,
    case: TrafficCase,
    data_size_bytes: int,
    n_chiplets: int,
    n_tpus: int,
    ar_parallel: bool = False,
    a2a_parallel: bool = False,
) -> tuple[int | float, int | float, str, dict[str, Any]]:
    """Return total logical bytes and the bandwidth numerator for one case."""
    total_case_bytes = len(case.active_pairs) * data_size_bytes
    metadata: dict[str, Any] = {
        "bandwidth_data_size_bytes": data_size_bytes,
    }

    if benchmark in {"p2p", "p2p-rdma"}:
        if len(case.active_pairs) != 1:
            raise ValueError(
                f"{benchmark} split cases must contain exactly one active "
                "chiplet link, "
                f"got {len(case.active_pairs)} for case {case.label}"
            )
        metadata["bandwidth_formula"] = "data_size"
        metadata["bandwidth_data_size_bytes_per_link"] = data_size_bytes
        if benchmark == "p2p-rdma":
            metadata["rdma_direction"] = "directed_src_to_dst"
            metadata["rdma_transport"] = "pallas_remote_dma"
        return total_case_bytes, data_size_bytes, "single_chiplet_pair", metadata

    if benchmark == "a2a":
        # Uniform all-to-all: half the ranks send half their input across
        # the cut in ONE direction. Parallel chiplet groups share the cut;
        # sum their bytes, never sum their concurrently measured durations.
        group_count = CHIPLETS_PER_TPU if a2a_parallel else 1
        group_ranks = n_tpus if a2a_parallel else n_chiplets
        bisection_bytes = data_size_bytes * group_ranks * group_count / 4
        bisection_metadata = {
            "bandwidth_formula": "data_size * ranks_per_group / 4 * parallel_groups",
            "bandwidth_formula_scope": "one_way_bisection",
            "bandwidth_formula_participants": "selected_block_chiplets",
            "bandwidth_formula_n_devices": n_chiplets,
            "bandwidth_formula_n_devices_per_group": group_ranks,
            "bandwidth_formula_parallel_groups": group_count,
            "bandwidth_accounting_mode": "uniform_alltoall_one_way_bisection",
            "bandwidth_accounting_unit": "bisection",
            "bisection_traffic_bytes": bisection_bytes,
        }
        if a2a_parallel:
            if n_tpus <= 1:
                raise ValueError("parallel all-to-all requires at least 2 TPU chips")
            chunk_bytes = data_size_bytes // n_tpus
            per_chiplet_ici_bytes = chunk_bytes * (n_tpus - 1)
            per_tpu_chip_ici_bytes = per_chiplet_ici_bytes * CHIPLETS_PER_TPU
            total_inter_tpu_bytes = (
                data_size_bytes * (n_tpus - 1) * CHIPLETS_PER_TPU
            )
            metadata.update({
                "all_to_all_parallel": True,
                "all_to_all_parallel_group_count": CHIPLETS_PER_TPU,
                "all_to_all_parallel_group_size": n_tpus,
                "all_to_all_input_bytes_per_chiplet": data_size_bytes,
                "all_to_all_chunk_bytes": chunk_bytes,
                "all_to_all_remote_tpu_chunks_per_chiplet": n_tpus - 1,
                "all_to_all_inter_tpu_bytes_per_chiplet": per_chiplet_ici_bytes,
                "all_to_all_inter_tpu_bytes_per_tpu_chip": (
                    per_tpu_chip_ici_bytes
                ),
                "all_to_all_total_inter_tpu_bytes": total_inter_tpu_bytes,
            })
            return (
                total_inter_tpu_bytes,
                bisection_bytes,
                "one_way_bisection",
                {**metadata, **bisection_metadata},
            )

        # Retain logical inter-chip byte counts as diagnostics only. They
        # exclude forwarded traffic and are NOT the bandwidth numerator.
        remote_chunks_per_chiplet = n_chiplets - CHIPLETS_PER_TPU
        chunk_bytes = data_size_bytes // n_chiplets
        per_chiplet_ici_bytes = chunk_bytes * remote_chunks_per_chiplet
        per_tpu_chip_ici_bytes = per_chiplet_ici_bytes * CHIPLETS_PER_TPU
        total_inter_tpu_bytes = per_chiplet_ici_bytes * n_chiplets
        metadata.update({
            "all_to_all_parallel": False,
            "all_to_all_input_bytes_per_chiplet": data_size_bytes,
            "all_to_all_chunk_bytes": chunk_bytes,
            "all_to_all_remote_tpu_chunks_per_chiplet": (
                remote_chunks_per_chiplet
            ),
            "all_to_all_inter_tpu_bytes_per_chiplet": (
                per_chiplet_ici_bytes
            ),
            "all_to_all_inter_tpu_bytes_per_tpu_chip": (
                per_tpu_chip_ici_bytes
            ),
            "all_to_all_total_inter_tpu_formula": (
                "data_size * (n_devices - chiplets_per_tpu)"
            ),
            "all_to_all_total_inter_tpu_bytes": total_inter_tpu_bytes,
        })
        return (
            total_inter_tpu_bytes,
            bisection_bytes,
            "one_way_bisection",
            {**metadata, **bisection_metadata},
        )

    if benchmark == "ar":
        if n_chiplets <= 1:
            raise ValueError("all-reduce requires at least 2 chiplets")

        if ar_parallel:
            # Parallel AllReduce splits the 2-D mesh by chiplet index:
            #   group 0: chiplet 0 across all TPU chips
            #   group 1: chiplet 1 across all TPU chips
            #
            # Each group contains n_tpus ranks. The NCCL-style bus numerator
            # per participant is data_size * 2 * (n_tpus - 1) / n_tpus. Because
            # the two chiplet groups run in parallel, the effective single-TPU
            # chip ICI bandwidth sums both chiplets on the chip.
            per_chiplet_bus_bytes = (
                data_size_bytes * 2 * (n_tpus - 1) / n_tpus
            )
            per_tpu_chip_bus_bytes = per_chiplet_bus_bytes * CHIPLETS_PER_TPU
            total_bus_bytes = (
                data_size_bytes * 2 * (n_tpus - 1) * CHIPLETS_PER_TPU
            )
            metadata.update({
                "bandwidth_formula": (
                    "data_size * 2 * (n_devices_per_group - 1) / "
                    "n_devices_per_group * parallel_groups"
                ),
                "bandwidth_formula_scope": (
                    "single_tpu_chip_parallel_allreduce_bus"
                ),
                "bandwidth_formula_participants": (
                    "two_parallel_chiplet_groups"
                ),
                "bandwidth_formula_n_devices_per_group": n_tpus,
                "bandwidth_formula_parallel_groups": CHIPLETS_PER_TPU,
                "bandwidth_accounting_mode": (
                    "parallel_per_tpu_chip_aggregate"
                ),
                "bandwidth_accounting_unit": "tpu_chip",
                "all_reduce_parallel": True,
                "all_reduce_parallel_group_count": CHIPLETS_PER_TPU,
                "all_reduce_parallel_group_size": n_tpus,
                "all_reduce_payload_bytes_per_device": data_size_bytes,
                "all_reduce_bus_factor_per_group": (
                    2 * (n_tpus - 1) / n_tpus
                ),
                "all_reduce_bus_bytes_per_chiplet": per_chiplet_bus_bytes,
                "all_reduce_bus_bytes_per_tpu_chip": per_tpu_chip_bus_bytes,
                "all_reduce_total_bus_formula": (
                    "data_size * 2 * (n_devices_per_group - 1) * "
                    "parallel_groups"
                ),
                "all_reduce_total_bus_bytes": total_bus_bytes,
            })
            return (
                total_bus_bytes,
                per_tpu_chip_bus_bytes,
                "single_tpu_chip_parallel_allreduce_bus",
                metadata,
            )

        # Standard AllReduce bus-bandwidth numerator per participating device.
        # In this benchmark n_devices is the number of JAX devices/chiplets.
        # This is the 2*(N-1)/N factor used by NCCL-style busbw:
        # reduce-scatter contributes (N-1)/N of the payload and all-gather
        # contributes another (N-1)/N. The reported bandwidth is average
        # per-device bus bandwidth, not the sum across all devices.
        total_bus_bytes = data_size_bytes * 2 * (n_chiplets - 1)
        if total_bus_bytes % n_chiplets == 0:
            per_device_bus_bytes: int | float = total_bus_bytes // n_chiplets
        else:
            per_device_bus_bytes = total_bus_bytes / n_chiplets

        metadata.update({
            "bandwidth_formula": (
                "data_size * 2 * (n_devices - 1) / n_devices"
            ),
            "bandwidth_formula_scope": "average_per_device_allreduce_bus",
            "bandwidth_formula_participants": "jax_devices_chiplets",
            "bandwidth_formula_n_devices": n_chiplets,
            "bandwidth_accounting_mode": (
                "non_parallel_per_chiplet_average"
            ),
            "bandwidth_accounting_unit": "chiplet",
            "all_reduce_bus_factor": (
                2 * (n_chiplets - 1) / n_chiplets
            ),
            "all_reduce_payload_bytes_per_device": data_size_bytes,
            "all_reduce_bus_bytes_per_device": per_device_bus_bytes,
            "all_reduce_total_bus_formula": (
                "data_size * 2 * (n_devices - 1)"
            ),
            "all_reduce_total_bus_bytes": total_bus_bytes,
        })
        return (
            total_bus_bytes,
            per_device_bus_bytes,
            "average_per_device_allreduce_bus",
            metadata,
        )

    metadata["bandwidth_formula"] = "active_links * data_size"
    metadata["bandwidth_data_size_bytes_per_link"] = data_size_bytes
    return total_case_bytes, total_case_bytes, "aggregate_logical", metadata


def format_topology(topology: tuple[int, int, int]) -> str:
    """Format ``(x, y, z)`` topology dimensions as ``"XxYxZ"``."""
    return "x".join(str(dim) for dim in topology)


def _int_tuple(value: Any) -> tuple[int, ...] | None:
    """Convert a device coordinate-like object to an integer tuple."""
    if callable(value):
        value = value()
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return None
    try:
        return tuple(int(v) for v in value)
    except (TypeError, ValueError):
        return None


def get_device_coords(device: Any) -> tuple[int, int, int] | None:
    """Return TPU chip coordinates exposed by a JAX device if available."""
    for attr_name in ("coords", "coordinates", "coordinate"):
        if hasattr(device, attr_name):
            coords = _int_tuple(getattr(device, attr_name))
            if coords and len(coords) >= 3:
                return coords[:3]
    return None


def infer_tpu_topology(
    devices: list[Any] | None = None,
    *,
    normalize_origin: bool = False,
) -> tuple[tuple[int, int, int], str, list[tuple[int, int, int]]]:
    """Infer TPU chip topology from JAX device coordinates."""
    jax_devices = list(jax.devices() if devices is None else devices)
    if not jax_devices:
        raise ValueError("No JAX devices are available")

    chip_coords = []
    for index, device in enumerate(jax_devices):
        coords = get_device_coords(device)
        if coords is None:
            raise ValueError(
                "Cannot infer TPU topology because JAX device "
                f"{index} does not expose chip coordinates: {device!r}"
            )
        chip_coords.append(coords)

    topology_source = "tpu_chip_coords"
    if normalize_origin:
        origin = tuple(
            min(coord[axis] for coord in chip_coords)
            for axis in range(3)
        )
        chip_coords = [
            tuple(coord[axis] - origin[axis] for axis in range(3))
            for coord in chip_coords
        ]
        topology_source = "local_tpu_chip_coords"

    if len(chip_coords) % CHIPLETS_PER_TPU != 0:
        raise ValueError(
            f"JAX reports {len(chip_coords)} devices, which is not divisible by "
            f"{CHIPLETS_PER_TPU} chiplets per TPU chip"
        )

    ordered_chip_coords = []
    for group_start in range(0, len(chip_coords), CHIPLETS_PER_TPU):
        group = chip_coords[group_start:group_start + CHIPLETS_PER_TPU]
        # Matrix index i maps to jax.devices()[2*i] and [2*i+1]. Both must
        # share the same TPU chip coordinate; otherwise TPU-level matrices
        # cannot be expanded safely to chiplet-level links.
        if len(set(group)) != 1:
            raise ValueError(
                "Cannot map TPU-level traffic matrix to JAX devices because "
                "logical devices for one TPU chip are not consecutive in "
                f"jax.devices(): group starting at {group_start} has {group}"
            )
        ordered_chip_coords.append(group[0])
    if len(set(ordered_chip_coords)) != len(ordered_chip_coords):
        raise ValueError(
            "Cannot map TPU-level traffic matrix to JAX devices because "
            f"consecutive chip groups contain duplicate coordinates: "
            f"{ordered_chip_coords}"
        )

    unique_coords = sorted(set(chip_coords))
    expected_chips = len(chip_coords) // CHIPLETS_PER_TPU
    if len(unique_coords) != expected_chips:
        raise ValueError(
            "Cannot infer TPU topology because unique chip coordinates "
            f"({len(unique_coords)}) do not match expected TPU chip count "
            f"({expected_chips}) from {len(chip_coords)} JAX devices"
        )
    coord_counts = {
        coord: chip_coords.count(coord)
        for coord in unique_coords
    }
    invalid_counts = {
        coord: count
        for coord, count in coord_counts.items()
        if count != CHIPLETS_PER_TPU
    }
    if invalid_counts:
        raise ValueError(
            "Cannot infer TPU topology because chip coordinate fanout is not "
            f"{CHIPLETS_PER_TPU} logical devices per chip: {invalid_counts}"
        )

    axis_values = [
        sorted(set(coord[axis] for coord in unique_coords))
        for axis in range(3)
    ]
    for axis, values in enumerate(axis_values):
        expected = list(range(len(values)))
        if values != expected:
            raise ValueError(
                "Cannot infer TPU topology because coordinate axis "
                f"{axis} is not zero-based and continuous: "
                f"got {values}, expected {expected}"
            )

    topology = tuple(len(values) for values in axis_values)
    inferred_chips = topology[0] * topology[1] * topology[2]
    if inferred_chips != len(unique_coords):
        raise ValueError(
            "Cannot infer TPU topology because device coordinates do not form "
            "a dense cuboid: "
            f"{len(unique_coords)} unique chip coordinates do not fill "
            f"inferred topology {format_topology(topology)}"
        )

    return topology, topology_source, ordered_chip_coords


def parse_traffic_matrix(matrix_str: str) -> list[list[int]]:
    """Parse traffic matrix string like ``"#1,0,1#0,1,0"`` into 2-D list.

    Each ``#`` starts a new row; values are comma-separated 0 or 1.
    Matrix index ``i`` maps to the ``i``-th consecutive TPU chip group in
    ``jax.devices()``. Each chip group must contain two logical devices with
    the same chip coordinates.

    Raises:
        ValueError: if format is invalid, matrix is empty, not square,
            or contains values other than 0/1.
    """
    # Example for 2 TPU chips: "#0,1#1,0" enables TPU0->TPU1 and TPU1->TPU0.
    if not matrix_str.startswith("#"):
        raise ValueError(f"Traffic matrix must start with '#', got: {matrix_str!r}")

    rows = []
    row_strings = matrix_str.split("#")[1:]
    for row_idx, row_str in enumerate(row_strings):
        if not row_str:
            raise ValueError(f"Traffic matrix row {row_idx} is empty")
        try:
            row = [int(x) for x in row_str.split(",")]
        except ValueError:
            raise ValueError(f"Traffic matrix values must be integers: {row_str!r}")
        if any(v not in (0, 1) for v in row):
            raise ValueError(f"Traffic matrix values must be 0 or 1: {row!r}")
        rows.append(row)

    if not rows:
        raise ValueError("Traffic matrix is empty")

    n = len(rows)
    for i, row in enumerate(rows):
        if len(row) != n:
            raise ValueError(
                f"Traffic matrix must be square. Row {i} has "
                f"{len(row)} elements, expected {n}"
            )
    return rows


def format_traffic_matrix_count_label(traffic_matrix: list[list[int]]) -> str:
    """Return a compact label as ``tm_<active-links>-<total-links>``."""
    active_links = sum(sum(row) for row in traffic_matrix)
    total_links = len(traffic_matrix) * len(traffic_matrix)
    return f"tm_{active_links}-{total_links}"


def build_builtin_tpu_traffic_matrix_str(
    n_tpus: int,
    include_diagonal: bool,
) -> str:
    """Build a TPU-level built-in matrix."""
    if n_tpus <= 0:
        raise ValueError(f"n_tpus must be positive, got {n_tpus}")
    rows = []
    for src in range(n_tpus):
        rows.append(
            ",".join(
                "1" if include_diagonal or src != dst else "0"
                for dst in range(n_tpus)
            )
        )
    return "#" + "#".join(rows)


def get_active_pairs(traffic_matrix: list[list[int]]) -> list[tuple[int, int]]:
    """Extract ``(src, dst)`` pairs where ``traffic_matrix[src][dst] == 1``."""
    return [
        (src, dst)
        for src, row in enumerate(traffic_matrix)
        for dst, v in enumerate(row)
        if v == 1
    ]


def expand_to_chiplet_level(tpu_matrix: np.ndarray) -> np.ndarray:
    """Expand TPU-level traffic matrix (NxN) to chiplet-level (2Nx2N).

    Each TPU chip has 2 chiplets. If ``TPU[i][j]=1``, then all 4 chiplet
    links are active::

        chiplet[2i]   -> chiplet[2j]
        chiplet[2i]   -> chiplet[2j+1]
        chiplet[2i+1] -> chiplet[2j]
        chiplet[2i+1] -> chiplet[2j+1]

    Self-links (``TPU[i][i]=1``) expand to intra-chip communication and are
    included. Use ``get_active_pairs`` on the result and filter if needed.
    """
    n_tpus = tpu_matrix.shape[0]
    n_chiplets = n_tpus * CHIPLETS_PER_TPU
    chiplet_matrix = np.zeros((n_chiplets, n_chiplets), dtype=np.int32)

    for src_tpu in range(n_tpus):
        for dst_tpu in range(n_tpus):
            if tpu_matrix[src_tpu, dst_tpu] == 1:
                # TPU-level 1 expands to 2x2 chiplet links because each TPU
                # chip exposes two JAX devices. Diagonal entries are kept here;
                # callers decide whether same-chip traffic is meaningful.
                for src_off in range(CHIPLETS_PER_TPU):
                    for dst_off in range(CHIPLETS_PER_TPU):
                        src_c = src_tpu * CHIPLETS_PER_TPU + src_off
                        dst_c = dst_tpu * CHIPLETS_PER_TPU + dst_off
                        chiplet_matrix[src_c, dst_c] = 1

    return chiplet_matrix


def validate_ragged_all_to_all_bounds(
    traffic_matrix_np: np.ndarray,
    payload_rows_per_link: int,
    context: str,
) -> np.ndarray:
    """Return row-count matrix after checking size/offset bounds."""
    # ragged_all_to_all uses int32 row counts/offsets in this benchmark, so the
    # check must consider both per-link rows and fanout-accumulated offsets.
    tm_rows = traffic_matrix_np.astype(np.int64) * int(payload_rows_per_link)
    max_link_rows = int(tm_rows.max())
    max_send_rows = int(tm_rows.sum(axis=1).max())
    max_recv_rows = int(tm_rows.sum(axis=0).max())
    max_input_offset = int((np.cumsum(tm_rows, axis=1) - tm_rows).max())
    max_output_offset = int((np.cumsum(tm_rows, axis=0) - tm_rows).max())

    checks = {
        "per-link rows": max_link_rows,
        "per-device send rows": max_send_rows,
        "per-device receive rows": max_recv_rows,
        "input offset rows": max_input_offset,
        "output offset rows": max_output_offset,
    }
    overflow = {
        name: value
        for name, value in checks.items()
        if value > RAGGED_INDEX_MAX
    }
    if overflow:
        max_active_fanout = max(
            1,
            int((traffic_matrix_np > 0).sum(axis=1).max()),
            int((traffic_matrix_np > 0).sum(axis=0).max()),
        )
        max_safe_rows = RAGGED_INDEX_MAX // max_active_fanout
        overflow_text = ", ".join(
            f"{name}={value}" for name, value in overflow.items()
        )
        raise ValueError(
            f"{context} traffic volume exceeds ragged_all_to_all int32 "
            f"size/offset bounds: {overflow_text}, limit={RAGGED_INDEX_MAX}. "
            f"--data-size is per active chiplet link in bytes and maps to "
            f"payload rows of {PAYLOAD_ROW_BYTES} bytes; "
            f"this matrix has max fanout {max_active_fanout}, so use "
            f"--data-size <= {max_safe_rows * PAYLOAD_ROW_BYTES} bytes "
            f"per link, "
            "or reduce the active links in --traffic-matrix."
        )

    return tm_rows


def build_traffic_cases(
    chiplet_matrix_np: np.ndarray,
    execution_shape: str,
    tpu_chip_order: list[tuple[int, int, int]] | None = None,
    topology: tuple[int, int, int] | None = None,
) -> list[TrafficCase]:
    """Build concrete chiplet-level matrices for the shared runner."""
    active_pairs = get_active_pairs(chiplet_matrix_np.tolist())
    if execution_shape == "single_matrix":
        return [
            TrafficCase(
                label="matrix",
                matrix=chiplet_matrix_np,
                active_pairs=active_pairs,
                metadata={},
            )
        ]

    if execution_shape != "split_pairs":
        raise ValueError(f"Unknown traffic execution shape: {execution_shape}")

    n_chiplets = chiplet_matrix_np.shape[0]
    cases = []
    for pair_idx, (src, dst) in enumerate(active_pairs):
        single_matrix = np.zeros((n_chiplets, n_chiplets), dtype=np.int32)
        single_matrix[src, dst] = 1
        src_tpu, src_chiplet = divmod(src, CHIPLETS_PER_TPU)
        dst_tpu, dst_chiplet = divmod(dst, CHIPLETS_PER_TPU)
        metadata: dict[str, Any] = {
            "pair_index": pair_idx,
            "src_chiplet_id": src,
            "dst_chiplet_id": dst,
            "src_tpu_id": src_tpu,
            "dst_tpu_id": dst_tpu,
            "pair_scope": (
                "inter_tpu" if src_tpu != dst_tpu else "local_or_same_tpu"
            ),
            "src_chiplet_offset": src_chiplet,
            "dst_chiplet_offset": dst_chiplet,
            "src_core": src_chiplet,
            "dst_core": dst_chiplet,
            "core_pair": f"{src_chiplet}->{dst_chiplet}",
        }

        if tpu_chip_order is not None and topology is not None:
            src_coord = tpu_chip_order[src_tpu]
            dst_coord = tpu_chip_order[dst_tpu]
            src_endpoint = {
                "tpu_coord": list(src_coord),
                "core": src_chiplet,
            }
            dst_endpoint = {
                "tpu_coord": list(dst_coord),
                "core": dst_chiplet,
            }
            coord_delta = [
                dst_axis - src_axis
                for src_axis, dst_axis in zip(src_coord, dst_coord)
            ]
            axis_distances = torus_axis_distances(
                src_coord,
                dst_coord,
                topology,
            )
            metadata.update({
                "src_tpu_coord": list(src_coord),
                "dst_tpu_coord": list(dst_coord),
                "src_endpoint": src_endpoint,
                "dst_endpoint": dst_endpoint,
                "tpu_coord_delta": coord_delta,
                "tpu_torus_axis_distances": axis_distances,
                "tpu_distance_metric": "torus_manhattan",
                "tpu_torus_manhattan_distance": sum(axis_distances),
                "logical_pair": (
                    f"src{format_endpoint_display(src_coord, src_chiplet)}"
                    f"->dst{format_endpoint_display(dst_coord, dst_chiplet)}"
                ),
            })
            label = (
                f"pair{pair_idx}_src_"
                f"{format_endpoint_label(src_coord, src_chiplet)}_dst_"
                f"{format_endpoint_label(dst_coord, dst_chiplet)}"
            )
        else:
            label = f"pair{pair_idx}_chiplet{src}_chiplet{dst}"

        cases.append(
            TrafficCase(
                label=label,
                matrix=single_matrix,
                active_pairs=[(src, dst)],
                metadata=metadata,
            )
        )
    return cases
