"""Runtime helpers shared by TPU benchmark entrypoints."""

from __future__ import annotations

from datetime import datetime
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DumpHloConfig:
    """Resolved XLA HLO dump configuration."""

    dump_dir: str | None


@dataclass(frozen=True)
class BenchmarkDirs:
    """Common output directories for one benchmark run."""

    output_dir: str | None
    metrics_dir: str | None
    trace_dir: str | None
    dump_hlo_dir: str | None


TPU_LOG_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
TPU_TEMP_ROOT = os.path.join(TPU_LOG_ROOT, ".tmp")
TPU_PROFILE_HLO_DIR_ENV = "COMMPILOT_PROFILE_HLO_DIR"

_DEFAULT_PROFILE_TESTCASE_BY_PREFIX = {
    "gemm": "gemm",
    "ici": "ici",
    "hbm": "tpubandwidth",
    "vmem": "tpubandwidth",
}


def _resolve_output_path(path: str) -> str:
    """Return a normalized absolute output path without restricting its root."""
    return os.path.realpath(os.path.abspath(path))


def default_profile_result_dir(testcase: str) -> str:
    """Return the standard retained-artifact root for one testcase."""
    return os.path.join(TPU_LOG_ROOT, testcase, "profile")


def benchmark_temporary_directory(prefix: str):
    """Create an auto-cleaned timing workspace below the repository ``logs/.tmp`` directory."""
    os.makedirs(TPU_TEMP_ROOT, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix=prefix, dir=TPU_TEMP_ROOT)


def _argv_option_value(argv: list[str], option: str) -> str | None:
    """Read a simple ``--option value`` or ``--option=value`` CLI argument."""
    for index, value in enumerate(argv):
        if value.startswith(f"{option}="):
            return value.split("=", 1)[1]
        if value == option:
            if index + 1 >= len(argv):
                raise ValueError(f"{option} requires a value")
            return argv[index + 1]
    return None


TPU_BENCHMARK_ENV_PROFILES: dict[str, dict[str, Any]] = {
    # Mirrors accelerator-microbenchmarks' gemm_numerics flags. These are most
    # useful for FP8/BF16/FP16 GEMM where XLA lowers dot_general to a TPU
    # convolution and may need to fuse downcast/layout conversion into inputs.
    "gemm": {
        "libtpu_flags": [
            # Newer libtpu rejects continuation fusion on TPU7x (non-Viperlite).
            "--xla_tpu_enable_async_collective_fusion=false",
            "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true",
            "--xla_tpu_enable_async_collective_fusion_multiple_steps=true",
            "--xla_tpu_overlap_compute_collective_tc=true",
            "--xla_enable_async_all_gather=true",
            "--xla_enable_async_collective_permute=true",
            "--xla_tpu_enable_all_experimental_scheduler_features=true",
            "--xla_tpu_accumulate_into_mrb=true",
            "--xla_tpu_scoped_vmem_limit_kib=65536",
            "--xla_tpu_vmem_scavenging_mode=NONE",
            "--xla_tpu_allow_conv_input_fusion_with_downcast_convert=true",
            "--xla_tpu_dvfs_p_state=7",
        ],
    },
    "ici": {
        "libtpu_flags": [
            "--xla_jf_debug_level=3",
            "--xla_tpu_collect_sflag_wait_stats_trace=true",
            "--xla_tpu_force_global_barriers=true",
            "--xla_tpu_ragged_all_to_all_max_rdma_size_kib=-1",
            "--xla_tpu_dvfs_p_state=7",
        ],
    },
    "ici_send_recv": {
        "libtpu_flags": [
            "--xla_tpu_collect_sflag_wait_stats_trace=true",
            "--xla_tpu_force_global_barriers=true",
            "--xla_tpu_ragged_all_to_all_max_rdma_size_kib=-1",
            "--xla_tpu_dvfs_p_state=7",
        ],
    },
    "ici_all_to_all": {
        "libtpu_flags": [
            "--xla_jf_debug_level=3",
            "--xla_tpu_dvfs_p_state=7",
        ],
    },
    "ici_psum": {
        "libtpu_flags": [
            "--xla_jf_debug_level=3",
            "--xla_sc_disable_megacore_partitioning=true",
            "--xla_tpu_disable_sparse_core_collective_offload_remover=true",
            "--xla_tpu_enable_all_reduce_offload_tracing=true",
            "--xla_tpu_enable_all_reduce_scatter_fusion=false",
            "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true",
            "--xla_tpu_pad_operations_input_tiles=true",
            "--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=0",
            "--xla_tpu_use_tc_device_shape_on_sc=true",
            "--xla_tpu_dvfs_p_state=7",
        ],
    },
    "pcie": {
        "libtpu_flags": [
            "--xla_tpu_dvfs_p_state=7",
        ],
        "env": {
            "TPU_PREMAPPED_BUFFER_SIZE": "68719476736",
            "TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES": "68719476736",
        },
    },
    "hbm": {
        "libtpu_flags": [
            "--xla_tpu_scoped_vmem_limit_kib=65536",
            "--xla_jf_bounds_check=false",
            "--xla_tpu_dvfs_p_state=7",
        ],
    },
    "vmem": {
        "libtpu_flags": [
            "--xla_tpu_scoped_vmem_limit_kib=65536",
            "--xla_jf_bounds_check=false",
            "--xla_tpu_dvfs_p_state=7",
        ],
    },
}


def _flag_key(flag: str) -> str:
    """Return the comparable key for an XLA flag string."""
    return flag.split("=", 1)[0]


def _append_unique_flags(env_name: str, flags: list[str]) -> None:
    """Append default flags without overriding caller-provided same-name flags."""
    existing = os.environ.get(env_name, "").split()
    existing_keys = {_flag_key(flag) for flag in existing}
    merged = list(existing)
    for flag in flags:
        key = _flag_key(flag)
        if key not in existing_keys:
            merged.append(flag)
            existing_keys.add(key)
    if merged:
        os.environ[env_name] = " ".join(merged)


def configure_tpu_benchmark_env(profile: str) -> None:
    """Apply TPU benchmark flags/env before importing JAX.

    Defaults are adapted from AI-Hypercomputer/accelerator-microbenchmarks'
    ``op_flags.yaml``. Existing caller-provided flags with the same name win, so
    users can still tune or disable an individual setting from the shell.
    """
    config = TPU_BENCHMARK_ENV_PROFILES.get(profile)
    if config is None:
        raise ValueError(f"Unknown TPU benchmark env profile: {profile}")

    _append_unique_flags("LIBTPU_INIT_ARGS", list(config.get("libtpu_flags", [])))
    for key, value in config.get("env", {}).items():
        os.environ.setdefault(key, str(value))


def configure_dump_hlo_from_argv(
    prefix: str,
    argv: list[str] | None = None,
    flags: tuple[str, ...] = ("--profile", "--dump-hlo"),
) -> DumpHloConfig:
    """Configure a persistent profile HLO directory before JAX is imported."""
    argv = sys.argv if argv is None else argv
    if not any(flag in argv for flag in flags):
        os.environ.pop(TPU_PROFILE_HLO_DIR_ENV, None)
        return DumpHloConfig(dump_dir=None)

    existing_xla_flags = os.environ.get("XLA_FLAGS", "")
    existing_dump_dir = re.search(r"(?:^|\s)--xla_dump_to=(\S+)", existing_xla_flags)
    if existing_dump_dir:
        dump_dir = _resolve_output_path(existing_dump_dir.group(1))
        os.makedirs(dump_dir, exist_ok=True)
        os.environ[TPU_PROFILE_HLO_DIR_ENV] = dump_dir
        return DumpHloConfig(dump_dir=dump_dir)

    result_dir_arg = _argv_option_value(list(argv), "--result-dir")
    if result_dir_arg:
        profile_root = _resolve_output_path(result_dir_arg)
    else:
        testcase = _DEFAULT_PROFILE_TESTCASE_BY_PREFIX.get(prefix, prefix)
        profile_root = default_profile_result_dir(testcase)

    session_name = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
        f"{prefix}_profile_p{os.getpid()}"
    )
    dump_dir = os.path.join(profile_root, "hlo", session_name)
    os.makedirs(dump_dir, exist_ok=True)
    os.environ[TPU_PROFILE_HLO_DIR_ENV] = dump_dir

    dump_hlo_flag = f"--xla_dump_to={dump_dir}"
    if existing_xla_flags:
        os.environ["XLA_FLAGS"] = f"{existing_xla_flags} {dump_hlo_flag}"
    else:
        os.environ["XLA_FLAGS"] = dump_hlo_flag
    return DumpHloConfig(dump_dir=dump_dir)


def prepare_benchmark_dirs(
    result_dir: str | None,
    run_name: str,
    dump_hlo: bool = False,
    create_metrics: bool = True,
    create_trace: bool = True,
) -> BenchmarkDirs:
    """Create common output directories for one benchmark run."""
    if result_dir is None:
        if dump_hlo:
            raise ValueError("--profile requires --result-dir")
        return BenchmarkDirs(
            output_dir=None,
            metrics_dir=None,
            trace_dir=None,
            dump_hlo_dir=None,
        )

    result_dir = _resolve_output_path(result_dir)
    output_dir = os.path.join(result_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    metrics_dir = os.path.join(output_dir, "metrics") if create_metrics else None
    trace_dir = (
        os.path.join(result_dir, "xprof", run_name)
        if create_trace else None
    )
    dump_hlo_dir = (
        os.environ.get(TPU_PROFILE_HLO_DIR_ENV)
        or os.path.join(result_dir, "hlo", run_name)
    ) if dump_hlo else None

    for directory in (metrics_dir, trace_dir, dump_hlo_dir):
        if directory:
            os.makedirs(directory, exist_ok=True)

    return BenchmarkDirs(
        output_dir=output_dir,
        metrics_dir=metrics_dir,
        trace_dir=trace_dir,
        dump_hlo_dir=dump_hlo_dir,
    )


def validate_positive(name: str, value: int | float) -> None:
    """Raise if a numeric parameter is not positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_non_negative(name: str, value: int | float) -> None:
    """Raise if a numeric parameter is negative."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def initialize_jax_distributed(
    logger,
    *,
    coordinator_address: str,
    process_count: int,
    process_id: int,
) -> None:
    """Initialize one explicitly defined JAX distributed process group."""
    import jax

    try:
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=process_count,
            process_id=process_id,
        )
        logger.info(
            "JAX distributed initialized successfully: coordinator=%s "
            "process_count=%d process_id=%d",
            coordinator_address,
            process_count,
            process_id,
        )
    except Exception as exc:
        logger.error("JAX distributed initialization failed: %s", exc)
        raise RuntimeError(
            "JAX distributed initialization failed; check the coordinator, "
            "process count, process id, and connectivity between all selected hosts"
        ) from exc


def configure_logging(level: int = logging.INFO) -> None:
    """Configure consistent benchmark CLI logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def initialize_jax_runtime(
    logger,
    *,
    coordinator_address: str,
    process_count: int,
    process_id: int,
) -> dict[str, Any]:
    """Initialize JAX distributed runtime and log the device view."""
    initialize_jax_distributed(
        logger,
        coordinator_address=coordinator_address,
        process_count=process_count,
        process_id=process_id,
    )
    return log_jax_runtime_view(logger)


def initialize_local_jax_runtime(logger) -> dict[str, Any]:
    """Log the local JAX device view without joining a distributed runtime.

    GEMM and TPUbandwidth are host-local benchmarks.  They must remain usable
    when a multi-host job exports coordinator variables for other testcases,
    so these entrypoints deliberately avoid ``jax.distributed.initialize``.
    """
    logger.info("Using host-local JAX runtime; distributed initialization is disabled")
    return log_jax_runtime_view(logger)


def get_local_devices_or_raise() -> list[Any]:
    """Return local JAX devices, failing clearly when none are available."""
    import jax

    devices = list(jax.local_devices())
    if not devices:
        raise ValueError("No local JAX devices are available")
    return devices


def _device_attr(device: Any, name: str, default: Any = None) -> Any:
    """Read a JAX device attribute that may be exposed as a value or method."""
    value = getattr(device, name, default)
    if callable(value):
        try:
            return value()
        except TypeError:
            return value
    return value


def device_metadata(device: Any, device_index: int) -> dict[str, Any]:
    """Return JSON-safe metadata for one JAX device."""
    coords = _device_attr(device, "coords", []) or []
    return {
        "device_index": device_index,
        "device": str(device),
        "process_index": _device_attr(device, "process_index"),
        "coords": list(coords),
        "core_on_chip": _device_attr(device, "core_on_chip"),
    }


def log_device_separator(
    logger: logging.Logger,
    label: str,
    device_index: int,
    device: Any,
) -> None:
    """Log a readable section header before a per-device benchmark."""
    metadata = device_metadata(device, device_index)
    title = (
        f"{label} {device_index} | "
        f"global_device={metadata.get('device')} | "
        f"coords={metadata.get('coords')} | "
        f"core_on_chip={metadata.get('core_on_chip')}"
    )
    separator = "=" * max(80, len(title))
    logger.info(separator)
    logger.info(title)
    logger.info(separator)


def log_jax_runtime_view(logger) -> dict[str, Any]:
    """Log JAX distributed/device state and return the captured values."""
    import jax

    fields = {
        "process_index": jax.process_index,
        "process_count": jax.process_count,
        "local_device_count": jax.local_device_count,
        "global_device_count": jax.device_count,
        "local_devices": jax.local_devices,
        "global_devices": jax.devices,
    }
    runtime_view = {}
    for name, getter in fields.items():
        try:
            runtime_view[name] = getter()
            logger.info("JAX %s: %s", name, runtime_view[name])
        except Exception as exc:
            runtime_view[name] = f"<unavailable: {exc}>"
            logger.warning("JAX %s unavailable: %s", name, exc)
    return runtime_view
