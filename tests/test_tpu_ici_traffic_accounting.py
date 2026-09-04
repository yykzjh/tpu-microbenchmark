from __future__ import annotations

import importlib
import ast
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def traffic_module():
    """Load the pure traffic helpers without installing the TPU JAX runtime."""
    source_root = (
        Path(__file__).resolve().parents[1]
        / "src"
    )
    previous_jax = sys.modules.get("jax")
    previous_numpy = sys.modules.get("numpy")
    sys.modules["jax"] = ModuleType("jax")
    numpy_stub = ModuleType("numpy")
    numpy_stub.int32 = int
    numpy_stub.iinfo = lambda _dtype: type("IInfo", (), {"max": 2**31 - 1})()
    sys.modules["numpy"] = numpy_stub
    sys.path.insert(0, str(source_root))
    try:
        module = importlib.import_module("ici.traffic")
        yield module
    finally:
        sys.path.remove(str(source_root))
        sys.modules.pop("ici.traffic", None)
        sys.modules.pop("ici", None)
        if previous_jax is None:
            sys.modules.pop("jax", None)
        else:
            sys.modules["jax"] = previous_jax
        if previous_numpy is None:
            sys.modules.pop("numpy", None)
        else:
            sys.modules["numpy"] = previous_numpy


def _collective_case(traffic_module):
    return traffic_module.TrafficCase(
        label="collective",
        matrix=None,
        active_pairs=[],
        metadata={},
    )


@pytest.mark.parametrize("n_tpus", [2, 4, 8, 16, 32, 64])
def test_all_to_all_modes_use_one_way_bisection_of_selected_block(
    traffic_module, n_tpus,
) -> None:
    data_size = 32 * 4096
    case = _collective_case(traffic_module)

    non_parallel = traffic_module.compute_case_traffic_bytes(
        benchmark="a2a",
        case=case,
        data_size_bytes=data_size,
        n_chiplets=n_tpus * 2,
        n_tpus=n_tpus,
    )
    parallel = traffic_module.compute_case_traffic_bytes(
        benchmark="a2a",
        case=case,
        data_size_bytes=data_size,
        n_chiplets=n_tpus * 2,
        n_tpus=n_tpus,
        a2a_parallel=True,
    )

    assert non_parallel[0] == data_size * (n_tpus - 1) * 2
    assert non_parallel[1] == data_size * (n_tpus * 2) / 4
    assert non_parallel[2] == "one_way_bisection"
    assert non_parallel[3]["bandwidth_accounting_unit"] == "bisection"

    assert parallel[0] == non_parallel[0]
    assert parallel[1] == non_parallel[1]
    assert parallel[2] == "one_way_bisection"
    assert parallel[3]["bandwidth_accounting_unit"] == "bisection"
    for result in (parallel, non_parallel):
        assert result[1] == result[3]["bisection_traffic_bytes"]
    assert non_parallel[3]["bandwidth_formula_n_devices_per_group"] == n_tpus * 2
    assert non_parallel[3]["bandwidth_formula_parallel_groups"] == 1
    assert parallel[3]["bandwidth_formula_n_devices_per_group"] == n_tpus
    assert parallel[3]["bandwidth_formula_parallel_groups"] == 2


def test_all_reduce_parallel_and_non_parallel_use_different_units(
    traffic_module,
) -> None:
    data_size = 16 * 4096
    case = _collective_case(traffic_module)

    non_parallel = traffic_module.compute_case_traffic_bytes(
        benchmark="ar",
        case=case,
        data_size_bytes=data_size,
        n_chiplets=16,
        n_tpus=8,
    )
    parallel = traffic_module.compute_case_traffic_bytes(
        benchmark="ar",
        case=case,
        data_size_bytes=data_size,
        n_chiplets=16,
        n_tpus=8,
        ar_parallel=True,
    )

    assert non_parallel[0] == data_size * 2 * 15
    assert non_parallel[1] == data_size * 2 * 15 // 16
    assert non_parallel[2] == "average_per_device_allreduce_bus"
    assert non_parallel[3]["bandwidth_accounting_unit"] == "chiplet"

    assert parallel[0] == data_size * 2 * 7 * 2
    assert parallel[1] == data_size * 2 * 7 // 8 * 2
    assert parallel[2] == "single_tpu_chip_parallel_allreduce_bus"
    assert parallel[3]["bandwidth_accounting_unit"] == "tpu_chip"


@pytest.mark.parametrize("parallel,duration_ms,expected_gbps", [
    (False, 23.0884, 372.04547),
    (True, 23.1511, 371.03786),
])
def test_full_slice_bisection_uses_eight_gib_and_unchanged_duration(
    traffic_module, parallel, duration_ms, expected_gbps,
):
    _, numerator, _, _ = traffic_module.compute_case_traffic_bytes(
        benchmark="a2a", case=_collective_case(traffic_module),
        data_size_bytes=1024**3, n_chiplets=32, n_tpus=16, a2a_parallel=parallel,
    )
    assert numerator == 8 * 1024**3
    source = (Path(__file__).resolve().parents[1]
              / "src/utils/metrics.py").read_text()
    fn = next(node for node in ast.parse(source).body
              if isinstance(node, ast.FunctionDef) and node.name == "compute_bandwidth_GBps")
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "metrics.py", "exec"), namespace)
    assert abs(namespace[fn.name](numerator, duration_ms) - expected_gbps) < 0.0001


@pytest.mark.parametrize("parallel", [False, True])
def test_alltoall_structured_metric_carries_bisection_accounting(traffic_module, parallel):
    source = (Path(__file__).resolve().parents[1]
              / "src/ici/runner.py").read_text()
    fn = next(node for node in ast.parse(source).body
              if isinstance(node, ast.FunctionDef) and node.name == "iter_commpilot_log_metrics")
    namespace = {"Any": object}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "runner.py", "exec"), namespace)
    _, numerator, scope, metadata = traffic_module.compute_case_traffic_bytes(
        benchmark="a2a", case=_collective_case(traffic_module),
        data_size_bytes=65536, n_chiplets=16, n_tpus=8, a2a_parallel=parallel,
    )
    rows = list(namespace[fn.name]({
        "metadata": {"benchmark": "a2a", **metadata},
        "metrics": {"bandwidth_GBps": 100, "bandwidth_scope": scope,
                    "bandwidth_traffic_bytes": numerator},
    }))
    assert rows[0][0] == ("alltoall_parallel_bisection" if parallel else "alltoall_bisection")
    assert rows[0][2]["bandwidth_accounting_unit"] == "bisection"
    assert rows[0][2]["bandwidth_traffic_bytes"] == 262144
    assert rows[0][2]["bisection_traffic_bytes"] == 262144
