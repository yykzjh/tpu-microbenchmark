"""CPU-only regression checks for the TPU timing boundary and output policy."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager, nullcontext

import pytest


SOURCE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
)


@pytest.mark.parametrize("file", ["ici/runner.py", "ici/kernels.py"])
def test_all_ici_paths_skip_zero_crop_probe_and_consumer(file):
    tree = ast.parse((SOURCE_ROOT / file).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            assert node.id not in {"maybe_zero_crop", "detect_zero_crop_available"}
        if isinstance(node, ast.ImportFrom):
            assert node.module != "ici.zero_crop"


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("benchmark", ["all_to_all", "all_reduce"])
def test_collectives_take_resident_input_and_return_real_output(benchmark, parallel):
    tree = ast.parse((SOURCE_ROOT / "ici/kernels.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == f"compile_{benchmark}_kernel")
    captured = {}
    payload, output = object(), object()

    def collective(value, **kwargs):
        assert value is payload
        captured["collective"] = kwargs
        return output

    def jit(fn):
        def lower(spec):
            captured["spec"] = spec
            return SimpleNamespace(compile=lambda: fn)
        return SimpleNamespace(lower=lower)

    def shard_map(fn, **kwargs):
        captured["sharding"] = kwargs
        return fn

    namespace = {
        "Mesh": object,
        "P": lambda *axes: axes,
        "NamedSharding": lambda mesh, spec: (mesh, spec),
        "PAYLOAD_MID_DIM": 8, "PAYLOAD_LAST_DIM": 128,
        "MARKER": "marker",
        "jnp": SimpleNamespace(float32="float32"),
        "jax": SimpleNamespace(
            jit=jit, shard_map=shard_map, named_scope=lambda _: nullcontext(),
            ShapeDtypeStruct=lambda shape, dtype, **kw: SimpleNamespace(shape=shape, **kw),
            lax=SimpleNamespace(all_to_all=collective, psum=collective),
        ),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "kernels.py", "exec"), namespace)
    mesh = object()
    count = 8 if parallel and benchmark == "all_to_all" else 16
    fn = namespace[function.name](mesh, count, 32, parallel=parallel)
    assert fn(payload) is output
    rows = 8 if parallel else 16
    assert captured["spec"].shape == (32 * rows, 8, 128)
    expected = ("d", None, None) if parallel else ("d",)
    assert captured["sharding"]["in_specs"] == expected
    assert captured["sharding"]["out_specs"] == expected


def test_ragged_kernel_accepts_prepared_send_buffer():
    tree = ast.parse((SOURCE_ROOT / "ici/kernels.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "compile_traffic_matrix_kernel")
    kernel = next(n for n in function.body if isinstance(n, ast.FunctionDef)
                  and n.name == "kernel")
    assert [a.arg for a in kernel.args.args] == ["traffic_matrix", "payload"]
    # Small offset vectors remain kernel arguments/operations; large payload
    # broadcasts must not be silently reintroduced inside the timed program.
    assignments = {
        target.id for n in ast.walk(kernel) if isinstance(n, ast.Assign)
        for target in n.targets if isinstance(target, ast.Name)
    }
    assert "payload" not in assignments
    assert "output" in assignments  # Required zeroed, non-received regions.
    assert any(isinstance(n, ast.Return) and isinstance(n.value, ast.Name)
               and n.value.id == "result" for n in ast.walk(kernel))
    donation = [kw.value for n in ast.walk(function) if isinstance(n, ast.Call)
                for kw in n.keywords if kw.arg == "donate_argnums"]
    assert donation == []


def test_ragged_buffers_are_reused_and_capacity_changes_release_old_pair():
    tree = ast.parse((SOURCE_ROOT / "ici/runner.py").read_text())
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and isinstance(n.test, ast.Compare)
                  and isinstance(n.test.left, ast.Name)
                  and n.test.left.id == "ragged_buffer_shape")
    created, released = [], []

    def prepare(mesh, count, rows, **kwargs):
        result = object()
        created.append((rows, kwargs.get("fill_zero", False), result))
        return result

    namespace = dict(ragged_buffer_shape=None, ragged_buffers=[None],
                     max_send=8, max_recv=16, mesh=object(), n_chiplets=16,
                     make_axis_sharded_payload=prepare,
                     delete_device_object=lambda pair: released.append(list(pair)))
    code = compile(ast.Module(body=[branch], type_ignores=[]), "runner.py", "exec")
    exec(code, namespace)
    original = namespace["ragged_buffers"]
    exec(code, namespace)
    assert namespace["ragged_buffers"] is original
    assert [(rows, zero) for rows, zero, _ in created] == [(8, False)]
    namespace["max_send"] = 32
    exec(code, namespace)
    assert released[-1] == original
    assert namespace["ragged_buffers"] is not original
    assert len(created) == 2


@pytest.mark.parametrize("parallel", [False, True])
def test_payload_is_initialized_on_device_with_matching_shard_values(parallel):
    tree = ast.parse((SOURCE_ROOT / "ici/kernels.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "make_axis_sharded_payload")
    captured = {}

    def shard_map(kernel, **kwargs):
        captured.update(kwargs)
        return kernel

    namespace = {
        "Mesh": object, "P": lambda *a: a,
        "PAYLOAD_MID_DIM": 8, "PAYLOAD_LAST_DIM": 128,
        "jax": SimpleNamespace(Array=object, jit=lambda f: f, shard_map=shard_map,
                               lax=SimpleNamespace(axis_index=lambda _: 3)),
        "jnp": SimpleNamespace(float32="float32",
                               full=lambda shape, value, **kw: (shape, value)),
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "kernels.py", "exec"), namespace)
    rows = 8 if parallel else 16
    spec = ("d", None, None) if parallel else ("d",)
    result = namespace[fn.name](SimpleNamespace(shape={"d": rows}), 16, 32,
                                partition_spec=spec, row_shard_count=rows)
    assert result == ((32, 8, 128), 4)
    assert captured["in_specs"] == ()
    assert captured["out_specs"] == spec


def test_cpu_timer_blocks_live_out_before_stopping_and_releases_afterwards():
    # Exercise the real synchronized loop with a fake clock/runtime: removing
    # ZeroCrop must not turn the measurement into asynchronous dispatch time.
    tree = ast.parse((SOURCE_ROOT / "utils/profiling.py").read_text())
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_synchronized_iterations"
    )
    events = []
    inputs, output = (object(),), object()
    ticks = iter([1.0, 1.081, 2.0, 2.082])

    def clock():
        events.append("clock")
        return next(ticks)

    def block(value):
        events.append("input_ready" if value is inputs else "output_ready")
        assert value is inputs or value is output

    def launch(*args):
        assert args == inputs
        events.append("launch")
        return output

    def release(value):
        assert value is output
        events.append("release")

    namespace = {
        "jax": SimpleNamespace(block_until_ready=block),
        "time": SimpleNamespace(perf_counter=clock),
        "delete_device_object": release,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "profiling.py", "exec"),
        namespace,
    )
    durations = namespace["run_synchronized_iterations"](launch, lambda: inputs, 2)
    assert durations == pytest.approx([81.0, 82.0])
    assert events == [
        "input_ready", "clock", "launch", "output_ready", "clock", "release",
    ] * 2


def test_profiled_cpu_timer_excludes_annotation_setup_teardown():
    tree = ast.parse((SOURCE_ROOT / "utils/profiling.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "run_profiled_iterations")
    events = []
    inputs, output = (object(),), object()
    ticks = iter([1.0, 1.081])

    @contextmanager
    def annotation(*args, **kwargs):
        events.append("annotation_start")
        yield
        events.append("annotation_end")

    def clock():
        events.append("clock")
        return next(ticks)

    def launch(*args):
        assert args == inputs
        events.append("launch")
        return output

    namespace = {
        "TraceTimingConfig": object,
        "benchmark_temporary_directory": lambda _: nullcontext("tmp"),
        "_trace_context": lambda *a, **kw: nullcontext(),
        "time": SimpleNamespace(perf_counter=clock),
        "jax": SimpleNamespace(
            block_until_ready=lambda x: events.append("input_ready" if x is inputs else "output_ready"),
            profiler=SimpleNamespace(StepTraceAnnotation=annotation),
        ),
        "delete_device_object": lambda _: events.append("release"),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "profiling.py", "exec"), namespace)
    config = SimpleNamespace(trace_only_xla=True, task_name="test", cleanup_trace=True)
    actual = namespace[function.name](launch, lambda: inputs, 1, config, False)
    assert actual == pytest.approx([81.0])
    assert events == ["input_ready", "annotation_start", "clock", "launch",
                      "output_ready", "clock", "annotation_end", "release"]
