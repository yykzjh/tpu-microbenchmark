# TPU Microbenchmark

Standalone TPU benchmarks for GEMM, host/device transfers, HBM, VMEM and ICI.
The Python programs and Bash launchers run directly on TPU VMs or existing
TPU containers; no acceptance service or scheduler integration is required.

## Install on every host

Requires Linux x86_64, Python 3.12+, Bash 4+, and a working TPU runtime/device
assignment. Distributed launchers also require OpenSSH client/server, `tar`,
`awk`, coreutils and `ip` (iproute2). The optional network test requires
iperf **2** and numactl on both hosts. This repository does not provision VMs,
start containers, install host TPU drivers or create SSH keys.

```bash
git clone https://github.com/yykzjh/tpu-microbenchmark.git
cd tpu-microbenchmark
uv sync --locked
```

The dependency file is `pyproject.toml`; `uv.lock` records the reproducible
resolution. Runtime versions are JAX/JAXlib 0.11.1, libtpu 0.0.46 and tpu-info
0.14.2, matching the synchronized benchmark environment. Dependencies resolve
from public sources only. TPU execution still requires real TPU hardware;
CPU-only regression tests do not verify TPU performance.

For SSH launchers, clone the same revision at the **same absolute path** on
every host and run `uv sync --locked` in each clone. Each launcher automatically
uses its own repository's `.venv/bin` and `src/`. No global `PYTHONPATH`,
`/root/bin` installation or platform-generated environment file is required.
Use the host/container environment supplied for that Slice; do not copy a
single host's TPU worker identity to other hosts.

## Input files and SSH requirements

| Entry point | Required input file |
| --- | --- |
| Direct Python GEMM / PCIe / HBM / VMEM | None |
| Direct Python ICI with `--runtime-scope local` | None |
| Direct Python ICI with `--runtime-scope slice` | None; launch every host with explicit coordinator, process count and rank |
| Shell GEMM / TPUBandwidth launchers | `--hostfile` and `--master` |
| Shell multi-host ICI | `--hostfile` |
| Shell iperf | `--hostfile`, with a two-host selection |

Create your own `hostfile` in the repository, one reachable host/IP per line:

```text
tpu-host-0
tpu-host-1
```

Blank lines and lines starting with `#` are ignored. Use unique bare hostnames
or IPv4 addresses, **not** `user@host`, ports, YAML or JSON. Replace the example
names with addresses reachable between the test containers/VMs. Real hostfiles
are gitignored and should not be published.

For ICI, the first host is rank 0 and the coordinator; **run the shell command
on that host**, and keep the hostfile in a stable order. The launcher reads the
file once and forwards the host list, so remote hosts do not need a copy of
the hostfile. All hosts must have the same checkout and Python environment.

SSH must work noninteractively between all participating hosts (binary-tree
fan-out), using the same login and port. Defaults: current user and port 22.
For existing root containers, use `--ssh-user root --ssh-port 2222` if that
is their configured SSH port. Host-key checking uses `accept-new`; changed
keys fail. Distribute/verify keys using your normal administration process.
TCP 8476 (the JAX coordinator) and the TPU runtime communication paths must
also be reachable. Do not run concurrent benchmarks against the same Slice.

## Retest AllReduce on a two-host 2x2x2 Slice

For the original block-to-block observation, exact two-host reproduction steps,
SparseCore on/off control, bandwidth accounting, and anonymized measurements,
see [AllReduce performance variance](docs/allreduce-performance-variance.md).

Use an **independent two-host 2x2x2 Slice**, with four TPU chips per host.
Put both hosts in the two-line hostfile above. The launcher derives the process
count from this file; Python discovers the device topology at runtime. Omitting
`--block-range` tests the entire Slice. The hostfile does not provision or resize
a Slice: selecting two hosts from an existing four-host Slice is not equivalent.

From host 0, inside the repository:

```bash
bash scripts/ici/test_multinode_ici.sh \
  --hostfile ./hostfile \
  --mode allreduce_parallel \
  --allreduce-data-size 4GiB \
  --warmup 2 --iterations 5 \
  --xprof-timing
```

This launches only parallel AllReduce, with temporary Xprof device timing.
For root containers append `--ssh-user root --ssh-port 2222`.
The root collects one log per host under `logs/ici/`. Successful execution
means the benchmark completed and produced metrics, not that a performance
threshold was met. Both hosts participate in this full-Slice retest.

To compare CPU timing, omit `--xprof-timing`; do not change any other parameters.

Without SSH, execute the Python program **on both hosts concurrently**,
using the same coordinator and count, with rank 0 on the first host and rank 1
on the second. Replace the placeholder coordinator hostname below locally:

```bash
# Set RANK=0 on the first host and RANK=1 on the second host.
RANK=0
uv run --locked python src/ici/test_ici.py ar \
  --runtime-scope slice \
  --coordinator-address tpu-host-0:8476 \
  --process-count 2 --process-id "$RANK" \
  --data-size 4GiB --warmup 2 --iteration 5 \
  --parallel --xprof-timing
```

Python writes results to stdout; redirect stdout/stderr yourself if using this
manual launch method. Only the shell launcher collects per-host logs.

## Other benchmarks

Direct Python commands run on the current host only and need no hostfile:

```bash
uv run --locked python src/gemm/test_gemm.py \
  --m 32768 --n 32768 --k 32768 --dtype bf16 --warmup 2 --iteration 5
uv run --locked python src/pcie/test_pcie.py h2d \
  --data-size 1GiB --target-devices 8 --pcie-mode one_to_one
uv run --locked python src/memory/test_hbm.py \
  --mode write --data-size 4GiB --block-shape 2048 1024 --dtype float32
uv run --locked python src/memory/test_vmem.py \
  --data-size 4GiB --block-shape 2048 1024 --dtype float32
uv run --locked python src/ici/test_ici.py ar \
  --runtime-scope local --data-size 4GiB --parallel
```

Use `--dtype fp16/bf16/fp8` for GEMM; `h2d/d2h` for PCIe; and
`--mode read/write/copy` for HBM. GEMM and memory programs use local devices;
ICI distinguishes independent `local` from distributed `slice` execution.
A local collective does not validate the full cross-host Slice.

For single-node tests across several hosts, use the dedicated shell launchers
from the host named by `--master`. Each host runs its local devices independently:

```bash
bash scripts/gemm/test_gemm.sh \
  --hostfile ./hostfile --master tpu-host-0 --dtype fp16,bf16,fp8
bash scripts/tpubandwidth/test_tpubandwidth.sh \
  --hostfile ./hostfile --master tpu-host-0 \
  --mode h2d,d2h,hbm_write --xprof-modes hbm_write
```

A one-line hostfile runs these shell launchers on just that host.

Full-Slice P2P and collective examples (run sequentially from host 0):

```bash
bash scripts/ici/test_multinode_ici.sh \
  --hostfile ./hostfile --mode p2p --p2p-pair-mode neighbors --xprof-timing
bash scripts/ici/test_multinode_ici.sh \
  --hostfile ./hostfile \
  --mode allreduce,allreduce_parallel,alltoall,alltoall_parallel \
  --xprof-modes allreduce,allreduce_parallel
```

`neighbors` skips Self, tests both chiplets bidirectionally within every chip,
and tests chiplet 0 to chiplet 0 on adjacent chips. For a 2x2x2 mesh this is
16 Die-to-Die plus 24 directed Chip-to-Chip pairs. Axes up to length 4 do not
wrap. Larger Slice adjacency is deliberately rejected in this mode.

## Logs and timing

Shell logs are timestamped and include node identity, per-stage wall times,
and `[COMMPILOT_METRIC]` structured results for compatibility. They live in
`logs/gemm/`, `logs/tpubandwidth/`, `logs/ici/` or `logs/iperf/`.
No result log is automatically uploaded.
Logs contain real node identities and runtime details; keep them private or
redact those fields before sharing. Only placeholder hostnames belong in public
examples; never commit real hostfiles, node identifiers or private IP addresses.

CPU timing is the default. `--xprof-timing` selects device-trace timing;
shell `--xprof-modes <csv>` selects individual subtests and, when nonempty,
overrides the all-subtests switch. Xprof collection adds wall-clock overhead.
Temporary traces are removed after timing; `--profile` explicitly retains
trace/HLO artifacts. The updated collective kernels use resident inputs and
return real outputs without the old ZeroCrop timing consumer.

The optional `scripts/oss/upload-logs-to-oss.sh` needs an installed `ossutil`
and your own `OSS_PATH/OSS_AK/OSS_SK/OSS_ENDPOINT/OSS_REGION` environment
variables. It is not a test prerequisite. Never commit credentials or logs.
Use `--logs-dir` if uploading logs outside this repository.

`scripts/cleanup_tpu_tests.sh --dry-run` previews matching local benchmark
processes. Review before executing without `--dry-run`: it stops benchmarks
and removes the libtpu lock file. Never use it on a host with another active
TPU workload.

## Regression tests

```bash
uv run --locked --group dev pytest -q
```

These CPU-only tests cover timing boundaries, traffic accounting and launcher
contracts. Actual TPU execution and measured bandwidth require hardware.
