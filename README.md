# TPU Microbenchmark

TPU microbenchmark suite for measuring key performance metrics of TPU hardware, covering memory bandwidth, matrix computation throughput, PCIe transfer bandwidth, and inter-chip interconnect bandwidth.

## Directory Structure

```
├── pyproject.toml              # uv project configuration and dependencies
├── scripts/
│   └── cleanup_tpu_tests.sh    # Test process cleanup script
└── src/
    ├── memory/                 # HBM / VMEM memory bandwidth tests
    │   ├── test_hbm.py         # HBM bandwidth benchmark (read/write/copy)
    │   ├── test_vmem.py        # VMEM bandwidth benchmark
    │   ├── kernels.py          # Pallas TPU kernels
    │   ├── arrays.py           # Test array construction
    │   ├── benchmark.py        # Benchmark execution orchestration
    │   └── options.py          # CLI arguments and result output
    ├── gemm/                   # Matrix multiplication (GEMM) compute throughput tests
    │   └── test_gemm.py
    ├── pcie/                   # PCIe host-device transfer bandwidth tests
    │   └── test_pcie.py
    ├── ici/                    # ICI inter-chip interconnect bandwidth tests
    │   ├── test_ici.py         # CLI entry point (raw/p2p/p2p-rdma/a2a/ar)
    │   ├── runner.py           # Benchmark orchestration
    │   ├── kernels.py          # JAX/Pallas communication kernels
    │   ├── traffic.py          # Topology inference and traffic matrix
    │   ├── phases.py           # Execution phases (warmup/timed)
    │   ├── constants.py        # Shared constants
    │   ├── cleanup.py          # Runtime memory management
    │   └── zero_crop.py        # ZeroCrop FFI custom primitive
    └── utils/                  # Shared utilities
        ├── runtime.py          # TPU environment setup and JAX initialization
        ├── profiling.py        # xprof tracing and HLO dump
        ├── metrics.py          # Statistical metrics and JSONL output
        └── units.py            # Data size parsing and unit conversion
```

## Prerequisites

- **Hardware**: TPU device (ICI tests require multi-chip TPU; other tests work with a single chip)
- **OS**: Linux x86_64 (JAX TPU only supports Linux)
- **Python**: >= 3.12
- **Package Manager**: [uv](https://docs.astral.sh/uv/)

## Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync
```

uv will automatically install all dependencies from PyPI and Google JAX releases, including `jax[tpu]`, `jaxlib`, `libtpu`, etc.

## Recommended Test Commands

All tests are standalone CLI scripts, run via `uv run python -m` (not pytest). Each test automatically configures TPU/XLA environment variables before running.

### GEMM Compute Throughput Test

Measures matrix multiplication (GEMM) TFLOP/s.

```bash
# fp16
uv run python -m src.gemm.test_gemm \
  --m 32768 \
  --n 32768 \
  --k 32768 \
  --dtype fp16 \
  --warmup 2 \
  --iteration 5 \
  --result-dir ./commpilot_results \
  --dump-hlo

# bf16
uv run python -m src.gemm.test_gemm \
  --m 32768 \
  --n 32768 \
  --k 32768 \
  --dtype bf16 \
  --warmup 2 \
  --iteration 5 \
  --result-dir ./commpilot_results \
  --dump-hlo

# fp8
uv run python -m src.gemm.test_gemm \
  --m 32768 \
  --n 32768 \
  --k 32768 \
  --dtype fp8 \
  --warmup 2 \
  --iteration 5 \
  --result-dir ./commpilot_results \
  --dump-hlo
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--m` | (required) | Matrix M dimension |
| `--n` | (required) | Matrix N dimension |
| `--k` | (required) | Matrix K dimension |
| `--dtype` | `bf16` | Input data type: `fp32`, `fp16`, `bf16`, `fp8` |
| `--warmup` | `2` | Number of warmup iterations |
| `--iteration` | `5` | Number of timed iterations |
| `--result-dir` | `./results` | Output directory for results |
| `--dump-hlo` | `false` | Collect XLA HLO dump |
| `--cleanup-trace` | `false` | Delete xprof trace after extracting timings |

### PCIe Transfer Bandwidth Test

Measures CPU <-> TPU PCIe transfer bandwidth (GB/s).

```bash
# Host-to-Device (CPU -> TPU)
uv run python -m src.pcie.test_pcie h2d \
  --data-size 1GiB \
  --warmup 2 \
  --iteration 5 \
  --result-dir ./commpilot_results

# Device-to-Host (TPU -> CPU)
uv run python -m src.pcie.test_pcie d2h \
  --data-size 1GiB \
  --warmup 2 \
  --iteration 5 \
  --result-dir ./commpilot_results
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `h2d` / `d2h` | (required) | Subcommand: `h2d` (host->device) or `d2h` (device->host) |
| `--data-size` | (required) | Transfer size per target device, must be aligned to 512 bytes |
| `--warmup` | `2` | Number of warmup iterations |
| `--iteration` | `5` | Number of timed iterations |
| `--target-devices` | `8` | Number of TPU chiplet devices to test (default 8 = 4 chips x 2 chiplets) |
| `--result-dir` | `./results` | Output directory for results |
| `--cleanup-trace` | `false` | Delete xprof trace after extracting timings |

Each direction tests two modes:
- **one_to_one**: Sequential transfer per device
- **one_to_many**: Concurrent transfer across all devices

### HBM Bandwidth Test

Measures HBM (High Bandwidth Memory) read/write/copy bandwidth.

```bash
# Read
uv run python -m src.memory.test_hbm \
  --mode read \
  --data-size 4GiB \
  --block-shape 4096 1024 \
  --dtype float32 \
  --warmup 2 \
  --iteration 5 \
  --result-dir ./commpilot_results \
  --dump-hlo

# Write
uv run python -m src.memory.test_hbm \
  --mode write \
  --data-size 4GiB \
  --block-shape 4096 1024 \
  --dtype float32 \
  --warmup 2 \
  --iteration 5 \
  --result-dir ./commpilot_results \
  --dump-hlo

# Copy
uv run python -m src.memory.test_hbm \
  --mode copy \
  --data-size 4GiB \
  --block-shape 4096 1024 \
  --dtype float32 \
  --warmup 2 \
  --iteration 5 \
  --result-dir ./commpilot_results \
  --dump-hlo
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | `all` | Test mode: `read`, `write`, `copy`, `all` |
| `--data-size` | `256MiB` | HBM data size, must be aligned to block shape |
| `--block-shape` | `128 1024` | Pallas block shape; last dimension must be divisible by 128, second-to-last dimension must be divisible by 8. Recommended: `4096 1024` for HBM, `2048 1024` for VMEM |
| `--dtype` | `float32` | Data type: `float32`, `float16`, `bfloat16` |
| `--warmup` | `2` | Number of warmup iterations |
| `--iteration` | `5` | Number of timed iterations |
| `--result-dir` | `./results` | Output directory for results |
| `--dump-hlo` | `false` | Collect XLA HLO dump |
| `--cleanup-trace` | `false` | Delete xprof trace after extracting timings |

### VMEM Bandwidth Test

Measures VMEM (Vector Memory / on-chip vector memory) copy bandwidth.

```bash
uv run python -m src.memory.test_vmem \
  --data-size 4GiB \
  --block-shape 2048 1024 \
  --dtype float32 \
  --warmup 2 \
  --iteration 5 \
  --result-dir ./commpilot_results \
  --dump-hlo
```

Parameters are the same as the HBM test (VMEM test does not include `--mode`; it is fixed to VMEM copy mode).

### ICI Interconnect Bandwidth Test

Measures TPU inter-chip interconnect (ICI) bandwidth. Requires a multi-chip TPU environment.

```bash
# P2P (using raw subcommand + custom traffic matrix, 8x8 TPU chips)
uv run python -m src.ici.test_ici \
  raw \
  --traffic-matrix "#1,0,0,0,0,0,0,1#0,0,0,0,0,0,0,0#0,0,0,0,0,0,0,0#0,0,0,0,0,0,0,0#0,0,0,0,0,0,0,0#0,0,0,0,0,0,0,0#0,0,0,0,0,0,0,0#0,0,0,0,0,0,0,0" \
  --data-size 1GiB \
  --warmup 2 \
  --iteration 5 \
  --result-dir ./commpilot_results \
  --dump-hlo

# All-to-All collective communication
uv run python -m src.ici.test_ici a2a \
  --data-size 1GiB \
  --warmup 2 \
  --iteration 5 \
  --result-dir ./commpilot_results \
  --dump-hlo

# All-Reduce (psum) collective communication
uv run python -m src.ici.test_ici ar \
  --data-size 4GiB \
  --warmup 2 \
  --iteration 5 \
  --result-dir ./commpilot_results \
  --dump-hlo
```

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `raw` | User-defined traffic matrix, supports `--traffic-matrix` and `--concurrent` |
| `p2p` | Built-in all-ones traffic matrix, per-link breakdown test |
| `p2p-rdma` | Per-link P2P test using Pallas Remote DMA |
| `a2a` | All-to-All collective communication, reports inter-TPU ICI bandwidth |
| `ar` | All-Reduce (psum), reports bus bandwidth |

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data-size` | (required) | Payload size, must be aligned to 4096 bytes (8x128 float32) |
| `--warmup` | `2` | Number of warmup iterations |
| `--iteration` | `5` | Number of timed iterations |
| `--result-dir` | `./results` | Output directory for results |
| `--dump-hlo` | `false` | Collect XLA HLO dump |
| `--cleanup-trace` | `false` | Delete xprof trace after extracting timings |
| `--traffic-matrix` | (raw only) | Traffic matrix, format: `#0,1#1,0` (each row starts with `#`, separated by `,`) |
| `--concurrent` | (raw only) | Run in single-matrix mode (default splits into per-link tests) |

## Output Results

Each test creates a timestamped output directory under `--result-dir`, containing:

- `metrics/metrics_report.jsonl` -- Structured JSONL metrics, including bandwidth, latency percentiles, etc.
- `trace/` -- xprof trace files
- `dump_hlo/` -- XLA HLO dump (requires `--dump-hlo`)

## Test Cleanup

After tests complete (especially after abnormal exits), run the cleanup script to release TPU resources:

```bash
# Clean up residual test processes and libtpu lock files
bash scripts/cleanup_tpu_tests.sh

# Preview what will be cleaned (without actually executing)
bash scripts/cleanup_tpu_tests.sh --dry-run
```

The cleanup script will:
1. Terminate residual processes matching TPU test patterns (`test_hbm`, `test_vmem`, `test_gemm`, `test_pcie`, `test_ici`)
2. Delete `/tmp/libtpu_lockfile` lock files (which may be left behind by crashed tests)

## Test Isolation Notes

- **Each test must run independently**: Each test sets different `LIBTPU_INIT_ARGS` XLA compilation flags before importing JAX; these flags cannot be changed within the same process
- **Sequential execution**: Tests should run serially to avoid TPU resource contention
- **Recommended execution order**: HBM -> VMEM -> GEMM -> PCIe -> ICI, running the cleanup script between each test
- **ICI tests must be isolated**: Calls `jax.distributed.initialize()` to establish global multi-host state, which cannot be re-initialized within the same process
