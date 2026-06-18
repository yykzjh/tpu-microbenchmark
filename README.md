# TPU Microbenchmark

TPU 微基准测试套件，用于测量 TPU 硬件的关键性能指标，涵盖内存带宽、矩阵计算吞吐、PCIe 传输带宽和芯片间互联带宽。

## 目录结构

```
├── pyproject.toml              # uv 项目配置与依赖
├── scripts/
│   └── cleanup_tpu_tests.sh    # 测试进程清理脚本
└── src/
    ├── memory/                 # HBM / VMEM 内存带宽测试
    │   ├── test_hbm.py         # HBM 带宽基准 (read/write/copy)
    │   ├── test_vmem.py        # VMEM 带宽基准
    │   ├── kernels.py          # Pallas TPU 内核
    │   ├── arrays.py           # 测试数组构造
    │   ├── benchmark.py        # 基准执行编排
    │   └── options.py          # CLI 参数与结果输出
    ├── gemm/                   # 矩阵乘法 (GEMM) 计算吞吐测试
    │   └── test_gemm.py
    ├── pcie/                   # PCIe 主机-设备传输带宽测试
    │   └── test_pcie.py
    ├── ici/                    # ICI 芯片间互联带宽测试
    │   ├── test_ici.py         # CLI 入口 (raw/p2p/p2p-rdma/a2a/ar)
    │   ├── runner.py           # 基准编排
    │   ├── kernels.py          # JAX/Pallas 通信内核
    │   ├── traffic.py          # 拓扑推断与流量矩阵
    │   ├── phases.py           # 执行阶段 (warmup/timed)
    │   ├── constants.py        # 共享常量
    │   ├── cleanup.py          # 运行时内存管理
    │   └── zero_crop.py        # ZeroCrop FFI 自定义原语
    └── utils/                  # 共享工具
        ├── runtime.py          # TPU 环境配置与 JAX 初始化
        ├── profiling.py        # xprof 追踪与 HLO dump
        ├── metrics.py          # 统计指标与 JSONL 输出
        └── units.py            # 数据大小解析与单位转换
```

## 环境要求

- **硬件**: TPU 设备（ICI 测试需要多芯片 TPU，其他测试单芯片即可）
- **OS**: Linux x86_64（JAX TPU 仅支持 Linux）
- **Python**: >= 3.12
- **包管理**: [uv](https://docs.astral.sh/uv/)

## 环境安装

```bash
# 安装 uv (如果未安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境并安装依赖
uv sync
```

uv 会自动从 PyPI 和 Google JAX releases 安装所有依赖，包括 `jax[tpu]`、`jaxlib`、`libtpu` 等。

## 推荐测试命令

所有测试都是独立的 CLI 脚本，通过 `uv run python -m` 运行（不是 pytest）。每个测试在运行前会自动配置 TPU/XLA 环境变量。

### GEMM 计算吞吐测试

测量矩阵乘法 (GEMM) 的 TFLOP/s。

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

**参数说明:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--m` | (必填) | 矩阵 M 维度 |
| `--n` | (必填) | 矩阵 N 维度 |
| `--k` | (必填) | 矩阵 K 维度 |
| `--dtype` | `bf16` | 输入数据类型: `fp32`, `fp16`, `bf16`, `fp8` |
| `--warmup` | `2` | 预热迭代次数 |
| `--iteration` | `5` | 计时迭代次数 |
| `--result-dir` | `./results` | 结果输出目录 |
| `--dump-hlo` | `false` | 收集 XLA HLO dump |
| `--cleanup-trace` | `false` | 提取时间后删除 xprof trace |

### PCIe 传输带宽测试

测量 CPU <-> TPU 的 PCIe 传输带宽 (GB/s)。

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

**参数说明:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `h2d` / `d2h` | (必填) | 子命令: `h2d` (主机->设备) 或 `d2h` (设备->主机) |
| `--data-size` | (必填) | 每个目标设备的传输大小，需对齐到 512 字节 |
| `--warmup` | `2` | 预热迭代次数 |
| `--iteration` | `5` | 计时迭代次数 |
| `--target-devices` | `8` | 测试的 TPU chiplet 设备数 (默认 8 = 4 chips x 2 chiplets) |
| `--result-dir` | `./results` | 结果输出目录 |
| `--cleanup-trace` | `false` | 提取时间后删除 xprof trace |

每个方向会测试两种模式:
- **one_to_one**: 逐设备串行传输
- **one_to_many**: 所有设备并发传输

### HBM 带宽测试

测量 HBM (High Bandwidth Memory) 的读/写/拷贝带宽。

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

**参数说明:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `all` | 测试模式: `read`, `write`, `copy`, `all` |
| `--data-size` | `256MiB` | HBM 数据大小，需对齐到 block shape |
| `--block-shape` | `128 1024` | Pallas block shape，末维须被 128 整除，次末维须被 8 整除。推荐 HBM 使用 `4096 1024`，VMEM 使用 `2048 1024` |
| `--dtype` | `float32` | 数据类型: `float32`, `float16`, `bfloat16` |
| `--warmup` | `2` | 预热迭代次数 |
| `--iteration` | `5` | 计时迭代次数 |
| `--result-dir` | `./results` | 结果输出目录 |
| `--dump-hlo` | `false` | 收集 XLA HLO dump |
| `--cleanup-trace` | `false` | 提取时间后删除 xprof trace |

### VMEM 带宽测试

测量 VMEM (Vector Memory / 片上向量内存) 的拷贝带宽。

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

参数与 HBM 测试相同（VMEM 测试不含 `--mode`，固定为 VMEM copy 模式）。

### ICI 互联带宽测试

测量 TPU 芯片间互联 (Inter-Chip Interconnect) 带宽。需要多芯片 TPU 环境。

```bash
# P2P (使用 raw 子命令 + 自定义流量矩阵，8x8 TPU 芯片)
uv run python -m src.ici.test_ici \
  raw \
  --traffic-matrix "#1,0,0,0,0,0,0,1#0,0,0,0,0,0,0,0#0,0,0,0,0,0,0,0#0,0,0,0,0,0,0,0#0,0,0,0,0,0,0,0#0,0,0,0,0,0,0,0#0,0,0,0,0,0,0,0#0,0,0,0,0,0,0,0" \
  --data-size 1GiB \
  --warmup 2 \
  --iteration 5 \
  --result-dir ./commpilot_results \
  --dump-hlo

# All-to-All 集合通信
uv run python -m src.ici.test_ici a2a \
  --data-size 1GiB \
  --warmup 2 \
  --iteration 5 \
  --result-dir ./commpilot_results \
  --dump-hlo

# All-Reduce (psum) 集合通信
uv run python -m src.ici.test_ici ar \
  --data-size 4GiB \
  --warmup 2 \
  --iteration 5 \
  --result-dir ./commpilot_results \
  --dump-hlo
```

**子命令:**

| 子命令 | 说明 |
|--------|------|
| `raw` | 用户自定义流量矩阵，支持 `--traffic-matrix` 和 `--concurrent` |
| `p2p` | 内置全 1 流量矩阵，逐链路拆分测试 |
| `p2p-rdma` | 使用 Pallas Remote DMA 的逐链路 P2P 测试 |
| `a2a` | All-to-All 集合通信，报告 inter-TPU ICI 带宽 |
| `ar` | All-Reduce (psum)，报告总线带宽 |

**参数说明:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-size` | (必填) | Payload 大小，需对齐到 4096 字节 (8x128 float32) |
| `--warmup` | `2` | 预热迭代次数 |
| `--iteration` | `5` | 计时迭代次数 |
| `--result-dir` | `./results` | 结果输出目录 |
| `--dump-hlo` | `false` | 收集 XLA HLO dump |
| `--cleanup-trace` | `false` | 提取时间后删除 xprof trace |
| `--traffic-matrix` | (仅 raw) | 流量矩阵，格式: `#0,1#1,0` (每行 `#` 开头，`,` 分隔) |
| `--concurrent` | (仅 raw) | 以单矩阵模式运行（默认拆分为逐链路测试） |

## 输出结果

每个测试在 `--result-dir` 下创建带时间戳的输出目录，包含:

- `metrics/metrics_report.jsonl` -- 结构化 JSONL 指标，包含带宽、延迟百分位等
- `trace/` -- xprof 追踪文件
- `dump_hlo/` -- XLA HLO dump（需启用 `--dump-hlo`）

## 测试清理

测试结束后（特别是异常退出时），运行清理脚本释放 TPU 资源:

```bash
# 清理残留测试进程和 libtpu 锁文件
bash scripts/cleanup_tpu_tests.sh

# 预览将要清理的内容（不实际执行）
bash scripts/cleanup_tpu_tests.sh --dry-run
```

清理脚本会:
1. 终止匹配 TPU 测试模式 (`test_hbm`, `test_vmem`, `test_gemm`, `test_pcie`, `test_ici`) 的残留进程
2. 删除 `/tmp/libtpu_lockfile` 锁文件（崩溃的测试可能遗留）

## 测试隔离说明

- **每个测试必须独立运行**：各测试在 JAX 导入前设置不同的 `LIBTPU_INIT_ARGS` XLA 编译标志，这些标志在进程内不可更改
- **顺序执行**: 测试之间应串行运行，避免 TPU 资源争用
- **推荐执行顺序**: HBM -> VMEM -> GEMM -> PCIe -> ICI，每个测试之间运行清理脚本
- **ICI 测试必须隔离**: 调用 `jax.distributed.initialize()` 建立全局多主机状态，不可在同一进程中重新初始化
