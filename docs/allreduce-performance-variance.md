# L20G7 分块 AllReduce 性能差异复现说明

本文记录 2026-09-04 在 L20G7（TPU v7 / Ironwood）上发现的 AllReduce
Parallel 分块性能差异，以及随后在独立两节点 Slice 上完成的 SparseCore
AllReduce 下放开关对照。节点名称、IP、平台 Task/Run 标识和私有存储路径均已移除。

## 1. 现象与实测结果

最初测试使用一个四节点 `2x2x4` Slice。每个节点包含 4 个 TPU chip，整个
Slice 共 16 个 TPU chip / 32 个 JAX device（每个 chip 有两个 chiplet）。在
Z 轴上以 stride=1 滑动一个 `2x2x2` 窗口，得到三个 8-chip 分块。三个分块
使用完全相同的 AllReduce 参数并按顺序执行：

- `jax.lax.psum`；
- Parallel 模式：两个 chiplet 组并行，每组 8 个参与者；
- 每个 chiplet 输入 4 GiB FP32；
- warmup=2、iteration=5；
- CPU 同步计时（每轮对真实输出执行 `block_until_ready`）。

| `2x2x2` 分块 | AllReduce Parallel | 相对两个参考分块均值 |
| --- | ---: | ---: |
| `0:2,0:2,0:2` | **185.30 GB/s** | **-29.95%** |
| `0:2,0:2,1:3` | 264.51 GB/s | 基准样本 |
| `0:2,0:2,2:4` | 264.52 GB/s | 基准样本 |

两个参考分块均值为 264.515 GB/s，首个分块只有其约 70.05%。三组 P50
分别为 81.1234、56.8317、56.8280 ms。同轮
AllToAll Parallel 的三个分块分别为 159.70、159.69、159.69 GB/s，没有
出现对应的分块差异，因此现象集中在 AllReduce 的具体实现路径，不能直接
归纳为整个分块或全部 ICI 集合通信异常。

原始三分块测试每个分块只执行一组 warmup=2 / iteration=5，没有对每个分块
做多次独立进程复测。源码的 `ici_psum` profile 默认配置包含 SC AllReduce
下放开启，但当时日志没有记录最终生效的完整 `LIBTPU_INIT_ARGS`，也没有保留
HLO/trace；因此只能确认源码默认值，不能仅凭原日志证明异常分块实际完成了
SparseCore 下放。

历史上相同规模 `2x2x2`、4 GiB、warmup=2、iteration=5 的 AllReduce
Parallel 还测得过 279.38 GB/s。该数值用于说明正常路径可达到的量级，不能
替代同一轮三个分块之间的直接比较。

## 2. 分块测试是怎样执行的

原测试在完整四节点 Slice 上启动所有 JAX 分布式进程。程序先从
`jax.devices()` 自动推断完整 `2x2x4` 物理坐标，再用半开区间
`--block-range x0:x1,y0:y1,z0:z1` 选择分块中的 TPU chip。只有选中分块的
16 个 JAX device 组成 benchmark mesh；没有选中 device 的 host 仍加入 JAX
分布式进程并参与进程级同步，但不加入该分块的 `psum`。

Parallel 模式使用形状为 `(8, 2)`、轴名为 `("d", "chiplet")` 的 mesh，
沿 `d` 轴执行 `jax.lax.psum`，因此两个 chiplet 位置形成两个并行的 8-device
AllReduce 组。测试输入在计时前常驻 TPU，计时结果等待真实 AllReduce 输出
ready；当前实现不包含旧版 ZeroCrop 输出消费者。

脚本以二叉 SSH 树启动全部 host，并在根节点的 `logs/ici/` 下收集每个 host
的日志。公开仓库不包含原始日志，因为日志会带真实节点标识和运行环境信息；
[脱敏汇总日志](allreduce-performance-variance-results.txt) 只保留复核所需字段。

## 3. 推荐的独立两节点复测

为了向外部复现且不依赖 CommPilot，请申请一个独立的两节点 `2x2x2` Slice，
每节点 4 个 TPU chip。不要只从现有四节点 Slice 的 hostfile 中删掉两个节点；
那不会把物理 Slice 重新切成独立的两节点 Slice。

在两个 host 上检出相同 commit，并在相同绝对路径安装依赖：

```bash
git clone https://github.com/yykzjh/tpu-microbenchmark.git
cd tpu-microbenchmark
uv sync --locked
```

创建不提交到 Git 的 `hostfile`：

```text
tpu-host-0
tpu-host-1
```

确认两个 host 之间无密码 SSH 可用，且仓库绝对路径、Python 环境和代码版本
相同。从 host 0 执行：

```bash
bash scripts/ici/test_multinode_ici.sh \
  --hostfile ./hostfile \
  --mode allreduce_parallel \
  --allreduce-data-size 4GiB \
  --warmup 2 \
  --iterations 5 \
  --xprof-timing
```

根容器若使用 SSH 2222 端口，追加
`--ssh-user root --ssh-port 2222`。两节点独立 Slice 应省略 `--block-range`，
让程序测试完整的 `2x2x2` Slice。

结果写入 `logs/ici/`。可用以下命令提取标准指标：

```bash
rg 'COMMPILOT_METRIC.*allreduce_parallel' logs/ici
```

日志同时应显示 `slice_topology=2x2x2`、16 个 chiplet、两个并行组、Xprof
计时来源以及 4 GiB 输入。若任一项不同，不应将结果与本文数值直接比较。

## 4. SparseCore AllReduce 下放的最小 A/B 对照

当前 `ici_psum` 默认参数包含：

```text
--xla_sc_disable_megacore_partitioning=true
--xla_tpu_disable_sparse_core_collective_offload_remover=true
--xla_tpu_enable_all_reduce_offload_tracing=true
--xla_tpu_enable_all_reduce_scatter_fusion=false
--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true
--xla_tpu_pad_operations_input_tiles=true
--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=0
--xla_tpu_use_tc_device_shape_on_sc=true
```

`src/utils/runtime.py` 的合并规则是“调用方已有同名 flag 优先”，所以可只覆盖
目标 flag，其他默认参数保持不变。为了保证两个 host 的实际 flags 完全一致，
以下 A/B 建议分别在两个终端并发直接启动 Python，而不是只在 SSH 根进程设置
环境变量。

两组都先在两个 host 上设置同一值：

```bash
# A 组：两个 host 都设置 true；B 组：两个 host 都设置 false。
export SC_ALLREDUCE_OFFLOAD=true
export LIBTPU_INIT_ARGS="--xla_tpu_enable_sparse_core_collective_offload_all_reduce=${SC_ALLREDUCE_OFFLOAD}"
```

然后在 host 0 执行：

```bash
uv run --locked python src/ici/test_ici.py ar \
  --runtime-scope slice \
  --coordinator-address tpu-host-0:8476 \
  --process-count 2 \
  --process-id 0 \
  --data-size 4GiB \
  --warmup 2 \
  --iteration 5 \
  --parallel \
  --xprof-timing \
  --profile \
  --result-dir ./results/sc-${SC_ALLREDUCE_OFFLOAD}/rank0
```

同时在 host 1 执行相同命令，只把 `--process-id` 和结果目录改为：

```text
--process-id 1
--result-dir ./results/sc-${SC_ALLREDUCE_OFFLOAD}/rank1
```

先完整执行 `true` 组，确认两个进程都退出后再执行 `false` 组。不要并发运行
两组，也不要在同一 Slice 上同时运行其他 TPU workload。`--profile` 会保留
Xprof trace 和 HLO；如果只需要性能数值，可去掉 `--profile`，但仍保留
`--xprof-timing`。

受控两节点 `2x2x2` Slice 对照结果如下。两组只切换上述一个 flag，输入 HLO
在优化前字节级一致：

| 指标 | SC 开启 | SC 关闭 | 差异 |
| --- | ---: | ---: | ---: |
| AllReduce 设备耗时 P50 | 53.80 ms | 80.85 ms | 开启后减少 33.46% |
| 每 TPU 有效单向 ICI 带宽 | **279.40 GB/s** | **185.94 GB/s** | 开启后提高 50.26% |
| 完整设备程序耗时（含 Copy） | 56.57 ms | 80.93 ms | 开启后减少 30.10% |
| 单次程序墙钟时间 | 19.76 s | 18.33 s | profile 固定开销主导，不作带宽依据 |

优化后 HLO 显示，开启组被转换为 `sparsecore` 异步执行线程和
`OFFLOAD_COLLECTIVE`，其配置包含 `use_n_dimension_strategy=true`；关闭组保留
普通 AllReduce 实现。开启组即使多出约 2.68 ms 的输出 Copy，完整设备程序
仍明显更快。两组 trace 都没有 ZeroCrop，也没有 GEMM 与通信重叠，因此本次
差异不是 CPU 计时边界、ZeroCrop 或计算重叠造成的假象。

## 5. 带宽口径

Parallel 模式把每个 chiplet 的 `D=4 GiB` 输入分成两个独立并行组。每组
`N=8` 个参与者，标准 AllReduce bus traffic 为 `D * 2 * (N-1) / N`，两个
组相加后，以一个 TPU chip（两个 chiplet）的总 ICI 带宽报告：

```text
traffic_bytes = D * 2 * (N - 1) / N * 2
bandwidth_GBps = traffic_bytes / Xprof_device_duration
```

这里 GB/s 使用十进制 `10^9 bytes/s`。4 GiB、N=8 时分子为 14 GiB；以
53.80 ms 计算约为 279.4 GB/s。这个口径不是单个 chiplet 的带宽，也不是
整个 Slice 所有 TPU 的聚合带宽。

## 6. 当前结论边界

现有证据证明：在受控两节点测试中，SC 开启会选择更快的设备端集合通信实现；
关闭 SC 后的 185.94 GB/s 又与原异常分块 185.30 GB/s 很接近。但“数值接近”
不是根因证明，不能据此断言原四节点首分块当时一定没有成功下放到 SparseCore。

要最终解释原始分块差异，需要在同一个四节点 `2x2x4` Slice 上保存三个分块
各自的优化后 HLO 和 Xprof trace，核对异常分块是否缺少
`OFFLOAD_COLLECTIVE` / `use_n_dimension_strategy=true`，并结合运行时 ICI
路由及硬件错误计数器排除链路或路由因素。AllToAll 同轮稳定只能缩小范围，
不能单独排除某条仅由 AllReduce 路径使用的硬件或路由资源。
