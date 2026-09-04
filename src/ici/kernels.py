"""JAX and Pallas kernel compilers for TPU ICI benchmarks."""

from __future__ import annotations

from typing import Any

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
import jax.experimental.pallas.tpu as pltpu
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ici.constants import (
    PAYLOAD_LAST_DIM,
    PAYLOAD_MID_DIM,
)
from utils.profiling import MARKER

def create_device_mesh(
    num_devices: int,
    devices: list[Any] | None = None,
) -> Mesh:
    """Create 1-D JAX device mesh over all devices.

    A flat 1-D mesh ``("d",)`` is used because the communication pattern
    is entirely defined by the traffic matrix; the physical topology
    (torus coordinates) is used for inferred metadata.

    TPU matrix index ``i`` maps to logical devices ``2*i`` and ``2*i+1`` in
    this mesh. :func:`infer_tpu_topology` validates that those consecutive
    device pairs expose the same TPU chip coordinates.
    """
    selected_devices = list(jax.devices() if devices is None else devices)
    if len(selected_devices) < num_devices:
        raise ValueError(
            f"Need {num_devices} devices, only {len(selected_devices)} available. "
            f"Check that jax.distributed.initialize() succeeded on all hosts."
        )
    return Mesh(np.asarray(selected_devices[:num_devices]), ("d",))


def create_parallel_chiplet_mesh(
    num_devices: int,
    devices: list[Any] | None = None,
) -> Mesh:
    """Create a 2-D mesh that separates the two chiplets on each TPU chip.

    The mesh is created through JAX's topology-aware mesh builder instead of
    manually reshaping ``jax.devices()``. This gives the runtime a better chance
    to place the intra-chip chiplet dimension on the last mesh axis. A
    collective over axis ``"d"`` therefore creates two parallel replica groups,
    one per chiplet coordinate on the ``"chiplet"`` axis.
    """
    if num_devices % 2 != 0:
        raise ValueError(
            f"Parallel chiplet mesh requires an even device count, got {num_devices}"
        )
    selected_devices = list(jax.devices() if devices is None else devices)
    if len(selected_devices) < num_devices:
        raise ValueError(
            f"Need {num_devices} devices, only {len(selected_devices)} available. "
            f"Check that jax.distributed.initialize() succeeded on all hosts."
        )
    return jax.make_mesh(
        (num_devices // 2, 2),
        ("d", "chiplet"),
        devices=selected_devices[:num_devices],
    )


def make_axis_sharded_payload(
    mesh: Mesh,
    num_devices: int,
    payload_rows_per_link: int,
    partition_spec: P | None = None,
    row_shard_count: int | None = None,
) -> jax.Array:
    """Prepare immutable payloads on TPU, outside the measured executable.

    Materializing every local GiB shard in NumPy adds host allocation and H2D
    upload to each case. A separate device initializer creates the same values
    once; the collective still receives a real dynamic input, not a constant
    that the compiler can fold into its measured executable.
    """
    partition_spec = partition_spec or P("d")
    row_shard_count = row_shard_count or num_devices
    if row_shard_count != mesh.shape["d"]:
        raise ValueError("Payload row shard count must match the mesh d axis")

    def initialize():
        return jnp.full(
            (payload_rows_per_link, PAYLOAD_MID_DIM, PAYLOAD_LAST_DIM),
            jax.lax.axis_index("d") + 1,
            dtype=jnp.float32,
        )

    return jax.jit(jax.shard_map(
        initialize, mesh=mesh, in_specs=(), out_specs=partition_spec,
        check_vma=False,
    ))()


def compile_traffic_matrix_kernel(
    mesh: Mesh,
    num_devices: int,
    max_send_rows: int,
    max_recv_rows: int,
):
    """Compile a ragged_all_to_all P2P kernel.

    The kernel takes a traffic matrix (num_devices x num_devices, int32, in
    **payload row counts**) as a dynamic input. This allows the same compiled
    function to be reused with different traffic patterns without
    recompilation, as long as no device exceeds the buffer capacity.

    Args:
        mesh: 1-D JAX device mesh with axis ``"d"``.
        num_devices: total number of devices in the mesh.
        max_send_rows: max total send rows across any single device
            (compile-time constant, determines operand buffer shape).
        max_recv_rows: max total recv rows across any single device
            (compile-time constant, determines output buffer shape).
    Returns:
        A compiled function: ``(traffic_matrix, payload) -> buffers``.
    """

    def kernel(traffic_matrix, payload):
        me = jax.lax.axis_index("d")

        # --- send/recv sizes (payload rows, NOT bytes) ---
        # One payload row has shape (8, 128) and contains 4096 bytes of
        # float32 data. ragged_all_to_all sizes/offsets index the first
        # dimension of payload/output.
        # send_sizes[dst]: how many rows this device sends to dst
        send_sizes = traffic_matrix[me, :]
        # recv_sizes[src]: how many rows this device receives from src
        recv_sizes = traffic_matrix[:, me]

        # --- input_offsets: where in the operand each dst's data starts ---
        # Cumulative sum of send_sizes gives the starting position for each
        # destination's data within the operand's first dimension.
        # Example: send_sizes [0, 4, 4] -> offsets [0, 0, 4].
        input_offsets = jnp.concatenate([
            jnp.zeros((1,), jnp.int32),
            jnp.cumsum(send_sizes)[:-1],
        ])

        # --- output_offsets: per JAX ragged_all_to_all semantics ---
        # output_offsets[dst] = the offset in device `dst`'s output buffer
        # where this sender's data will be written. This is the sender's
        # declaration of write position, NOT the receiver's local offset.
        #
        # prefix_by_src[src, me] = sum of traffic[0..src-1, me], i.e. the
        # cumulative elements that sources 0..src-1 send TO device `me`.
        # When device `me` acts as sender to `dst`, it declares offset =
        # prefix_by_src[me, dst] = sum of traffic[0..me-1, dst].
        prefix_by_src = jnp.cumsum(traffic_matrix, axis=0) - traffic_matrix
        output_offsets = prefix_by_src[me, :]

        # Ragged transfers leave non-received regions unchanged. Keep the
        # zero initialization local to this executable: passing a large
        # external output buffer introduces TPU copies even with donation.
        output = jnp.zeros(
            (max_recv_rows, PAYLOAD_MID_DIM, PAYLOAD_LAST_DIM), jnp.float32,
        )

        with jax.named_scope(MARKER):
            # Return the real output: a live-out is sufficient to keep the
            # collective, without an additional full-buffer FFI consumer.
            result = jax.lax.ragged_all_to_all(
                operand=payload,
                output=output,
                input_offsets=input_offsets,
                send_sizes=send_sizes,
                output_offsets=output_offsets,
                recv_sizes=recv_sizes,
                axis_name="d",
            )

        return result

    return (
        jax.jit(
            jax.shard_map(
                kernel,
                mesh=mesh,
                in_specs=(P(), P("d")),
                out_specs=P("d"),
                check_vma=False,
            ),
        )
        .lower(
            jax.ShapeDtypeStruct((num_devices, num_devices), jnp.int32),
            jax.ShapeDtypeStruct(
                (max_send_rows * num_devices, PAYLOAD_MID_DIM, PAYLOAD_LAST_DIM),
                jnp.float32, sharding=NamedSharding(mesh, P("d")),
            ),
        )
        .compile()
    )


def compile_self_copy_kernel(
    mesh: Mesh,
    num_devices: int,
    payload_rows_per_link: int,
    target_chiplet: int,
):
    """Compile a local JAX elementwise kernel for one self P2P chiplet case.

    ``p2p`` self pairs are not real ICI transfers. They model a local copy from
    one HBM-backed array location to another on the same chiplet. Plain
    ``Array.copy()`` can lower to an identity in XLA, so use ``x + 1.0`` to
    force a local read/write kernel while preserving the same data-size
    bandwidth numerator as the former self-copy path.
    """
    if target_chiplet < 0 or target_chiplet >= num_devices:
        raise ValueError(
            f"target_chiplet must be in [0, {num_devices}), got "
            f"{target_chiplet}"
        )

    local_payload_shape = (
        payload_rows_per_link,
        PAYLOAD_MID_DIM,
        PAYLOAD_LAST_DIM,
    )
    global_payload_shape = (
        payload_rows_per_link * num_devices,
        PAYLOAD_MID_DIM,
        PAYLOAD_LAST_DIM,
    )

    def kernel(payload):
        def shard_fn(local_payload):
            me = jax.lax.axis_index("d")

            def copy_like_read_write(x):
                return x + jnp.float32(1.0)

            def keep_local(x):
                return x

            with jax.named_scope(MARKER):
                return jax.lax.cond(
                    me == target_chiplet,
                    copy_like_read_write,
                    keep_local,
                    local_payload,
                )

        return jax.shard_map(
            shard_fn,
            mesh=mesh,
            in_specs=P("d"),
            out_specs=P("d"),
            check_vma=False,
        )(payload)

    return (
        jax.jit(kernel)
        .lower(jax.ShapeDtypeStruct(global_payload_shape, jnp.float32))
        .compile()
    )


def compile_remote_dma_p2p_kernel(
    mesh: Mesh,
    num_devices: int,
    payload_rows_per_link: int,
    src_chiplet: int,
    dst_chiplet: int,
):
    """Compile a directed Pallas remote-DMA P2P kernel for one chiplet pair.

    The kernel runs SPMD over the same 1-D logical mesh as the other ICI modes,
    but only ``src_chiplet`` starts a remote DMA and only ``dst_chiplet`` waits
    for the receive semaphore. Non-endpoint devices still enter the compiled
    program, which keeps multi-host execution synchronized without contributing
    to local metrics.
    """
    if src_chiplet == dst_chiplet:
        raise ValueError("p2p-rdma requires distinct src/dst chiplets")

    local_payload_shape = (
        payload_rows_per_link,
        PAYLOAD_MID_DIM,
        PAYLOAD_LAST_DIM,
    )
    global_payload_shape = (
        payload_rows_per_link * num_devices,
        PAYLOAD_MID_DIM,
        PAYLOAD_LAST_DIM,
    )

    def rdma_kernel(input_ref, output_ref, send_sem, recv_sem):
        me = lax.axis_index("d")
        remote_copy = pltpu.make_async_remote_copy(
            src_ref=input_ref,
            dst_ref=output_ref,
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=(dst_chiplet,),
            device_id_type=pl.DeviceIdType.MESH,
        )

        @pl.when(me == src_chiplet)
        def _send():
            remote_copy.start()
            remote_copy.wait_send()

        @pl.when(me == dst_chiplet)
        def _recv():
            remote_copy.wait_recv()

    rdma_call = pl.pallas_call(
        rdma_kernel,
        out_shape=jax.ShapeDtypeStruct(local_payload_shape, jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
            out_specs=pl.BlockSpec(memory_space=pl.ANY),
            scratch_shapes=([pltpu.SemaphoreType.DMA] * 2),
        ),
    )

    def kernel(payload):
        def shard_fn(local_payload):
            with jax.named_scope(MARKER):
                result = rdma_call(local_payload)
            return result

        return jax.shard_map(
            shard_fn,
            mesh=mesh,
            in_specs=P("d"),
            out_specs=P("d"),
            check_vma=False,
        )(payload)

    return (
        jax.jit(kernel)
        .lower(jax.ShapeDtypeStruct(global_payload_shape, jnp.float32))
        .compile()
    )


def compile_all_to_all_kernel(
    mesh: Mesh,
    num_devices: int,
    payload_rows_per_chiplet: int,
    parallel: bool = False,
):
    """Compile a dense JAX all_to_all kernel for the built-in a2a benchmark.

    Each device owns one pre-split payload of ``payload_rows_per_chiplet`` rows.
    ``jax.lax.all_to_all(..., tiled=True)`` splits that axis into equal chunks
    and exchanges one chunk with every device on mesh axis ``"d"``. In parallel
    mode the two chiplet coordinates form independent, concurrent groups.
    """
    if payload_rows_per_chiplet % num_devices != 0:
        raise ValueError(
            "a2a --data-size must divide evenly into one all_to_all chunk per "
            f"JAX device: payload_rows={payload_rows_per_chiplet}, "
            f"n_devices={num_devices}"
        )
    output_spec = P("d", None, None) if parallel else P("d")

    def kernel(payload):
        # all_to_all with tiled=True splits each chiplet's pre-split payload
        # into num_devices equal chunks. Only the chunks targeting other TPU
        # chips are counted in the ICI bandwidth numerator.
        with jax.named_scope(MARKER):
            result = jax.lax.all_to_all(
                payload,
                axis_name="d",
                split_axis=0,
                concat_axis=0,
                tiled=True,
            )

        return result

    return (
        jax.jit(
            jax.shard_map(
                kernel,
                mesh=mesh,
                in_specs=output_spec,
                out_specs=output_spec,
                check_vma=False,
            )
        )
        .lower(jax.ShapeDtypeStruct(
            (payload_rows_per_chiplet * num_devices, PAYLOAD_MID_DIM, PAYLOAD_LAST_DIM),
            jnp.float32,
            sharding=NamedSharding(mesh, output_spec),
        ))
        .compile()
    )


def compile_all_reduce_kernel(
    mesh: Mesh,
    num_devices: int,
    payload_rows_per_link: int,
    parallel: bool = False,
):
    """Compile a JAX psum kernel for the built-in all-reduce benchmark."""
    partition_spec = P("d", None, None) if parallel else P("d")
    collective_axis = "d"
    row_shard_count = num_devices // 2 if parallel else num_devices
    input_shape = (
        payload_rows_per_link * row_shard_count,
        PAYLOAD_MID_DIM,
        PAYLOAD_LAST_DIM,
    )
    input_spec = jax.ShapeDtypeStruct(
        input_shape,
        jnp.float32,
        sharding=NamedSharding(mesh, partition_spec),
    )

    def kernel(payload):
        with jax.named_scope(MARKER):
            # psum is the JAX collective used as AllReduce here; bandwidth is
            # reported with the standard busbw-style traffic factor.
            result = jax.lax.psum(payload, axis_name=collective_axis)

        return result

    return (
        jax.jit(
            jax.shard_map(
                kernel,
                mesh=mesh,
                in_specs=partition_spec,
                out_specs=partition_spec,
                check_vma=False,
            )
        )
        .lower(input_spec)
        .compile()
    )
