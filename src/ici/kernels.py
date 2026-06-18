"""JAX and Pallas kernel compilers for TPU ICI benchmarks."""

from __future__ import annotations

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
from ici.zero_crop import maybe_zero_crop
from utils.profiling import MARKER

def create_device_mesh(num_devices: int) -> Mesh:
    """Create 1-D JAX device mesh over all devices.

    A flat 1-D mesh ``("d",)`` is used because the communication pattern
    is entirely defined by the traffic matrix; the physical topology
    (torus coordinates) is used for inferred metadata.

    TPU matrix index ``i`` maps to logical devices ``2*i`` and ``2*i+1`` in
    this mesh. :func:`infer_tpu_topology` validates that those consecutive
    device pairs expose the same TPU chip coordinates.
    """
    devices = jax.devices()
    if len(devices) < num_devices:
        raise ValueError(
            f"Need {num_devices} devices, only {len(devices)} available. "
            f"Check that jax.distributed.initialize() succeeded on all hosts."
        )
    return Mesh(np.asarray(devices[:num_devices]), ("d",))


def make_axis_sharded_payload(
    mesh: Mesh,
    num_devices: int,
    payload_rows_per_link: int,
) -> jax.Array:
    """Create one sharded payload chunk per device for all-reduce input."""
    sharding = NamedSharding(mesh, P("d"))
    # Global axis 0 is partitioned by mesh axis d. Each device receives exactly
    # payload_rows_per_link rows, which is the AllReduce payload per chiplet.
    global_shape = (
        payload_rows_per_link * num_devices,
        PAYLOAD_MID_DIM,
        PAYLOAD_LAST_DIM,
    )

    def data_callback(index):
        axis0 = index[0] if isinstance(index, tuple) else index
        if not isinstance(axis0, slice):
            raise ValueError(f"Unexpected all-reduce payload shard index: {index}")
        start = axis0.start or 0
        stop = axis0.stop or global_shape[0]
        shard_rows = stop - start
        if shard_rows <= 0:
            raise ValueError(f"Invalid all-reduce payload shard index: {index}")
        device_index = start // payload_rows_per_link
        # Fill each shard with a different value so a debugger can distinguish
        # participant shards in traces or dumps without adding extra reductions
        # to the benchmark path.
        return (
            np.ones(
                (shard_rows, PAYLOAD_MID_DIM, PAYLOAD_LAST_DIM),
                dtype=np.float32,
            )
            * (device_index + 1)
        )

    return jax.make_array_from_callback(global_shape, sharding, data_callback)


def compile_traffic_matrix_kernel(
    mesh: Mesh,
    num_devices: int,
    max_send_rows: int,
    max_recv_rows: int,
    use_zero_crop: bool,
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
        A compiled function: ``(traffic_matrix,) -> received buffers``.
    """

    def kernel(traffic_matrix):
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

        payload = (
            jnp.ones(
                (max_send_rows, PAYLOAD_MID_DIM, PAYLOAD_LAST_DIM),
                dtype=jnp.float32,
            )
            * (me + 1)
        )
        output = jnp.zeros(
            (max_recv_rows, PAYLOAD_MID_DIM, PAYLOAD_LAST_DIM),
            dtype=jnp.float32,
        )

        with jax.named_scope(MARKER):
            # The MARKER wraps only the collective primitive. Input payload
            # creation above is inside the compiled function but outside the
            # named scope, so xprof extraction focuses on communication.
            result = jax.lax.ragged_all_to_all(
                operand=payload,
                output=output,
                input_offsets=input_offsets,
                send_sizes=send_sizes,
                output_offsets=output_offsets,
                recv_sizes=recv_sizes,
                axis_name="d",
            )
            result = maybe_zero_crop(result, use_zero_crop)

        return result

    return (
        jax.jit(
            jax.shard_map(
                kernel,
                mesh=mesh,
                in_specs=P(),
                out_specs=P("d"),
                check_vma=False,
            )
        )
        .lower(jnp.zeros((num_devices, num_devices), dtype=jnp.int32))
        .compile()
    )


def compile_remote_dma_p2p_kernel(
    mesh: Mesh,
    num_devices: int,
    payload_rows_per_link: int,
    src_chiplet: int,
    dst_chiplet: int,
    use_zero_crop: bool,
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
                result = maybe_zero_crop(result, use_zero_crop)
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
    use_zero_crop: bool,
):
    """Compile a dense JAX all_to_all kernel for the built-in a2a benchmark.

    Each device owns one pre-split payload of ``payload_rows_per_chiplet`` rows.
    ``jax.lax.all_to_all(..., tiled=True)`` splits that axis into equal chunks
    and exchanges one chunk with every device in the mesh.
    """
    if payload_rows_per_chiplet % num_devices != 0:
        raise ValueError(
            "a2a --data-size must divide evenly into one all_to_all chunk per "
            f"JAX device: payload_rows={payload_rows_per_chiplet}, "
            f"n_devices={num_devices}"
        )
    local_rows = payload_rows_per_chiplet

    def kernel():
        me = jax.lax.axis_index("d")
        # all_to_all with tiled=True splits each chiplet's pre-split payload
        # into num_devices equal chunks. Only the chunks targeting other TPU
        # chips are counted in the ICI bandwidth numerator.
        payload = (
            jnp.ones(
                (local_rows, PAYLOAD_MID_DIM, PAYLOAD_LAST_DIM),
                dtype=jnp.float32,
            )
            * (me + 1)
        )

        with jax.named_scope(MARKER):
            result = jax.lax.all_to_all(
                payload,
                axis_name="d",
                split_axis=0,
                concat_axis=0,
                tiled=True,
            )
            result = maybe_zero_crop(result, use_zero_crop)

        return result

    return (
        jax.jit(
            jax.shard_map(
                kernel,
                mesh=mesh,
                in_specs=(),
                out_specs=P("d"),
                check_vma=False,
            )
        )
        .lower()
        .compile()
    )


def compile_all_reduce_kernel(
    mesh: Mesh,
    num_devices: int,
    payload_rows_per_link: int,
    use_zero_crop: bool,
):
    """Compile a JAX psum kernel for the built-in all-reduce benchmark."""

    def kernel(payload):
        with jax.named_scope(MARKER):
            # psum is the JAX collective used as AllReduce here; bandwidth is
            # reported with the standard busbw-style traffic factor.
            result = jax.lax.psum(payload, axis_name="d")
            result = maybe_zero_crop(result, use_zero_crop)

        return result

    return (
        jax.jit(
            jax.shard_map(
                kernel,
                mesh=mesh,
                in_specs=P("d"),
                out_specs=P("d"),
                check_vma=False,
            )
        )
        .lower(jax.ShapeDtypeStruct(
            (
                payload_rows_per_link * num_devices,
                PAYLOAD_MID_DIM,
                PAYLOAD_LAST_DIM,
            ),
            jnp.float32,
        ))
        .compile()
    )
