"""Pallas kernels for TPU memory bandwidth benchmarks."""

from __future__ import annotations

import jax.numpy as jnp
import jax.experimental.pallas.tpu as pltpu

# ---------------------------------------------------------------------------
# HBM Read Kernel
# ---------------------------------------------------------------------------

def hbm_read_kernel(
    x_hbm_ref,       # HBM input (memory_space=HBM)
    o_hbm_ref,       # HBM output placeholder
    vmem_0,          # VMEM bank 0
    vmem_1,          # VMEM bank 1
    vmem_2,          # VMEM bank 2
    vmem_3,          # VMEM bank 3
    sem_0,           # DMA semaphore for bank 0
    sem_1,           # DMA semaphore for bank 1
    sem_2,           # DMA semaphore for bank 2
    sem_3,           # DMA semaphore for bank 3
    *,
    n_iters: int,
):
    """Four-bank HBM read prefetch pipeline."""
    if n_iters >= 1:
        copy_a = pltpu.make_async_copy(x_hbm_ref.at[0, :, :], vmem_0, sem_0)
        copy_a.start()
        if n_iters >= 2:
            copy_b = pltpu.make_async_copy(x_hbm_ref.at[1, :, :], vmem_1, sem_1)
            copy_b.start()
        if n_iters >= 3:
            copy_c = pltpu.make_async_copy(x_hbm_ref.at[2, :, :], vmem_2, sem_2)
            copy_c.start()
        if n_iters >= 4:
            copy_d = pltpu.make_async_copy(x_hbm_ref.at[3, :, :], vmem_3, sem_3)
            copy_d.start()
        for chunk_base in range(0, n_iters, 4):
            if chunk_base < n_iters:
                copy_a.wait()
                pltpu.touch(vmem_0)
                next_chunk = chunk_base + 4
                if next_chunk < n_iters:
                    copy_a = pltpu.make_async_copy(
                        x_hbm_ref.at[next_chunk, :, :],
                        vmem_0,
                        sem_0,
                    )
                    copy_a.start()
            if chunk_base + 1 < n_iters:
                copy_b.wait()
                pltpu.touch(vmem_1)
                next_chunk = chunk_base + 5
                if next_chunk < n_iters:
                    copy_b = pltpu.make_async_copy(
                        x_hbm_ref.at[next_chunk, :, :],
                        vmem_1,
                        sem_1,
                    )
                    copy_b.start()
            if chunk_base + 2 < n_iters:
                copy_c.wait()
                pltpu.touch(vmem_2)
                next_chunk = chunk_base + 6
                if next_chunk < n_iters:
                    copy_c = pltpu.make_async_copy(
                        x_hbm_ref.at[next_chunk, :, :],
                        vmem_2,
                        sem_2,
                    )
                    copy_c.start()
            if chunk_base + 3 < n_iters:
                copy_d.wait()
                pltpu.touch(vmem_3)
                next_chunk = chunk_base + 7
                if next_chunk < n_iters:
                    copy_d = pltpu.make_async_copy(
                        x_hbm_ref.at[next_chunk, :, :],
                        vmem_3,
                        sem_3,
                    )
                    copy_d.start()


# ---------------------------------------------------------------------------
# HBM Write Kernel
# ---------------------------------------------------------------------------

def hbm_write_kernel(
    o_hbm_ref,       # HBM output
    vmem_0,          # VMEM bank 0
    vmem_1,          # VMEM bank 1
    vmem_2,          # VMEM bank 2
    vmem_3,          # VMEM bank 3
    sem_0,           # DMA semaphore for bank 0
    sem_1,           # DMA semaphore for bank 1
    sem_2,           # DMA semaphore for bank 2
    sem_3,           # DMA semaphore for bank 3
    *,
    n_iters: int,
):
    """Four-bank HBM write pipeline."""
    vmem_0[...] = jnp.full_like(vmem_0[...], jnp.float32(1.0))
    vmem_1[...] = jnp.full_like(vmem_1[...], jnp.float32(2.0))
    vmem_2[...] = jnp.full_like(vmem_2[...], jnp.float32(3.0))
    vmem_3[...] = jnp.full_like(vmem_3[...], jnp.float32(4.0))

    if n_iters >= 1:
        copy_a = pltpu.make_async_copy(vmem_0, o_hbm_ref.at[0, :, :], sem_0)
        copy_a.start()
        if n_iters >= 2:
            copy_b = pltpu.make_async_copy(vmem_1, o_hbm_ref.at[1, :, :], sem_1)
            copy_b.start()
        if n_iters >= 3:
            copy_c = pltpu.make_async_copy(vmem_2, o_hbm_ref.at[2, :, :], sem_2)
            copy_c.start()
        if n_iters >= 4:
            copy_d = pltpu.make_async_copy(vmem_3, o_hbm_ref.at[3, :, :], sem_3)
            copy_d.start()
        for chunk_base in range(0, n_iters, 4):
            if chunk_base < n_iters:
                copy_a.wait()
                next_chunk = chunk_base + 4
                if next_chunk < n_iters:
                    copy_a = pltpu.make_async_copy(
                        vmem_0,
                        o_hbm_ref.at[next_chunk, :, :],
                        sem_0,
                    )
                    copy_a.start()
            if chunk_base + 1 < n_iters:
                copy_b.wait()
                next_chunk = chunk_base + 5
                if next_chunk < n_iters:
                    copy_b = pltpu.make_async_copy(
                        vmem_1,
                        o_hbm_ref.at[next_chunk, :, :],
                        sem_1,
                    )
                    copy_b.start()
            if chunk_base + 2 < n_iters:
                copy_c.wait()
                next_chunk = chunk_base + 6
                if next_chunk < n_iters:
                    copy_c = pltpu.make_async_copy(
                        vmem_2,
                        o_hbm_ref.at[next_chunk, :, :],
                        sem_2,
                    )
                    copy_c.start()
            if chunk_base + 3 < n_iters:
                copy_d.wait()
                next_chunk = chunk_base + 7
                if next_chunk < n_iters:
                    copy_d = pltpu.make_async_copy(
                        vmem_3,
                        o_hbm_ref.at[next_chunk, :, :],
                        sem_3,
                    )
                    copy_d.start()


# ---------------------------------------------------------------------------
# HBM Copy Kernel
# ---------------------------------------------------------------------------

def hbm_copy_kernel(
    x_hbm_ref,       # HBM input
    o_hbm_ref,       # HBM output
    read_vmem_0,     # HBM→VMEM read bank 0
    read_vmem_1,     # HBM→VMEM read bank 1
    write_vmem_0,    # VMEM→HBM write bank 0
    write_vmem_1,    # VMEM→HBM write bank 1
    read_sem_0,      # HBM→VMEM DMA semaphore for read bank 0
    read_sem_1,      # HBM→VMEM DMA semaphore for read bank 1
    write_sem_0,     # VMEM→HBM DMA semaphore for write bank 0
    write_sem_1,     # VMEM→HBM DMA semaphore for write bank 1
    *,
    n_iters: int,
):
    """Single-program HBM read/write benchmark with two banks per direction."""
    write_vmem_0[...] = jnp.full_like(write_vmem_0[...], jnp.float32(1.0))
    write_vmem_1[...] = jnp.full_like(write_vmem_1[...], jnp.float32(2.0))

    if n_iters >= 1:
        read_copy_0 = pltpu.make_async_copy(
            x_hbm_ref.at[0, :, :],
            read_vmem_0,
            read_sem_0,
        )
        read_copy_0.start()
        write_copy_0 = pltpu.make_async_copy(
            write_vmem_0,
            o_hbm_ref.at[0, :, :],
            write_sem_0,
        )
        write_copy_0.start()

        if n_iters >= 2:
            read_copy_1 = pltpu.make_async_copy(
                x_hbm_ref.at[1, :, :],
                read_vmem_1,
                read_sem_1,
            )
            read_copy_1.start()
            write_copy_1 = pltpu.make_async_copy(
                write_vmem_1,
                o_hbm_ref.at[1, :, :],
                write_sem_1,
            )
            write_copy_1.start()

        for chunk_base in range(0, n_iters, 2):
            read_copy_0.wait()
            pltpu.touch(read_vmem_0)
            next_read_0 = chunk_base + 2
            if next_read_0 < n_iters:
                read_copy_0 = pltpu.make_async_copy(
                    x_hbm_ref.at[next_read_0, :, :],
                    read_vmem_0,
                    read_sem_0,
                )
                read_copy_0.start()

            write_copy_0.wait()
            next_write_0 = chunk_base + 2
            if next_write_0 < n_iters:
                write_copy_0 = pltpu.make_async_copy(
                    write_vmem_0,
                    o_hbm_ref.at[next_write_0, :, :],
                    write_sem_0,
                )
                write_copy_0.start()

            if chunk_base + 1 < n_iters:
                read_copy_1.wait()
                pltpu.touch(read_vmem_1)
                next_read_1 = chunk_base + 3
                if next_read_1 < n_iters:
                    read_copy_1 = pltpu.make_async_copy(
                        x_hbm_ref.at[next_read_1, :, :],
                        read_vmem_1,
                        read_sem_1,
                    )
                    read_copy_1.start()

                write_copy_1.wait()
                next_write_1 = chunk_base + 3
                if next_write_1 < n_iters:
                    write_copy_1 = pltpu.make_async_copy(
                        write_vmem_1,
                        o_hbm_ref.at[next_write_1, :, :],
                        write_sem_1,
                    )
                    write_copy_1.start()


# ---------------------------------------------------------------------------
# VMEM Copy Kernel
# ---------------------------------------------------------------------------

def vmem_copy_kernel(
    o_hbm_ref,       # HBM output
    vmem_src_0,      # VMEM source/destination pair 0
    vmem_dst_0,
    vmem_src_1,      # VMEM source/destination pair 1
    vmem_dst_1,
    vmem_src_2,      # VMEM source/destination pair 2
    vmem_dst_2,
    vmem_src_3,      # VMEM source/destination pair 3
    vmem_dst_3,
    out_sem,         # VMEM→HBM output semaphore
    *,
    n_iters: int,
):
    """VMEM copy bandwidth kernel.

    Initializes eight VMEM buffers, then rotates the read/write mapping with
    an 8-phase permutation. Each loop still performs four VMEM read+write
    copies, but the source and destination buffers cover more pairings than a
    fixed srcN->dstN pattern.
    """
    vmem_src_0[...] = jnp.full_like(vmem_src_0[...], jnp.float32(1.0))
    vmem_src_1[...] = jnp.full_like(vmem_src_1[...], jnp.float32(2.0))
    vmem_src_2[...] = jnp.full_like(vmem_src_2[...], jnp.float32(3.0))
    vmem_src_3[...] = jnp.full_like(vmem_src_3[...], jnp.float32(4.0))

    for i in range(n_iters):
        # Rotate across both direction and destination offset. This keeps four
        # independent copy pairs per inner iteration while avoiding one fixed
        # pairing pattern dominating the generated VMEM access sequence.
        if i % 8 == 0:
            vmem_dst_0[...] = vmem_src_0[...]
            vmem_dst_1[...] = vmem_src_1[...]
            vmem_dst_2[...] = vmem_src_2[...]
            vmem_dst_3[...] = vmem_src_3[...]
            pltpu.touch(vmem_dst_0)
            pltpu.touch(vmem_dst_1)
            pltpu.touch(vmem_dst_2)
            pltpu.touch(vmem_dst_3)
        elif i % 8 == 1:
            vmem_src_1[...] = vmem_dst_0[...]
            vmem_src_2[...] = vmem_dst_1[...]
            vmem_src_3[...] = vmem_dst_2[...]
            vmem_src_0[...] = vmem_dst_3[...]
            pltpu.touch(vmem_src_1)
            pltpu.touch(vmem_src_2)
            pltpu.touch(vmem_src_3)
            pltpu.touch(vmem_src_0)
        elif i % 8 == 2:
            vmem_dst_2[...] = vmem_src_1[...]
            vmem_dst_3[...] = vmem_src_2[...]
            vmem_dst_0[...] = vmem_src_3[...]
            vmem_dst_1[...] = vmem_src_0[...]
            pltpu.touch(vmem_dst_2)
            pltpu.touch(vmem_dst_3)
            pltpu.touch(vmem_dst_0)
            pltpu.touch(vmem_dst_1)
        elif i % 8 == 3:
            vmem_src_3[...] = vmem_dst_2[...]
            vmem_src_0[...] = vmem_dst_3[...]
            vmem_src_1[...] = vmem_dst_0[...]
            vmem_src_2[...] = vmem_dst_1[...]
            pltpu.touch(vmem_src_3)
            pltpu.touch(vmem_src_0)
            pltpu.touch(vmem_src_1)
            pltpu.touch(vmem_src_2)
        elif i % 8 == 4:
            vmem_dst_1[...] = vmem_src_0[...]
            vmem_dst_2[...] = vmem_src_1[...]
            vmem_dst_3[...] = vmem_src_2[...]
            vmem_dst_0[...] = vmem_src_3[...]
            pltpu.touch(vmem_dst_1)
            pltpu.touch(vmem_dst_2)
            pltpu.touch(vmem_dst_3)
            pltpu.touch(vmem_dst_0)
        elif i % 8 == 5:
            vmem_src_2[...] = vmem_dst_0[...]
            vmem_src_3[...] = vmem_dst_1[...]
            vmem_src_0[...] = vmem_dst_2[...]
            vmem_src_1[...] = vmem_dst_3[...]
            pltpu.touch(vmem_src_2)
            pltpu.touch(vmem_src_3)
            pltpu.touch(vmem_src_0)
            pltpu.touch(vmem_src_1)
        elif i % 8 == 6:
            vmem_dst_3[...] = vmem_src_0[...]
            vmem_dst_0[...] = vmem_src_1[...]
            vmem_dst_1[...] = vmem_src_2[...]
            vmem_dst_2[...] = vmem_src_3[...]
            pltpu.touch(vmem_dst_3)
            pltpu.touch(vmem_dst_0)
            pltpu.touch(vmem_dst_1)
            pltpu.touch(vmem_dst_2)
        else:
            vmem_src_0[...] = vmem_dst_0[...]
            vmem_src_1[...] = vmem_dst_1[...]
            vmem_src_2[...] = vmem_dst_2[...]
            vmem_src_3[...] = vmem_dst_3[...]
            pltpu.touch(vmem_src_0)
            pltpu.touch(vmem_src_1)
            pltpu.touch(vmem_src_2)
            pltpu.touch(vmem_src_3)

    copy_out = pltpu.make_async_copy(vmem_src_0, o_hbm_ref, out_sem)
    copy_out.start()
    copy_out.wait()
