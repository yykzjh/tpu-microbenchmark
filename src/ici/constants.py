"""Shared constants for TPU ICI benchmarks."""

from __future__ import annotations

import numpy as np

# Size of one float32 element in bytes.
FLOAT32_BYTES = 4

# Payload shape is (m, 8, 128). ragged_all_to_all sizes/offsets are counted
# along the first dimension, so one transfer row contains 8 * 128 float32s.
# Example: --data-size 128MiB maps to m = 128MiB / (8 * 128 * 4) = 32768.
PAYLOAD_MID_DIM = 8
PAYLOAD_LAST_DIM = 128
PAYLOAD_ROW_ELEMENTS = PAYLOAD_MID_DIM * PAYLOAD_LAST_DIM
PAYLOAD_ROW_BYTES = PAYLOAD_ROW_ELEMENTS * FLOAT32_BYTES

# Each TPU chip contains 2 logical devices (chiplets)
CHIPLETS_PER_TPU = 2
CORE_PAIR_LABELS = ("0->0", "0->1", "1->0", "1->1")

# ragged_all_to_all size/offset operands in this benchmark are passed as int32.
RAGGED_INDEX_MAX = np.iinfo(np.int32).max
