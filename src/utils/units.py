"""Unit parsing helpers for TPU benchmark parameters."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ParsedByteSize:
    """Normalized byte-size CLI value."""

    input: str
    bytes: int
    label: str

    def metadata(self, prefix: str = "data_size") -> dict[str, str | int]:
        """Return flat metadata fields for metrics dimensions."""
        return {
            f"{prefix}_input": self.input,
            f"{prefix}_bytes": self.bytes,
            f"{prefix}_label": self.label,
        }


def parse_byte_size(
    value: str | int,
    *,
    parameter_name: str = "data_size",
    alignment_bytes: int = 1,
    alignment_description: str | None = None,
) -> int:
    """Parse a byte size with optional binary unit suffixes."""
    if isinstance(value, int):
        size_bytes = value
    else:
        text = str(value).strip()
        # Examples accepted by this parser: "1048576", "512MiB", "1G".
        # All suffixes are interpreted as binary units because TPU payload
        # alignment is normally expressed in powers of two.
        match = re.fullmatch(r"([0-9]+)\s*([A-Za-z]*)", text)
        if not match:
            raise ValueError(
                f"{parameter_name} must be an integer byte count with an "
                "optional binary unit suffix: K, KB, KiB, M, MB, MiB, G, GB, "
                "or GiB"
            )

        number = int(match.group(1))
        unit = match.group(2).lower()
        multipliers = {
            "": 1,
            "k": 1024,
            "kb": 1024,
            "kib": 1024,
            "m": 1024**2,
            "mb": 1024**2,
            "mib": 1024**2,
            "g": 1024**3,
            "gb": 1024**3,
            "gib": 1024**3,
        }
        if unit not in multipliers:
            raise ValueError(
                f"Unsupported {parameter_name} unit {match.group(2)!r}; "
                "supported units: K, KB, KiB, M, MB, MiB, G, GB, GiB"
            )
        size_bytes = number * multipliers[unit]

    if size_bytes <= 0:
        raise ValueError(f"{parameter_name} must be positive, got {value!r}")
    if alignment_bytes <= 0:
        raise ValueError(f"alignment_bytes must be positive, got {alignment_bytes}")
    if size_bytes % alignment_bytes != 0:
        alignment_text = (
            f" ({alignment_description})" if alignment_description else ""
        )
        # Alignment failures are reported with the normalized byte count so a
        # caller can adjust the original value without guessing unit expansion.
        raise ValueError(
            f"{parameter_name}={size_bytes} bytes is not aligned to "
            f"{alignment_bytes} bytes{alignment_text}."
        )
    return size_bytes


def parse_data_size(
    value: str | int,
    *,
    alignment_bytes: int = 1,
    alignment_description: str | None = None,
) -> ParsedByteSize:
    """Parse the common ``--data-size`` argument and keep display metadata."""
    size_bytes = parse_byte_size(
        value,
        parameter_name="data_size",
        alignment_bytes=alignment_bytes,
        alignment_description=alignment_description,
    )
    return ParsedByteSize(
        input=str(value),
        bytes=size_bytes,
        label=format_byte_size_label(size_bytes),
    )


def data_size_help(
    description: str,
    *,
    alignment_bytes: int | None = None,
    default: str | None = None,
) -> str:
    """Build consistent help text for the common ``--data-size`` argument."""
    parts = [
        description.rstrip("."),
        (
            "Supports plain bytes or binary K/KB/KiB/M/MB/MiB/G/GB/GiB "
            "suffixes, e.g. 512MiB or 1G"
        ),
    ]
    if alignment_bytes is not None:
        parts.append(f"Must be a multiple of {alignment_bytes} bytes")
    if default is not None:
        parts.append(f"Default: {default}")
    return ". ".join(parts) + "."


def format_byte_size_label(size_bytes: int) -> str:
    """Return a compact filesystem-safe label for a byte size."""
    for suffix, unit_bytes in (
        ("GiB", 1024**3),
        ("MiB", 1024**2),
        ("KiB", 1024),
    ):
        if size_bytes % unit_bytes == 0:
            return f"{size_bytes // unit_bytes}{suffix}"
    return f"{size_bytes}B"
