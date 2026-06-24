from __future__ import annotations

from pathlib import Path
from typing import Any


def format_bytes(size: int) -> str:
    # Human-readable byte counts make example output useful at a glance.
    value = float(size)
    for unit in ("bytes", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            if unit == "bytes":
                return f"{int(value)} bytes"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size} bytes"


def block_payload_stats(reader: Any) -> tuple[int, int]:
    # Python readers expose block objects directly. Other readers may only have
    # the summarized values stored in metadata, so support both shapes.
    blocks = getattr(reader, "blocks", None)
    if blocks is not None:
        compressed = sum(int(block.size) for block in blocks)
        uncompressed = sum(int(block.uncompressed_size) for block in blocks)
        return compressed, uncompressed

    metadata = getattr(reader, "metadata", {}) or {}
    return int(metadata.get("block_payload_bytes") or 0), int(metadata.get("block_uncompressed_bytes") or 0)


def compression_summary(compressed: int, uncompressed: int) -> str:
    # Compare bytes on disk with the bytes that would have been stored before
    # block compression. This is a quick way to see whether the codec helped.
    if uncompressed <= 0:
        return "n/a"
    if compressed <= 0:
        return f"{format_bytes(compressed)} stored from {format_bytes(uncompressed)} raw block payload"

    if compressed <= uncompressed:
        factor = uncompressed / compressed
        savings = (1.0 - (compressed / uncompressed)) * 100.0
        return (
            f"{format_bytes(compressed)} stored from {format_bytes(uncompressed)} raw block payload "
            f"({factor:.2f}x smaller, {savings:.1f}% savings)"
        )

    factor = compressed / uncompressed
    overhead = ((compressed / uncompressed) - 1.0) * 100.0
    return (
        f"{format_bytes(compressed)} stored from {format_bytes(uncompressed)} raw block payload "
        f"({factor:.2f}x larger, {overhead:.1f}% overhead)"
    )


def compact_list(values: list[Any], *, limit: int = 5) -> str:
    # Summaries should stay readable even if a dataset has thousands of row
    # names or sensors.
    shown = [str(value) for value in values[:limit]]
    if len(values) > limit:
        shown.append(f"... +{len(values) - limit} more")
    return ", ".join(shown) if shown else "none"


def image_preview(image: dict[str, Any]) -> str:
    # RowPack image payloads carry enough metadata to reshape bytes into
    # [height, width, channels] tensors later.
    height = int(image.get("height") or 0)
    width = int(image.get("width") or 0)
    channels = int(image.get("channels") or 0)
    storage = image.get("storage") or "encoded"
    size = len(image.get("bytes") or b"")
    shape = f"{height}x{width}x{channels}" if height and width and channels else "shape unknown"
    return f"{shape}, {storage}, {format_bytes(size)}"


def print_reader_summary(
    reader: Any,
    path: str | Path,
    *,
    title: str = "RowPack summary",
    include_first_row: bool = True,
) -> None:
    # Keep example scripts focused on the write/read logic by sharing one small,
    # friendly summary printer.
    metadata = reader.metadata
    compressed, uncompressed = block_payload_stats(reader)
    print()
    print(title)
    print(f"  path: {Path(path)}")
    print(f"  file size: {format_bytes(reader.size_bytes)}")
    print(f"  rows: {len(reader)}")
    print(f"  blocks: {metadata.get('block_count', reader.header.block_count)}")
    print(f"  rows per block setting: {metadata.get('rows_per_block')}")
    print(f"  payload format: {metadata.get('payload_format')}")
    print(f"  block codec: {metadata.get('block_codec')}")
    print(f"  block payload: {compression_summary(compressed, uncompressed)}")

    dataset_name = metadata.get("dataset_name")
    if dataset_name:
        print(f"  dataset name: {dataset_name}")
    description = metadata.get("description")
    if description:
        print(f"  description: {description}")

    sensors = metadata.get("sensors") or []
    if sensors:
        sensor_labels = [f"{sensor.get('name')} ({sensor.get('type')})" for sensor in sensors]
        print(f"  sensors: {compact_list(sensor_labels)}")

    row_names = [name for name in metadata.get("row_names") or [] if name]
    print(f"  row names: {compact_list(row_names)}")

    if include_first_row and len(reader):
        row = reader.read_row(0)
        print("  first row:")
        print(f"    keys: {', '.join(sorted(row))}")
        if "timestamp_ns" in row:
            print(f"    timestamp_ns: {row['timestamp_ns']}")
        if row.get("sensors"):
            print(f"    sensor fields: {compact_list(sorted(row['sensors']))}")
        images = row.get("images") or []
        if images:
            print(f"    first image: {image_preview(images[0])}")
