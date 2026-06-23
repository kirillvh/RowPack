from __future__ import annotations

import json
import io
import mmap
import os
import warnings
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Iterable, Iterator

from .format import (
    BLOCK_INDEX_STRUCT,
    CODEC_IDS,
    CODEC_LZAV_DEFAULT,
    CODEC_LZAV_HI,
    CODEC_NAMES,
    CODEC_NONE,
    FLAG_UNCOMPRESSED,
    HEADER_SIZE,
    ROW_INDEX_STRUCT,
    ROW_PREFIX_STRUCT,
    BlockIndexEntry,
    Header,
    RowIndexEntry,
    empty_header,
    pack_header,
    unpack_header,
)
from .native import load_native


class RowPackError(RuntimeError):
    pass


@dataclass
class PendingBlock:
    start_row: int
    row_count: int
    offset: int
    size: int
    uncompressed_size: int
    codec: int


class RowPackWriter:
    """Streaming writer for RowPack v0 files."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        rows_per_block: int = 32,
        metadata: dict[str, Any] | None = None,
        payload_format: str = "json",
        block_codec: str = "none",
        native_module_dir: str | None = None,
        overwrite: bool = False,
    ):
        self.path = Path(path)
        if self.path.exists() and not overwrite:
            raise FileExistsError(self.path)
        if rows_per_block < 1:
            raise ValueError("rows_per_block must be >= 1")
        if payload_format not in {"json", "cista"}:
            raise ValueError("payload_format must be 'json' or 'cista'")
        if block_codec not in CODEC_IDS:
            raise ValueError(f"Unsupported RowPack block_codec {block_codec!r}")

        self.rows_per_block = rows_per_block
        self.payload_format = payload_format
        self.block_codec = block_codec
        self.block_codec_id = CODEC_IDS[block_codec]
        self.native = load_native(native_module_dir) if payload_format == "cista" or block_codec != "none" else None
        self.metadata = dict(metadata or {})
        self.row_index: list[RowIndexEntry] = []
        self.blocks: list[PendingBlock] = []
        self.row_names: list[str | None] = []
        self.aliases: dict[str, dict[str, Any]] = {}
        self._closed = False
        self._next_row_id = 0
        self._pending_start_row = 0
        self._pending_payload = bytearray()
        self._pending_rows: list[RowIndexEntry] = []

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w+b")
        self._file.write(pack_header(empty_header()))

    def __enter__(self) -> "RowPackWriter":
        return self

    def __exit__(self, exc_type, _exc, _tb) -> None:
        if exc_type is None:
            self.finish()
        else:
            self.close()

    def append_row(
        self,
        row: dict[str, Any],
        *,
        name: str | None = None,
        aliases: Iterable[str] | None = None,
    ) -> int:
        if self._closed:
            raise RowPackError("Cannot append to a closed RowPackWriter")

        if len(self._pending_rows) >= self.rows_per_block:
            self._flush_block()

        row_id = self._next_row_id
        if not self._pending_rows:
            self._pending_start_row = row_id

        payload = serialize_row(row, row_id=row_id, payload_format=self.payload_format, native=self.native)
        block_id = len(self.blocks)
        row_in_block = len(self._pending_rows)
        offset = len(self._pending_payload)
        self._pending_payload.extend(payload)
        self._pending_rows.append(
            RowIndexEntry(offset=offset, size=len(payload), block_id=block_id, row_in_block=row_in_block)
        )
        self._next_row_id += 1

        self.row_names.append(name)
        for alias in aliases or []:
            if name is None:
                raise ValueError("aliases require a canonical row name")
            self.aliases[alias] = {
                "canonical": name,
                "status": "non_canonical",
                "message": f"{alias!r} is an alias for canonical row name {name!r}",
            }

        return row_id

    def finish(self) -> None:
        if self._closed:
            return

        self._flush_block()
        metadata_offset = self._file.tell()
        observed_compressions = sorted({CODEC_NAMES[block.codec] for block in self.blocks})
        metadata = {
            "format": "RowPack",
            "format_version": "0.1",
            "storage": "row-major",
            "compression": self.block_codec,
            "block_codec": self.block_codec,
            "observed_compressions": observed_compressions,
            "payload_format": self.payload_format,
            "rows_per_block": self.rows_per_block,
            "row_count": len(self.row_index),
            "block_count": len(self.blocks),
            "row_names": self.row_names,
            "aliases": self.aliases,
            **self.metadata,
        }
        metadata_bytes = json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._file.write(metadata_bytes)

        block_index_offset = self._file.tell()
        for block in self.blocks:
            self._file.write(
                BLOCK_INDEX_STRUCT.pack(
                    block.start_row,
                    block.row_count,
                    block.offset,
                    block.size,
                    block.uncompressed_size,
                    block.codec,
                    0,
                )
            )
        block_index_size = self._file.tell() - block_index_offset

        row_index_offset = self._file.tell()
        for entry in self.row_index:
            self._file.write(
                ROW_INDEX_STRUCT.pack(entry.offset, entry.size, entry.block_id, entry.row_in_block)
            )
        row_index_size = self._file.tell() - row_index_offset

        header = Header(
            major=0,
            minor=1,
            header_size=HEADER_SIZE,
            flags=FLAG_UNCOMPRESSED if all(block.codec == CODEC_NONE for block in self.blocks) else 0,
            row_count=len(self.row_index),
            block_count=len(self.blocks),
            data_offset=HEADER_SIZE,
            metadata_offset=metadata_offset,
            metadata_size=len(metadata_bytes),
            block_index_offset=block_index_offset,
            block_index_size=block_index_size,
            row_index_offset=row_index_offset,
            row_index_size=row_index_size,
        )
        self._file.seek(0)
        self._file.write(pack_header(header))
        self.close()

    def _flush_block(self) -> None:
        if not self._pending_rows:
            return

        uncompressed = bytes(self._pending_payload)
        codec = self.block_codec_id
        payload = uncompressed
        if codec != CODEC_NONE:
            if self.native is None:
                raise RowPackError("LZAV RowPack blocks require rowpack_native")
            payload = self.native.lzav_compress(uncompressed, self.block_codec)

        block_offset = self._file.tell()
        self._file.write(payload)
        block_id = len(self.blocks)
        self.blocks.append(
            PendingBlock(
                start_row=self._pending_start_row,
                row_count=len(self._pending_rows),
                offset=block_offset,
                size=len(payload),
                uncompressed_size=len(uncompressed),
                codec=codec,
            )
        )

        for entry in self._pending_rows:
            offset = entry.offset if codec != CODEC_NONE else block_offset + entry.offset
            self.row_index.append(
                RowIndexEntry(offset=offset, size=entry.size, block_id=block_id, row_in_block=entry.row_in_block)
            )

        self._pending_payload.clear()
        self._pending_rows.clear()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._file.close()


class RowPackReader:
    """Memory-mapped reader for RowPack v0 files."""

    def __init__(self, path: str | os.PathLike[str], *, native_module_dir: str | None = None):
        self.path = Path(path)
        self._file = self.path.open("rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self.header = unpack_header(self._mmap[:HEADER_SIZE])
        self.metadata = self._read_metadata()
        self.payload_format = self.metadata.get("payload_format", "json")
        self.native_module_dir = native_module_dir
        self.native = None
        self.blocks = self._read_block_index()
        self.row_index = self._read_row_index()
        self.name_to_row = self._build_name_index()
        self.aliases = self.metadata.get("aliases") or {}
        self._block_cache: dict[int, bytes] = {}
        self._block_cache_order: list[int] = []
        self._block_cache_size = 4

    def __enter__(self) -> "RowPackReader":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def __len__(self) -> int:
        return self.header.row_count

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size

    def close(self) -> None:
        if getattr(self, "_mmap", None) is not None:
            self._mmap.close()
            self._mmap = None
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None

    def read_row(self, index: int) -> dict[str, Any]:
        entry = self.row_index[self._normalize_index(index)]
        payload = self._read_row_payload(entry)
        return deserialize_row(payload, payload_format=self.payload_format, native=self._native())

    def read_window(self, start: int, count: int) -> list[dict[str, Any]]:
        if count < 0:
            raise ValueError("count must be >= 0")
        if start < 0:
            start += len(self)
        if start < 0 or start > len(self):
            raise IndexError(start)
        stop = min(start + count, len(self))
        return [self.read_row(index) for index in range(start, stop)]

    def iter_rows(
        self,
        *,
        read_pattern: str = "sequential",
        max_rows: int | None = None,
        read_block_size: int = 32,
        seed: int = 0,
    ) -> Iterator[dict[str, Any]]:
        if read_pattern == "random_block":
            yield from self._iter_random_blocks(max_rows=max_rows, read_block_size=read_block_size, seed=seed)
        elif read_pattern == "sequential":
            limit = min(max_rows if max_rows is not None else len(self), len(self))
            for index in range(limit):
                yield self.read_row(index)
        else:
            raise ValueError(f"Unsupported RowPack read_pattern {read_pattern!r}")

    def row_id_for_name(self, name: str, *, warn_alias: bool = True) -> int:
        if name in self.name_to_row:
            return self.name_to_row[name]

        alias = self.aliases.get(name)
        if alias:
            canonical = alias.get("canonical")
            if canonical in self.name_to_row:
                if warn_alias and alias.get("status") != "canonical":
                    warnings.warn(alias.get("message") or f"{name!r} is an alias for {canonical!r}", stacklevel=2)
                return self.name_to_row[canonical]

        candidates = sorted(set(self.name_to_row) | set(self.aliases))
        close = get_close_matches(name, candidates, n=5)
        suffix = f" Did you mean: {', '.join(close)}?" if close else ""
        raise KeyError(f"Unknown RowPack row name {name!r}.{suffix}")

    def _normalize_index(self, index: int) -> int:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return index

    def _native(self):
        if self.payload_format != "cista" and not any(block.codec != CODEC_NONE for block in self.blocks):
            return None
        if self.native is None:
            self.native = load_native(self.native_module_dir)
        return self.native

    def _read_row_payload(self, entry: RowIndexEntry) -> bytes | memoryview:
        block = self.blocks[entry.block_id]
        if block.codec == CODEC_NONE:
            return self._mmap[entry.offset : entry.offset + entry.size]

        block_payload = self._read_uncompressed_block(entry.block_id)
        start = entry.offset
        stop = start + entry.size
        if stop > len(block_payload):
            raise RowPackError("RowPack row slice exceeds decompressed block size")
        return block_payload[start:stop]

    def _read_uncompressed_block(self, block_id: int) -> bytes:
        cached = self._block_cache.get(block_id)
        if cached is not None:
            return cached

        block = self.blocks[block_id]
        if block.codec not in {CODEC_LZAV_DEFAULT, CODEC_LZAV_HI}:
            raise RowPackError(f"Unsupported RowPack block codec {block.codec}")
        compressed = self._mmap[block.offset : block.offset + block.size]
        payload = self._native().lzav_decompress(bytes(compressed), block.uncompressed_size)
        if len(payload) != block.uncompressed_size:
            raise RowPackError("LZAV decompressed size mismatch")

        self._block_cache[block_id] = payload
        self._block_cache_order.append(block_id)
        while len(self._block_cache_order) > self._block_cache_size:
            old = self._block_cache_order.pop(0)
            if old != block_id:
                self._block_cache.pop(old, None)
        return payload

    def _read_metadata(self) -> dict[str, Any]:
        start = self.header.metadata_offset
        stop = start + self.header.metadata_size
        return json.loads(self._mmap[start:stop].decode("utf-8"))

    def _read_block_index(self) -> list[BlockIndexEntry]:
        if self.header.block_index_size != self.header.block_count * BLOCK_INDEX_STRUCT.size:
            raise RowPackError("Corrupt RowPack block index size")

        blocks = []
        start = self.header.block_index_offset
        for idx in range(self.header.block_count):
            offset = start + idx * BLOCK_INDEX_STRUCT.size
            blocks.append(BlockIndexEntry(*BLOCK_INDEX_STRUCT.unpack(self._mmap[offset : offset + BLOCK_INDEX_STRUCT.size])))
        return blocks

    def _read_row_index(self) -> list[RowIndexEntry]:
        if self.header.row_index_size != self.header.row_count * ROW_INDEX_STRUCT.size:
            raise RowPackError("Corrupt RowPack row index size")

        rows = []
        start = self.header.row_index_offset
        for idx in range(self.header.row_count):
            offset = start + idx * ROW_INDEX_STRUCT.size
            rows.append(RowIndexEntry(*ROW_INDEX_STRUCT.unpack(self._mmap[offset : offset + ROW_INDEX_STRUCT.size])))
        return rows

    def _build_name_index(self) -> dict[str, int]:
        names = self.metadata.get("row_names") or []
        return {name: idx for idx, name in enumerate(names) if name}

    def _iter_random_blocks(
        self,
        *,
        max_rows: int | None,
        read_block_size: int,
        seed: int,
    ) -> Iterator[dict[str, Any]]:
        total_rows = len(self)
        target_rows = max_rows if max_rows is not None else total_rows
        read_block_size = max(1, read_block_size)
        yielded = 0
        rng = TorchLikeRandom(seed)

        while yielded < target_rows:
            window_size = min(read_block_size, target_rows - yielded)
            max_start = max(0, total_rows - window_size)
            start = rng.randint_inclusive(max_start) if max_start else 0
            for index in range(start, start + window_size):
                yield self.read_row(index)
                yielded += 1
                if yielded >= target_rows:
                    break


def serialize_row(
    row: dict[str, Any],
    *,
    row_id: int,
    payload_format: str = "json",
    native: Any = None,
) -> bytes:
    if payload_format == "cista":
        if native is None:
            raise RowPackError("CISTA RowPack payloads require rowpack_native")
        data, images = split_row_payload(row)
        extra = dict(data)
        turns = extra.pop("data", [])
        extra_json = json.dumps(extra, ensure_ascii=False, separators=(",", ":"))
        return native.encode_cista_payload(row_id, turns, images, extra_json)

    if payload_format != "json":
        raise ValueError(f"Unsupported RowPack payload_format {payload_format!r}")

    data, image_payloads = split_row_payload(row)
    images = [image["bytes"] for image in image_payloads]
    data_bytes = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    image_lengths = [len(image) for image in images]

    chunks = [
        ROW_PREFIX_STRUCT.pack(len(data_bytes), len(images), row_id),
        b"".join(length.to_bytes(8, "little") for length in image_lengths),
        data_bytes,
        *images,
    ]
    return b"".join(chunks)


def deserialize_row(
    payload: bytes | memoryview,
    *,
    payload_format: str = "json",
    native: Any = None,
) -> dict[str, Any]:
    if payload_format == "cista":
        if native is None:
            raise RowPackError("CISTA RowPack payloads require rowpack_native")
        row_id, turns, images, extra_json = native.decode_cista_payload(bytes(payload))
        data = json.loads(extra_json) if extra_json else {}
        data["data"] = turns
        data["images"] = [normalize_decoded_image(image) for image in images]
        data.setdefault("_rowpack", {})["row_id"] = row_id
        return data

    if payload_format != "json":
        raise ValueError(f"Unsupported RowPack payload_format {payload_format!r}")

    data_json_size, image_count, row_id = ROW_PREFIX_STRUCT.unpack(payload[: ROW_PREFIX_STRUCT.size])
    cursor = ROW_PREFIX_STRUCT.size
    image_lengths = []
    for _ in range(image_count):
        image_lengths.append(int.from_bytes(payload[cursor : cursor + 8], "little"))
        cursor += 8

    data_start = cursor
    data_stop = data_start + data_json_size
    data = json.loads(bytes(payload[data_start:data_stop]).decode("utf-8"))
    cursor = data_stop

    images = []
    for length in image_lengths:
        images.append({"bytes": bytes(payload[cursor : cursor + length]), "path": None})
        cursor += length

    data["images"] = images
    data.setdefault("_rowpack", {})["row_id"] = row_id
    return data


def split_row_payload(row: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = {key: value for key, value in row.items() if key != "images"}
    images = [coerce_image_payload(image) for image in row.get("images") or []]
    return data, images


def coerce_image_payload(image: Any) -> dict[str, Any]:
    if isinstance(image, dict) and "bytes" in image and any(
        key in image for key in ("height", "width", "channels", "storage")
    ):
        return {
            "bytes": coerce_image_bytes(image["bytes"]),
            "height": int(image.get("height") or 0),
            "width": int(image.get("width") or 0),
            "channels": int(image.get("channels") or 0),
            "storage": image.get("storage") or "encoded",
        }
    return {
        "bytes": coerce_image_bytes(image),
        "height": 0,
        "width": 0,
        "channels": 0,
        "storage": "encoded",
    }


def normalize_decoded_image(image: Any) -> dict[str, Any]:
    if isinstance(image, dict):
        return {
            "bytes": coerce_image_bytes(image.get("bytes") or b""),
            "path": image.get("path"),
            "height": int(image.get("height") or 0),
            "width": int(image.get("width") or 0),
            "channels": int(image.get("channels") or 0),
            "storage": image.get("storage") or "encoded",
        }
    return {"bytes": bytes(image), "path": None, "height": 0, "width": 0, "channels": 0, "storage": "encoded"}


def coerce_image_bytes(image: Any) -> bytes:
    if isinstance(image, bytes):
        return image
    if isinstance(image, bytearray):
        return bytes(image)
    if isinstance(image, memoryview):
        return image.tobytes()
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            return coerce_image_bytes(image["bytes"])
        if image.get("path") is not None:
            return Path(image["path"]).read_bytes()
    if hasattr(image, "save"):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    raise TypeError(f"Unsupported image payload for RowPack: {type(image)!r}")


class TorchLikeRandom:
    def __init__(self, seed: int):
        try:
            import torch
        except Exception:
            torch = None

        self.torch = torch
        if torch is None:
            import random

            self._fallback = random.Random(seed)
        else:
            self._generator = torch.Generator()
            self._generator.manual_seed(seed)

    def randint_inclusive(self, high: int) -> int:
        if self.torch is None:
            return self._fallback.randint(0, high)
        return int(self.torch.randint(high + 1, (1,), generator=self._generator).item())
