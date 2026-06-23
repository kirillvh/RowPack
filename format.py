from __future__ import annotations

import struct
from dataclasses import dataclass


MAGIC = b"ROWPACK\0"
VERSION_MAJOR = 0
VERSION_MINOR = 1
HEADER_SIZE = 128

FLAG_UNCOMPRESSED = 1 << 0

CODEC_NONE = 0
CODEC_LZAV_DEFAULT = 1
CODEC_LZAV_HI = 2

CODEC_NAMES = {
    CODEC_NONE: "none",
    CODEC_LZAV_DEFAULT: "lzav_default",
    CODEC_LZAV_HI: "lzav_hi",
}
CODEC_IDS = {name: codec for codec, name in CODEC_NAMES.items()}

HEADER_FORMAT = "<8sHHI10Q32s"
ROW_INDEX_FORMAT = "<QQII"
BLOCK_INDEX_FORMAT = "<QQQQQII"
ROW_PREFIX_FORMAT = "<IIQ"

HEADER_STRUCT = struct.Struct(HEADER_FORMAT)
ROW_INDEX_STRUCT = struct.Struct(ROW_INDEX_FORMAT)
BLOCK_INDEX_STRUCT = struct.Struct(BLOCK_INDEX_FORMAT)
ROW_PREFIX_STRUCT = struct.Struct(ROW_PREFIX_FORMAT)

assert HEADER_STRUCT.size == HEADER_SIZE


@dataclass(frozen=True)
class Header:
    major: int
    minor: int
    header_size: int
    flags: int
    row_count: int
    block_count: int
    data_offset: int
    metadata_offset: int
    metadata_size: int
    block_index_offset: int
    block_index_size: int
    row_index_offset: int
    row_index_size: int


@dataclass(frozen=True)
class RowIndexEntry:
    offset: int
    size: int
    block_id: int
    row_in_block: int


@dataclass(frozen=True)
class BlockIndexEntry:
    start_row: int
    row_count: int
    offset: int
    size: int
    uncompressed_size: int
    codec: int
    reserved: int = 0


def pack_header(header: Header) -> bytes:
    return HEADER_STRUCT.pack(
        MAGIC,
        header.major,
        header.minor,
        header.header_size,
        header.flags,
        header.row_count,
        header.block_count,
        header.data_offset,
        header.metadata_offset,
        header.metadata_size,
        header.block_index_offset,
        header.block_index_size,
        header.row_index_offset,
        header.row_index_size,
        b"\0" * 32,
    )


def unpack_header(data: bytes) -> Header:
    if len(data) != HEADER_SIZE:
        raise ValueError(f"RowPack header must be {HEADER_SIZE} bytes, got {len(data)}")

    (
        magic,
        major,
        minor,
        header_size,
        flags,
        row_count,
        block_count,
        data_offset,
        metadata_offset,
        metadata_size,
        block_index_offset,
        block_index_size,
        row_index_offset,
        row_index_size,
        _reserved,
    ) = HEADER_STRUCT.unpack(data)

    if magic != MAGIC:
        raise ValueError(f"Not a RowPack file: bad magic {magic!r}")
    if header_size != HEADER_SIZE:
        raise ValueError(f"Unsupported RowPack header size {header_size}")
    if major != VERSION_MAJOR:
        raise ValueError(f"Unsupported RowPack major version {major}")

    return Header(
        major=major,
        minor=minor,
        header_size=header_size,
        flags=flags,
        row_count=row_count,
        block_count=block_count,
        data_offset=data_offset,
        metadata_offset=metadata_offset,
        metadata_size=metadata_size,
        block_index_offset=block_index_offset,
        block_index_size=block_index_size,
        row_index_offset=row_index_offset,
        row_index_size=row_index_size,
    )


def empty_header() -> Header:
    return Header(
        major=VERSION_MAJOR,
        minor=VERSION_MINOR,
        header_size=HEADER_SIZE,
        flags=FLAG_UNCOMPRESSED,
        row_count=0,
        block_count=0,
        data_offset=HEADER_SIZE,
        metadata_offset=0,
        metadata_size=0,
        block_index_offset=0,
        block_index_size=0,
        row_index_offset=0,
        row_index_size=0,
    )
