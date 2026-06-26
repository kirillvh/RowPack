from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .authoring import MetadataBuilder
from .convert_jsonl import (
    build_metadata,
    continuation_row_name,
    expand_jsonl_inputs,
    filter_record,
    rowpack_row_from_json_record,
    split_record,
)
from .convert_parquet import row_aliases_from_record
from .format import (
    BLOCK_INDEX_STRUCT,
    CODEC_IDS,
    CODEC_NAMES,
    CODEC_NONE,
    FLAG_UNCOMPRESSED,
    HEADER_SIZE,
    ROW_INDEX_STRUCT,
    BlockIndexEntry,
    Header,
    RowIndexEntry,
    empty_header,
    pack_header,
)
from .io import serialize_row
from .native import load_native
from .search_index import DocumentIndexBuilder


@dataclass
class EncodedBlock:
    block_id: int
    start_row: int
    row_count: int
    payload: bytes
    uncompressed_size: int
    codec_id: int
    row_entries: list[RowIndexEntry]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert large JSONL files to one RowPack with parallel block encoding.")
    parser.add_argument("--input", action="append", required=True, help="JSONL file, .jsonl.gz file, directory, or glob. Repeat as needed.")
    parser.add_argument("--output", required=True, help="Output .rowpack path")
    parser.add_argument("--columns", nargs="+", default=None, help="Optional JSON keys to keep")
    parser.add_argument("--drop-column", action="append", default=[], help="JSON key to exclude. Repeat as needed.")
    parser.add_argument("--name-column", default=None, help="JSON key or dotted path to use as RowPack row name")
    parser.add_argument("--alias-column", action="append", default=[], help="JSON key to use as non-canonical row aliases")
    parser.add_argument("--index-column", default=None, help="JSON key or dotted path used for searchable document ranges")
    parser.add_argument("--index-label-column", action="append", default=[], help="JSON key or dotted path to search in the document index")
    parser.add_argument("--index-alias-column", action="append", default=[], help="JSON key or dotted path containing alternate document ids/names")
    parser.add_argument("--index-name", default="documents")
    parser.add_argument("--split-column", action="append", default=[], help="String column to split across continuation rows")
    parser.add_argument("--split-max-chars", type=int, default=0)
    parser.add_argument("--split-overlap-chars", type=int, default=0)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--description", default=None)
    parser.add_argument("--rows-per-block", type=int, default=128)
    parser.add_argument("--payload-format", default="json", choices=["json", "cista"])
    parser.add_argument("--block-codec", default="lzav_hi", choices=["none", "lzav_default", "lzav_hi"])
    parser.add_argument("--native-module-dir", default=None)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--max-inflight-blocks", type=int, default=None)
    parser.add_argument("--max-input-lines", type=int, default=None, help="Stop after this many source JSONL records")
    parser.add_argument("--max-input-bytes", type=int, default=None, help="Stop after reading at least this many source bytes")
    parser.add_argument("--progress-every-blocks", type=int, default=256)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    metrics = convert_jsonl_to_rowpack_parallel(
        expand_jsonl_inputs(args.input),
        output=args.output,
        columns=args.columns,
        drop_columns=set(args.drop_column),
        name_column=args.name_column,
        alias_columns=args.alias_column,
        index_column=args.index_column,
        index_label_columns=args.index_label_column,
        index_alias_columns=args.index_alias_column,
        index_name=args.index_name,
        split_columns=set(args.split_column),
        split_max_chars=args.split_max_chars,
        split_overlap_chars=args.split_overlap_chars,
        dataset_name=args.dataset_name,
        description=args.description,
        rows_per_block=args.rows_per_block,
        payload_format=args.payload_format,
        block_codec=args.block_codec,
        native_module_dir=args.native_module_dir,
        workers=args.workers,
        max_inflight_blocks=args.max_inflight_blocks,
        max_input_lines=args.max_input_lines,
        max_input_bytes=args.max_input_bytes,
        progress_every_blocks=args.progress_every_blocks,
        overwrite=args.overwrite,
    )
    text = json.dumps(metrics, indent=2)
    print(text)
    if args.summary_json:
        Path(args.summary_json).write_text(text + "\n", encoding="utf-8")
    return 0


def convert_jsonl_to_rowpack_parallel(
    jsonl_paths: Iterable[str | os.PathLike[str]],
    *,
    output: str | os.PathLike[str],
    columns: list[str] | None = None,
    drop_columns: set[str] | None = None,
    name_column: str | None = None,
    alias_columns: Iterable[str] = (),
    index_column: str | None = None,
    index_label_columns: Iterable[str] = (),
    index_alias_columns: Iterable[str] = (),
    index_name: str = "documents",
    split_columns: set[str] | None = None,
    split_max_chars: int = 0,
    split_overlap_chars: int = 0,
    dataset_name: str | None = None,
    description: str | None = None,
    rows_per_block: int = 128,
    payload_format: str = "json",
    block_codec: str = "lzav_hi",
    native_module_dir: str | None = None,
    workers: int = 1,
    max_inflight_blocks: int | None = None,
    max_input_lines: int | None = None,
    max_input_bytes: int | None = None,
    progress_every_blocks: int = 256,
    overwrite: bool = False,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    paths = [Path(path) for path in jsonl_paths]
    if not paths:
        raise ValueError("No JSONL input files matched")
    if rows_per_block < 1:
        raise ValueError("rows_per_block must be >= 1")
    if block_codec not in CODEC_IDS:
        raise ValueError(f"Unsupported block codec {block_codec!r}")
    if payload_format not in {"json", "cista"}:
        raise ValueError(f"Unsupported payload format {payload_format!r}")

    output_path = Path(output)
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    drop_columns = set(drop_columns or set())
    split_columns = set(split_columns or set())
    alias_columns = list(alias_columns)
    index_label_columns = list(index_label_columns)
    index_alias_columns = list(index_alias_columns)
    workers = max(1, int(workers))
    max_inflight_blocks = max_inflight_blocks or max(1, workers * 3)
    codec_id = CODEC_IDS[block_codec]

    metadata_builder = build_metadata(
        paths,
        output=output_path,
        columns=columns,
        drop_columns=drop_columns,
        image_columns=set(),
        name_column=name_column,
        alias_columns=alias_columns,
        index_column=index_column,
        index_label_columns=index_label_columns,
        index_alias_columns=index_alias_columns,
        index_name=index_name,
        split_columns=split_columns,
        split_max_chars=split_max_chars,
        split_overlap_chars=split_overlap_chars,
        dataset_name=dataset_name,
        description=description,
        rows_per_block=rows_per_block,
        block_codec=block_codec,
        payload_format=payload_format,
    )
    metadata_extra = metadata_builder.to_dict() if isinstance(metadata_builder, MetadataBuilder) else dict(metadata_builder)
    metadata_extra["converter"] = "rowpack.convert_jsonl_parallel"
    metadata_extra["parallel_workers"] = workers
    metadata_extra["max_input_lines"] = max_input_lines
    metadata_extra["max_input_bytes"] = max_input_bytes

    document_index = DocumentIndexBuilder(index_name=index_name) if index_column else None
    row_names: list[str | None] | None = [] if name_column else None
    aliases: dict[str, dict[str, Any]] = {}
    block_entries: list[BlockIndexEntry] = []
    row_count = 0
    row_id_counter = 0
    input_records = 0
    input_bytes = 0
    next_block_id = 0
    block_rows: list[dict[str, Any]] = []
    block_start_row = 0
    row_index_temp_path: Path | None = None

    def source_blocks() -> Iterator[tuple[int, int, list[dict[str, Any]]]]:
        nonlocal block_rows, block_start_row, input_records, input_bytes, next_block_id, row_id_counter

        for path, source_line, source_record, raw_size in iter_source_records(
            paths,
            max_input_lines=max_input_lines,
            max_input_bytes=max_input_bytes,
        ):
            input_records += 1
            input_bytes += raw_size
            record = filter_record(source_record, columns=columns, drop_columns=drop_columns)
            base_name = field_text(source_record, name_column) if name_column else None
            split_records = split_record(
                record,
                split_columns=split_columns,
                split_max_chars=split_max_chars,
                split_overlap_chars=split_overlap_chars,
                source_file=str(path),
                source_line=source_line,
                base_name=base_name,
            )

            for part_index, part_record in enumerate(split_records):
                row, images = rowpack_row_from_json_record(
                    part_record,
                    drop_columns=drop_columns,
                    image_columns=set(),
                    image_base_dir=path.parent,
                )
                if images:
                    row["images"] = images

                row_id = row_id_counter
                row_id_counter += 1
                if not block_rows:
                    block_start_row = row_id

                if row_names is not None:
                    row_name = continuation_row_name(base_name, part_index, len(split_records))
                    row_names.append(row_name)
                    if row_name is not None and part_index == 0:
                        for alias in row_aliases_from_record(source_record, alias_columns):
                            aliases[alias] = {
                                "canonical": row_name,
                                "status": "non_canonical",
                                "message": f"{alias!r} is an alias for canonical row name {row_name!r}",
                            }

                if document_index is not None:
                    document_index.observe(
                        row_id,
                        field_value(source_record, index_column),
                        labels=[field_value(source_record, column) for column in index_label_columns],
                        aliases=[field_value(source_record, column) for column in index_alias_columns],
                        metadata={"source_file": str(path), "source_line": source_line} if part_index == 0 else None,
                    )

                block_rows.append(row)
                if len(block_rows) >= rows_per_block:
                    rows = block_rows
                    yield next_block_id, block_start_row, rows
                    next_block_id += 1
                    block_rows = []

        if block_rows:
            rows = block_rows
            yield next_block_id, block_start_row, rows
            next_block_id += 1
            block_rows = []

    def write_encoded_block(encoded: EncodedBlock, out, row_index_temp) -> None:
        nonlocal row_count
        block_offset = out.tell()
        out.write(encoded.payload)
        block_entries.append(
            BlockIndexEntry(
                start_row=encoded.start_row,
                row_count=encoded.row_count,
                offset=block_offset,
                size=len(encoded.payload),
                uncompressed_size=encoded.uncompressed_size,
                codec=encoded.codec_id,
                reserved=0,
            )
        )
        for entry in encoded.row_entries:
            offset = entry.offset if encoded.codec_id != CODEC_NONE else block_offset + entry.offset
            row_index_temp.write(ROW_INDEX_STRUCT.pack(offset, entry.size, encoded.block_id, entry.row_in_block))
        row_count += encoded.row_count

        if progress_every_blocks and len(block_entries) % progress_every_blocks == 0:
            elapsed = max(time.perf_counter() - start_time, 1e-9)
            print(
                f"converted {input_records} source records -> {row_count} rows, "
                f"{len(block_entries)} blocks, {input_bytes / (1024 * 1024):.1f} MiB read, "
                f"{row_count / elapsed:.1f} rows/s",
                flush=True,
            )

    def run_inline(out, row_index_temp) -> None:
        for current_block_id, start_row, rows in source_blocks():
            encoded = encode_block(current_block_id, start_row, rows, payload_format, block_codec, native_module_dir)
            write_encoded_block(encoded, out, row_index_temp)

    def run_pool(out, row_index_temp) -> None:
        next_write_block = 0
        pending: dict[int, EncodedBlock] = {}
        in_flight: dict[concurrent.futures.Future[EncodedBlock], int] = {}

        def drain(done: set[concurrent.futures.Future[EncodedBlock]]) -> None:
            nonlocal next_write_block
            for future in done:
                in_flight.pop(future, None)
                encoded = future.result()
                pending[encoded.block_id] = encoded
            while next_write_block in pending:
                write_encoded_block(pending.pop(next_write_block), out, row_index_temp)
                next_write_block += 1

        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for current_block_id, start_row, rows in source_blocks():
                future = executor.submit(
                    encode_block,
                    current_block_id,
                    start_row,
                    rows,
                    payload_format,
                    block_codec,
                    native_module_dir,
                )
                in_flight[future] = current_block_id
                if len(in_flight) >= max_inflight_blocks:
                    done, _not_done = concurrent.futures.wait(
                        in_flight,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    drain(done)

            while in_flight:
                done, _not_done = concurrent.futures.wait(
                    in_flight,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                drain(done)

    try:
        with tempfile.NamedTemporaryFile(
            prefix=output_path.stem + "_row_index_",
            suffix=".bin",
            dir=output_path.parent,
            delete=False,
        ) as row_index_temp:
            row_index_temp_path = Path(row_index_temp.name)
            with output_path.open("w+b") as out:
                out.write(pack_header(empty_header()))
                if workers == 1:
                    run_inline(out, row_index_temp)
                else:
                    run_pool(out, row_index_temp)

                metadata_offset = out.tell()
                metadata = build_final_metadata(
                    metadata_extra,
                    block_codec=block_codec,
                    payload_format=payload_format,
                    rows_per_block=rows_per_block,
                    row_count=row_count,
                    block_entries=block_entries,
                    row_names=row_names,
                    aliases=aliases,
                    document_index=document_index,
                    index_name=index_name,
                    index_column=index_column,
                    index_label_columns=index_label_columns,
                    index_alias_columns=index_alias_columns,
                )
                metadata_bytes = json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                out.write(metadata_bytes)

                block_index_offset = out.tell()
                for entry in block_entries:
                    out.write(
                        BLOCK_INDEX_STRUCT.pack(
                            entry.start_row,
                            entry.row_count,
                            entry.offset,
                            entry.size,
                            entry.uncompressed_size,
                            entry.codec,
                            entry.reserved,
                        )
                    )
                block_index_size = out.tell() - block_index_offset

                row_index_offset = out.tell()
                row_index_temp.flush()
                row_index_temp.seek(0)
                while True:
                    chunk = row_index_temp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                row_index_size = out.tell() - row_index_offset

                header = Header(
                    major=0,
                    minor=1,
                    header_size=HEADER_SIZE,
                    flags=FLAG_UNCOMPRESSED if codec_id == CODEC_NONE else 0,
                    row_count=row_count,
                    block_count=len(block_entries),
                    data_offset=HEADER_SIZE,
                    metadata_offset=metadata_offset,
                    metadata_size=len(metadata_bytes),
                    block_index_offset=block_index_offset,
                    block_index_size=block_index_size,
                    row_index_offset=row_index_offset,
                    row_index_size=row_index_size,
                )
                out.seek(0)
                out.write(pack_header(header))
    except Exception:
        if output_path.exists():
            output_path.unlink()
        raise
    finally:
        if row_index_temp_path is not None and row_index_temp_path.exists():
            row_index_temp_path.unlink()

    elapsed_s = time.perf_counter() - start_time
    return {
        "output": str(output_path),
        "input_files": [str(path) for path in paths],
        "input_records": input_records,
        "input_bytes_read": input_bytes,
        "rows_written": row_count,
        "blocks_written": len(block_entries),
        "rows_per_block": rows_per_block,
        "payload_format": payload_format,
        "block_codec": block_codec,
        "workers": workers,
        "elapsed_s": elapsed_s,
        "source_mib_per_s": (input_bytes / (1024 * 1024)) / elapsed_s if elapsed_s else 0.0,
        "output_rows_per_s": row_count / elapsed_s if elapsed_s else 0.0,
        "output_bytes": output_path.stat().st_size,
    }


def iter_source_records(
    paths: list[Path],
    *,
    max_input_lines: int | None,
    max_input_bytes: int | None,
) -> Iterator[tuple[Path, int, dict[str, Any], int]]:
    seen_lines = 0
    seen_bytes = 0
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                if max_input_lines is not None and seen_lines >= max_input_lines:
                    return
                if not line.strip():
                    continue
                seen_lines += 1
                seen_bytes += len(line)
                value = json.loads(line)
                record = value if isinstance(value, dict) else {"value": value}
                yield path, line_number, record, len(line)
                if max_input_bytes is not None and seen_bytes >= max_input_bytes:
                    return


def encode_block(
    block_id: int,
    start_row: int,
    rows: list[dict[str, Any]],
    payload_format: str,
    block_codec: str,
    native_module_dir: str | None,
) -> EncodedBlock:
    needs_native = payload_format == "cista" or block_codec != "none"
    native = load_native(native_module_dir) if needs_native else None
    row_entries: list[RowIndexEntry] = []
    payload_parts: list[bytes] = []
    offset = 0
    for row_in_block, row in enumerate(rows):
        payload = serialize_row(
            row,
            row_id=start_row + row_in_block,
            payload_format=payload_format,
            native=native,
        )
        payload_parts.append(payload)
        row_entries.append(
            RowIndexEntry(offset=offset, size=len(payload), block_id=block_id, row_in_block=row_in_block)
        )
        offset += len(payload)

    uncompressed = b"".join(payload_parts)
    codec_id = CODEC_IDS[block_codec]
    payload = uncompressed
    if codec_id != CODEC_NONE:
        if native is None:
            raise RuntimeError("LZAV block compression requires rowpack_native")
        payload = bytes(native.lzav_compress(uncompressed, block_codec))

    return EncodedBlock(
        block_id=block_id,
        start_row=start_row,
        row_count=len(rows),
        payload=payload,
        uncompressed_size=len(uncompressed),
        codec_id=codec_id,
        row_entries=row_entries,
    )


def build_final_metadata(
    metadata_extra: dict[str, Any],
    *,
    block_codec: str,
    payload_format: str,
    rows_per_block: int,
    row_count: int,
    block_entries: list[BlockIndexEntry],
    row_names: list[str | None] | None,
    aliases: dict[str, dict[str, Any]],
    document_index: DocumentIndexBuilder | None,
    index_name: str,
    index_column: str | None,
    index_label_columns: list[str],
    index_alias_columns: list[str],
) -> dict[str, Any]:
    observed_compressions = sorted({CODEC_NAMES[block.codec] for block in block_entries})
    block_payload_bytes = sum(block.size for block in block_entries)
    block_uncompressed_bytes = sum(block.uncompressed_size for block in block_entries)
    metadata = {
        "format": "RowPack",
        "format_version": "0.1",
        "storage": "row-major",
        "compression": block_codec,
        "block_codec": block_codec,
        "observed_compressions": observed_compressions,
        "block_payload_bytes": block_payload_bytes,
        "block_uncompressed_bytes": block_uncompressed_bytes,
        "block_compression_ratio": block_payload_bytes / block_uncompressed_bytes if block_uncompressed_bytes else None,
        "payload_format": payload_format,
        "rows_per_block": rows_per_block,
        "row_count": row_count,
        "block_count": len(block_entries),
        "aliases": aliases,
        **metadata_extra,
    }
    if row_names is not None:
        metadata["row_names"] = row_names
    if document_index is not None and index_column:
        metadata.setdefault("search_indexes", {})[index_name] = document_index.finish()
        metadata.setdefault("search_index_schema", {})[index_name] = document_index.metadata_schema(
            key_column=index_column,
            label_columns=index_label_columns,
            alias_columns=index_alias_columns,
        )
    return metadata


def field_value(record: dict[str, Any], path: str | None) -> Any:
    if not path:
        return None
    value: Any = record
    for part in path.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            return None
    return value


def field_text(record: dict[str, Any], path: str | None) -> str | None:
    value = field_value(record, path)
    return None if value is None else str(value)


if __name__ == "__main__":
    raise SystemExit(main())
