from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

from .authoring import MetadataBuilder, RowPackDatasetBuilder
from .convert_parquet import (
    AUTO_IMAGE_COLUMNS,
    extract_images,
    row_aliases_from_record,
    row_name_from_record,
    to_jsonable,
)
from .search_index import DocumentIndexBuilder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert generic JSONL files to a RowPack dataset.")
    parser.add_argument("--input", action="append", required=True, help="JSONL file, .jsonl.gz file, directory, or glob. Repeat as needed.")
    parser.add_argument("--output", required=True, help="Output .rowpack path")
    parser.add_argument("--columns", nargs="+", default=None, help="Optional JSON keys to keep")
    parser.add_argument("--drop-column", action="append", default=[], help="JSON key to exclude. Repeat as needed.")
    parser.add_argument("--image-column", action="append", default=[], help="JSON key containing image bytes/structs/paths. Repeat as needed.")
    parser.add_argument("--no-auto-image-columns", action="store_true", help="Do not auto-detect keys named image/images/img/imgs.")
    parser.add_argument("--image-base-dir", default=None, help="Base directory for relative image paths stored in JSONL rows")
    parser.add_argument("--name-column", default=None, help="JSON key to use as the RowPack row name")
    parser.add_argument("--alias-column", action="append", default=[], help="JSON key to use as non-canonical row aliases")
    parser.add_argument("--index-column", default=None, help="JSON key that groups rows into searchable document ranges")
    parser.add_argument("--index-label-column", action="append", default=[], help="Extra JSON key to search in the document index, such as title or author")
    parser.add_argument("--index-alias-column", action="append", default=[], help="JSON key containing alternate document ids/names")
    parser.add_argument("--index-name", default="documents", help="Metadata search index name")
    parser.add_argument("--split-column", action="append", default=[], help="String column to split across continuation rows. Repeat as needed.")
    parser.add_argument("--split-max-chars", type=int, default=0, help="Maximum characters per split chunk. 0 disables splitting.")
    parser.add_argument("--split-overlap-chars", type=int, default=0, help="Characters to overlap between adjacent chunks")
    parser.add_argument("--dataset-name", default=None, help="Dataset name to store in RowPack metadata")
    parser.add_argument("--description", default=None, help="Dataset description to store in RowPack metadata")
    parser.add_argument("--rows-per-block", type=int, default=32)
    parser.add_argument("--payload-format", default="json", choices=["json", "cista"])
    parser.add_argument("--block-codec", default="none", choices=["none", "lzav_default", "lzav_hi"])
    parser.add_argument("--native-module-dir", default=None, help="Directory containing rowpack_native")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    paths = expand_jsonl_inputs(args.input)
    image_columns = set(args.image_column)
    if not args.no_auto_image_columns:
        image_columns.update(auto_image_columns(paths, args.columns))

    rows_written = convert_jsonl_to_rowpack(
        paths,
        output=args.output,
        columns=args.columns,
        drop_columns=set(args.drop_column),
        image_columns=image_columns,
        image_base_dir=Path(args.image_base_dir) if args.image_base_dir else None,
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
        overwrite=args.overwrite,
    )
    print(f"converted {rows_written} rows from {len(paths)} JSONL file(s) to {args.output}")
    return 0


def convert_jsonl_to_rowpack(
    jsonl_paths: Iterable[str | os.PathLike[str]],
    *,
    output: str | os.PathLike[str],
    columns: list[str] | None = None,
    drop_columns: set[str] | None = None,
    image_columns: set[str] | None = None,
    image_base_dir: Path | None = None,
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
    rows_per_block: int = 32,
    payload_format: str = "json",
    block_codec: str = "none",
    native_module_dir: str | None = None,
    overwrite: bool = False,
) -> int:
    paths = [Path(path) for path in jsonl_paths]
    if not paths:
        raise ValueError("No JSONL input files matched")
    if split_overlap_chars < 0:
        raise ValueError("split_overlap_chars must be >= 0")
    if split_max_chars < 0:
        raise ValueError("split_max_chars must be >= 0")

    drop_columns = set(drop_columns or set())
    image_columns = set(image_columns or set())
    split_columns = set(split_columns or set())
    alias_columns = list(alias_columns)
    index_label_columns = list(index_label_columns)
    index_alias_columns = list(index_alias_columns)
    metadata = build_metadata(
        paths,
        output=output,
        columns=columns,
        drop_columns=drop_columns,
        image_columns=image_columns,
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

    rows_written = 0
    document_index = DocumentIndexBuilder(index_name=index_name) if index_column else None
    with RowPackDatasetBuilder(
        output,
        metadata=metadata,
        rows_per_block=rows_per_block,
        payload_format=payload_format,
        block_codec=block_codec,
        image_codec="encoded",
        native_module_dir=native_module_dir,
        overwrite=overwrite,
    ) as builder:
        for path in paths:
            for source_line, source_record in iter_jsonl_records(path):
                record = filter_record(source_record, columns=columns, drop_columns=drop_columns)
                base_name = row_name_from_record(source_record, name_column)
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
                        image_columns=image_columns,
                        image_base_dir=image_base_dir or path.parent,
                    )
                    if images:
                        row["images"] = images
                    row_name = continuation_row_name(base_name, part_index, len(split_records))
                    aliases = row_aliases_from_record(source_record, alias_columns) if part_index == 0 and row_name is not None else []
                    row_id = builder.append_row(row, name=row_name, aliases=aliases)
                    if document_index is not None:
                        document_index.observe(
                            row_id,
                            json_field_value(source_record, index_column),
                            labels=[json_field_value(source_record, column) for column in index_label_columns],
                            aliases=[json_field_value(source_record, column) for column in index_alias_columns],
                            metadata={"source_file": str(path), "source_line": source_line} if part_index == 0 else None,
                        )
                    rows_written += 1
        if document_index is not None:
            builder.writer.metadata.setdefault("search_indexes", {})[index_name] = document_index.finish()
            builder.writer.metadata.setdefault("search_index_schema", {})[index_name] = document_index.metadata_schema(
                key_column=index_column,
                label_columns=index_label_columns,
                alias_columns=index_alias_columns,
            )
    return rows_written


def expand_jsonl_inputs(inputs: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        matches = [Path(match) for match in glob_like(item)]
        if not matches:
            candidate = Path(item)
            if candidate.is_dir():
                matches = sorted([*candidate.glob("*.jsonl"), *candidate.glob("*.jsonl.gz")])
            elif candidate.exists():
                matches = [candidate]
        for match in matches:
            if match.is_dir():
                paths.extend(sorted([*match.glob("*.jsonl"), *match.glob("*.jsonl.gz")]))
            else:
                paths.append(match)

    deduped = sorted(dict.fromkeys(path.resolve() for path in paths))
    missing = [str(path) for path in deduped if not path.exists()]
    if missing:
        raise FileNotFoundError("JSONL input path(s) not found: " + ", ".join(missing))
    if not deduped:
        raise FileNotFoundError("No JSONL input files matched")
    return deduped


def glob_like(pattern: str) -> list[str]:
    import glob

    if any(char in pattern for char in "*?[]"):
        return glob.glob(pattern)
    return []


def auto_image_columns(paths: list[Path], columns: list[str] | None) -> set[str]:
    available = set(columns or first_record_keys(paths))
    return {name for name in available if name.lower() in AUTO_IMAGE_COLUMNS}


def first_record_keys(paths: list[Path]) -> list[str]:
    for path in paths:
        for _line, record in iter_jsonl_records(path):
            return list(record)
    return []


def iter_jsonl_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                yield line_number, value
            else:
                yield line_number, {"value": value}


def filter_record(record: dict[str, Any], *, columns: list[str] | None, drop_columns: set[str]) -> dict[str, Any]:
    if columns is not None:
        record = {key: record.get(key) for key in columns if key in record}
    if drop_columns:
        record = {key: value for key, value in record.items() if key not in drop_columns}
    return record


def split_record(
    record: dict[str, Any],
    *,
    split_columns: set[str],
    split_max_chars: int,
    split_overlap_chars: int,
    source_file: str,
    source_line: int,
    base_name: str | None,
) -> list[dict[str, Any]]:
    if split_max_chars <= 0 or not split_columns:
        return [record]

    chunks_by_column: dict[str, list[str]] = {}
    for column in split_columns:
        value = record.get(column)
        if isinstance(value, str) and len(value) > split_max_chars:
            chunks_by_column[column] = split_text(value, max_chars=split_max_chars, overlap_chars=split_overlap_chars)

    if not chunks_by_column:
        return [record]

    part_count = max(len(chunks) for chunks in chunks_by_column.values())
    rows: list[dict[str, Any]] = []
    for part_index in range(part_count):
        if part_index == 0:
            row = dict(record)
        else:
            row = {}

        for column, chunks in chunks_by_column.items():
            if part_index < len(chunks):
                row[column] = chunks[part_index]

        row["_rowpack_split"] = {
            "source_file": source_file,
            "source_line": source_line,
            "parent_row_name": base_name,
            "part_index": part_index,
            "part_count": part_count,
            "columns": sorted(chunks_by_column),
            "is_continuation": part_index > 0,
        }
        rows.append(row)
    return rows


def split_text(text: str, *, max_chars: int, overlap_chars: int = 0) -> list[str]:
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(len(text), start + max_chars)
        if hard_end == len(text):
            split_at = hard_end
        else:
            split_at = text.rfind(" ", start + 1, hard_end + 1)
            if split_at <= start:
                split_at = hard_end
        chunks.append(text[start:split_at])
        if split_at >= len(text):
            break
        next_start = max(split_at - overlap_chars, start + 1)
        if next_start <= start:
            next_start = split_at
        start = next_start
    return chunks


def rowpack_row_from_json_record(
    record: dict[str, Any],
    *,
    drop_columns: set[str],
    image_columns: set[str],
    image_base_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row: dict[str, Any] = {}
    images: list[dict[str, Any]] = []
    for key, value in record.items():
        if key in drop_columns:
            continue
        if key in image_columns:
            images.extend(extract_images(value, base_dir=image_base_dir))
        else:
            row[key] = to_jsonable(value)
    return row, images


def continuation_row_name(base_name: str | None, part_index: int, part_count: int) -> str | None:
    if base_name is None:
        return None
    if part_count <= 1 or part_index == 0:
        return base_name
    return f"{base_name}::part_{part_index:04d}"


def json_field_value(record: dict[str, Any], path: str | None) -> Any:
    if not path:
        return None
    value: Any = record
    for part in path.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            return None
    return value


def build_metadata(
    paths: list[Path],
    *,
    output: str | os.PathLike[str],
    columns: list[str] | None,
    drop_columns: set[str],
    image_columns: set[str],
    name_column: str | None,
    alias_columns: list[str],
    index_column: str | None,
    index_label_columns: list[str],
    index_alias_columns: list[str],
    index_name: str,
    split_columns: set[str],
    split_max_chars: int,
    split_overlap_chars: int,
    dataset_name: str | None,
    description: str | None,
    rows_per_block: int,
    block_codec: str,
    payload_format: str,
) -> MetadataBuilder:
    metadata = (
        MetadataBuilder()
        .dataset_name(dataset_name or Path(output).stem)
        .description(description or "Converted from JSONL with rowpack.convert_jsonl")
        .compression(block_codec=block_codec, rows_per_block=rows_per_block)
        .image_codec("encoded")
        .extra("source_format", "jsonl")
        .extra("source_jsonl_files", [str(path) for path in paths])
        .extra("payload_format_requested", payload_format)
        .extra("image_columns", sorted(image_columns))
        .extra("drop_columns", sorted(drop_columns))
    )
    if columns:
        metadata.extra("selected_columns", columns)
    if name_column:
        metadata.extra("name_column", name_column)
    if alias_columns:
        metadata.extra("alias_columns", alias_columns)
    if index_column:
        metadata.extra(
            "search_index_config",
            {
                "index_name": index_name,
                "kind": "document_ranges",
                "key_column": index_column,
                "label_columns": index_label_columns,
                "alias_columns": index_alias_columns,
            },
        )
    if split_columns and split_max_chars > 0:
        metadata.extra(
            "split_policy",
            {
                "columns": sorted(split_columns),
                "max_chars": split_max_chars,
                "overlap_chars": split_overlap_chars,
                "continuation_rows": "Continuation rows omit unrelated columns and include _rowpack_split metadata.",
            },
        )

    known_columns = columns or sorted(set(first_record_keys(paths)) | set(image_columns) | set(split_columns))
    for column in known_columns:
        if column in drop_columns:
            continue
        if column in image_columns:
            meaning = "JSONL image payload column"
        elif column in split_columns:
            meaning = "JSONL text column that may be split into continuation rows"
        else:
            meaning = "JSONL source column"
        metadata.row_field(column, "dynamic", meaning)
    return metadata


if __name__ == "__main__":
    raise SystemExit(main())
