from __future__ import annotations

import argparse
import base64
import datetime as dt
import decimal
import glob
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

from .authoring import MetadataBuilder, RowPackDatasetBuilder
from .search_index import DocumentIndexBuilder


AUTO_IMAGE_COLUMNS = {"image", "images", "img", "imgs"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert generic Parquet files to a RowPack dataset.")
    parser.add_argument("--input", action="append", required=True, help="Parquet file, directory, or glob. Repeat for many inputs.")
    parser.add_argument("--output", required=True, help="Output .rowpack path")
    parser.add_argument("--columns", nargs="+", default=None, help="Optional Parquet columns to read")
    parser.add_argument("--drop-column", action="append", default=[], help="Column to exclude from RowPack rows. Repeat as needed.")
    parser.add_argument("--image-column", action="append", default=[], help="Column containing image bytes/structs. Repeat as needed.")
    parser.add_argument("--no-auto-image-columns", action="store_true", help="Do not auto-detect columns named image/images/img/imgs.")
    parser.add_argument("--image-base-dir", default=None, help="Base directory for relative image paths stored in Parquet rows")
    parser.add_argument("--name-column", default=None, help="Column to use as the RowPack row name")
    parser.add_argument("--alias-column", action="append", default=[], help="Column to use as non-canonical row aliases")
    parser.add_argument("--index-column", default=None, help="Column that groups rows into searchable document ranges")
    parser.add_argument("--index-label-column", action="append", default=[], help="Extra column to search in the document index, such as title or author")
    parser.add_argument("--index-alias-column", action="append", default=[], help="Column containing alternate document ids/names")
    parser.add_argument("--index-name", default="documents", help="Metadata search index name")
    parser.add_argument("--dataset-name", default=None, help="Dataset name to store in RowPack metadata")
    parser.add_argument("--description", default=None, help="Dataset description to store in RowPack metadata")
    parser.add_argument("--rows-per-block", type=int, default=32)
    parser.add_argument("--payload-format", default="json", choices=["json", "cista"])
    parser.add_argument("--block-codec", default="none", choices=["none", "lzav_default", "lzav_hi"])
    parser.add_argument("--native-module-dir", default=None, help="Directory containing rowpack_native")
    parser.add_argument("--batch-size", type=int, default=1024, help="Rows to pull from Parquet at a time")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    paths = expand_inputs(args.input)
    image_columns = set(args.image_column)
    if not args.no_auto_image_columns:
        image_columns.update(auto_image_columns(paths, args.columns))

    rows_written = convert_parquet_to_rowpack(
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
        dataset_name=args.dataset_name,
        description=args.description,
        rows_per_block=args.rows_per_block,
        payload_format=args.payload_format,
        block_codec=args.block_codec,
        native_module_dir=args.native_module_dir,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    print(f"converted {rows_written} rows from {len(paths)} Parquet file(s) to {args.output}")
    return 0


def convert_parquet_to_rowpack(
    parquet_paths: Iterable[str | os.PathLike[str]],
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
    dataset_name: str | None = None,
    description: str | None = None,
    rows_per_block: int = 32,
    payload_format: str = "json",
    block_codec: str = "none",
    native_module_dir: str | None = None,
    batch_size: int = 1024,
    overwrite: bool = False,
) -> int:
    pq = import_pyarrow_parquet()
    paths = [Path(path) for path in parquet_paths]
    if not paths:
        raise ValueError("No Parquet input files matched")

    drop_columns = set(drop_columns or set())
    image_columns = set(image_columns or set())
    alias_columns = list(alias_columns)
    index_label_columns = list(index_label_columns)
    index_alias_columns = list(index_alias_columns)
    selected_columns = requested_columns(
        columns,
        image_columns,
        name_column,
        alias_columns,
        index_column=index_column,
        index_label_columns=index_label_columns,
        index_alias_columns=index_alias_columns,
    )
    schema = merged_schema(paths, selected_columns)
    metadata = build_metadata(
        paths,
        schema=schema,
        output=output,
        image_columns=image_columns,
        name_column=name_column,
        alias_columns=alias_columns,
        index_column=index_column,
        index_label_columns=index_label_columns,
        index_alias_columns=index_alias_columns,
        index_name=index_name,
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
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(batch_size=batch_size, columns=selected_columns):
                for record in batch.to_pylist():
                    row, images = rowpack_row_from_parquet_record(
                        record,
                        drop_columns=drop_columns,
                        image_columns=image_columns,
                        image_base_dir=image_base_dir or path.parent,
                    )
                    if images:
                        row["images"] = images
                    row_name = row_name_from_record(record, name_column)
                    aliases = row_aliases_from_record(record, alias_columns) if row_name is not None else []
                    row_id = builder.append_row(row, name=row_name, aliases=aliases)
                    if document_index is not None:
                        document_index.observe(
                            row_id,
                            record.get(index_column),
                            labels=[record.get(column) for column in index_label_columns],
                            aliases=[record.get(column) for column in index_alias_columns],
                            metadata={"source_file": str(path)},
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


def import_pyarrow_parquet():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "Parquet conversion requires pyarrow. Install it with `pip install pyarrow` "
            "or use your project's dependency manager."
        ) from exc
    return pq


def expand_inputs(inputs: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        matches = [Path(match) for match in glob.glob(item)]
        if not matches:
            candidate = Path(item)
            if candidate.is_dir():
                matches = sorted(candidate.glob("*.parquet"))
            elif candidate.exists():
                matches = [candidate]
        for match in matches:
            if match.is_dir():
                paths.extend(sorted(match.glob("*.parquet")))
            else:
                paths.append(match)

    deduped = sorted(dict.fromkeys(path.resolve() for path in paths))
    missing = [str(path) for path in deduped if not path.exists()]
    if missing:
        raise FileNotFoundError("Parquet input path(s) not found: " + ", ".join(missing))
    if not deduped:
        raise FileNotFoundError("No Parquet input files matched")
    return deduped


def auto_image_columns(paths: list[Path], columns: list[str] | None) -> set[str]:
    pq = import_pyarrow_parquet()
    available = set(columns or pq.ParquetFile(paths[0]).schema.names)
    return {name for name in available if name.lower() in AUTO_IMAGE_COLUMNS}


def requested_columns(
    columns: list[str] | None,
    image_columns: set[str],
    name_column: str | None,
    alias_columns: Iterable[str],
    *,
    index_column: str | None = None,
    index_label_columns: Iterable[str] = (),
    index_alias_columns: Iterable[str] = (),
) -> list[str] | None:
    if columns is None:
        return None
    requested = list(
        dict.fromkeys(
            [
                *columns,
                *image_columns,
                *(alias_columns or []),
                *(index_label_columns or []),
                *(index_alias_columns or []),
            ]
        )
    )
    if name_column:
        requested.append(name_column)
    if index_column:
        requested.append(index_column)
    return list(dict.fromkeys(requested))


def merged_schema(paths: list[Path], columns: list[str] | None):
    pq = import_pyarrow_parquet()
    first = pq.ParquetFile(paths[0]).schema_arrow
    if columns is None:
        return first
    return first.select([first.get_field_index(column) for column in columns if first.get_field_index(column) >= 0])


def build_metadata(
    paths: list[Path],
    *,
    schema: Any,
    output: str | os.PathLike[str],
    image_columns: set[str],
    name_column: str | None,
    alias_columns: list[str],
    index_column: str | None,
    index_label_columns: list[str],
    index_alias_columns: list[str],
    index_name: str,
    dataset_name: str | None,
    description: str | None,
    rows_per_block: int,
    block_codec: str,
    payload_format: str,
) -> MetadataBuilder:
    metadata = (
        MetadataBuilder()
        .dataset_name(dataset_name or Path(output).stem)
        .description(description or "Converted from Parquet with rowpack.convert_parquet")
        .compression(block_codec=block_codec, rows_per_block=rows_per_block)
        .image_codec("encoded")
        .extra("source_format", "parquet")
        .extra("source_parquet_files", [str(path) for path in paths])
        .extra("payload_format_requested", payload_format)
        .extra("image_columns", sorted(image_columns))
    )
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

    for field in schema:
        meaning = "Parquet image payload column" if field.name in image_columns else "Parquet source column"
        metadata.row_field(field.name, str(field.type), meaning)
    return metadata


def rowpack_row_from_parquet_record(
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


def extract_images(value: Any, *, base_dir: Path) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        images: list[dict[str, Any]] = []
        for item in value:
            images.extend(extract_images(item, base_dir=base_dir))
        return images
    if isinstance(value, tuple):
        return extract_images(list(value), base_dir=base_dir)
    if isinstance(value, dict):
        if "bytes" in value or "path" in value:
            return [image_payload_from_mapping(value, base_dir=base_dir)]
        return []
    if isinstance(value, (bytes, bytearray, memoryview)):
        return [encoded_image_payload(bytes(value))]
    if isinstance(value, str):
        return [image_payload_from_path(value, base_dir=base_dir)]
    return []


def image_payload_from_mapping(value: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    payload = dict(value)
    if payload.get("bytes") is None and payload.get("path"):
        return image_payload_from_path(str(payload["path"]), base_dir=base_dir, metadata=payload)
    payload["bytes"] = bytes(payload.get("bytes") or b"")
    payload.setdefault("path", None)
    payload["height"] = int(payload.get("height") or 0)
    payload["width"] = int(payload.get("width") or 0)
    payload["channels"] = int(payload.get("channels") or 0)
    payload["storage"] = payload.get("storage") or "encoded"
    return payload


def encoded_image_payload(data: bytes) -> dict[str, Any]:
    return {"bytes": data, "path": None, "height": 0, "width": 0, "channels": 0, "storage": "encoded"}


def image_payload_from_path(path_value: str, *, base_dir: Path, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(path_value)
    if not path.is_absolute():
        path = base_dir / path
    payload = dict(metadata or {})
    payload["bytes"] = path.read_bytes()
    payload["path"] = str(path)
    payload["height"] = int(payload.get("height") or 0)
    payload["width"] = int(payload.get("width") or 0)
    payload["channels"] = int(payload.get("channels") or 0)
    payload["storage"] = payload.get("storage") or "encoded"
    return payload


def row_name_from_record(record: dict[str, Any], name_column: str | None) -> str | None:
    if not name_column:
        return None
    value = record.get(name_column)
    if value is None:
        return None
    return str(value)


def row_aliases_from_record(record: dict[str, Any], alias_columns: Iterable[str]) -> list[str]:
    aliases: list[str] = []
    for column in alias_columns:
        value = record.get(column)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            aliases.extend(str(item) for item in value if item is not None)
        else:
            aliases.append(str(value))
    return aliases


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        return {"_rowpack_type": "bytes_base64", "data": base64.b64encode(data).decode("ascii"), "size": len(data)}
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "as_py"):
        return to_jsonable(value.as_py())
    if hasattr(value, "item"):
        try:
            return to_jsonable(value.item())
        except Exception:
            pass
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


if __name__ == "__main__":
    raise SystemExit(main())
