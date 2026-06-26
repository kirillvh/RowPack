from __future__ import annotations

import argparse
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PARENT = SOURCE_ROOT.parent
if str(SOURCE_PARENT) not in sys.path:
    sys.path.insert(0, str(SOURCE_PARENT))

from rowpack import MetadataBuilder, RowPackDatasetBuilder, RowPackReader


def main() -> int:
    parser = argparse.ArgumentParser(description="Store an already-encoded AVIF file as a RowPack video chunk.")
    parser.add_argument("--input", default="examples/sampleavif.avif", help="AVIF or animated AVIF file to store")
    parser.add_argument("--output", default="build/examples/sample_avif_chunks.rowpack", help="Output .rowpack path")
    parser.add_argument("--payload-format", default="json", choices=["json", "cista"])
    parser.add_argument("--block-codec", default="none", choices=["none", "lzav_default", "lzav_hi"])
    parser.add_argument("--native-module-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = Path(args.input)
    if not source.is_absolute():
        source = Path(__file__).resolve().parents[1] / source

    metadata = (
        MetadataBuilder()
        .dataset_name("sample_avif_chunks")
        .description("Example RowPack with a generic binary AVIF video chunk")
        .row_field("files", "file[]", "Binary attachments stored without base64 expansion")
        .row_field("_rowpack_continuation", "json", "Continuation metadata for media chunk rows")
        .compression(block_codec=args.block_codec, rows_per_block=8)
    )

    with RowPackDatasetBuilder(
        args.output,
        metadata=metadata,
        payload_format=args.payload_format,
        block_codec=args.block_codec,
        native_module_dir=args.native_module_dir,
        overwrite=args.overwrite,
    ) as builder:
        builder.append_video_chunk_row(
            stream="sample_camera",
            chunk={"bytes": source.read_bytes(), "path": str(source), "name": source.name},
            chunk_index=0,
            codec="avif",
            mime_type="image/avif",
            start_timestamp_ns=0,
            end_timestamp_ns=15_000_000_000,
            frame_count=0,
            fps=None,
        )

    with RowPackReader(args.output, native_module_dir=args.native_module_dir) as reader:
        row = reader.read_row(0)
        file_payload = row["files"][0]
        print(f"wrote {args.output}")
        print(f"  rows: {len(reader)}")
        print(f"  stored file: {file_payload.get('name')} ({file_payload.get('mime_type')})")
        print(f"  bytes: {len(file_payload['bytes'])}")
        print(f"  exact byte roundtrip: {file_payload['bytes'] == source.read_bytes()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
