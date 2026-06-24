from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Run directly from a checkout without requiring `pip install .`.
SOURCE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PARENT = SOURCE_ROOT.parent
if str(SOURCE_PARENT) not in sys.path:
    sys.path.insert(0, str(SOURCE_PARENT))

from rowpack import MetadataBuilder, RowPackDatasetBuilder, RowPackReader

from _summary import print_reader_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a tiny synthetic RowPack-only write/read benchmark.")
    parser.add_argument("--output", default="build/examples/quick_benchmark.rowpack", help="Output .rowpack path")
    parser.add_argument("--rows", type=int, default=512, help="Synthetic rows to write and read")
    parser.add_argument("--rows-per-block", type=int, default=32)
    parser.add_argument("--payload-format", default="json", choices=["json", "cista"])
    parser.add_argument("--block-codec", default="none", choices=["none", "lzav_default", "lzav_hi"])
    parser.add_argument("--native-module-dir", default=None, help="Directory containing rowpack_native")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # This benchmark is intentionally synthetic. It measures RowPack write/read
    # overhead for a small VQA-like row shape; it does not compare against
    # Parquet or exercise a real dataset loader.
    metadata = (
        MetadataBuilder()
        .dataset_name("quick_benchmark")
        .description("Synthetic RowPack smoke benchmark")
        .compression(block_codec=args.block_codec, rows_per_block=args.rows_per_block)
        .image_codec("raw_rgb")
    )
    image = {"bytes": bytes([17, 23, 42]) * 64, "height": 8, "width": 8, "channels": 3}

    # The write phase includes row serialization and optional block compression.
    # `rows_per_block` controls how many neighboring rows are compressed
    # together, which is the same unit random-block training reads back later.
    write_start = time.perf_counter()
    with RowPackDatasetBuilder(
        output,
        metadata=metadata,
        rows_per_block=args.rows_per_block,
        payload_format=args.payload_format,
        block_codec=args.block_codec,
        image_codec="raw_rgb",
        native_module_dir=args.native_module_dir,
        overwrite=True,
    ) as dataset:
        for index in range(args.rows):
            # A VQA-style row stores turns plus images. The exact text is fake,
            # but the structure matches what a multimodal training loop expects.
            dataset.append_vqa_row(
                turns=[
                    {"role": "user", "modality": "text", "data": f"What is in frame {index}?"},
                    {"role": "assistant", "modality": "text", "data": "A synthetic test pattern."},
                ],
                images=[image],
                extra={"sample_id": index},
                name=f"sample_{index:06d}",
            )
    write_seconds = time.perf_counter() - write_start

    # Random-block reading samples a window, then walks sequentially inside it.
    # That mirrors shuffled training better than pure sequential scans, while
    # still giving storage formats a chance to amortize block reads.
    read_start = time.perf_counter()
    with RowPackReader(output, native_module_dir=args.native_module_dir) as reader:
        read_rows = sum(1 for _row in reader.iter_rows(read_pattern="random_block", read_block_size=args.rows_per_block))
        print_reader_summary(reader, output, title="Synthetic benchmark dataset", include_first_row=False)
    read_seconds = time.perf_counter() - read_start

    samples_per_second = read_rows / read_seconds if read_seconds else float("inf")
    print("  timing:")
    print(f"    rows iterated: {read_rows}")
    print(f"    write time: {write_seconds:.4f} s")
    print(f"    random-block read time: {read_seconds:.4f} s")
    print(f"    random-block read throughput: {samples_per_second:.2f} rows/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
