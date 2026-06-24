from __future__ import annotations

import argparse
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PARENT = SOURCE_ROOT.parent
if str(SOURCE_PARENT) not in sys.path:
    sys.path.insert(0, str(SOURCE_PARENT))

from rowpack import MetadataBuilder, RowPackBlockDataset, RowPackDatasetBuilder, RowPackReader


def main() -> int:
    parser = argparse.ArgumentParser(description="RowPack Python smoke test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--native-module-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rowpack_path = output_dir / "python_smoke.rowpack"

    metadata = (
        MetadataBuilder()
        .dataset_name("python_smoke")
        .row_field("sample_id", "int64")
        .compression(block_codec="lzav_hi", rows_per_block=2)
        .image_codec("raw_rgb")
    )
    image = {"bytes": bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]), "height": 2, "width": 2, "channels": 3}

    with RowPackDatasetBuilder(
        rowpack_path,
        metadata=metadata,
        rows_per_block=2,
        payload_format="cista",
        block_codec="lzav_hi",
        image_codec="raw_rgb",
        native_module_dir=args.native_module_dir,
        overwrite=True,
    ) as dataset:
        for index in range(3):
            dataset.append_vqa_row(
                turns=[
                    {"role": "user", "modality": "text", "data": f"question {index}"},
                    {"role": "assistant", "modality": "text", "data": f"answer {index}"},
                ],
                images=[image],
                extra={"sample_id": index},
                name=f"sample_{index}",
            )

    with RowPackReader(rowpack_path, native_module_dir=args.native_module_dir) as reader:
        assert len(reader) == 3
        assert reader.metadata["block_codec"] == "lzav_hi"
        assert reader.metadata["block_payload_bytes"] > 0
        assert reader.metadata["block_uncompressed_bytes"] > 0
        assert reader.metadata["block_compression_ratio"] is not None
        second = reader.read_row(1)
        assert second["sample_id"] == 1
        assert second["data"][0]["data"] == "question 1"
        assert second["images"][0]["height"] == 2
        assert reader.row_id_for_name("sample_2") == 2

    list_path = output_dir / "rowpacks.txt"
    list_path.write_text(rowpack_path.name + "\n", encoding="utf-8")
    dataset = RowPackBlockDataset(list_path, mode="sequential", return_format="row", native_module_dir=args.native_module_dir)
    rows = list(dataset)
    assert len(rows) == 3
    assert rows[2]["sample_id"] == 2

    print(f"python smoke ok: {rowpack_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
