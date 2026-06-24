from __future__ import annotations

import argparse
import sys
from pathlib import Path

# These examples are meant to run directly from a fresh checkout, before the
# package has been installed. Adding the repository parent lets `import rowpack`
# work while still avoiding collisions with standard-library modules like `io`.
SOURCE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PARENT = SOURCE_ROOT.parent
if str(SOURCE_PARENT) not in sys.path:
    sys.path.insert(0, str(SOURCE_PARENT))

from rowpack import MetadataBuilder, RowPackDatasetBuilder, RowPackReader

from _summary import print_reader_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a tiny RowPack dataset from Python.")
    parser.add_argument("--output", default="build/examples/robot_demo.rowpack", help="Output .rowpack path")
    parser.add_argument("--native-module-dir", default=None, help="Directory containing rowpack_native")
    parser.add_argument("--payload-format", default="json", choices=["json", "cista"])
    parser.add_argument("--block-codec", default="none", choices=["none", "lzav_default", "lzav_hi"])
    parser.add_argument("--rows-per-block", type=int, default=2)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Metadata is not just decoration. It is where a future reader learns what
    # each row means, which sensors produced it, and how the payload was stored.
    metadata = (
        MetadataBuilder()
        .dataset_name("robot_demo")
        .description("Minimal RowPack authoring example")
        .row_field("timestamp_ns", "int64", "Synchronized row timestamp")
        .sensor("front_camera", "rgb8", topic="/camera/front", frame_id="camera")
        .sensor("imu", "sensor_msgs.msg:Imu", topic="/imu")
        .calibration("front_camera", fx=620.0, fy=620.0, cx=320.0, cy=240.0)
        .compression(block_codec=args.block_codec, rows_per_block=args.rows_per_block)
        .image_codec("raw_rgb")
    )

    # A 2x2 RGB image: four pixels, three uint8 color channels per pixel.
    # Real robot/camera data would usually arrive from a sensor message or a
    # NumPy array; this tiny literal keeps the example completely self-contained.
    raw_rgb = bytes(
        [
            255,
            0,
            0,
            0,
            255,
            0,
            0,
            0,
            255,
            255,
            255,
            255,
        ]
    )

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
        for index in range(3):
            # Each append creates one row. Here a row has JSON-like sensor data,
            # one image, a timestamp, and a stable human-readable row name.
            dataset.append_sensor_row(
                {"imu": {"angular_velocity": [0.0, 0.1 * index, 0.0]}},
                images=[{"bytes": raw_rgb, "height": 2, "width": 2, "channels": 3}],
                timestamp_ns=123456789 + index,
                name=f"frame_{index:06d}",
            )

    # Reopen the file immediately so the example proves both sides of the API:
    # writing the RowPack and reading it back.
    with RowPackReader(output, native_module_dir=args.native_module_dir) as reader:
        print_reader_summary(reader, output, title="Created RowPack dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
