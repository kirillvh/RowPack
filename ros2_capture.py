from __future__ import annotations

import argparse
import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .authoring import MetadataBuilder, RowPackDatasetBuilder


@dataclass(frozen=True)
class TopicConfig:
    name: str
    type: str
    field: str
    role: str = "json"
    codec: str | None = None
    jpeg_quality: int | None = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record synchronized ROS2 sensor rows into a RowPack dataset.")
    parser.add_argument("config", help="JSON capture config")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    return run_capture(config)


def run_capture(config: dict[str, Any]) -> int:
    try:
        import rclpy
        from rclpy.node import Node
    except ImportError as exc:
        raise SystemExit("RowPack ROS2 capture requires ROS2 Python packages such as rclpy.") from exc

    topics = [TopicConfig(**item) for item in config["topics"]]
    metadata = build_metadata(config, topics)

    builder = RowPackDatasetBuilder(
        config["output"],
        metadata=metadata,
        rows_per_block=int(config.get("rows_per_block", 32)),
        payload_format=config.get("payload_format", "cista"),
        block_codec=config.get("block_codec", "lzav_hi"),
        image_codec=config.get("image_codec", "encoded"),
        jpeg_quality=int(config.get("jpeg_quality", 90)),
        native_module_dir=config.get("native_module_dir"),
        overwrite=bool(config.get("overwrite", False)),
    )

    class RowPackCaptureNode(Node):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__(config.get("node_name", "rowpack_capture"))
            self.latest: dict[str, tuple[int, Any]] = {}
            self.slop_ns = int(float(config.get("sync", {}).get("slop_s", 0.02)) * 1_000_000_000)
            self.row_count = 0
            qos_depth = int(config.get("qos_depth", 10))

            for topic in topics:
                msg_type = load_message_type(topic.type)
                self.create_subscription(msg_type, topic.name, self._callback(topic), qos_depth)
                self.get_logger().info(f"RowPack capture subscribed to {topic.name} as {topic.field}")

        def _callback(self, topic: TopicConfig):
            def handle(msg: Any) -> None:
                self.latest[topic.name] = (message_stamp_ns(msg), msg)
                self._try_emit_row()

            return handle

        def _try_emit_row(self) -> None:
            if len(self.latest) != len(topics):
                return

            stamps = [stamp for stamp, _msg in self.latest.values()]
            if max(stamps) - min(stamps) > self.slop_ns:
                oldest = min(self.latest, key=lambda name: self.latest[name][0])
                self.latest.pop(oldest, None)
                return

            sensors: dict[str, Any] = {}
            images: list[dict[str, Any]] = []
            timestamp_ns = max(stamps)

            for topic in topics:
                _stamp, msg = self.latest[topic.name]
                if topic.role == "image":
                    images.append(image_message_to_rowpack(builder, topic, msg))
                else:
                    sensors[topic.field] = message_to_builtin(msg)

            builder.append_sensor_row(
                sensors,
                images=images,
                timestamp_ns=timestamp_ns,
                name=f"t{timestamp_ns}",
            )
            self.row_count += 1
            self.latest.clear()
            if self.row_count % int(config.get("log_every_rows", 100)) == 0:
                self.get_logger().info(f"RowPack capture wrote {self.row_count} synchronized rows")

        def close(self) -> None:
            builder.finish()

    rclpy.init()
    node = RowPackCaptureNode()
    try:
        rclpy.spin(node)
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()
    return 0


def build_metadata(config: dict[str, Any], topics: list[TopicConfig]) -> MetadataBuilder | dict[str, Any]:
    raw_metadata = config.get("metadata")
    if isinstance(raw_metadata, dict):
        metadata = MetadataBuilder(raw_metadata)
    else:
        metadata = MetadataBuilder()

    metadata.dataset_name(config.get("dataset_name", "rowpack_ros2_capture"))
    if config.get("description"):
        metadata.description(config["description"])
    metadata.compression(block_codec=config.get("block_codec", "lzav_hi"), rows_per_block=int(config.get("rows_per_block", 32)))
    metadata.image_codec(config.get("image_codec", "encoded"), jpeg_quality=int(config.get("jpeg_quality", 90)))
    metadata.row_field("timestamp_ns", "int64", "Synchronized row timestamp in nanoseconds")
    metadata.row_field("sensors", "json", "Non-image ROS messages converted to JSON-compatible values")
    metadata.row_field("images", "image[]", "Image topics encoded according to image_codec settings")
    for topic in topics:
        metadata.sensor(topic.field, topic.type, topic=topic.name, role=topic.role)
    return metadata


def load_message_type(type_spec: str) -> type:
    if ":" in type_spec:
        module_name, class_name = type_spec.split(":", 1)
    else:
        module_name, class_name = type_spec.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def message_stamp_ns(msg: Any) -> int:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None and hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return time.time_ns()


def image_message_to_rowpack(builder: RowPackDatasetBuilder, topic: TopicConfig, msg: Any) -> dict[str, Any]:
    if hasattr(msg, "format") and hasattr(msg, "data") and not hasattr(msg, "height"):
        return {
            "bytes": bytes(msg.data),
            "path": None,
            "height": 0,
            "width": 0,
            "channels": 0,
            "storage": "encoded",
            "format": str(msg.format),
            "topic": topic.name,
            "field": topic.field,
        }

    raw, height, width, channels = ros_image_to_raw(msg)
    return builder.encode_image(
        raw,
        codec=topic.codec or builder.image_codec,
        height=height,
        width=width,
        channels=channels,
        jpeg_quality=topic.jpeg_quality,
    ) | {"topic": topic.name, "field": topic.field}


def ros_image_to_raw(msg: Any) -> tuple[bytes, int, int, int]:
    height = int(msg.height)
    width = int(msg.width)
    encoding = str(msg.encoding).lower()
    data = bytes(msg.data)

    if encoding in {"rgb8", "bgr8"}:
        channels = 3
    elif encoding in {"rgba8", "bgra8"}:
        channels = 4
    elif encoding in {"mono8", "8uc1"}:
        channels = 1
    else:
        raise ValueError(f"Unsupported ROS image encoding {msg.encoding!r}; start with rgb8/bgr8/rgba8/bgra8/mono8")

    row_stride = width * channels
    step = int(getattr(msg, "step", row_stride) or row_stride)
    if step != row_stride:
        data = b"".join(data[row * step : row * step + row_stride] for row in range(height))

    if encoding in {"bgr8", "bgra8"}:
        data = swap_bgr_to_rgb(data, channels)
    return data, height, width, channels


def swap_bgr_to_rgb(data: bytes, channels: int) -> bytes:
    swapped = bytearray(data)
    for idx in range(0, len(swapped), channels):
        swapped[idx], swapped[idx + 2] = swapped[idx + 2], swapped[idx]
    return bytes(swapped)


def message_to_builtin(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        return repr(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"_bytes_len": len(value)}
    if isinstance(value, (list, tuple)):
        return [message_to_builtin(item, depth=depth + 1) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "get_fields_and_field_types"):
        return {
            field: message_to_builtin(getattr(value, field), depth=depth + 1)
            for field in value.get_fields_and_field_types()
            if field != "data"
        }
    slots = getattr(value, "__slots__", None)
    if slots:
        return {
            field.lstrip("_"): message_to_builtin(getattr(value, field), depth=depth + 1)
            for field in slots
            if hasattr(value, field) and field.lstrip("_") != "data"
        }
    return repr(value)


if __name__ == "__main__":
    raise SystemExit(main())
