from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .audio import encode_audio_payload
from .io import RowPackWriter, coerce_file_payload, coerce_image_bytes
from .native import load_native


@dataclass
class MetadataBuilder:
    """Small convenience builder for RowPack dataset metadata."""

    values: dict[str, Any] = field(default_factory=dict)
    row_schema: list[dict[str, Any]] = field(default_factory=list)
    sensors: list[dict[str, Any]] = field(default_factory=list)

    def dataset_name(self, value: str) -> "MetadataBuilder":
        self.values["dataset_name"] = value
        return self

    def description(self, value: str) -> "MetadataBuilder":
        self.values["description"] = value
        return self

    def date_taken(self, value: str) -> "MetadataBuilder":
        self.values["date_taken"] = value
        return self

    def row_field(self, name: str, type: str, meaning: str | None = None, **extra: Any) -> "MetadataBuilder":
        field_info = {"name": name, "type": type}
        if meaning is not None:
            field_info["meaning"] = meaning
        field_info.update(extra)
        self.row_schema.append(field_info)
        return self

    def sensor(
        self,
        name: str,
        type: str,
        description: str | None = None,
        *,
        topic: str | None = None,
        frame_id: str | None = None,
        **extra: Any,
    ) -> "MetadataBuilder":
        sensor_info = {"name": name, "type": type}
        if description is not None:
            sensor_info["description"] = description
        if topic is not None:
            sensor_info["topic"] = topic
        if frame_id is not None:
            sensor_info["frame_id"] = frame_id
        sensor_info.update(extra)
        self.sensors.append(sensor_info)
        return self

    def calibration(self, name: str, **values: Any) -> "MetadataBuilder":
        calibration = self.values.setdefault("calibration", {})
        calibration[name] = values
        return self

    def compression(self, *, block_codec: str = "lzav_hi", rows_per_block: int = 32, **extra: Any) -> "MetadataBuilder":
        self.values["compression_settings"] = {
            "block_codec": block_codec,
            "rows_per_block": rows_per_block,
            **extra,
        }
        return self

    def image_codec(self, codec: str, **options: Any) -> "MetadataBuilder":
        self.values["image_codec_settings"] = {"codec": codec, **options}
        return self

    def audio_codec(self, codec: str, **options: Any) -> "MetadataBuilder":
        self.values["audio_codec_settings"] = {"codec": codec, **options}
        return self

    def search_index(
        self,
        name: str,
        entries: Iterable[dict[str, Any]],
        *,
        schema: dict[str, Any] | None = None,
    ) -> "MetadataBuilder":
        self.values.setdefault("search_indexes", {})[name] = [dict(entry) for entry in entries]
        if schema is not None:
            self.values.setdefault("search_index_schema", {})[name] = dict(schema)
        return self

    def extra(self, key: str, value: Any) -> "MetadataBuilder":
        self.values[key] = value
        return self

    def to_dict(self) -> dict[str, Any]:
        out = dict(self.values)
        if self.row_schema:
            out["row_schema"] = list(self.row_schema)
        if self.sensors:
            out["sensors"] = list(self.sensors)
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))


class RowPackDatasetBuilder:
    """High-level streaming RowPack authoring API.

    Defaults are chosen for publishable training datasets: CISTA row payloads,
    row-major LZAV high-ratio blocks, and source image bytes kept as encoded
    JPEG/PNG/WebP unless the caller asks for raw, QOI, or JPEG re-encoding.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        metadata: MetadataBuilder | dict[str, Any] | None = None,
        rows_per_block: int = 32,
        payload_format: str = "cista",
        block_codec: str = "lzav_hi",
        image_codec: str = "encoded",
        jpeg_quality: int = 90,
        audio_codec: str = "encoded",
        audio_backend: str = "auto",
        opus_bitrate: str = "64k",
        flac_compression_level: int = 5,
        audio_tool: str | None = None,
        native_module_dir: str | None = None,
        overwrite: bool = False,
    ):
        if isinstance(metadata, MetadataBuilder):
            metadata_dict = metadata.to_dict()
        else:
            metadata_dict = dict(metadata or {})
        metadata_dict.setdefault("authoring_api", "RowPackDatasetBuilder")
        metadata_dict.setdefault(
            "image_codec_settings",
            {"codec": image_codec, "jpeg_quality": jpeg_quality} if image_codec == "jpeg_lossy" else {"codec": image_codec},
        )
        audio_settings = {
            "codec": audio_codec,
            "backend": audio_backend,
            "opus_bitrate": opus_bitrate,
            "flac_compression_level": flac_compression_level,
        }
        if audio_tool:
            audio_settings["audio_tool"] = "rowpack_audio_tool"
        metadata_dict.setdefault("audio_codec_settings", audio_settings)

        self.image_codec = image_codec
        self.jpeg_quality = jpeg_quality
        self.audio_codec = audio_codec
        self.audio_backend = audio_backend
        self.opus_bitrate = opus_bitrate
        self.flac_compression_level = flac_compression_level
        self.audio_tool = audio_tool
        self.native_module_dir = native_module_dir
        self._native = None
        self.writer = RowPackWriter(
            path,
            rows_per_block=rows_per_block,
            metadata=metadata_dict,
            payload_format=payload_format,
            block_codec=block_codec,
            native_module_dir=native_module_dir,
            overwrite=overwrite,
        )

    def __enter__(self) -> "RowPackDatasetBuilder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.writer.__exit__(exc_type, exc, tb)

    def append_row(
        self,
        row: dict[str, Any],
        *,
        name: str | None = None,
        aliases: Iterable[str] | None = None,
    ) -> int:
        return self.writer.append_row(row, name=name, aliases=aliases)

    def append_vqa_row(
        self,
        *,
        turns: Iterable[dict[str, Any]],
        images: Iterable[Any] = (),
        files: Iterable[Any] = (),
        extra: dict[str, Any] | None = None,
        name: str | None = None,
        aliases: Iterable[str] | None = None,
    ) -> int:
        row = dict(extra or {})
        row["data"] = list(turns)
        row["images"] = [self.encode_image(image) for image in images]
        encoded_files = [self.encode_file(file) for file in files]
        if encoded_files:
            row["files"] = encoded_files
        return self.append_row(row, name=name, aliases=aliases)

    def append_sensor_row(
        self,
        sensors: dict[str, Any],
        *,
        images: Iterable[Any] = (),
        files: Iterable[Any] = (),
        timestamp_ns: int | None = None,
        name: str | None = None,
        aliases: Iterable[str] | None = None,
    ) -> int:
        row = {"sensors": sensors, "images": [self.encode_image(image) for image in images]}
        encoded_files = [self.encode_file(file) for file in files]
        if encoded_files:
            row["files"] = encoded_files
        if timestamp_ns is not None:
            row["timestamp_ns"] = int(timestamp_ns)
        return self.append_row(row, name=name, aliases=aliases)

    def append_file_row(
        self,
        files: Iterable[Any],
        *,
        extra: dict[str, Any] | None = None,
        name: str | None = None,
        aliases: Iterable[str] | None = None,
    ) -> int:
        row = dict(extra or {})
        row["files"] = [self.encode_file(file) for file in files]
        return self.append_row(row, name=name, aliases=aliases)

    def append_audio_row(
        self,
        audios: Iterable[Any],
        *,
        codec: str | None = None,
        extra: dict[str, Any] | None = None,
        name: str | None = None,
        aliases: Iterable[str] | None = None,
        **audio_options: Any,
    ) -> int:
        row = dict(extra or {})
        row["files"] = [self.encode_audio(audio, codec=codec, **audio_options) for audio in audios]
        row["_rowpack_audio"] = {"kind": "audio", "codec": codec or self.audio_codec}
        return self.append_row(row, name=name, aliases=aliases)

    def append_video_chunk_row(
        self,
        *,
        stream: str,
        chunk: Any,
        chunk_index: int,
        codec: str = "avif",
        mime_type: str = "image/avif",
        start_timestamp_ns: int | None = None,
        end_timestamp_ns: int | None = None,
        frame_count: int | None = None,
        fps: float | None = None,
        extra: dict[str, Any] | None = None,
        name: str | None = None,
        aliases: Iterable[str] | None = None,
    ) -> int:
        existing_file_name = chunk.get("name") if isinstance(chunk, dict) else None
        file_name = existing_file_name or f"{stream}_chunk_{int(chunk_index):06d}{video_file_extension(codec, mime_type)}"
        file_payload = self.encode_file(
            chunk,
            name=file_name,
            mime_type=mime_type,
            role="video_chunk",
            codec=codec,
            stream=stream,
            chunk_index=int(chunk_index),
            start_timestamp_ns=start_timestamp_ns,
            end_timestamp_ns=end_timestamp_ns,
            frame_count=frame_count,
            fps=fps,
        )
        row = dict(extra or {})
        row["files"] = [file_payload]
        row["_rowpack_continuation"] = {
            "kind": "video_chunk",
            "stream": stream,
            "chunk_index": int(chunk_index),
            "is_continuation": True,
        }
        if start_timestamp_ns is not None:
            row["timestamp_ns"] = int(start_timestamp_ns)
            row["_rowpack_continuation"]["start_timestamp_ns"] = int(start_timestamp_ns)
        if end_timestamp_ns is not None:
            row["_rowpack_continuation"]["end_timestamp_ns"] = int(end_timestamp_ns)
        return self.append_row(row, name=name or f"{stream}::chunk_{int(chunk_index):06d}", aliases=aliases)

    def encode_image(
        self,
        image: Any,
        *,
        codec: str | None = None,
        height: int | None = None,
        width: int | None = None,
        channels: int | None = None,
        jpeg_quality: int | None = None,
    ) -> dict[str, Any]:
        codec = codec or self.image_codec
        if codec == "encoded":
            if isinstance(image, dict):
                payload = dict(image)
                payload["bytes"] = coerce_image_bytes(payload)
                payload.setdefault("storage", "encoded")
                payload["height"] = int(height or payload.get("height") or 0)
                payload["width"] = int(width or payload.get("width") or 0)
                payload["channels"] = int(channels or payload.get("channels") or 0)
                return payload
            return {"bytes": coerce_image_bytes(image), "path": None, "height": 0, "width": 0, "channels": 0, "storage": "encoded"}

        raw, raw_height, raw_width, raw_channels = raw_image_bytes(
            image,
            height=height,
            width=width,
            channels=channels,
        )

        if codec == "raw_rgb":
            return {
                "bytes": raw,
                "path": None,
                "height": raw_height,
                "width": raw_width,
                "channels": raw_channels,
                "storage": "raw_rgb",
            }
        if codec == "qoi_lossless":
            return {
                "bytes": raw,
                "path": None,
                "height": raw_height,
                "width": raw_width,
                "channels": raw_channels,
                "storage": "qoi_lossless",
            }
        if codec == "jpeg_lossy":
            quality = self.jpeg_quality if jpeg_quality is None else jpeg_quality
            encoded = self.native().jpeg_encode_rgb(raw, raw_height, raw_width, raw_channels, int(quality))
            return {
                "bytes": bytes(encoded),
                "path": None,
                "height": raw_height,
                "width": raw_width,
                "channels": raw_channels,
                "storage": "encoded",
                "codec": "jpeg",
                "jpeg_quality": int(quality),
            }

        raise ValueError(f"Unsupported RowPack image codec {codec!r}")

    def encode_file(
        self,
        file: Any,
        *,
        name: str | None = None,
        mime_type: str | None = None,
        role: str = "attachment",
        codec: str | None = None,
        **metadata: Any,
    ) -> dict[str, Any]:
        payload = coerce_file_payload(file)
        if name is not None:
            payload["name"] = name
        if mime_type is not None:
            payload["mime_type"] = mime_type
        payload["role"] = role
        if codec is not None:
            payload["codec"] = codec
        for key, value in metadata.items():
            if value is not None:
                payload[key] = value
        payload["size"] = len(payload["bytes"])
        return payload

    def encode_audio(
        self,
        audio: Any,
        *,
        codec: str | None = None,
        backend: str | None = None,
        opus_bitrate: str | None = None,
        flac_compression_level: int | None = None,
        sample_rate: int | None = None,
        channels: int | None = None,
        name: str | None = None,
        **metadata: Any,
    ) -> dict[str, Any]:
        return encode_audio_payload(
            audio,
            codec=codec or self.audio_codec,
            backend=backend or self.audio_backend,
            opus_bitrate=opus_bitrate or self.opus_bitrate,
            flac_compression_level=(
                self.flac_compression_level if flac_compression_level is None else flac_compression_level
            ),
            sample_rate=sample_rate,
            channels=channels,
            name=name,
            audio_tool=self.audio_tool,
            **metadata,
        )

    def native(self):
        if self._native is None:
            self._native = load_native(self.native_module_dir)
        return self._native

    def finish(self) -> None:
        self.writer.finish()

    def close(self) -> None:
        self.writer.close()


def raw_image_bytes(
    image: Any,
    *,
    height: int | None = None,
    width: int | None = None,
    channels: int | None = None,
) -> tuple[bytes, int, int, int]:
    if isinstance(image, dict):
        raw = coerce_image_bytes(image)
        return validate_raw_shape(
            raw,
            height=int(height or image.get("height") or 0),
            width=int(width or image.get("width") or 0),
            channels=int(channels or image.get("channels") or 0),
        )

    if isinstance(image, (bytes, bytearray, memoryview)):
        return validate_raw_shape(coerce_image_bytes(image), height=height, width=width, channels=channels)

    if isinstance(image, (str, os.PathLike, Path)):
        return validate_raw_shape(Path(image).read_bytes(), height=height, width=width, channels=channels)

    if hasattr(image, "__array__"):
        import numpy as np

        array = np.asarray(image)
        if array.dtype != np.uint8:
            raise TypeError("RowPack raw image arrays must have dtype uint8")
        if array.ndim == 2:
            array = array[:, :, None]
        if array.ndim != 3:
            raise ValueError("RowPack raw image arrays must have shape [height, width, channels]")
        contiguous = np.ascontiguousarray(array)
        h, w, c = contiguous.shape
        return contiguous.tobytes(), int(h), int(w), int(c)

    if hasattr(image, "convert"):
        converted = image.convert("RGB")
        return converted.tobytes(), int(converted.height), int(converted.width), 3

    raise TypeError(f"Unsupported raw image object for RowPack: {type(image)!r}")


def validate_raw_shape(
    raw: bytes,
    *,
    height: int | None,
    width: int | None,
    channels: int | None,
) -> tuple[bytes, int, int, int]:
    if not height or not width or not channels:
        raise ValueError("Raw image payloads require height, width, and channels")
    if channels not in {1, 3, 4}:
        raise ValueError("Raw image payloads require 1, 3, or 4 channels")
    expected = int(height) * int(width) * int(channels)
    if len(raw) != expected:
        raise ValueError(f"Raw image byte length {len(raw)} does not match height*width*channels={expected}")
    return raw, int(height), int(width), int(channels)


def video_file_extension(codec: str, mime_type: str | None = None) -> str:
    normalized = codec.lower()
    if normalized in {"avif", "av1_avif"} or mime_type == "image/avif":
        return ".avif"
    if normalized in {"h264", "h.264", "avc", "h265", "h.265", "hevc"} or mime_type == "video/mp4":
        return ".mp4"
    return ".bin"
