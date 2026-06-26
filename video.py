from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .native import load_native


AVIF_CODECS = {"avif", "av1_avif"}


def is_avif_codec(codec: str) -> bool:
    return codec.lower() in AVIF_CODECS


def native_avif_unavailable_message(native_module_dir: str | None = None) -> str:
    location = f" from {native_module_dir!r}" if native_module_dir else ""
    return (
        f"Native RowPack AVIF support is not available{location}. "
        "Build RowPack with `cmake -S . -B build -DROWPACK_ENABLE_LIBAVIF=ON` "
        "and `cmake --build build --config Release`, then pass "
        "`--native-module-dir build/Release` if the module is not found automatically. "
        "Use encoder='ffmpeg' only when you intentionally want the system ffmpeg path."
    )


@dataclass
class VideoFrame:
    timestamp_ns: int
    data: bytes
    height: int
    width: int
    channels: int

    def rgb24(self) -> bytes:
        if self.channels == 3:
            return self.data
        if self.channels == 4:
            return bytes(value for index, value in enumerate(self.data) if index % 4 != 3)
        if self.channels == 1:
            out = bytearray()
            for value in self.data:
                out.extend([value, value, value])
            return bytes(out)
        raise ValueError("VideoFrame channels must be 1, 3, or 4")


class FfmpegVideoEncoder:
    """Encode raw RGB frame chunks with the system ffmpeg executable."""

    def __init__(
        self,
        *,
        executable: str = "ffmpeg",
        codec: str = "avif",
        crf: int = 30,
        preset: str | None = None,
        extra_args: list[str] | None = None,
    ):
        self.executable = executable
        self.codec = codec
        self.crf = int(crf)
        self.preset = preset
        self.extra_args = list(extra_args or [])

    def encode(self, frames: list[VideoFrame], *, fps: float | None = None) -> dict[str, Any]:
        if not frames:
            raise ValueError("Cannot encode an empty video chunk")
        self._check_available()

        first = frames[0]
        for frame in frames:
            if frame.height != first.height or frame.width != first.width:
                raise ValueError("All frames in a RowPack video chunk must have the same width and height")

        fps = fps or estimate_fps(frames)
        raw = b"".join(frame.rgb24() for frame in frames)
        with tempfile.TemporaryDirectory(prefix="rowpack_video_") as tmp:
            output = Path(tmp) / self.output_name()
            command = [
                self.executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{first.width}x{first.height}",
                "-r",
                format_fps(fps),
                "-i",
                "pipe:0",
                *self.codec_args(),
                *self.extra_args,
                str(output),
            ]
            result = subprocess.run(command, input=raw, capture_output=True)
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"ffmpeg video encode failed for codec {self.codec!r}: {stderr}")

            payload = output.read_bytes()

        return {
            "bytes": payload,
            "name": self.output_name(),
            "mime_type": self.mime_type(),
            "codec": self.codec,
            "encoder": "ffmpeg",
            "frame_count": len(frames),
            "fps": fps,
            "height": first.height,
            "width": first.width,
            "channels": 3,
            "start_timestamp_ns": frames[0].timestamp_ns,
            "end_timestamp_ns": frames[-1].timestamp_ns,
        }

    def codec_args(self) -> list[str]:
        codec = self.codec.lower()
        if codec in {"avif", "av1_avif"}:
            return ["-an", "-c:v", "libaom-av1", "-crf", str(self.crf), "-b:v", "0", "-f", "avif"]
        if codec in {"h264", "h.264", "avc"}:
            args = ["-an", "-c:v", "libx264", "-crf", str(self.crf), "-pix_fmt", "yuv420p"]
            if self.preset:
                args.extend(["-preset", self.preset])
            return args
        if codec in {"h265", "h.265", "hevc"}:
            args = ["-an", "-c:v", "libx265", "-crf", str(self.crf), "-pix_fmt", "yuv420p"]
            if self.preset:
                args.extend(["-preset", self.preset])
            return args
        raise ValueError(f"Unsupported ffmpeg RowPack video codec {self.codec!r}")

    def output_name(self) -> str:
        codec = self.codec.lower()
        if codec in {"avif", "av1_avif"}:
            return "chunk.avif"
        if codec in {"h264", "h.264", "avc"}:
            return "chunk_h264.mp4"
        if codec in {"h265", "h.265", "hevc"}:
            return "chunk_h265.mp4"
        return "chunk.bin"

    def mime_type(self) -> str:
        codec = self.codec.lower()
        if codec in {"avif", "av1_avif"}:
            return "image/avif"
        if codec in {"h264", "h.264", "avc", "h265", "h.265", "hevc"}:
            return "video/mp4"
        return "application/octet-stream"

    def _check_available(self) -> None:
        if shutil.which(self.executable) is None:
            raise RuntimeError(f"ffmpeg executable {self.executable!r} was not found on PATH")


class LibAvifVideoEncoder:
    """Encode raw RGB frame chunks with the native libavif backend."""

    def __init__(
        self,
        *,
        codec: str = "avif",
        quality: int | None = None,
        crf: int = 30,
        speed: int = 6,
        max_threads: int = 1,
        yuv_format: str = "yuv420",
        native_module_dir: str | None = None,
    ):
        if not is_avif_codec(codec):
            raise ValueError("LibAvifVideoEncoder only supports codec='avif'")
        self.codec = codec
        self.quality = int(quality if quality is not None else max(1, min(100, 100 - int(crf))))
        self.speed = int(speed)
        self.max_threads = int(max_threads)
        self.yuv_format = yuv_format
        self.native_module_dir = native_module_dir

    def encode(self, frames: list[VideoFrame], *, fps: float | None = None) -> dict[str, Any]:
        if not frames:
            raise ValueError("Cannot encode an empty video chunk")

        first = frames[0]
        for frame in frames:
            if frame.height != first.height or frame.width != first.width:
                raise ValueError("All frames in a RowPack video chunk must have the same width and height")

        fps = fps or estimate_fps(frames)
        native = load_native(self.native_module_dir)
        if not hasattr(native, "avif_encode_rgb_sequence"):
            raise RuntimeError(native_avif_unavailable_message(self.native_module_dir))

        payload = native.avif_encode_rgb_sequence(
            [frame.rgb24() for frame in frames],
            int(first.height),
            int(first.width),
            float(fps),
            int(self.quality),
            int(self.speed),
            int(self.max_threads),
            self.yuv_format,
        )

        return {
            "bytes": bytes(payload),
            "name": self.output_name(),
            "mime_type": self.mime_type(),
            "codec": self.codec,
            "encoder": "libavif",
            "quality": self.quality,
            "speed": self.speed,
            "max_threads": self.max_threads,
            "yuv_format": self.yuv_format,
            "frame_count": len(frames),
            "fps": fps,
            "height": first.height,
            "width": first.width,
            "channels": 3,
            "start_timestamp_ns": frames[0].timestamp_ns,
            "end_timestamp_ns": frames[-1].timestamp_ns,
        }

    def output_name(self) -> str:
        return "chunk.avif"

    def mime_type(self) -> str:
        return "image/avif"


class LibAvifVideoDecoder:
    """Decode AVIF image sequences to raw RGB frame buffers."""

    def __init__(self, *, max_threads: int = 1, native_module_dir: str | None = None):
        self.max_threads = int(max_threads)
        self.native_module_dir = native_module_dir

    def decode(self, payload: bytes | bytearray | memoryview | dict[str, Any]) -> dict[str, Any]:
        native = load_native(self.native_module_dir)
        if not hasattr(native, "avif_decode_rgb_sequence"):
            raise RuntimeError(
                "rowpack_native was built without libavif decode support. "
                "Rebuild with -DROWPACK_ENABLE_LIBAVIF=ON and a decoder backend such as dav1d."
            )
        data = payload.get("bytes") if isinstance(payload, dict) else payload
        if data is None:
            raise ValueError("AVIF decode payload is missing bytes")
        decoded = native.avif_decode_rgb_sequence(bytes(data), int(self.max_threads))
        return {
            "frames": [bytes(frame) for frame in decoded["frames"]],
            "width": int(decoded["width"]),
            "height": int(decoded["height"]),
            "channels": int(decoded["channels"]),
            "frame_count": int(decoded["frame_count"]),
            "timescale": int(decoded["timescale"]),
            "duration_in_timescales": int(decoded["duration_in_timescales"]),
            "duration_s": float(decoded["duration_s"]),
            "fps": float(decoded["fps"]),
        }


def decode_avif_chunk(
    payload: bytes | bytearray | memoryview | dict[str, Any],
    *,
    max_threads: int = 1,
    native_module_dir: str | None = None,
) -> dict[str, Any]:
    return LibAvifVideoDecoder(max_threads=max_threads, native_module_dir=native_module_dir).decode(payload)


def libavif_available(native_module_dir: str | None = None) -> bool:
    try:
        native = load_native(native_module_dir)
    except Exception:
        return False
    return hasattr(native, "avif_encode_rgb_sequence")


def libavif_decode_available(native_module_dir: str | None = None) -> bool:
    try:
        native = load_native(native_module_dir)
    except Exception:
        return False
    return hasattr(native, "avif_decode_rgb_sequence")


class VideoChunkBuffer:
    """Accumulate timestamped raw frames and flush them as encoded chunks."""

    def __init__(
        self,
        *,
        stream: str,
        chunk_seconds: float = 15.0,
        codec: str = "avif",
        encoder: str = "ffmpeg",
        fps: float | None = None,
        crf: int = 30,
        quality: int | None = None,
        speed: int = 6,
        max_threads: int = 1,
        yuv_format: str = "yuv420",
        preset: str | None = None,
        ffmpeg: str = "ffmpeg",
        native_module_dir: str | None = None,
        allow_ffmpeg_fallback: bool = False,
    ):
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be > 0")
        self.stream = stream
        self.chunk_seconds = float(chunk_seconds)
        self.fps = fps
        encoder_name = encoder.lower()
        if encoder_name == "auto":
            if is_avif_codec(codec):
                if libavif_available(native_module_dir):
                    encoder_name = "libavif"
                elif allow_ffmpeg_fallback:
                    encoder_name = "ffmpeg"
                else:
                    raise RuntimeError(native_avif_unavailable_message(native_module_dir))
            else:
                encoder_name = "ffmpeg"
        if encoder_name == "libavif":
            self.encoder = LibAvifVideoEncoder(
                codec=codec,
                quality=quality,
                crf=crf,
                speed=speed,
                max_threads=max_threads,
                yuv_format=yuv_format,
                native_module_dir=native_module_dir,
            )
        elif encoder_name == "ffmpeg":
            self.encoder = FfmpegVideoEncoder(executable=ffmpeg, codec=codec, crf=crf, preset=preset)
        else:
            raise ValueError("encoder must be 'auto', 'libavif', or 'ffmpeg'")
        self.encoder_name = encoder_name
        self.frames: list[VideoFrame] = []
        self.next_chunk_index = 0

    def add_frame(
        self,
        *,
        timestamp_ns: int,
        data: bytes,
        height: int,
        width: int,
        channels: int,
    ) -> dict[str, Any] | None:
        self.frames.append(
            VideoFrame(
                timestamp_ns=int(timestamp_ns),
                data=bytes(data),
                height=int(height),
                width=int(width),
                channels=int(channels),
            )
        )
        if self.duration_seconds() >= self.chunk_seconds:
            return self.flush()
        return None

    def flush(self) -> dict[str, Any] | None:
        if not self.frames:
            return None
        chunk = self.encoder.encode(self.frames, fps=self.fps)
        chunk["stream"] = self.stream
        chunk["chunk_index"] = self.next_chunk_index
        self.frames.clear()
        self.next_chunk_index += 1
        return chunk

    def duration_seconds(self) -> float:
        if len(self.frames) < 2:
            return 0.0
        return max(0.0, (self.frames[-1].timestamp_ns - self.frames[0].timestamp_ns) / 1_000_000_000.0)


def estimate_fps(frames: list[VideoFrame]) -> float:
    if len(frames) < 2:
        return 30.0
    duration = max(1e-9, (frames[-1].timestamp_ns - frames[0].timestamp_ns) / 1_000_000_000.0)
    return max(1.0, (len(frames) - 1) / duration)


def format_fps(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")
