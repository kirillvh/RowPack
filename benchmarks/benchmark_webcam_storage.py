from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


SOURCE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PARENT = SOURCE_ROOT.parent
if str(SOURCE_PARENT) not in sys.path:
    sys.path.insert(0, str(SOURCE_PARENT))

from rowpack import MetadataBuilder, NativeCistaVQARows, RowPackDatasetBuilder, RowPackReader, decode_avif_chunk
from rowpack.video import VideoChunkBuffer


@dataclass(frozen=True)
class CapturedFrame:
    index: int
    timestamp_ns: int
    rgb: bytes
    height: int
    width: int
    channels: int = 3


@dataclass(frozen=True)
class Variant:
    name: str
    kind: str
    image_codec: str | None = None
    description: str = ""


VARIANTS = [
    Variant("raw_rgb_frames", "images", "raw_rgb", "One RowPack row per raw RGB frame"),
    Variant("qoi_lossless_frames", "images", "qoi_lossless", "One RowPack row per QOI lossless frame"),
    Variant("jpeg_lossy_frames", "images", "jpeg_lossy", "One RowPack row per STB JPEG frame"),
    Variant("avif_video_chunks", "video", None, "Multi-frame AVIF chunks stored as RowPack file rows"),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark RowPack webcam storage as raw frames, QOI, JPEG, and chunked AVIF."
    )
    parser.add_argument("--output-dir", default="results/webcam_storage_benchmark")
    parser.add_argument("--source", default="webcam", choices=["webcam", "synthetic", "rowpack"])
    parser.add_argument(
        "--source-rowpack",
        default=None,
        help="When --source rowpack is used, replay raw RGB frames from this RowPack file.",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--fps", type=float, default=None, help="Override FPS used for chunk timestamps")
    parser.add_argument("--width", type=int, default=0, help="Requested webcam width")
    parser.add_argument("--height", type=int, default=0, help="Requested webcam height")
    parser.add_argument("--synthetic-width", type=int, default=320)
    parser.add_argument("--synthetic-height", type=int, default=180)
    parser.add_argument("--max-frames", type=int, default=0, help="Stop capture after this many frames. 0 disables.")
    parser.add_argument("--chunk-seconds", type=float, default=5.0)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--avif-quality", type=int, default=70)
    parser.add_argument("--avif-speed", type=int, default=6)
    parser.add_argument("--avif-max-threads", type=int, default=1)
    parser.add_argument("--yuv-format", default="yuv420", choices=["yuv420", "yuv422", "yuv444"])
    parser.add_argument("--rows-per-block", type=int, default=32)
    parser.add_argument("--block-codec", default="none", choices=["none", "lzav_default", "lzav_hi"])
    parser.add_argument("--read-pattern", default="sequential", choices=["sequential", "random_block"])
    parser.add_argument("--read-block-size", type=int, default=32)
    parser.add_argument("--read-repeats", type=int, default=3)
    parser.add_argument(
        "--profile-existing-dir",
        default=None,
        help="Skip capture/write and profile existing variant .rowpack files from this result directory.",
    )
    parser.add_argument("--cache-mode", default="warm", choices=["warm", "evict"])
    parser.add_argument("--cold-cache", action="store_true", help="Alias for --cache-mode evict")
    parser.add_argument(
        "--evict-file",
        default=None,
        help="Large file to stream before each timed read pass. Defaults to OUTPUT_DIR/_cache_evict.bin.",
    )
    parser.add_argument("--evict-mib", type=int, default=1024, help="Create/use this many MiB for cache eviction.")
    parser.add_argument(
        "--evict-read-mib",
        type=int,
        default=0,
        help="MiB to read from the eviction file before each timed pass. 0 reads the whole file.",
    )
    parser.add_argument("--evict-chunk-mib", type=int, default=16)
    parser.add_argument("--native-module-dir", default=None)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace benchmark outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    prepare_cache_mode(args, output_dir)

    if args.profile_existing_dir:
        input_dir = Path(args.profile_existing_dir)
        source_info = load_source_info(input_dir)
        source_info.update(cache_mode_info(args))
        rows = profile_existing_variants(input_dir, source_info, args)
        write_outputs(rows, source_info, output_dir)
        print(f"\nwrote benchmark results to {output_dir}")
        print_summary_table(rows)
        return 0

    frames, measured_fps, capture_seconds = capture_frames(args)
    if not frames:
        raise RuntimeError("No frames were captured")

    raw_source_bytes = sum(len(frame.rgb) for frame in frames)
    source_info = {
        "source": args.source,
        "frame_count": len(frames),
        "height": frames[0].height,
        "width": frames[0].width,
        "channels": frames[0].channels,
        "fps": measured_fps,
        "capture_seconds": capture_seconds,
        "raw_source_bytes": raw_source_bytes,
        "chunk_seconds": args.chunk_seconds,
    }
    source_info.update(cache_mode_info(args))
    (output_dir / "source_info.json").write_text(json.dumps(source_info, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        print(f"writing {variant.name}...")
        if variant.kind == "images":
            path, write_s, encoded_chunks = write_image_variant(
                frames,
                variant,
                output_dir,
                args,
            )
        else:
            path, write_s, encoded_chunks = write_avif_variant(
                frames,
                measured_fps,
                output_dir,
                args,
            )

        print(f"reading {variant.name}...")
        stored_profile = profile_stored_read(path, variant, args)
        decode_profile = profile_decode(path, variant, args)
        storage_profile = rowpack_storage_profile(path, args.native_module_dir)
        row = {
            "variant": variant.name,
            "kind": variant.kind,
            "description": variant.description,
            "path": str(path),
            "rowpack_bytes": path.stat().st_size,
            "rowpack_mib": path.stat().st_size / (1024 * 1024),
            "raw_source_mib": raw_source_bytes / (1024 * 1024),
            "compression_ratio_x": raw_source_bytes / path.stat().st_size if path.stat().st_size else 0.0,
            "write_s": write_s,
            "write_frames_per_s": len(frames) / write_s if write_s else 0.0,
            "frame_count": len(frames),
            "row_count": storage_profile["row_count"],
            "encoded_chunks": encoded_chunks,
            **storage_profile,
            **stored_profile,
            **decode_profile,
        }
        rows.append(row)

    write_outputs(rows, source_info, output_dir)
    print(f"\nwrote benchmark results to {output_dir}")
    print_summary_table(rows)
    return 0


def capture_frames(args: argparse.Namespace) -> tuple[list[CapturedFrame], float, float]:
    if args.source == "synthetic":
        fps = args.fps or 30.0
        frame_count = args.max_frames or max(1, int(round(args.seconds * fps)))
        frames = synthetic_frames(frame_count, args.synthetic_width, args.synthetic_height, fps)
        return frames, fps, frame_count / fps
    if args.source == "rowpack":
        return rowpack_source_frames(args)
    return webcam_frames(args)


def rowpack_source_frames(args: argparse.Namespace) -> tuple[list[CapturedFrame], float, float]:
    if not args.source_rowpack:
        raise ValueError("--source rowpack requires --source-rowpack")
    path = Path(args.source_rowpack)
    frames: list[CapturedFrame] = []
    with RowPackReader(path, native_module_dir=args.native_module_dir) as reader:
        for row_index, row in enumerate(reader.iter_rows(read_pattern="sequential")):
            images = row.get("images") or []
            if not images:
                continue
            image = images[0]
            data = image.get("bytes") or b""
            height = int(image.get("height") or 0)
            width = int(image.get("width") or 0)
            channels = int(image.get("channels") or 3)
            if not data or height <= 0 or width <= 0:
                continue
            expected = height * width * channels
            if len(data) != expected:
                raise ValueError(
                    f"{path} row {row_index} image is not raw packed pixels: "
                    f"got {len(data)} bytes, expected {expected}"
                )
            frame_index = int(row.get("frame_index", len(frames)))
            timestamp_ns = int(row.get("timestamp_ns", frame_index * 1_000_000_000 / (args.fps or 30.0)))
            frames.append(
                CapturedFrame(
                    index=frame_index,
                    timestamp_ns=timestamp_ns,
                    rgb=bytes(data),
                    height=height,
                    width=width,
                    channels=channels,
                )
            )
            if args.max_frames and len(frames) >= args.max_frames:
                break

    if not frames:
        raise RuntimeError(f"No raw RGB frames could be read from {path}")
    fps = args.fps or infer_fps(frames)
    capture_seconds = (frames[-1].timestamp_ns - frames[0].timestamp_ns) / 1_000_000_000 if len(frames) > 1 else 0.0
    if capture_seconds <= 0 and fps > 0:
        capture_seconds = len(frames) / fps
    return frames, fps, capture_seconds


def infer_fps(frames: list[CapturedFrame]) -> float:
    if len(frames) < 2:
        return 30.0
    first = frames[0].timestamp_ns
    last = frames[-1].timestamp_ns
    elapsed = (last - first) / 1_000_000_000
    if elapsed <= 0:
        return 30.0
    return (len(frames) - 1) / elapsed


def prepare_cache_mode(args: argparse.Namespace, output_dir: Path) -> None:
    if args.cold_cache:
        args.cache_mode = "evict"
    args._cache_evict_path = None
    args._cache_evict_bytes = 0
    args._cache_evict_read_bytes = 0
    args._cache_evict_checksum = 0
    if args.cache_mode != "evict":
        return
    if args.evict_mib <= 0:
        raise ValueError("--evict-mib must be positive when cache eviction is enabled")
    if args.evict_chunk_mib <= 0:
        raise ValueError("--evict-chunk-mib must be positive when cache eviction is enabled")

    evict_path = Path(args.evict_file) if args.evict_file else output_dir / "_cache_evict.bin"
    evict_bytes = args.evict_mib * 1024 * 1024
    ensure_evict_file(evict_path, evict_bytes, args.evict_chunk_mib * 1024 * 1024)
    args._cache_evict_path = evict_path
    args._cache_evict_bytes = evict_bytes
    read_bytes = args.evict_read_mib * 1024 * 1024 if args.evict_read_mib > 0 else evict_bytes
    args._cache_evict_read_bytes = min(read_bytes, evict_path.stat().st_size)


def ensure_evict_file(path: Path, target_bytes: int, chunk_bytes: int) -> None:
    if path.exists() and path.stat().st_size >= target_bytes:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pattern = os.urandom(min(chunk_bytes, target_bytes))
    written = 0
    with path.open("wb") as handle:
        while written < target_bytes:
            chunk = pattern[: min(len(pattern), target_bytes - written)]
            handle.write(chunk)
            written += len(chunk)


def evict_cache_for_read(args: argparse.Namespace) -> None:
    evict_path = getattr(args, "_cache_evict_path", None)
    read_bytes = int(getattr(args, "_cache_evict_read_bytes", 0) or 0)
    if evict_path is None or read_bytes <= 0:
        return
    chunk_bytes = max(1, args.evict_chunk_mib) * 1024 * 1024
    remaining = read_bytes
    checksum = int(getattr(args, "_cache_evict_checksum", 0) or 0)
    with Path(evict_path).open("rb", buffering=0) as handle:
        while remaining > 0:
            chunk = handle.read(min(chunk_bytes, remaining))
            if not chunk:
                handle.seek(0)
                continue
            checksum ^= chunk[0]
            checksum ^= chunk[-1]
            remaining -= len(chunk)
    args._cache_evict_checksum = checksum


def cache_mode_info(args: argparse.Namespace) -> dict[str, Any]:
    if args.cache_mode != "evict":
        return {"read_cache_mode": "warm"}
    evict_path = getattr(args, "_cache_evict_path", None)
    return {
        "read_cache_mode": "large_file_evict",
        "cache_evict_file": str(evict_path) if evict_path else None,
        "cache_evict_mib": getattr(args, "_cache_evict_bytes", 0) / (1024 * 1024),
        "cache_evict_read_mib": getattr(args, "_cache_evict_read_bytes", 0) / (1024 * 1024),
        "cache_evict_note": (
            "Portable cold-ish cache pressure. This streams a large separate file before each timed "
            "read pass, but it is not a privileged OS page-cache flush."
        ),
    }


def load_source_info(input_dir: Path) -> dict[str, Any]:
    source_path = input_dir / "source_info.json"
    if source_path.exists():
        return json.loads(source_path.read_text(encoding="utf-8"))
    return {
        "source": f"existing:{input_dir}",
        "frame_count": 0,
        "height": 0,
        "width": 0,
        "channels": 3,
        "fps": 0.0,
        "capture_seconds": 0.0,
        "raw_source_bytes": 0,
        "chunk_seconds": 0.0,
    }


def profile_existing_variants(
    input_dir: Path,
    source_info: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    prior_rows = load_prior_summary(input_dir)
    raw_source_bytes = int(source_info.get("raw_source_bytes") or 0)
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        path = input_dir / f"{variant.name}.rowpack"
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"reading {variant.name}...")
        stored_profile = profile_stored_read(path, variant, args)
        decode_profile = profile_decode(path, variant, args)
        storage_profile = rowpack_storage_profile(path, args.native_module_dir)
        prior = prior_rows.get(variant.name, {})
        write_s = float_value(prior, "write_s", 0.0)
        frame_count = int_value(prior, "frame_count", int(source_info.get("frame_count") or 0))
        if frame_count == 0:
            frame_count = int(decode_profile.get("read_decode_frames_per_s", 0.0) * decode_profile.get("read_decode_s", 0.0))
        row = {
            "variant": variant.name,
            "kind": variant.kind,
            "description": variant.description,
            "path": str(path),
            "rowpack_bytes": path.stat().st_size,
            "rowpack_mib": path.stat().st_size / (1024 * 1024),
            "raw_source_mib": raw_source_bytes / (1024 * 1024) if raw_source_bytes else 0.0,
            "compression_ratio_x": raw_source_bytes / path.stat().st_size if raw_source_bytes and path.stat().st_size else 0.0,
            "write_s": write_s,
            "write_frames_per_s": float_value(prior, "write_frames_per_s", frame_count / write_s if write_s else 0.0),
            "frame_count": frame_count,
            "row_count": storage_profile["row_count"],
            "encoded_chunks": int_value(prior, "encoded_chunks", storage_profile["row_count"]),
            **storage_profile,
            **stored_profile,
            **decode_profile,
        }
        rows.append(row)
    return rows


def load_prior_summary(input_dir: Path) -> dict[str, dict[str, str]]:
    summary_path = input_dir / "summary.csv"
    if not summary_path.exists():
        return {}
    with summary_path.open(newline="", encoding="utf-8") as handle:
        return {row.get("variant", ""): row for row in csv.DictReader(handle)}


def float_value(row: dict[str, Any], key: str, default: float) -> float:
    try:
        value = row.get(key, default)
        return float(value) if value not in {None, ""} else default
    except (TypeError, ValueError):
        return default


def int_value(row: dict[str, Any], key: str, default: int) -> int:
    try:
        value = row.get(key, default)
        return int(float(value)) if value not in {None, ""} else default
    except (TypeError, ValueError):
        return default


def webcam_frames(args: argparse.Namespace) -> tuple[list[CapturedFrame], float, float]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Webcam capture requires OpenCV Python: `python -m pip install opencv-python`") from exc

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open camera index {args.camera}")
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    raw_frames: list[bytes] = []
    height = 0
    width = 0
    started = time.perf_counter()
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            height, width = frame_rgb.shape[:2]
            raw_frames.append(frame_rgb.tobytes())

            if args.preview:
                cv2.imshow("RowPack storage benchmark capture", frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in {27, ord("q")}:
                    break

            if args.max_frames and len(raw_frames) >= args.max_frames:
                break
            if args.seconds > 0 and time.perf_counter() - started >= args.seconds:
                break
    finally:
        cap.release()
        if args.preview:
            cv2.destroyAllWindows()

    elapsed = max(time.perf_counter() - started, 1e-9)
    fps = args.fps or (len(raw_frames) / elapsed if raw_frames else float(cap.get(cv2.CAP_PROP_FPS) or 30.0))
    frames = [
        CapturedFrame(
            index=index,
            timestamp_ns=int(round(index * 1_000_000_000 / fps)),
            rgb=data,
            height=height,
            width=width,
        )
        for index, data in enumerate(raw_frames)
    ]
    return frames, fps, elapsed


def synthetic_frames(frame_count: int, width: int, height: int, fps: float) -> list[CapturedFrame]:
    frames: list[CapturedFrame] = []
    for index in range(frame_count):
        payload = bytearray(width * height * 3)
        box_size = max(12, min(width, height) // 5)
        box_x = (index * 5) % max(1, width - box_size)
        box_y = (index * 3) % max(1, height - box_size)
        cursor = 0
        for y in range(height):
            for x in range(width):
                base_r = (x * 255) // max(1, width - 1)
                base_g = (y * 255) // max(1, height - 1)
                base_b = ((x + y + index) * 255) // max(1, width + height + frame_count)
                if box_x <= x < box_x + box_size and box_y <= y < box_y + box_size:
                    base_r = 240
                    base_g = 48 + (index * 3) % 160
                    base_b = 32
                payload[cursor] = base_r
                payload[cursor + 1] = base_g
                payload[cursor + 2] = base_b
                cursor += 3
        frames.append(
            CapturedFrame(
                index=index,
                timestamp_ns=int(round(index * 1_000_000_000 / fps)),
                rgb=bytes(payload),
                height=height,
                width=width,
            )
        )
    return frames


def write_image_variant(
    frames: list[CapturedFrame],
    variant: Variant,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path, float, int]:
    assert variant.image_codec is not None
    path = output_dir / f"{variant.name}.rowpack"
    metadata = base_metadata(args, variant)
    if variant.image_codec == "jpeg_lossy":
        metadata.image_codec(variant.image_codec, jpeg_quality=args.jpeg_quality)
    else:
        metadata.image_codec(variant.image_codec)

    started = time.perf_counter()
    with RowPackDatasetBuilder(
        path,
        metadata=metadata,
        rows_per_block=args.rows_per_block,
        payload_format="cista",
        block_codec=args.block_codec,
        image_codec=variant.image_codec,
        jpeg_quality=args.jpeg_quality,
        native_module_dir=args.native_module_dir,
        overwrite=True,
    ) as builder:
        for frame in frames:
            builder.append_vqa_row(
                turns=[
                    {"role": "user", "modality": "text", "data": f"Describe webcam frame {frame.index}."},
                    {"role": "assistant", "modality": "text", "data": "A captured webcam frame."},
                ],
                images=[
                    {
                        "bytes": frame.rgb,
                        "height": frame.height,
                        "width": frame.width,
                        "channels": frame.channels,
                    }
                ],
                extra={"frame_index": frame.index, "timestamp_ns": frame.timestamp_ns},
                name=f"frame_{frame.index:06d}",
            )
    return path, time.perf_counter() - started, len(frames)


def write_avif_variant(
    frames: list[CapturedFrame],
    fps: float,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path, float, int]:
    path = output_dir / "avif_video_chunks.rowpack"
    variant = next(item for item in VARIANTS if item.name == "avif_video_chunks")
    metadata = (
        base_metadata(args, variant)
        .row_field("files", "file[]", "Encoded AVIF video chunks")
        .row_field("_rowpack_continuation", "json", "Video chunk continuation metadata")
        .extra(
            "avif_settings",
            {
                "quality": args.avif_quality,
                "speed": args.avif_speed,
                "max_threads": args.avif_max_threads,
                "yuv_format": args.yuv_format,
                "chunk_seconds": args.chunk_seconds,
            },
        )
    )
    buffer = VideoChunkBuffer(
        stream="webcam",
        chunk_seconds=args.chunk_seconds,
        codec="avif",
        encoder="auto",
        fps=fps,
        quality=args.avif_quality,
        speed=args.avif_speed,
        max_threads=args.avif_max_threads,
        yuv_format=args.yuv_format,
        native_module_dir=args.native_module_dir,
    )

    started = time.perf_counter()
    chunks = 0
    with RowPackDatasetBuilder(
        path,
        metadata=metadata,
        rows_per_block=1,
        payload_format="cista",
        block_codec=args.block_codec,
        native_module_dir=args.native_module_dir,
        overwrite=True,
    ) as builder:
        for frame in frames:
            chunk = buffer.add_frame(
                timestamp_ns=frame.timestamp_ns,
                data=frame.rgb,
                height=frame.height,
                width=frame.width,
                channels=frame.channels,
            )
            if chunk is not None:
                append_video_chunk(builder, chunk)
                chunks += 1
        chunk = buffer.flush()
        if chunk is not None:
            append_video_chunk(builder, chunk)
            chunks += 1
    return path, time.perf_counter() - started, chunks


def append_video_chunk(builder: RowPackDatasetBuilder, chunk: dict[str, Any]) -> None:
    builder.append_video_chunk_row(
        stream=str(chunk["stream"]),
        chunk=chunk,
        chunk_index=int(chunk["chunk_index"]),
        codec=str(chunk["codec"]),
        mime_type=str(chunk["mime_type"]),
        start_timestamp_ns=int(chunk["start_timestamp_ns"]),
        end_timestamp_ns=int(chunk["end_timestamp_ns"]),
        frame_count=int(chunk["frame_count"]),
        fps=float(chunk["fps"]),
        extra={"encoder": chunk.get("encoder"), "frame_count": int(chunk["frame_count"])},
    )


def base_metadata(args: argparse.Namespace, variant: Variant) -> MetadataBuilder:
    return (
        MetadataBuilder()
        .dataset_name(f"rowpack_webcam_storage_{variant.name}")
        .description(variant.description)
        .row_field("data", "turn[]", "Tiny VQA-style prompt used to exercise native CISTA rows")
        .row_field("images", "image[]", "Captured RGB frames")
        .compression(block_codec=args.block_codec, rows_per_block=args.rows_per_block)
        .sensor("webcam", "rgb8", role="camera")
        .extra(
            "benchmark",
            {
                "name": "webcam_storage",
                "variant": variant.name,
                "source": args.source,
                "chunk_seconds": args.chunk_seconds,
                "note": "Image variants store one independently encoded frame per row; AVIF stores many frames per chunk row.",
            },
        )
    )


def profile_stored_read(path: Path, variant: Variant, args: argparse.Namespace) -> dict[str, float]:
    total_frames = 0
    total_payload_bytes = 0
    elapsed = 0.0
    for repeat in range(max(1, args.read_repeats)):
        evict_cache_for_read(args)
        started = time.perf_counter()
        with RowPackReader(path, native_module_dir=args.native_module_dir) as reader:
            for row in reader.iter_rows(
                read_pattern=args.read_pattern,
                read_block_size=args.read_block_size,
                seed=repeat,
            ):
                if variant.kind == "images":
                    images = row.get("images") or []
                    total_frames += len(images)
                    total_payload_bytes += sum(len(image.get("bytes") or b"") for image in images)
                else:
                    files = row.get("files") or []
                    for file in files:
                        total_frames += int(file.get("frame_count") or 0)
                        total_payload_bytes += len(file.get("bytes") or b"")
        elapsed += time.perf_counter() - started
    return {
        "stored_read_s": elapsed,
        "stored_payload_mib": total_payload_bytes / (1024 * 1024),
        "stored_read_frames_per_s": total_frames / elapsed if elapsed else 0.0,
        "stored_read_mib_per_s": (total_payload_bytes / (1024 * 1024)) / elapsed if elapsed else 0.0,
    }


def profile_decode(path: Path, variant: Variant, args: argparse.Namespace) -> dict[str, float]:
    total_frames = 0
    total_decoded_bytes = 0
    elapsed = 0.0
    for repeat in range(max(1, args.read_repeats)):
        evict_cache_for_read(args)
        started = time.perf_counter()
        if variant.kind == "images":
            rows = NativeCistaVQARows(
                [str(path)],
                read_pattern=args.read_pattern,
                read_block_size=args.read_block_size,
                seed=repeat,
                native_module_dir=args.native_module_dir,
            )
            for _row_id, _pairs, images in rows:
                total_frames += len(images)
                total_decoded_bytes += sum(len(image.get("bytes") or b"") for image in images)
        else:
            with RowPackReader(path, native_module_dir=args.native_module_dir) as reader:
                for row in reader.iter_rows(
                    read_pattern=args.read_pattern,
                    read_block_size=args.read_block_size,
                    seed=repeat,
                ):
                    for file in row.get("files") or []:
                        decoded = decode_avif_chunk(file, max_threads=args.avif_max_threads, native_module_dir=args.native_module_dir)
                        frames = decoded.get("frames") or []
                        total_frames += len(frames)
                        total_decoded_bytes += sum(len(frame) for frame in frames)
        elapsed += time.perf_counter() - started
    result = {
        "decode_s": elapsed,
        "decoded_mib": total_decoded_bytes / (1024 * 1024),
        "decode_frames_per_s": total_frames / elapsed if elapsed else 0.0,
        "decode_mib_per_s": (total_decoded_bytes / (1024 * 1024)) / elapsed if elapsed else 0.0,
    }
    result.update(
        {
            "read_decode_s": result["decode_s"],
            "read_decode_frames_per_s": result["decode_frames_per_s"],
            "read_decode_mib_per_s": result["decode_mib_per_s"],
        }
    )
    return result


def row_count(path: Path, native_module_dir: str | None) -> int:
    with RowPackReader(path, native_module_dir=native_module_dir) as reader:
        return len(reader)


def rowpack_storage_profile(path: Path, native_module_dir: str | None) -> dict[str, Any]:
    with RowPackReader(path, native_module_dir=native_module_dir) as reader:
        return {
            "row_count": len(reader),
            "block_count": int(reader.metadata.get("block_count") or len(reader.blocks)),
            "rows_per_block": int(reader.metadata.get("rows_per_block") or 0),
            "block_codec": str(reader.metadata.get("block_codec") or reader.metadata.get("compression") or ""),
        }


def write_outputs(rows: list[dict[str, Any]], source_info: dict[str, Any], output_dir: Path) -> None:
    (output_dir / "source_info.json").write_text(json.dumps(source_info, indent=2), encoding="utf-8")
    fieldnames = [
        "variant",
        "kind",
        "rowpack_mib",
        "compression_ratio_x",
        "write_s",
        "write_frames_per_s",
        "stored_read_frames_per_s",
        "stored_read_mib_per_s",
        "read_decode_frames_per_s",
        "read_decode_mib_per_s",
        "frame_count",
        "row_count",
        "block_count",
        "rows_per_block",
        "block_codec",
        "encoded_chunks",
        "path",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        "# RowPack Webcam Storage Benchmark",
        "",
        f"- source: `{source_info['source']}`",
        f"- frames: {source_info['frame_count']}",
        f"- shape: {source_info['height']}x{source_info['width']}x{source_info['channels']}",
        f"- fps used for chunk timestamps: {source_info['fps']:.2f}",
        f"- raw RGB source size: {source_info['raw_source_bytes'] / (1024 * 1024):.2f} MiB",
        f"- AVIF chunk window: {source_info['chunk_seconds']:.2f} s",
        f"- read cache mode: `{source_info.get('read_cache_mode', 'warm')}`",
    ]
    if source_info.get("read_cache_mode") == "large_file_evict":
        lines.extend(
            [
                f"- cache eviction file size: {float(source_info.get('cache_evict_mib') or 0.0):.0f} MiB",
                f"- cache eviction read before each timed pass: {float(source_info.get('cache_evict_read_mib') or 0.0):.0f} MiB",
                "- cache eviction note: portable large-file eviction, not a privileged OS page-cache flush",
            ]
        )
    lines.extend(
        [
        "",
        "Image variants store one independently encoded frame per row. AVIF stores many neighboring frames in each chunk row, so it can exploit temporal redundancy between frames.",
        "",
        "| variant | file MiB | raw-to-file | block codec | rows/block | write frames/s | stored read frames/s | read+decode frames/s | rows | encoded units |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {variant} | {rowpack_mib:.2f} | {compression_ratio_x:.2f}x | {block_codec} | {rows_per_block} | "
            "{write_frames_per_s:.1f} | {stored_read_frames_per_s:.1f} | {read_decode_frames_per_s:.1f} | "
            "{row_count} | {encoded_chunks} |".format(**row)
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    make_charts(rows, output_dir)


def make_charts(rows: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception:
        return

    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    save_bar(
        rows,
        "rowpack_mib",
        "RowPack File Size",
        "MiB, log scale",
        chart_dir / "rowpack_size_mib.png",
        "{:.2f}",
        log=True,
        label_builder=lambda row, value: f"{value:.2f} MiB\n({float(row.get('compression_ratio_x') or 0.0):.2f}x)",
        label_position="outside",
        label_color="black",
    )
    save_bar(rows, "compression_ratio_x", "Raw RGB To RowPack Compression", "x smaller", chart_dir / "compression_ratio_x.png", "{:.2f}x")
    save_bar(rows, "write_frames_per_s", "Write Throughput", "frames/s", chart_dir / "write_frames_per_s.png", "{:.1f}")
    save_bar(rows, "stored_read_frames_per_s", "Stored Payload Read Throughput", "frames/s", chart_dir / "stored_read_frames_per_s.png", "{:.1f}")
    save_bar(
        rows,
        "read_decode_frames_per_s",
        "Read + Decode To RGB Throughput",
        "frames/s",
        chart_dir / "read_decode_frames_per_s.png",
        "{:.1f}",
        label_builder=lambda _row, value: f"{value:.1f} FPS",
        label_position="outside",
        label_color="black",
    )


def save_bar(
    rows: list[dict[str, Any]],
    key: str,
    title: str,
    ylabel: str,
    path: Path,
    label_format: str,
    *,
    log: bool = False,
    label_builder: Callable[[dict[str, Any], float], str] | None = None,
    label_position: str = "inside",
    label_color: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    labels = [row["variant"].replace("_", "\n") for row in rows]
    values = [float(row.get(key) or 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.45), 5.4))
    bars = ax.bar(labels, values, color=[variant_color(row["variant"]) for row in rows])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=0)
    if log and all(value > 0 for value in values):
        ax.set_yscale("log")
    else:
        apply_zoomed_ylim(ax, values)
    add_label_headroom(ax, values, log=log, label_position=label_position)
    label_bars(
        ax,
        bars,
        rows,
        values,
        label_format,
        log=log,
        label_builder=label_builder,
        label_position=label_position,
        label_color=label_color,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def variant_color(name: str) -> str:
    if "avif" in name:
        return "#f28e2b"
    if "jpeg" in name:
        return "#4e79a7"
    if "qoi" in name:
        return "#59a14f"
    return "#9c9c9c"


def apply_zoomed_ylim(ax: Any, values: list[float]) -> None:
    positive = [value for value in values if value > 0]
    if not positive:
        return
    min_value = min(positive)
    max_value = max(positive)
    margin = max((max_value - min_value) * 0.25, max_value * 0.03, 0.01)
    lower = max(0.0, min_value - margin)
    upper = max_value + margin
    if lower == 0.0 and min_value > 0:
        lower = min_value * 0.85
    if upper > lower:
        ax.set_ylim(lower, upper)


def add_label_headroom(ax: Any, values: list[float], *, log: bool, label_position: str) -> None:
    if label_position != "outside":
        return
    positive = [value for value in values if value > 0]
    if not positive:
        return
    lower, upper = ax.get_ylim()
    max_value = max(positive)
    if log:
        ax.set_ylim(lower, max(upper, max_value * 2.8))
    else:
        span = max(upper - lower, max_value, 1e-9)
        ax.set_ylim(lower, upper + span * 0.18)


def label_bars(
    ax: Any,
    bars: Iterable[Any],
    rows: list[dict[str, Any]],
    values: list[float],
    label_format: str,
    *,
    log: bool,
    label_builder: Callable[[dict[str, Any], float], str] | None = None,
    label_position: str = "inside",
    label_color: str | None = None,
) -> None:
    lower, upper = ax.get_ylim()
    for bar, row, value in zip(bars, rows, values):
        if label_position == "outside":
            if log:
                y = value * 1.2
            else:
                span = max(upper - lower, 1e-9)
                y = value + span * 0.02
            va = "bottom"
            color = label_color or "black"
        elif log:
            y = value * 0.82
            va = "top"
            color = label_color or "white"
        else:
            span = max(upper - lower, 1e-9)
            y = value - span * 0.04
            va = "top"
            color = label_color or "white"
        text = label_builder(row, value) if label_builder is not None else label_format.format(value)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            text,
            ha="center",
            va=va,
            rotation=0,
            fontsize=8,
            color=color,
            fontweight="bold",
            linespacing=1.15,
            clip_on=False,
        )


def print_summary_table(rows: list[dict[str, Any]]) -> None:
    print("variant                 size MiB  raw-to-file  read+decode frames/s")
    print("----------------------  --------  -----------  --------------------")
    for row in rows:
        print(
            f"{row['variant']:<22}  {row['rowpack_mib']:>8.2f}  "
            f"{row['compression_ratio_x']:>10.2f}x  {row['read_decode_frames_per_s']:>20.1f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
